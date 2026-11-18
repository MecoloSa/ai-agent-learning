"""固定召回、真实工具循环和在线模型的文本/结构化交付对照。"""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import time
from unittest.mock import patch

from langchain_core.documents import Document

import ai_agent.main as app
import ai_agent.resources.tools as tools
from ai_agent.config import get_settings
from ai_agent.learning.learning_state import LearningState
from ai_agent.persistence.session_materials import create_material_binding
from ai_agent.persistence.state_store import save_learning_state, load_learning_state
from ai_agent.retrieval.rag import LearningRAG
from tests.helpers.learning_closed_loop import worker, formal

ROOT = Path(__file__).resolve().parents[2]
FIXTURE = ROOT / "tests/fixtures/evaluation/rag_source_delivery_cases.json"
OLD_RULE = "结构化读取的来源由程序在文末列出，无需复制引用标记；"
NEW_RULE = "本轮工具返回的资料来源由程序在文末列出，无需复制引用标记；"


def run_case(case, followup, arm, directory, material_dir):
    if not app.needs_rag(str(material_dir)):
        raise RuntimeError("实验前提不成立：资料规模未触发真实RAG路由，禁止调用在线模型")
    directory.mkdir(parents=True, exist_ok=False)
    fixed_docs = [Document(page_content=text, metadata={"source": name, "chunk_index": 0})
                  for name, text in case["materials"].items()]
    fixed_docs.append(Document(page_content="无关背景。" * 1000,
                              metadata={"source": "excluded.md", "chunk_index": 0}))
    retrievals = []

    class FixedRAG:
        def __init__(self, *args, **kwargs):
            pass

        def retrieve_many(self, query_pairs, **kwargs):
            retrievals.append({"queries": list(query_pairs), "options": {
                key: sorted(value) if isinstance(value, set) else value
                for key, value in kwargs.items()
            }})
            return [doc.model_copy(deep=True) for doc in fixed_docs]

        def expand_neighbors(self, hits, *, window):
            return hits

        def format_documents(self, docs, **kwargs):
            return LearningRAG.format_documents(None, docs, **kwargs)

    state = LearningState(learning_goal="依据资料讨论学习问题", plan_markdown="原有正式安排，保持不变。",
                         material_binding=create_material_binding(str(material_dir)))
    save_learning_state(state, directory / "source.json")
    spec = {"source": str(directory / "source.json"), "current": str(directory / "current.json"),
            "archives": str(directory / "archives"), "observation": str(directory / "observation.json"),
            "events": [{"stage": "chat", "text": case["question"]},
                       {"stage": "chat", "text": followup}, {"stage": "chat", "text": "exit"}]}
    (directory / "input.json").write_text(json.dumps({**spec, "case": case, "arm": arm},
                                        ensure_ascii=False, indent=2), encoding="utf-8")
    original_factory, original_prompt = app.build_retrieve_learning_context_tool, app._build_interactive_system_prompt

    def factory(*args, **kwargs):
        kwargs["structured_result"] = arm == "structured"
        return original_factory(*args, **kwargs)

    def prompt(*args, **kwargs):
        text = original_prompt(*args, **kwargs)
        if arm == "text":
            text = text.replace(NEW_RULE, OLD_RULE).replace(
                "预载资料仍沿用已有位置引用，不补造位置。",
                "检索片段和预载资料仍沿用已有位置引用，不补造位置。")
        return text

    settings = replace(get_settings(), interactive_retrieval_max_context_chars=900)
    started = time.perf_counter()
    with patch.object(tools, "LearningRAG", FixedRAG), patch.object(app, "get_settings", lambda: settings), \
         patch.object(app, "build_retrieve_learning_context_tool", factory), \
         patch.object(app, "_build_interactive_system_prompt", prompt):
        status = worker(spec)
    elapsed = round(time.perf_counter() - started, 3)
    return summarize(case, arm, directory, status, elapsed, retrievals)


def summarize(case, arm, directory, status, elapsed=None, retrievals=None):
    state = load_learning_state(directory / "source.json")
    observation = json.loads((directory / "observation.json").read_text(encoding="utf-8"))
    final = load_learning_state(directory / "current.json") if (directory / "current.json").exists() else state
    models = observation.get("models", [])
    retrieval_executed = any(
        r.get("tool_name") == "retrieve_learning_context" and r.get("executed")
        for turn in observation.get("turns", [])
        for r in turn.get("audit_trace", {}).get("tool_results", [])
    )
    usage_known = all(isinstance(m.get("usage"), dict) for m in models) and bool(models)
    row = {"case_id": case["case_id"], "arm": arm, "status": status,
           "elapsed_s": elapsed, "model_calls": len(models),
           "model_elapsed_s": round(sum(m["elapsed_ms"] for m in models) / 1000, 3),
           "input_tokens": sum(m["usage"]["input_tokens"] for m in models) if usage_known else None,
           "output_tokens": sum(m["usage"]["output_tokens"] for m in models) if usage_known else None,
           "state_unchanged": formal(final) == formal(state), "retrievals": retrievals,
           "retrieval_executed": retrieval_executed,
           "responses": [t.get("response") for t in observation.get("turns", [])],
           "observation": str(directory / "observation.json"), "quality": "pending_manual_review"}
    (directory / "result.json").write_text(json.dumps(row, ensure_ascii=False, indent=2), encoding="utf-8")
    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--llm", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    fixture = json.loads(FIXTURE.read_text(encoding="utf-8"))
    if not args.llm:
        print(json.dumps({"mode": "preflight_only", "cases": fixture, "order": ["text", "structured", "structured", "text"],
                          "scope": "固定召回，不调用embedding或重排；每会话两问，真实交互循环"}, ensure_ascii=False, indent=2))
        return
    args.output_dir.mkdir(parents=True, exist_ok=args.resume)
    rows = []
    for case in fixture["cases"]:
        material_dir = args.output_dir / case["case_id"] / "materials"
        material_dir.mkdir(parents=True, exist_ok=args.resume)
        for name, content in case["materials"].items():
            (material_dir / name).write_text(content, encoding="utf-8")
        # 让真实资料规模满足项目路由；该长块仍被固定900字符预算排除。
        # 不以修改生产路由或无效PDF伪装满足实验条件。
        (material_dir / "excluded.md").write_text(
            "A" * max(get_settings().rag_large_file_bytes, 10000), encoding="utf-8")
        for index, arm in enumerate(["text", "structured", "structured", "text"], 1):
            directory = material_dir.parent / f"{index}-{arm}"
            if args.resume and (directory / "observation.json").exists():
                saved_input = json.loads((directory / "input.json").read_text(encoding="utf-8"))
                if saved_input["case"] != case or saved_input["arm"] != arm:
                    raise ValueError("已保存案例条件不同，不能复用")
                # 只恢复报告生成，不重新调用已经完成的模型。
                row = summarize(case, arm, directory, 0)
            else:
                row = run_case(case, fixture["followup"], arm, directory, material_dir)
            rows.append(row)
            (args.output_dir / "summary.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
            print(json.dumps({k: row[k] for k in ["case_id", "arm", "status", "model_calls", "input_tokens", "output_tokens", "state_unchanged"]}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
