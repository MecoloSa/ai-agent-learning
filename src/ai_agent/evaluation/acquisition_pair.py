"""获取限制配对实验：真实规划和聊天模型，冻结检索结果以隔离交接信息变量。

默认仅准备输入；--llm 才调用模型。不是端到端RAG召回评测，不自动判内容合格。
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time
from unittest.mock import patch

from langchain_core.documents import Document

from ai_agent.config import get_settings
from ai_agent.evaluation.interactive_metrics import summarize_chat_usage
from ai_agent.resources.materials import safe_material_path
import ai_agent.resources.tools as tools
from ai_agent.retrieval.rag import LearningRAG
import ai_agent.runtime.agents as agents
from ai_agent.runtime.context_handoff import pack_material_context


def digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def frozen_rag(documents: list[Document], calls: list | None = None):
    """保留真实格式化；仅替换索引/召回，避免向量和重排变化污染比较。"""
    class FrozenRAG:
        format_documents = LearningRAG.format_documents

        def __init__(self, *args, **kwargs):
            pass

        def retrieve_many(self, pairs, **kwargs):
            if calls is not None:
                calls.append({"queries": list(pairs), "returned_documents": len(documents)})
            return documents

        def expand_neighbors(self, hits, *, window):
            return hits

    return FrozenRAG


def load_documents(root: Path, sources: list[str]) -> list[Document]:
    return [Document(
        page_content=safe_material_path(str(root), source).read_text(encoding="utf-8"),
        metadata={"source": source, "chunk_index": 0, "retrieval_role": "hit"},
    ) for source in sources]


def prepare_case(case: dict, root: Path) -> dict:
    documents = load_documents(root, case["initial_sources"])
    budget = (
        len(LearningRAG.format_documents(None, documents[:1], max_context_chars=100000))
        if case["initial_budget"] == "first_block" else 16000
    )
    blocks, facts, initial_calls = [], [], []
    with patch.object(tools, "LearningRAG", frozen_rag(documents, initial_calls)):
        initial_tool = tools.build_retrieve_learning_context_tool(
            str(root), max_queries_per_call=case["initial_query_limit"],
            max_context_chars=budget, evidence_blocks=blocks, acquisition_facts=facts,
        )
        initial_result = initial_tool.invoke({"queries": case["initial_queries"], "neighbor_window": 0})
    evidence = "\n\n---\n\n".join(blocks)
    return {
        **case, "evidence": evidence, "acquisition_facts": facts,
        "evidence_sha256": digest(evidence), "initial_budget_chars": budget,
        "initial_tool_result": initial_result, "initial_tool_calls": initial_calls,
        "materials": [{"source": d.metadata["source"], "text": d.page_content} for d in documents],
        "supplement_materials": [{"source": d.metadata["source"], "text": d.page_content} for d in load_documents(root, case["supplement_sources"])],
    }


def message_record(message) -> dict:
    row = {"role": message.type, "content": message.content}
    if getattr(message, "tool_calls", None):
        row["tool_calls"] = message.tool_calls
    if getattr(message, "tool_call_id", None):
        row["tool_call_id"] = message.tool_call_id
    return row


def run_one(case: dict, root: Path, arm: str, repeat: int) -> dict:
    calls, responses, requests, trace = [], [], [], {}
    facts = case["acquisition_facts"] if arm == "with_facts" else []
    context = pack_material_context(case["evidence"], facts)
    invoke_original = agents._invoke_llm_raw

    def capture(llm, messages, *, stage):
        request = {"stage": stage, "messages": [message_record(m) for m in messages]}
        requests.append(request)
        started = time.perf_counter()
        response = invoke_original(llm, messages, stage=stage)
        # 不采集 reasoning_content 等私有推理字段；只保存公开回答和工具动作。
        responses.append({
            "stage": stage, "content": response.content,
            "tool_calls": response.tool_calls or [],
            "usage": response.usage_metadata or {},
            "latency_ms": round((time.perf_counter() - started) * 1000),
        })
        return response

    row = {"case_id": case["case_id"], "arm": arm, "repeat": repeat,
           "user_input": case["goal"], "evidence_sha256": case["evidence_sha256"],
           "acquisition_facts": facts}
    started = time.perf_counter()
    try:
        with patch.object(tools, "LearningRAG", frozen_rag(load_documents(root, case["supplement_sources"]), calls)), patch.object(agents, "needs_rag", lambda _: True), patch.object(agents, "_invoke_llm_raw", capture):
            row["answer"] = agents.create_study_plan(
                case["analysis"], context, case["goal"], 1,
                material_dir=str(root), mode="pipeline", plan_evidence_trace=trace,
            )
        row["status"] = "completed"
    except Exception as exc:
        # 避免把可能包含服务细节的异常原文写入公开实验材料。
        row.update(status="failed", error_type=type(exc).__name__)
    row.update(
        latency_ms=round((time.perf_counter() - started) * 1000),
        request_messages=requests, model_responses=responses,
        retrieval_calls=calls, plan_trace=trace,
        usage=summarize_chat_usage({"model_responses": responses}),
    )
    return row


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, default=Path("tests/fixtures/evaluation/acquisition_pair/cases.json"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--llm", action="store_true")
    args = parser.parse_args(argv)
    if args.repeats < 1 or args.repeats > 2:
        parser.error("初步实验仅允许1或2轮，不盲目扩大请求数")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results_path = args.output_dir / "results.jsonl"
    if results_path.exists():
        parser.error("结果文件已存在，请使用新输出目录，避免覆盖旧实验")
    definition = json.loads(args.cases.read_text(encoding="utf-8"))
    root = args.cases.parent / "materials"
    cases = [prepare_case(case, root) for case in definition["cases"]]
    settings = get_settings()
    source_files = [Path(agents.__file__), Path(tools.__file__), Path(LearningRAG.format_documents.__code__.co_filename)]
    import ai_agent.runtime.context_handoff as handoff
    source_files.extend([Path(handoff.__file__), Path(__file__)])
    source_hash = digest("".join(path.read_text(encoding="utf-8") for path in source_files))
    manifest = {
        "schema_version": "acquisition-pair-v1", "mode": "online" if args.llm else "preflight_only",
        "chat_model": settings.chat_model, "temperature": settings.llm_temperature,
        "timeout_s": settings.request_timeout, "repeats": args.repeats,
        "cases_sha256": digest(args.cases.read_text(encoding="utf-8")),
        "source_sha256": source_hash,
        "retrieval": "frozen_local_results", "plan_mode": "pipeline", "max_supplement_calls": 1,
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    (args.output_dir / "inputs.json").write_text(json.dumps(cases, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False), flush=True)
    if not args.llm:
        return 0
    with results_path.open("w", encoding="utf-8") as stream:
        for case in cases:
            for repeat in range(1, args.repeats + 1):
                # 对调AB顺序，降低先后顺序与偶发时延的干扰；每次均独立消息历史。
                order = ["without_facts", "with_facts"] if repeat % 2 else ["with_facts", "without_facts"]
                for arm in order:
                    row = run_one(case, root, arm, repeat)
                    row["source_unchanged"] = source_hash == digest("".join(path.read_text(encoding="utf-8") for path in source_files))
                    stream.write(json.dumps(row, ensure_ascii=False) + "\n")
                    stream.flush()
                    print(f"{case['case_id']} repeat={repeat} {arm}: {row['status']} retrievals={len(row['retrieval_calls'])} latency_ms={row['latency_ms']}", flush=True)
                    if row["status"] != "completed" or not row["source_unchanged"]:
                        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
