"""真实压缩与独立进程恢复验收；默认只预检，--llm才执行。"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import json
import hashlib
from pathlib import Path
import subprocess
import sys
import time
from unittest.mock import patch
from uuid import uuid4

import ai_agent.main as app
import ai_agent.persistence.memory as memory_module
from ai_agent.learning.learning_state import LearningState
from ai_agent.persistence.memory_state import MemorySnapshot
from ai_agent.persistence.session_materials import create_material_binding
from ai_agent.persistence.state_store import load_learning_state, save_learning_state

ROOT = Path(__file__).resolve().parents[2]
INPUTS = [
    "今天最多20分钟，只阅读和口头说明，不绘图、不编程。这里只讨论，不修改任何正式记录。",
    "更正：今天最多35分钟，其他条件保持。这里只更正对话条件，不修改任何计划或记录。",
    "请解释活动完成和已经理解为什么需要分开记录。只讨论，不更新记录。",
]
QUESTION = "请在来源1的历史用户消息中搜索“更正”，再读取命中的完整问答，逐字引用用户更正时间的完整原话并保留历史引用。不要修改任何计划、任务或反馈。"


def message_data(messages):
    return [{"role": m.type, "content": m.content, "tool_calls": getattr(m, "tool_calls", None),
             "tool_call_id": getattr(m, "tool_call_id", None)} for m in messages]


def worker(spec):
    state = load_learning_state(Path(spec["source"]))
    observation = {"inputs": spec["inputs"], "input_events": [], "model_requests": [],
                   "summary_requests": [], "compression_checks": [], "turns": []}
    inputs = iter([*spec["inputs"], "exit"])
    def read_input(prompt):
        value = next(inputs) if prompt == "你：" else "n"
        observation["input_events"].append({"prompt": prompt, "value": value,
                                            "is_chat": prompt == "你："})
        return value
    original_model = app.invoke_llm_response
    def model(llm, messages, *, stage):
        print("模型请求：" + stage, flush=True)
        observation["model_requests"].append({"stage": stage, "request_messages": message_data(messages)})
        return original_model(llm, messages, stage=stage)
    original_summary = memory_module._request_summary
    def summary(messages):
        print("摘要请求开始", flush=True)
        record = {"request_messages": message_data(messages)}
        observation["summary_requests"].append(record)
        try:
            response = original_summary(messages)
            record["content"] = response.content
            record["usage"] = getattr(response, "usage_metadata", None)
            return response
        except Exception as exc:
            record["error_type"] = type(exc).__name__
            raise
    original_compress = memory_module.ConversationMemory._maybe_compress
    def compress(memory):
        before = len(memory.messages)
        try:
            return original_compress(memory)
        finally:
            observation["compression_checks"].append({"before_messages": before,
                "after_messages": len(memory.messages), "adopted": len(memory.messages) < before,
                "summary_chars": len(memory.summary)})
    original_turn = app._invoke_interactive_turn
    def turn(**kwargs):
        trace = {}
        observation["turns"].append({"user_input": kwargs["user_input"], "audit_trace": trace})
        return original_turn(**kwargs, audit_trace=trace)
    try:
        with ExitStack() as stack:
            stack.enter_context(patch("builtins.input", read_input))
            stack.enter_context(patch.object(app, "needs_rag", lambda _: False))
            stack.enter_context(patch.object(app, "invoke_llm_response", model))
            stack.enter_context(patch.object(app, "_invoke_interactive_turn", turn))
            stack.enter_context(patch.object(memory_module, "_request_summary", summary))
            stack.enter_context(patch.object(memory_module.ConversationMemory, "_maybe_compress", compress))
            app.run_interactive_session("", state.plan_markdown, "", state.learning_goal, None,
                state.material_binding.root_dir, initial_state=state,
                current_state_path=Path(spec["current"]), archive_history=True,
                history_archive_root=Path(spec["archives"]))
        observation["status"] = "completed"
    except Exception as exc:
        observation.update(status="failed", error_type=type(exc).__name__)
    Path(spec["observation"]).write_text(json.dumps(observation, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0 if observation["status"] == "completed" else 1


def launch(spec, folder, name):
    path = folder / (name + "-input.json")
    path.write_text(json.dumps(spec, ensure_ascii=False), encoding="utf-8")
    # 子进程输出写入各自日志；父进程每阶段及时给出进度。
    with (folder / (name + "-console.log")).open("w", encoding="utf-8") as stream:
        return subprocess.run([sys.executable, "-X", "utf8", str(Path(__file__).resolve()),
            "--worker", str(path), "--llm"], cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT).returncode


def formal(state):
    return state.model_dump(mode="json", exclude={"conversation_memory"})


def run(output):
    run_id = uuid4().hex
    folder = output.parent / "history-lifecycle-artifacts" / run_id
    folder.mkdir(parents=True)
    materials = folder / "materials"
    materials.mkdir()
    (materials / "notes.md").write_text("合成资料；本例只核对对话历史。", encoding="utf-8")
    initial = LearningState(learning_goal="核对学习约定", plan_markdown="原计划，保持不变。",
        material_binding=create_material_binding(str(materials)),
        conversation_memory=MemorySnapshot(strategy="summary", max_chars=1, window_size=6))
    source = save_learning_state(initial, folder / "seed.json")
    first_path, resumed_path = folder / "first.json", folder / "resumed.json"
    common = {"archives": str(folder / "archives")}
    print("阶段1：独立进程进行三轮真实问答与摘要，结果保存在 " + str(folder), flush=True)
    first_code = launch({**common, "source": str(source), "current": str(first_path),
                        "inputs": INPUTS, "observation": str(folder / "first-observation.json")}, folder, "first")
    if first_code != 0:
        raise RuntimeError("第一阶段未完成，请检查first-observation.json和first-console.log")
    saved = load_learning_state(first_path)
    ref = saved.conversation_memory.history_archive.sources[0]
    db = folder / "archives" / (ref.archive_id + ".sqlite3")
    before = {path: path.read_bytes() for path in (first_path, db)}
    print("阶段2：新进程从磁盘恢复，搜索并完整回读原话。", flush=True)
    second_code = launch({**common, "source": str(first_path), "current": str(resumed_path),
                         "inputs": [QUESTION], "observation": str(folder / "resumed-observation.json")}, folder, "resumed")
    if second_code != 0:
        raise RuntimeError("恢复阶段未完成，请检查resumed-observation.json和resumed-console.log")
    after = load_learning_state(resumed_path)
    first = json.loads((folder / "first-observation.json").read_text(encoding="utf-8"))
    second = json.loads((folder / "resumed-observation.json").read_text(encoding="utf-8"))
    answer = after.conversation_memory.messages[-1].content
    tools = second["turns"][0]["audit_trace"]["tool_results"]
    reads = [json.loads(t["result"])["data"] for t in tools if t["tool_name"] == "read_conversation_turn"
             and json.loads(t["result"]).get("ok") is True]
    quote = INPUTS[1]
    row = {"run_id": run_id, "case_id": "history-lifecycle-online-007", "status": "completed",
        "manifest": {"summary_trigger_chars": 1, "separate_processes": True,
            "settings": {name: getattr(app.get_settings(), name) for name in
                         ("chat_model", "embedding_model", "llm_temperature", "request_timeout", "interactive_max_retrieval_calls", "memory_summary_max_chars")},
            "history_config": {
                "strategy": getattr(app.get_settings(), "history_retrieval_strategy", "bm25"),
                "min_similarity": getattr(app.get_settings(), "history_semantic_min_similarity", None),
            },
            "source_sha256": hashlib.sha256(b''.join(p.read_bytes() for p in sorted((ROOT/'src/ai_agent').rglob('*.py')))).hexdigest()},
        "artifacts": str(folder), "assessment": "not_evaluated", "final_answer": answer,
        "observations": {
            "first_stage_adopted_compressions": sum(x["adopted"] for x in first["compression_checks"]),
            "target_quote_in_saved_recent_messages": quote in str(saved.conversation_memory.messages),
            "target_quote_in_real_summary": quote in saved.conversation_memory.summary,
            "target_quote_in_resumed_first_request": quote in json.dumps(second["model_requests"][0]["request_messages"], ensure_ascii=False),
            "target_quote_in_answer": quote in answer,
            "target_turn_read_succeeded": any(d["turn_number"] == 2 and d["user_text"] == quote for d in reads),
            "returned_citation_in_answer": any(d["citation"] in answer for d in reads),
            "business_state_unchanged": formal(initial) == formal(saved) == formal(after),
            "parent_source_unchanged": all(path.read_bytes() == content for path, content in before.items()),
            "inherited_scope_unchanged": ref in after.conversation_memory.history_archive.sources,
            "unexpected_confirmations": sum(not e["is_chat"] for obs in (first, second) for e in obs["input_events"]),
        }, "tool_names": [t["tool_name"] for t in tools]}
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    return row


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--llm", action="store_true")
    parser.add_argument("--worker", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--output", type=Path, default=ROOT / ".evaluation-results/history-lifecycle-online-v1.jsonl")
    args = parser.parse_args(argv)
    if args.worker:
        if not args.llm:
            parser.error("worker也必须显式提供--llm")
        return worker(json.loads(args.worker.read_text(encoding="utf-8")))
    if not args.llm:
        print(json.dumps({"mode": "preflight_only", "first_process_inputs": [*INPUTS, "exit"],
            "second_process_inputs": [QUESTION, "exit"], "summary_trigger_chars": 1,
            "real_summary": True, "independent_processes": True,
            "note": "真实摘要可能保留原话；实际是否保留将单独观察，不人为修改摘要。"}, ensure_ascii=False, indent=2))
        return 0
    started = time.monotonic()
    row = run(args.output.resolve())
    print(json.dumps(row, ensure_ascii=False, indent=2))
    print("总耗时秒：" + str(round(time.monotonic() - started, 2)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
