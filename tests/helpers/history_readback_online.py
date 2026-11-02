"""真实模型回查验收。默认仅展示案例；--llm才运行一个案例，不改业务源码。"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import hashlib
import json
from pathlib import Path
import time
from unittest.mock import patch
from uuid import uuid4

import ai_agent.main as app
from ai_agent.learning.learning_state import LearningState
from ai_agent.persistence.history_session import ArchiveCapture
from ai_agent.persistence.memory_state import HistoryArchiveState, MemorySnapshot
from ai_agent.persistence.session_materials import create_material_binding
from ai_agent.persistence.state_store import load_learning_state, save_learning_state

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CASES = ROOT / "tests/fixtures/evaluation/history_readback_online_cases.json"
SOURCE_FILES = (
    "src/ai_agent/main.py", "src/ai_agent/config.py", "src/ai_agent/persistence/history_archive.py",
    "src/ai_agent/persistence/history_session.py", "src/ai_agent/persistence/history_readback.py",
    "src/ai_agent/persistence/memory.py", "src/ai_agent/runtime/execution_guard.py",
    "src/ai_agent/retrieval/semantic_history.py", "src/ai_agent/retrieval/history_embedding_cache.py",
)


def load_case(path: Path, case_id: str):
    fixture = json.loads(path.read_text(encoding="utf-8"))
    if fixture.get("schema_version") != "history-readback-online-cases-v1":
        raise ValueError("不支持的案例格式")
    history = fixture["history"]
    cases = fixture["cases"]
    ids = [case["case_id"] for case in cases]
    if len(set(ids)) != len(ids):
        raise ValueError("案例ID重复")
    if not 2 <= len(history) <= 50 or any(not isinstance(t.get(role), str) or not t[role].strip()
                                for t in history for role in ("user", "assistant")):
        raise ValueError("本验收要求2至50轮非空合成问答")
    case = next((c for c in cases if c["case_id"] == case_id), None)
    if case is None:
        raise ValueError("未找到指定案例")
    if type(case["through_turn"]) is not int or not 1 <= case["through_turn"] <= len(history):
        raise ValueError("案例范围不合法")
    if not isinstance(case["question"], str) or not case["question"].strip():
        raise ValueError("案例问题不能为空")
    target = case.get("target_turn", 2)
    if type(target) is not int or not 1 <= target <= len(history):
        raise ValueError("目标轮次不合法")
    if "sources" in fixture:
        specs = fixture["sources"]
        if not isinstance(specs, list) or not 1 <= len(specs) <= 20:
            raise ValueError("多来源案例范围不合法")
        for spec in specs:
            turns = spec["history"]
            cap = spec["through_turn"]
            if (not turns or type(cap) is not int or not 1 <= cap <= len(turns)
                    or any(not isinstance(t.get(role), str) or not t[role].strip()
                           for t in turns for role in ("user", "assistant"))):
                raise ValueError("多来源案例问答或范围不合法")
        source = case.get("target_source", 1)
        if type(source) is not int or not 1 <= source <= len(specs):
            raise ValueError("目标来源不合法")
        # 兼容旧观察字段时，目标原话必须与真实目标来源完全一致。
        if history != specs[source - 1]["history"]:
            raise ValueError("目标历史与来源内容不一致")
    return fixture, case


def manifest(path: Path):
    settings = app.get_settings()
    source = hashlib.sha256()
    for name in SOURCE_FILES:
        source.update(name.encode("utf-8"))
        source_path = ROOT / name
        source.update(source_path.read_bytes() if source_path.is_file() else b"<not-applied>")
    return {
        "schema_version": "history-readback-online-v1",
        "cases_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "source_py_sha256": source.hexdigest(),
        "settings": {**{name: getattr(settings, name) for name in (
            "chat_model", "llm_temperature", "request_timeout", "llm_retry_attempts",
            "interactive_max_retrieval_calls",
        )}, "embedding_model": settings.embedding_model,
            "history_retrieval_strategy": getattr(settings, "history_retrieval_strategy", "bm25"),
            "history_semantic_min_similarity": getattr(settings, "history_semantic_min_similarity", None),
            "interactive_max_no_progress_calls": getattr(settings, "interactive_max_no_progress_calls", 3)},
        "profile": {"memory_strategy": "summary", "recent_messages": 0,
                    "read_transport": "local", "retrieval": "disabled_for_isolated_history_test"},
    }


def _formal_state(state):
    # 本例允许聊天记忆与归档增长，正式目标、计划、任务和反馈必须保持。
    return state.model_dump(mode="json", exclude={"conversation_memory"})


def summarize_tool_usage(rows):
    """只汇总已观察到的回执，不把executed=True等同于业务成功。"""
    sequence = []
    read_positions = []
    budget_denials = 0
    for row in rows:
        result = json.loads(row["result"])
        success = row["executed"] is True and result.get("ok") is True
        code = result.get("error", {}).get("code")
        sequence.append({"tool_name": row["tool_name"], "executed": row["executed"],
                         "result_reused": row.get("result_reused", False),
                         "operational_progress": row.get("operational_progress"),
                         "consecutive_no_progress": row.get("consecutive_no_progress"),
                         "ok": result.get("ok"), "error_code": code})
        budget_denials += code == "budget_exceeded"
        if success and row["tool_name"] == "read_conversation_turn":
            data = result["data"]
            read_positions.append({"source_index": data["source_index"],
                                   "turn_number": data["turn_number"]})
    return {"tool_sequence": sequence, "successful_read_positions": read_positions,
            "budget_denial_count": budget_denials}


def run_case(fixture, case, output: Path, conditions):
    run_id = uuid4().hex
    folder = output.parent / "history-readback-online-artifacts" / run_id
    folder.mkdir(parents=True)
    materials = folder / "materials"
    materials.mkdir()
    (materials / "notes.md").write_text("合成资料；本案例只核对历史对话，不要求引用学习资料。", encoding="utf-8")
    # 各来源独立建档；数据库存在的轮次仍受快照各自上限限制。
    specs = fixture.get("sources", [{"history": fixture["history"], "through_turn": case["through_turn"]}])
    refs = []
    for spec in specs:
        capture = ArchiveCapture(folder / "archives", None, True)
        for turn in spec["history"]:
            capture.capture(turn["user"], turn["assistant"])
        refs.append(capture.state.sources[0].model_copy(update={"through_turn": spec["through_turn"]}))
    ref = refs[case.get("target_source", 1) - 1]
    state = LearningState(
        learning_goal="核对历史原话", plan_markdown="原计划，保持不变。",
        material_binding=create_material_binding(str(materials)),
        conversation_memory=MemorySnapshot(
            strategy="summary", max_chars=2000, window_size=6,
            summary="此前讨论过练习安排；这里没有保留具体条件和逐字原话。",
            history_archive=HistoryArchiveState(enabled=True, sources=tuple(refs)),
        ),
    )
    source_path = save_learning_state(state, folder / "source.json")
    archive_paths = [folder / "archives" / f"{r.archive_id}.sqlite3" for r in refs]
    original_files = {path: path.read_bytes() for path in [source_path, *archive_paths]}
    current_path = folder / "current.json"
    observation_path = folder / "observation.json"
    observation = {
        "user_input": case["question"], "initial_state": state.model_dump(mode="json"),
        "case_spec": case, "fixture_history": fixture["history"], "fixture_sources": fixture.get("sources"),
        "input_events": [], "model_requests": [], "audit_trace": {},
    }
    # 自动化只替代人机输入与观察边界；真实LLM、工具、状态和保存代码保持执行。
    inputs = iter([case["question"], "exit"])
    def input_text(prompt):
        is_chat = prompt == "你："
        answer = next(inputs) if is_chat else "n"
        observation["input_events"].append({"prompt": prompt, "value": answer,
                                            "kind": "chat" if is_chat else "decline_unexpected_confirmation"})
        return answer

    original_model = app.invoke_llm_response
    def observed_model(llm, messages, *, stage):
        observation["model_requests"].append({"stage": stage, "request_messages": [
            {"role": m.type, "content": m.content, "tool_calls": getattr(m, "tool_calls", None),
             "tool_call_id": getattr(m, "tool_call_id", None)} for m in messages
        ]})
        return original_model(llm, messages, stage=stage)

    original_turn = app._invoke_interactive_turn
    def observed_turn(**kwargs):
        return original_turn(**kwargs, audit_trace=observation["audit_trace"])

    started = time.monotonic()
    error_type = None
    status = "completed"
    try:
        with ExitStack() as stack:
            stack.enter_context(patch("builtins.input", input_text))
            stack.enter_context(patch.object(app, "needs_rag", lambda _: False))
            stack.enter_context(patch.object(app, "invoke_llm_response", observed_model))
            stack.enter_context(patch.object(app, "_invoke_interactive_turn", observed_turn))
            app.run_interactive_session("", state.plan_markdown, "", state.learning_goal, None, str(materials),
                initial_state=load_learning_state(source_path), current_state_path=current_path,
                archive_history=None, history_archive_root=folder / "archives")
    except Exception as exc:
        status, error_type = "failed", type(exc).__name__

    business_unchanged = None
    saved_assistant_message = None
    inherited_scope_unchanged = None
    try:
        saved = load_learning_state(current_path)
        business_unchanged = _formal_state(saved) == _formal_state(state)
        if saved.conversation_memory is not None:
            if saved.conversation_memory.messages:
                saved_assistant_message = saved.conversation_memory.messages[-1].content
            saved_archive = saved.conversation_memory.history_archive
            inherited_scope_unchanged = saved_archive is not None and all(r in saved_archive.sources for r in refs)
    except (OSError, ValueError):
        pass
    source_unchanged = all(path.is_file() and path.read_bytes() == content for path, content in original_files.items())
    audit = observation["audit_trace"]
    # 保存消息已经过主流程清理及确认处理；保留循环原始返回，便于发现两者差异。
    final = saved_assistant_message
    responses = audit.get("model_responses", [])
    last_model_answer = responses[-1].get("content") if responses else None
    rows = audit.get("tool_results", [])
    reads = []
    scope_denials = []
    successful_lists = []
    for row in rows:
        if row["tool_name"] not in {"list_conversation_sources", "read_conversation_turn"} or not row["executed"]:
            continue
        result = json.loads(row["result"])
        if row["tool_name"] == "list_conversation_sources":
            if result.get("ok") is True:
                successful_lists.append(row)
            continue
        if result.get("ok") is True:
            reads.append(result["data"])
        elif result.get("error", {}).get("code") == "turn_out_of_scope":
            scope_denials.append(row)
    target_turn = case.get("target_turn", 2)
    quote = fixture["history"][target_turn - 1]["user"]
    first_request = observation["model_requests"][0]["request_messages"] if observation["model_requests"] else None
    row = {
        "run_id": run_id, "case_id": case["case_id"], "mode": "online_llm", "status": status,
        "error_type": error_type, "manifest": conditions, "business_state_unchanged": business_unchanged,
        "source_unchanged": source_unchanged, "inherited_scope_unchanged": inherited_scope_unchanged,
        "latency_ms": round((time.monotonic() - started) * 1000),
        "artifacts": str(folder), "observation_path": str(observation_path), "assessment": "not_evaluated",
        "observations": {
            **summarize_tool_usage(rows),
            "first_request_contains_target_quote": quote in json.dumps(first_request, ensure_ascii=False) if first_request else None,
            "successful_read_count": len(reads), "scope_denial_count": len(scope_denials),
            "successful_source_list_count": len(successful_lists),
            "requested_read_succeeded": any(d["source_index"] == case.get("target_source", 1) and d["turn_number"] == target_turn for d in reads),
            "exact_target_quote_in_answer": quote in final if isinstance(final, str) else None,
            "returned_citation_in_answer": any(d["citation"] in final for d in reads) if isinstance(final, str) else None,
            # 程序可能附上真实来源；最终展示命中不能冒充模型自己复制正确。
            "model_returned_citation_in_answer": any(d["citation"] in last_model_answer for d in reads) if isinstance(last_model_answer, str) else None,
            "model_returned_reference_in_answer": any(d.get("citation_ref") and d["citation_ref"] in last_model_answer for d in reads) if isinstance(last_model_answer, str) else None,
            "contains_history_citation_notice": "【历史引用核验】" in final if isinstance(final, str) else None,
            "unexpected_confirmation_count": sum(e["kind"] != "chat" for e in observation["input_events"]),
        },
        "turn_answer": audit.get("final_answer"), "final_answer": final,
    }
    observation["result"] = row
    observation_path.write_text(json.dumps(observation, ensure_ascii=False, indent=2), encoding="utf-8")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    return row


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--case-id", default="history-readback-online-001")
    parser.add_argument("--output", type=Path, default=ROOT / ".evaluation-results/history-readback-online-v1.jsonl")
    parser.add_argument("--llm", action="store_true")
    args = parser.parse_args(argv)
    fixture, case = load_case(args.cases.resolve(), args.case_id)
    conditions = manifest(args.cases.resolve())
    if not args.llm:
        print(json.dumps({"mode": "preflight_only", "case": case, "fixture_history": fixture["history"], "fixture_sources": fixture.get("sources"),
                          "initial_context": {"messages": [], "summary": "此前讨论过练习安排；这里没有保留具体条件和逐字原话。",
                                              "snapshot_through_turn": case["through_turn"], "database_turns": len(fixture["history"])},
                          "automatic_input": [case["question"], "exit"],
                          "confirmation_policy": "任何意外确认均输入n并记录，不自动批准。",
                          "manifest": conditions}, ensure_ascii=False, indent=2))
        return 0
    row = run_case(fixture, case, args.output.resolve(), conditions)
    print(json.dumps(row, ensure_ascii=False, indent=2))
    return 0 if (row["status"] == "completed" and row["business_state_unchanged"] is True
                 and row["source_unchanged"] and row["inherited_scope_unchanged"] is True) else 1


if __name__ == "__main__":
    raise SystemExit(main())
