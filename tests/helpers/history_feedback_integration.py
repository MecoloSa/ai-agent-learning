"""历史回查→候选→确认→独立进程恢复；默认仅预检，--llm才真实生成。"""
from __future__ import annotations

import argparse
from contextlib import ExitStack, redirect_stdout
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
from unittest.mock import patch
from uuid import uuid4

import ai_agent.main as app
from ai_agent.evaluation.feedback_plan_input import ReplayInput, ReplayMismatch
from ai_agent.learning import confirmation
from ai_agent.learning.learning_state import LearningState
from ai_agent.persistence.history_session import ArchiveCapture
from ai_agent.persistence.memory_state import MemorySnapshot
from ai_agent.persistence.session_materials import create_material_binding
from ai_agent.persistence.state_store import load_learning_state, save_learning_state

ROOT = Path(__file__).resolve().parents[2]
FIXTURE = ROOT / "tests/fixtures/evaluation/history_feedback_integration.json"


def formal(state):
    return state.model_dump(mode="json", exclude={"conversation_memory"})


def load_fixture(path=FIXTURE):
    data = json.loads(path.read_text(encoding="utf-8"))
    if data["schema_version"] != "history-feedback-integration-v1":
        raise ValueError("案例版本不符")
    initial_path = (path.parent / data["initial_state"]).resolve()
    if not initial_path.is_relative_to(path.parent.resolve()):
        raise ValueError("初始状态越出案例目录")
    return data, load_learning_state(initial_path)


def resume_in_process(folder, source, archive_root, question):
    folder.mkdir()
    current, observation = folder / "current.json", folder / "request.json"
    spec = {"source": str(source), "current": str(current), "observation": str(observation),
            "history_root": str(archive_root), "inputs": [question, "exit"],
            "response": "这里只核对恢复记录，不产生候选或反馈。"}
    spec_path = folder / "input.json"
    spec_path.write_text(json.dumps(spec, ensure_ascii=False, indent=2), encoding="utf-8")
    completed = subprocess.run([sys.executable, "-X", "utf8", "-B",
        str(ROOT / "tests/helpers/feedback_plan_process_worker.py"), str(spec_path)],
        cwd=ROOT, capture_output=True, text=True, encoding="utf-8", timeout=60)
    (folder / "console.txt").write_text(completed.stdout + completed.stderr, encoding="utf-8")
    if completed.returncode != 0 or not current.is_file():
        raise RuntimeError("独立进程恢复失败")
    requests = json.loads(observation.read_text(encoding="utf-8"))
    return load_learning_state(current), requests


def run_case(folder, decision, *, resume=True):
    fixture, base = load_fixture()
    folder.mkdir(parents=True)
    materials, archives = folder / "materials", folder / "archives"
    materials.mkdir()
    (materials / "notes.md").write_text("合成评测资料，本轮检查历史条件与计划调整接线。", encoding="utf-8")
    capture = ArchiveCapture(archives, None, True)
    for turn in fixture["history"]:
        capture.capture(turn["user"], turn["assistant"])
    state = LearningState.model_validate({**base.model_dump(),
        "material_binding": create_material_binding(str(materials)).model_dump(),
        "conversation_memory": MemorySnapshot(strategy="summary", max_chars=20000, window_size=6,
            summary="此前讨论过学习安排，具体限制需查询历史原文。",
            history_archive=capture.state).model_dump()})
    source, current = save_learning_state(state, folder / "source.json"), folder / "current.json"
    original_files = {p: p.read_bytes() for p in [source, *archives.glob("*.sqlite3")]}
    reader = ReplayInput([fixture["question"], "v", "y" if decision == "adopt" else "n", "exit"])
    result = {"status": "running", "error_type": None, "decision_script": decision,
              "fixture": fixture, "initial_state": state.model_dump(mode="json"),
              "turns": [], "model_requests": [], "confirmations": [], "plan_changes": [],
              "input_events": [], "checks": {}, "assessment": "not_evaluated"}
    original_turn, original_model = app._invoke_interactive_turn, app.invoke_llm_response
    original_review, original_confirmation = app._review_context_change, app.read_confirmation

    def invoke_turn(**kwargs):
        trace = {}
        result["turns"].append({"user_input": kwargs["user_input"],
            "request_messages": [{"role": m.type, "content": m.content} for m in kwargs["request_messages"]],
            "audit_trace": trace})
        return original_turn(**kwargs, audit_trace=trace)

    def invoke_model(llm, messages, *, stage):
        result["model_requests"].append({"stage": stage, "request_messages": [
            {"role": m.type, "content": m.content, "tool_calls": getattr(m, "tool_calls", None),
             "tool_call_id": getattr(m, "tool_call_id", None)} for m in messages]})
        return original_model(llm, messages, stage=stage)

    def confirm(*args, **kwargs):
        with patch.object(confirmation, "input", lambda _: reader.read_plan_confirmation(), create=True):
            accepted = original_confirmation(*args, **kwargs)
        result["confirmations"].append({"accepted": accepted, "reason": (kwargs.get("trace") or {}).get("reason")})
        return accepted

    def review(before, **kwargs):
        event = {"proposed_plan": kwargs.get("plan_markdown"), "proposed_goal": kwargs.get("learning_goal"),
                 "decision": None, "confirmation_reason": None}
        result["plan_changes"].append(event)
        after, actual = original_review(before, **kwargs)
        event.update(decision=actual, confirmation_reason=(kwargs.get("confirmation_trace") or {}).get("reason"))
        return after, actual

    def unexpected(*args, **kwargs):
        raise ReplayMismatch("出现非预期输入入口")

    start = time.monotonic()
    try:
        with (folder / "console.txt").open("x", encoding="utf-8") as console, ExitStack() as stack:
            stack.enter_context(redirect_stdout(console))
            stack.enter_context(patch("builtins.input", unexpected))
            stack.enter_context(patch.object(app, "input", lambda _: reader.read_chat(), create=True))
            stack.enter_context(patch.object(app, "read_confirmation", confirm))
            stack.enter_context(patch.object(app, "_review_context_change", review))
            stack.enter_context(patch.object(app, "_invoke_interactive_turn", invoke_turn))
            stack.enter_context(patch.object(app, "invoke_llm_response", invoke_model))
            stack.enter_context(patch.object(app, "needs_rag", lambda _: False))
            app.run_interactive_session("", state.plan_markdown, "", state.learning_goal, None,
                str(materials), initial_state=load_learning_state(source), current_state_path=current,
                history_archive_root=archives)
        reader.assert_finished()
        saved = load_learning_state(current)
        result["final_state"] = saved.model_dump(mode="json")
        events, confirmations = result["plan_changes"], result["confirmations"]
        adopted = decision == "adopt"
        result["checks"].update(
            confirmation_matches=len(confirmations) == 1 and confirmations[0] == {
                "accepted": adopted, "reason": "user_confirmed" if adopted else "user_declined"},
            candidate_matches=len(events) == 1 and events[0]["decision"] == ("confirmed" if adopted else "cancelled"),
            records_preserved=all(getattr(saved, key) == getattr(state, key) for key in (
                "learning_goal", "initial_learning_goal", "initial_plan_markdown", "tasks", "feedback_records", "material_binding")),
            plan_persisted=len(events) == 1 and saved.plan_markdown == (events[0]["proposed_plan"] if adopted else state.plan_markdown),
            revision_matches=saved.context_revision == state.context_revision + int(adopted),
            review_flag_matches=saved.tasks_need_review is adopted,
            business_state_unchanged=formal(saved) == formal(state))
        tools = [r for t in result["turns"] for r in t["audit_trace"]["tool_results"]]
        target_quote = fixture["history"][fixture["target_turn"] - 1]["user"]
        reads = [json.loads(r["result"]) for r in tools if r["tool_name"] == "read_conversation_turn" and r["executed"]]
        result["checks"]["latest_history_actually_read"] = any(r.get("ok") is True and r["data"].get("user_text") == target_quote for r in reads)
        initial_messages = result["turns"][0]["request_messages"]
        result["checks"]["target_quote_absent_initial_context"] = target_quote not in json.dumps(initial_messages, ensure_ascii=False)
        result["checks"]["archive_scope_preserved"] = all(s in saved.conversation_memory.history_archive.sources for s in capture.state.sources)
        first_bytes = current.read_bytes()
        if resume:
            archive_before_resume = {p: p.read_bytes() for p in archives.glob("*.sqlite3")}
            restored, requests = resume_in_process(folder / "resume", current, archives, fixture["resume_question"])
            result["resume_requests"] = requests
            # 新进程模型是替身，只检查真正恢复的数据和消息，不评自然语言复述质量。
            result["checks"]["resume_state_matches"] = formal(restored) == formal(saved)
            result["checks"]["resume_source_unchanged"] = current.read_bytes() == first_bytes
            result["checks"]["resume_archives_unchanged"] = all(p.read_bytes() == content for p, content in archive_before_resume.items())
            result["checks"]["resume_archive_scope_preserved"] = all(s in restored.conversation_memory.history_archive.sources for s in saved.conversation_memory.history_archive.sources)
            views = [json.loads(m["content"]) for m in requests[0]["request_messages"]
                     if m["type"] == "human" and isinstance(m["content"], str)
                     and m["content"].startswith('{"context_type": "current_learning_state"')]
            result["checks"]["resume_input_matches"] = len(requests) == 1 and len(views) == 1 and views[0]["plan_markdown"] == saved.plan_markdown and views[0]["learning_goal"] == saved.learning_goal and views[0]["tasks"] == [t.model_dump(mode="json") for t in saved.tasks] and views[0]["context_revision"] == saved.context_revision
            result["checks"]["resume_confirmation_context_present"] = any(m["type"] == "ai" and ("confirmed" if adopted else "cancelled") in m["content"] for m in requests[0]["request_messages"])
        result["status"] = "completed"
    except Exception as exc:
        result.update(status="script_mismatch" if isinstance(exc, ReplayMismatch) else "failed", error_type=type(exc).__name__)
    # 脚本不匹配也可能已经保存普通问答；单独观察真实落盘状态，
    # 不能因整条链路未完成，就丢掉“正式记录是否改变”的事实。
    if "final_state" not in result and current.is_file():
        try:
            saved = load_learning_state(current)
            result["final_state"] = saved.model_dump(mode="json")
            result["checks"]["business_state_unchanged"] = formal(saved) == formal(state)
            result["checks"]["records_preserved"] = all(getattr(saved, key) == getattr(state, key) for key in (
                "learning_goal", "initial_learning_goal", "initial_plan_markdown", "tasks", "feedback_records", "material_binding"))
        except (OSError, ValueError) as exc:
            result["state_observation_error"] = type(exc).__name__
    result["checks"]["source_unchanged"] = all(p.is_file() and p.read_bytes() == content for p, content in original_files.items())
    result["input_events"] = reader.events
    result["latency_ms"] = round((time.monotonic() - start) * 1000)
    (folder / "observation.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--decision", choices=("adopt", "decline"), default="adopt")
    parser.add_argument("--output", type=Path, default=ROOT / ".evaluation-results/history-feedback-integration-20261031.json")
    parser.add_argument("--llm", action="store_true")
    args = parser.parse_args(argv)
    fixture, state = load_fixture()
    settings = app.get_settings()
    manifest = {"fixture_sha256": hashlib.sha256(FIXTURE.read_bytes()).hexdigest(),
                "helper_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "chat_model": settings.chat_model, "temperature": settings.llm_temperature,
                "history_strategy": settings.history_retrieval_strategy,
                "history_threshold": settings.history_semantic_min_similarity,
                "tool_budget": settings.interactive_max_retrieval_calls,
                "no_progress_limit": settings.interactive_max_no_progress_calls}
    files = sorted((ROOT / "src/ai_agent").rglob("*.py"))
    manifest["source_py_sha256"] = hashlib.sha256(json.dumps([
        (str(p.relative_to(ROOT)), hashlib.sha256(p.read_bytes()).hexdigest()) for p in files
    ]).encode()).hexdigest()
    manifest["initial_state_sha256"] = hashlib.sha256((FIXTURE.parent / fixture["initial_state"]).read_bytes()).hexdigest()
    manifest["profile"] = {"rag": "disabled", "memory_strategy": "summary", "trigger_chars": 20000,
                           "recent_messages": 0, "resume_model": "stub", "read_transport": "local"}
    if not args.llm:
        print(json.dumps({"mode": "preflight_only", "manifest": manifest, "fixture": fixture,
            "initial_state": state.model_dump(mode="json"),
            "automatic_inputs": [fixture["question"], "v", "y" if args.decision == "adopt" else "n", "exit"],
            "resume": "新进程恢复；模型替身，仅验证数据和请求，不评价在线复述。"}, ensure_ascii=False, indent=2))
        return 0
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as output:
        folder = args.output.parent / "history-feedback-integration-artifacts" / uuid4().hex
        result = run_case(folder, args.decision)
        result.update(manifest=manifest, artifacts=str(folder))
        json.dump(result, output, ensure_ascii=False, indent=2)
    # 采用时business_state_unchanged=false是预期事实，不是保护失败。
    controls = {k: v for k, v in result["checks"].items() if k != "business_state_unchanged"}
    print(json.dumps({"status": result["status"], "error_type": result["error_type"], "checks": result["checks"],
                      "output": str(args.output)}, ensure_ascii=False, indent=2))
    return 0 if result["status"] == "completed" and controls and all(v is True for v in controls.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
