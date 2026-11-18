"""真实分析/规划/交互/独立进程恢复；默认预检，不替换模型决策。"""
import argparse
import builtins
from contextlib import ExitStack
import json
from pathlib import Path
import subprocess
import sys
import time
from unittest.mock import patch

import ai_agent.main as app
import ai_agent.runtime.agents as agents
from ai_agent.learning.learning_state import LearningState
from ai_agent.persistence.session_materials import create_material_binding
from ai_agent.persistence.state_store import save_learning_state, load_learning_state
from ai_agent.evaluation.acquisition_pair import message_record
from tests.helpers.receipt_view_compare import fingerprint

ROOT = Path(__file__).resolve().parents[2]
CASES = [
    {"case_id": "technical", "goal": "理解正常结束与保护性停止的区别",
     "material": "正常结束指模型未再请求工具，程序结束循环，不证明回答正确。\n保护性停止指程序因预算或执行保护终止，不证明学习目标已完成。\n可通过阅读定义、比较例子和口头解释区别进行学习。\n"},
    {"case_id": "literature", "goal": "区分叙述视角与人物态度",
     "material": "叙述视角指谁在讲述以及读者能接触哪些信息；人物态度指角色对事情的看法。\n“我看到她把信藏了起来，却不知道她为何这样做。”体现叙述者的信息限制，不能据此认定她的真实动机。\n可以阅读例句、比较解释并口头说明依据；不要求编程、绘图或书面产出。\n"},
]
DISCUSSION = "今天最多20分钟，只阅读和口头说明。请依据已记录的困难讨论如何学习，只讨论，不修改计划、任务和反馈。"
CANDIDATE = "请依据已有反馈调整后续学习计划，总计最多20分钟，只阅读和口头说明，保留学习目标和任务反馈，给出完整候选计划供我确认。"
RESUME = "请复述当前生效计划及read-001的活动进度和理解反馈，并从来源1的历史中搜索“今天最多20分钟”，读取命中的完整问答后逐字引用我的限制原话。不要修改任何计划、任务或反馈。"


def formal(state):
    return state.model_dump(mode="json", exclude={"conversation_memory"})


def events():
    commands = [
        '/task {"task_id":"read-001","title":"阅读资料并口头比较"}',
        '/feedback {"task_id":"read-001","text":"阅读已完成，但我仍分不清两个概念。","activity_status":"completed","understanding_status":"self_reported_unclear"}',
        DISCUSSION, CANDIDATE,
    ]
    return [{"stage": "chat", "text": text} for text in commands] + [
        {"stage": "confirm", "text": "n"}, {"stage": "chat", "text": CANDIDATE},
        {"stage": "confirm", "text": "y"}, {"stage": "chat", "text": "exit"}]


def worker(spec):
    state = load_learning_state(Path(spec["source"]))
    data = {"inputs": spec["events"], "input_events": [], "turns": [], "models": [], "plan_reviews": []}
    pending = iter(spec["events"])
    original_turn, original_model, original_review = app._invoke_interactive_turn, agents._invoke_llm_raw, app._review_context_change

    def read_input(prompt):
        # 每次读取下一条输入前，观察上一轮已落盘的正式状态。
        current_path = Path(spec["current"])
        observed = load_learning_state(current_path) if current_path.exists() else state
        data.setdefault("state_observations", []).append({"prompt": prompt, "state": formal(observed)})
        if spec.get("observe_only"):
            # 观察模型是否提出候选，不预设它一定进入确认；所有候选一律拒绝。
            stage = "chat" if prompt == "你：" else "confirm"
            text = (spec["probe_input"] if not data["input_events"] else "exit") if stage == "chat" else "n"
            data["input_events"].append({"stage": stage, "text": text, "prompt": prompt})
            return text
        entry = next(pending)
        stage = "chat" if prompt == "你：" else "confirm"
        if entry["stage"] != stage:
            raise RuntimeError(f"输入阶段不匹配：需要{entry['stage']}，实际{stage}")
        data["input_events"].append({**entry, "prompt": prompt})
        return entry["text"]

    def model(llm, messages, *, stage):
        entry = {"stage": stage, "request_messages": [message_record(m) for m in messages]}
        data["models"].append(entry)
        started = time.perf_counter()
        try:
            response = original_model(llm, messages, stage=stage)
        finally:
            entry["elapsed_ms"] = round((time.perf_counter() - started) * 1000, 2)
        entry.update(content=response.content, tool_calls=response.tool_calls, usage=response.usage_metadata,
            invocation_audit=response.response_metadata.get("invocation_audit"))
        return response

    def turn(**kwargs):
        trace = {}
        entry = {"user_input": kwargs["user_input"], "audit_trace": trace}
        data["turns"].append(entry)
        entry["response"] = original_turn(**kwargs, audit_trace=trace)
        return entry["response"]

    def review(current, **kwargs):
        before = formal(current)
        updated, decision = original_review(current, **kwargs)
        data["plan_reviews"].append({"before": before, "after": formal(updated), "decision": decision})
        return updated, decision

    try:
        with ExitStack() as stack:
            stack.enter_context(patch.object(builtins, "input", read_input))
            stack.enter_context(patch.object(agents, "_invoke_llm_raw", model))
            stack.enter_context(patch.object(app, "_invoke_interactive_turn", turn))
            stack.enter_context(patch.object(app, "_review_context_change", review))
            app.run_interactive_session(spec.get("analysis", ""), state.plan_markdown,
                spec.get("context", ""), state.learning_goal, "window", state.material_binding.root_dir,
                initial_state=state, current_state_path=Path(spec["current"]), archive_history=True,
                history_archive_root=Path(spec["archives"]))
        if next(pending, None) is not None:
            raise RuntimeError("脚本未执行完")
        data["status"] = "completed"
    except Exception as exc:
        data.update(status="failed", error_type=type(exc).__name__, error_message=str(exc))
    Path(spec["observation"]).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0 if data["status"] == "completed" else 1


def launch(spec, directory, name):
    spec_path = directory / f"{name}-input.json"
    spec_path.write_text(json.dumps(spec, ensure_ascii=False, indent=2), encoding="utf-8")
    with (directory / f"{name}-console.log").open("w", encoding="utf-8") as stream:
        return subprocess.run([sys.executable, "-X", "utf8", "-m", "tests.helpers.learning_closed_loop",
            "--worker", str(spec_path), "--llm"], cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT,
            timeout=900).returncode


def run_case(case, directory, reuse_initial=None, *, flow_events=None, resume_text=RESUME):
    directory.mkdir(parents=True, exist_ok=False)
    materials = directory / "materials"
    materials.mkdir()
    (materials / "notes.md").write_text(case["material"], encoding="utf-8")
    initial_trace = []
    original = agents._invoke_llm_raw
    def model(llm, messages, *, stage):
        started = time.perf_counter()
        response = original(llm, messages, stage=stage)
        initial_trace.append({"stage": stage, "request_messages": [message_record(m) for m in messages],
            "content": response.content, "tool_calls": response.tool_calls,
            "usage": response.usage_metadata, "invocation_audit": response.response_metadata.get("invocation_audit"),
            "elapsed_ms": round((time.perf_counter() - started) * 1000, 2)})
        return response
    if reuse_initial is not None:
        reused = json.loads((reuse_initial / case["case_id"] / "initial-workflow.json").read_text(encoding="utf-8"))
        analysis, plan, context = reused["analysis"], reused["plan"], reused["context"]
    else:
        with patch.object(agents, "_invoke_llm_raw", model):
            analysis, plan, context = agents.run_learning_assistant(str(materials), case.get("planning_request", case["goal"]), 1, "agent")
    state = LearningState(learning_goal=case["goal"], plan_markdown=plan,
        material_binding=create_material_binding(str(materials)))
    source = save_learning_state(state, directory / "initial.json")
    (directory / "initial-workflow.json").write_text(json.dumps({"analysis": analysis, "plan": plan,
        "context": context, "models": initial_trace, "reused_from": str(reuse_initial) if reuse_initial else None}, ensure_ascii=False, indent=2), encoding="utf-8")
    common = {"archives": str(directory / "archives"), "analysis": analysis, "context": context}
    first = directory / "first.json"
    print(case["case_id"] + "：初始分析和规划完成；进入反馈、拒绝和采用", flush=True)
    code = launch({**common, "source": str(source), "current": str(first), "events": events() if flow_events is None else flow_events,
        "observation": str(directory / "first-observation.json")}, directory, "first")
    if code:
        return {"case_id": case["case_id"], "status": "failed", "phase": "first_session", "artifacts": str(directory)}
    saved = load_learning_state(first)
    first_bytes = first.read_bytes()
    resumed = directory / "resumed.json"
    print(case["case_id"] + "：进入独立新进程恢复与历史回查", flush=True)
    code = launch({**common, "source": str(first), "current": str(resumed),
        "events": [{"stage": "chat", "text": resume_text}, {"stage": "chat", "text": "exit"}],
        "observation": str(directory / "resumed-observation.json")}, directory, "resumed")
    if code:
        return {"case_id": case["case_id"], "status": "failed", "phase": "resume", "artifacts": str(directory)}
    after = load_learning_state(resumed)
    first_obs = json.loads((directory / "first-observation.json").read_text(encoding="utf-8"))
    second_obs = json.loads((directory / "resumed-observation.json").read_text(encoding="utf-8"))
    reviews = first_obs["plan_reviews"]
    checks = {"two_reviews": len(reviews) == 2,
        "decline_preserves_state": bool(reviews) and reviews[0]["decision"] == "cancelled" and reviews[0]["before"] == reviews[0]["after"],
        "adopt_revision_once": saved.context_revision == 2,
        "task_feedback_preserved": len(saved.tasks) == 1 and saved.tasks[0].activity_status == "completed"
            and saved.tasks[0].understanding_status == "self_reported_unclear" and len(saved.feedback_records) == 1,
        "resume_business_unchanged": formal(saved) == formal(after),
        "resume_source_unchanged": first.read_bytes() == first_bytes,
        "quote_in_answer": DISCUSSION in second_obs["turns"][0]["response"]}
    return {"case_id": case["case_id"], "status": "completed", "checks": checks,
        "quality_status": "pending_manual_review", "artifacts": str(directory)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--worker", type=Path)
    parser.add_argument("--llm", action="store_true")
    parser.add_argument("--reuse-initial", type=Path)
    args = parser.parse_args()
    if args.worker:
        if not args.llm:
            parser.error("worker必须显式--llm")
        return worker(json.loads(args.worker.read_text(encoding="utf-8")))
    if not args.output_dir:
        parser.error("需要--output-dir")
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    manifest = output / "manifest.json"
    if manifest.exists():
        parser.error("不覆盖已有实验")
    before = fingerprint()
    manifest.write_text(json.dumps({"mode": "online" if args.llm else "preflight_only", "cases": CASES,
        "events": events(), "resume": RESUME, "source_sha256": before}, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.llm:
        with (output / "results.jsonl").open("x", encoding="utf-8") as stream:
            for case in CASES:
                try:
                    row = run_case(case, output / case["case_id"], args.reuse_initial.resolve() if args.reuse_initial else None)
                except Exception as exc:
                    row = {"case_id": case["case_id"], "status": "failed", "error_type": type(exc).__name__}
                row["source_unchanged"] = fingerprint() == before
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
                stream.flush()
                print(json.dumps(row, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
