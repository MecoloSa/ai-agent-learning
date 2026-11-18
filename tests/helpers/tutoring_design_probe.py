"""Test-side comparisons for source quotes, plan delivery and exercise continuity."""

import argparse
import json
from pathlib import Path
import re
from time import perf_counter
from unittest.mock import patch
from uuid import uuid4

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import StructuredTool
from pydantic import StrictInt

from ai_agent import main as app
from ai_agent.learning.workspace_service import WorkspaceCommand, WorkspaceService
from ai_agent.learning.workspace_store import WorkspaceStore
from ai_agent.persistence.memory_state import MemorySnapshot, StoredMessage
from ai_agent.persistence.session_materials import create_material_binding
from ai_agent.resources import tools
from ai_agent.resources import document_structure
from tests.helpers.wink_observer import Observer, review_with_deepseek
from tests.helpers.interactive_request_view import project
from tests.helpers.short_read_recovery import recover_short_range


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("quote-ok", "quote-once", "quote-down", "quote-located", "plan-chat",
                                         "plan-draft", "plan-retry", "plan-replay-recovery", "exercise-base", "exercise-record"))
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--variant", choices=("baseline", "feedback", "defaults", "lean"), default="baseline")
    parser.add_argument("--materials", type=Path)
    parser.add_argument("--question")
    parser.add_argument("--expected-source", action="append", default=[])
    parser.add_argument("--recover-coverage", action="store_true")
    parser.add_argument("--signal-coverage", action="store_true")
    parser.add_argument("--structure-scope", action="store_true")
    parser.add_argument("--observer", choices=("off", "program", "facts", "precise", "targeted", "catalog", "catalog_guard", "attribution_self_check", "agent", "grounded", "gated", "async", "async3"), default="off")
    parser.add_argument("--recover-short-read", action="store_true")
    parser.add_argument("--observer-replay", type=Path)
    parser.add_argument("--live-after-prefix", action="store_true")
    parser.add_argument("--request-view", action="store_true")
    parser.add_argument("--request-view-gap", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    rows = [json.loads(line) for line in (args.source / "turns.jsonl").read_text(encoding="utf8").splitlines()]
    saved = WorkspaceStore(args.source / "study.sqlite3").load()
    plan = args.mode.startswith("plan")
    quote = args.mode.startswith("quote")
    history_rows = rows[:-1] if plan else rows[:1] if quote else rows[:2]
    turn = rows[-1] if plan else rows[1] if quote else rows[2]
    history = tuple(message for row in history_rows if row["request"]["action"] == "chat"
                    for message in (
                        StoredMessage(role="human", content=row["request"]["payload"]["text"]),
                        StoredMessage(role="ai", content=row["event"]["message"]),
                    ))
    memory = MemorySnapshot(strategy="window", max_chars=12000, window_size=12, messages=history)
    state_data = (turn["before"] | {"conversation_memory": memory.model_dump()})
    state = type(saved.state).model_validate(state_data)
    if args.materials:
        state = state.model_copy(update={"material_binding": create_material_binding(str(args.materials.resolve()))})
    original_turn = app._invoke_interactive_turn
    original_read = tools.read_material_lines
    original_builder = app.build_read_material_lines_tool
    original_bind = app.create_llm_with_tools
    original_inspect = document_structure.inspect_document_structure
    audits = []
    reads = 0

    def inspect(*pos, **kw):
        data = original_inspect(*pos, **kw)
        if data.get("mode") == "whole_document":
            return data
        # 描述这份回执的覆盖面，保留工具选择和原有预算。
        return {"section_list_scope": "仅列主要层级；遗漏的小节不能据此判定不存在。", **data}

    def covered_sources(audit):
        covered = set()
        for call in audit.get("tool_results", []):
            if not call["executed"] or call["tool_name"] not in {"read_material_lines", "inspect_document_structure"}:
                continue
            result = json.loads(call["result"])
            if not result.get("ok"):
                continue
            data = result["data"]
            has_text = (any(line["text"].strip() for line in data.get("lines", []))
                        if call["tool_name"] == "read_material_lines"
                        else data.get("mode") == "whole_document" and bool(data.get("text", "").strip()))
            if has_text:
                covered.add(data["source"])
        return covered

    def observe(**kwargs):
        audit = {}
        kwargs["audit_trace"] = audit
        if args.signal_coverage and args.expected_source:
            kwargs["request_messages"] = [*kwargs["request_messages"], HumanMessage(
                content="本轮逐份对照的资料：" + "、".join(args.expected_source) +
                "。分别说明实际核实的依据；如果有资料不可读，说明缺口。")]
        if args.variant == "lean":
            kwargs["extra_tools"] = tuple(tool for tool in kwargs["extra_tools"]
                                          if tool.name != "inspect_document_structure")
        if args.mode == "quote-located":
            # 模拟结构工具已返回章节定位，正文仍由模型自行读取。
            kwargs["request_messages"] = [*kwargs["request_messages"], HumanMessage(
                content="资料位置索引：从重读到回想.md，第4节，19–21行。")]
        if args.mode == "exercise-record":
            # 只提供先前真实发给用户的题目，不注入答案或判分标签。
            kwargs["request_messages"] = [*kwargs["request_messages"], HumanMessage(content=json.dumps({
                "context_type": "open_exercise",
                "question_presented": rows[0]["event"]["message"],
                "status": "awaiting_learner_response",
            }, ensure_ascii=False))]
        hook = Observer(args.observer, review_with_deepseek, audit, kwargs["user_input"],
                        expected_sources=args.expected_source)
        original_model_call = app.invoke_llm_response
        prefix = (json.loads(args.observer_replay.read_text(encoding="utf8"))["audits"][0]["model_responses"]
                  if args.observer_replay else [])
        records = iter(prefix[:-1] if args.live_after_prefix else prefix)

        def replay_or_call(model, messages, **options):
            record = next(records, None)
            if record is None:
                if args.request_view or args.request_view_gap:
                    transformed = project(messages, show_evidence_gap=args.request_view_gap)
                    audit.setdefault("request_view_changes", []).append({
                        "stage": options["stage"],
                        "changed_messages": sum(a.content != b.content for a, b in zip(messages, transformed)),
                        "added_messages": len(transformed) - len(messages),
                    })
                    messages = transformed
                return original_model_call(model, messages, **options)
            if options["stage"] != record["stage"]:
                raise ValueError("Recorded prefix stage did not match live execution")
            return AIMessage(content=record["content"], tool_calls=record["tool_calls"])

        def monitored(model, messages, **options):
            return hook.invoke(replay_or_call, model, messages, **options)
        try:
            with patch.object(app, "invoke_llm_response", monitored):
                return original_turn(**kwargs)
        finally:
            audit["observer_events"] = hook.events
            hook.close()
            audits.append(audit)

    def read(*pos, **kw):
        nonlocal reads
        reads += 1
        if args.mode == "quote-down" or (args.mode == "quote-once" and reads == 1):
            raise ValueError("模拟原文读取暂时失败，请重试。")
        return original_read(*pos, **kw)

    def builder(root, **kwargs):
        original = original_builder(root, **kwargs)
        if args.recover_short_read:
            def recover(relative_path, start_line, end_line):
                return recover_short_range(original.func, relative_path, start_line, end_line)
            return original.model_copy(update={"func": recover})

        def read_range(relative_path: str, start_line: StrictInt = 1, end_line: StrictInt = 200) -> str:
            try:
                return original.func(relative_path, start_line, end_line)
            except ValueError as exc:
                if str(exc) != "起始行超出文件范围，没有返回正文":
                    raise
                # 只在已验证文件的行范围错误后提供可恢复动作，不把正文提前塞入上下文。
                return json.dumps({"ok": False, "error": {"code": "line_out_of_range",
                    "message": "起始行已超过文末。可以从第1行开始读取短文件，或改用定位工具。"}}, ensure_ascii=False)

        if args.variant == "feedback":
            return original.model_copy(update={"func": read_range})
        return StructuredTool.from_function(
            read_range, name=original.name,
            description="读取资料目录内 TXT、MD 或 PY 的原文。已知行号可指定范围；短文件可只传相对路径，从开头读。每次最多200行。",
        )

    def bind(bound, **kwargs):
        if args.variant == "lean":
            bound = [tool for tool in bound if tool.name != "inspect_document_structure"]
        return original_bind(bound, **kwargs)

    def attempt(action, index, *, followup=None):
        store = WorkspaceStore(args.output / f"{index}.sqlite3", None if followup else state)
        service = WorkspaceService(store, args.output / f"archives-{index}")
        if plan:
            base = store.load()
            # 恢复原实验中的真实已拒绝候选及其提交事件，去掉末轮重提。
            rejected = tuple(item.model_copy(update={"base_version": base.version}) for item in saved.proposals
                             if item.status == "rejected")
            store.commit(base, base.model_copy(update={"proposals": rejected}),
                         uuid4().hex, "probe-seed", saved.events[-2])
        question = followup or args.question or turn["request"]["payload"]["text"]
        payload = {"text": question} if action == "chat" else {"mode": "adjust", "instructions": question}
        request = WorkspaceCommand(action=action, version=store.load().version, request_id=uuid4().hex, payload=payload)
        clock = perf_counter()
        event = service.execute(request)
        after = store.load()
        return {"request": request.model_dump(mode="json"), "event": event.model_dump(mode="json"),
                "formal_state_unchanged": after.state.model_dump(exclude={"conversation_memory"}) ==
                state.model_dump(exclude={"conversation_memory"}), "seconds": round(perf_counter() - clock, 3)}

    with (patch.object(app, "_invoke_interactive_turn", observe), patch.object(tools, "read_material_lines", read),
          patch.object(document_structure, "inspect_document_structure", inspect if args.structure_scope else original_inspect),
          patch.object(app, "build_read_material_lines_tool", original_builder
                       if args.variant == "baseline" and not args.recover_short_read else builder),
          patch.object(app, "create_llm_with_tools", bind)):
        if args.mode == "plan-replay-recovery":
            # 重放已记录的缺交付结果，只重新执行后续恢复请求。
            assert not turn["event"]["proposal_id"]
            attempts = [{"event": turn["event"], "replayed": True, "seconds": turn["seconds"]},
                        attempt("draft", 1)]
        else:
            attempts = [attempt("draft" if args.mode == "plan-draft" else "chat", 1)]
        if args.mode == "plan-retry" and not attempts[0]["event"]["proposal_id"]:
            attempts.append(attempt("draft", 2))
        missing = sorted(set(args.expected_source) - covered_sources(audits[0]))
        if args.recover_coverage and missing:
            # 一次独立的补查请求复用本次会话，不指定行号或指定工具。
            attempts.append(attempt("chat", 1, followup=(
                "请补查刚才尚未取得正文的资料：" + "、".join(missing) +
                "。结合上轮已查证内容回答原问题；如仍不可读，请明确区分已经核实与尚未核实的内容。"
            )))
    result = {"mode": args.mode, "variant": args.variant, "attempts": attempts, "audits": audits, "read_attempts": reads}
    result["source_coverage"] = [sorted(covered_sources(audit)) for audit in audits]
    result["missing_after_first_turn"] = missing
    if quote:
        raw = attempts[-1]["event"]["message"]
        excerpts = re.findall(r'“([^”\n]+)”|"([^"\n]+)"', raw)
        evidence = []
        for audit in audits:
            for tool in audit.get("tool_results", []):
                if tool["tool_name"] == "read_material_lines" and app.tool_result_succeeded("read_material_lines", tool["result"]):
                    data = json.loads(tool["result"])["data"]
                    evidence.extend(line["text"] for line in data["lines"] if not line.get("partial"))
                elif tool["tool_name"] == "inspect_document_structure" and tool["executed"]:
                    data = json.loads(tool["result"]).get("data", {})
                    if data.get("mode") == "whole_document" and isinstance(data.get("text"), str):
                        evidence.extend(data["text"].splitlines())
        # 这是评测候选规则。短关键词引号留给人工复核，长引语需要能在原文找到。
        result["quote_check"] = [{"text": left or right, "verified": any((left or right).replace("**", "") in line for line in evidence)}
                                 for left, right in excerpts if len(left or right) >= 8]
        result["evidence_lines"] = len(evidence)
    (args.output / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, default=str), encoding="utf8")


if __name__ == "__main__":
    main()
