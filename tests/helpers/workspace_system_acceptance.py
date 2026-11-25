"""Measure the real Web chat endpoint with isolated SQLite workspaces."""

import argparse
import json
from pathlib import Path
import re
from time import perf_counter
from types import SimpleNamespace
from unittest.mock import patch
from urllib.request import Request, urlopen

import ai_agent.main as app
from ai_agent.learning.learning_state import LearningState
from ai_agent.learning.workspace_service import WorkspaceService
from ai_agent.learning.workspace_store import WorkspaceStore
from ai_agent.persistence.session_materials import create_material_binding
from ai_agent.runtime import delegated_reading
from tests.helpers.task_delegation_smoke import CASES as TEXTBOOK_CASES
from tests.test_web_workspace import command, serving


CASES = {
    "advice": "我经常当天懂、隔天讲不清楚。请根据一般经验给三个复习办法，不查本地教材，也不要修改计划。",
    "quote": TEXTBOOK_CASES["requested_quote"],
    "cross_book": TEXTBOOK_CASES["requested_cross_book"],
    "missing": TEXTBOOK_CASES["missing_branch"],
    "long_excerpt": TEXTBOOK_CASES["long_excerpt"],
}
MODES = {
    "fast": {"deep_mode": False},
    "deep": {"deep_mode": True, "worker_thinking": True},
    "deep_no_thinking": {"deep_mode": True, "worker_thinking": False},
}


def run(output, case, mode, repeat, state):
    directory = output / f"{case}-{mode}-{repeat}"
    directory.mkdir(parents=True, exist_ok=False)
    store = WorkspaceStore(directory / "workspace.sqlite3", state)
    service = WorkspaceService(store, directory / "archives")
    audit = {}
    delegation_calls = []
    delegation_errors = []
    original = app._invoke_interactive_turn
    build_graph = delegated_reading.build_graph

    def traced(**kwargs):
        target = kwargs["audit_trace"] if kwargs["audit_trace"] is not None else audit
        result = original(**{**kwargs, "audit_trace": target})
        if target is not audit:
            audit.update(target)
        return result

    def instrumented(root, scope, ask, trace, **options):
        def record(stage, invoke):
            async def timed():
                entry = {"stage": stage}
                delegation_calls.append(entry)
                started = perf_counter()
                try:
                    response = await invoke()
                    entry["usage"] = response.usage_metadata
                    if stage == "planner":
                        entry["content"] = response.content
                    return response
                finally:
                    entry["seconds"] = round(perf_counter() - started, 3)
            return timed()

        async def observed(role, stage, messages, tools=()):
            return await record(stage, lambda: ask(role, stage, messages, tools))

        fast_report = options["fast_report"]

        async def observed_report(messages):
            return await record("fast_report", lambda: fast_report(messages))

        graph = build_graph(root, scope, observed, trace, **{**options, "fast_report": observed_report})

        async def invoke(state):
            try:
                return await graph.ainvoke(state)
            except Exception as error:
                delegation_errors.append({"type": type(error).__name__, "message": str(error).split("\n", 1)[0]})
                raise

        return SimpleNamespace(ainvoke=invoke)

    row = {"case": case, "mode": mode, "repeat": repeat, "user_input": CASES[case], "events": []}
    with (patch.object(app, "_invoke_interactive_turn", traced),
          patch.object(delegated_reading, "build_graph", instrumented), serving(service) as url):
        with urlopen(url, timeout=10) as response:
            token = re.search(r'name="workspace-token" content="([0-9a-f]+)"', response.read().decode())[1]
        payload = command("chat", store.load().version, text=CASES[case], **MODES[mode])
        request = Request(url + "/api/action", data=payload.model_dump_json().encode("utf8"), headers={
            "Origin": url, "X-Workspace-Token": token, "Content-Type": "application/json",
            "Accept": "application/x-ndjson",
        })
        started = perf_counter()
        try:
            with urlopen(request, timeout=240) as response:
                for line in response:
                    event = json.loads(line)
                    elapsed = round(perf_counter() - started, 3)
                    row["events"].append({"seconds": elapsed, "type": event["type"],
                                          **({"data": event["data"]} if event["type"] != "delta" else {})})
                    if event["type"] == "delta" and event["data"].strip() and "first_delta" not in row:
                        row["first_delta"] = elapsed
                    if event["type"] == "result":
                        row["status"] = event["data"]["status"]
                        row["message"] = event["data"].get("event", {}).get("message")
                        row["error"] = event["data"].get("error")
        except (OSError, ValueError) as error:
            row["error_type"] = type(error).__name__
        row["seconds"] = round(perf_counter() - started, 3)
    row["audit"] = audit
    row["delegation_calls"] = delegation_calls
    row["delegation_errors"] = delegation_errors
    row["state_preserved"] = store.load().state.model_dump(exclude={"conversation_memory"}) == state.model_dump(
        exclude={"conversation_memory"},
    )
    (directory / "result.json").write_text(json.dumps(row, ensure_ascii=False, indent=2), encoding="utf8")
    print(case, mode, repeat, row.get("status"), row["seconds"], row.get("first_delta"),
          row["state_preserved"], flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("materials")
    parser.add_argument("output", type=Path)
    parser.add_argument("--cases", nargs="+", choices=CASES, default=list(CASES))
    parser.add_argument("--modes", nargs="+", choices=MODES, default=list(MODES))
    parser.add_argument("--repeats", type=int, default=1)
    args = parser.parse_args()
    initial = LearningState(
        learning_goal="理解智能体的工具调用与反馈", plan_markdown="阅读工具调用章节，再口头解释关键机制。",
        material_binding=create_material_binding(str(Path(args.materials).resolve())),
    )
    for repeat in range(args.repeats):
        modes = args.modes[repeat % len(args.modes):] + args.modes[:repeat % len(args.modes)]
        for case in args.cases:
            for mode in modes:
                run(args.output, case, mode, repeat, initial)
