import asyncio
import json

import pytest

from tests.helpers.langgraph_reading_compare import BOOKS, build_graph
from tests.helpers.reading_dependency import execute_ready, validate_plan


def test_ready_queue_runs_d_while_b_is_waiting():
    async def scenario():
        d_started = asyncio.Event()
        tasks = [{"id": name, "depends_on": deps} for name, deps in
                 (("A", []), ("B", []), ("C", ["A", "B"]), ("D", ["A"]))]

        async def worker(task, inputs):
            if task["id"] == "B":
                await d_started.wait()
            if task["id"] == "D":
                assert list(inputs) == ["A"]
                d_started.set()
            return {"status": "done"}

        _, events = await asyncio.wait_for(execute_ready(tasks, worker), 3)
        timeline = [(event["id"], event["status"]) for event in events]
        assert timeline.index(("D", "started")) < timeline.index(("B", "done"))
        assert timeline.index(("B", "done")) < timeline.index(("C", "started"))

    asyncio.run(scenario())


@pytest.mark.parametrize("dependencies", [["A"], ["missing"]])
def test_invalid_dependencies_rejected_before_execution(dependencies):
    with pytest.raises(ValueError):
        validate_plan([{"id": "A", "book": BOOKS[0], "page": 36,
                        "depends_on": dependencies}], BOOKS)


def test_graph_preserves_successful_evidence_when_comparison_is_skipped():
    tasks = [
        {"id": "A", "book": BOOKS[0], "page": 36, "depends_on": [], "goal": "read A"},
        {"id": "B", "book": BOOKS[1], "page": 9999, "depends_on": [], "goal": "read B"},
        {"id": "C", "book": None, "page": None, "depends_on": ["A", "B"], "goal": "compare"},
    ]

    class Read:
        def invoke(self, args):
            if args["start_page"] == 9999:
                raise ValueError("out of range")
            return json.dumps({"ok": True, "data": {"pages": [{"text": "excerpt"}]}})

    calls = []

    async def ask(role, stage, instructions, payload):
        calls.append(stage)
        if stage == "planner":
            return json.dumps({"tasks": tasks})
        if stage == "deliver":
            assert [item["page"] for item in payload["evidence"]] == [36]
            assert [row["status"] for row in payload["reports"]] == ["done", "unavailable", "skipped"]
        return "answer"

    state = asyncio.run(build_graph(Read(), ask).ainvoke({"question": "compare if possible"}))
    assert calls == ["planner", "A", "deliver"]
    assert state["answer"] == "answer"
