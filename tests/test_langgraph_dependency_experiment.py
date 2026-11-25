"""在受控分支时序下验证 LangGraph 的并行与条件路由。"""

import asyncio
from operator import add
from typing import Annotated, Literal, TypedDict

import pytest
from langgraph.graph import END, START, StateGraph
from langgraph.types import Send


class FanOutState(TypedDict, total=False):
    branch: str
    approved: Annotated[list[str], add]


def test_send_waits_for_parallel_superstep_before_starting_downstream():
    async def scenario():
        fast_finished = asyncio.Event()
        release_slow = asyncio.Event()
        slow_started = asyncio.Event()
        downstream_started = asyncio.Event()

        async def read(state: FanOutState):
            branch = state["branch"]
            if branch == "A":
                fast_finished.set()
            else:
                slow_started.set()
                await release_slow.wait()
            return {"approved": [branch]}

        def follow_up(state: FanOutState):
            downstream_started.set()
            return {}

        graph = StateGraph(FanOutState)
        graph.add_node("read", read)
        graph.add_node("follow_up", follow_up)
        graph.add_conditional_edges(
            START, lambda _: [Send("read", {"branch": key}) for key in ("A", "B")], ["read"]
        )
        graph.add_edge("read", "follow_up")
        graph.add_edge("follow_up", END)
        run = asyncio.create_task(graph.compile().ainvoke({"approved": []}))
        try:
            await asyncio.wait_for(fast_finished.wait(), 5)
            await asyncio.wait_for(slow_started.wait(), 5)
            with pytest.raises(TimeoutError):
                await asyncio.wait_for(downstream_started.wait(), 0.1)
        finally:
            release_slow.set()
            result = await asyncio.wait_for(run, 5)
        assert downstream_started.is_set()
        assert sorted(result["approved"]) == ["A", "B"]

    asyncio.run(scenario())


def test_ready_queue_can_start_a_only_consumer_before_independent_b_finishes():
    async def scenario():
        release_b = asyncio.Event()
        started_d = asyncio.Event()

        async def read_b():
            await release_b.wait()

        async def follow_a():
            started_d.set()

        async def read_a():
            return "approved"

        a = asyncio.create_task(read_a())
        b = asyncio.create_task(read_b())
        try:
            assert await asyncio.wait_for(a, 5) == "approved"
            await asyncio.wait_for(follow_a(), 5)
            assert started_d.is_set() and not b.done()
        finally:
            release_b.set()
            await asyncio.wait_for(b, 5)

    asyncio.run(scenario())


class ConditionalState(TypedDict, total=False):
    a_verdict: Literal["confirmed", "insufficient", "failed"]
    b_status: Literal["approved", "skipped"]
    delivery: Literal["compare", "report_gap"]


@pytest.mark.parametrize("verdict", ["confirmed", "insufficient", "failed"])
def test_condition_terminates_and_routes_to_the_correct_delivery(verdict):
    reads = []
    graph = StateGraph(ConditionalState)

    def read_b(state):
        reads.append("B")
        return {"b_status": "approved"}

    graph.add_node("read_b", read_b)
    graph.add_node("skip_b", lambda _: {"b_status": "skipped"})
    graph.add_node("deliver", lambda state: {
        "delivery": "compare" if state["b_status"] == "approved" else "report_gap"
    })
    graph.add_conditional_edges(
        START, lambda state: "read_b" if state["a_verdict"] == "confirmed" else "skip_b",
        ["read_b", "skip_b"],
    )
    graph.add_edge("read_b", "deliver")
    graph.add_edge("skip_b", "deliver")
    graph.add_edge("deliver", END)
    result = graph.compile().invoke({"a_verdict": verdict})

    assert result["delivery"] == ("compare" if verdict == "confirmed" else "report_gap")
    assert reads == (["B"] if verdict == "confirmed" else [])
