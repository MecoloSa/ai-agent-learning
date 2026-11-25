"""Run the actual retrieval and verified-outline tools through one-worker graphs."""

import asyncio
import json
import sys
from pathlib import Path
from time import perf_counter
from unittest.mock import patch

from langchain_core.messages import AIMessage

from ai_agent.retrieval import rag
from tests.helpers import delegated_reading as probe


async def check(root, *, source, question, tool_name, arguments, select_evidence, options):
    trace = []
    started = perf_counter()

    async def ask(role, stage, messages, tools=()):
        if stage == "planner":
            return AIMessage(content=json.dumps({"tasks": [{
                "id": "check", "instruction": question, "sources": [source], "depends_on": [],
            }]}))
        if stage == "deliver":
            assert json.loads(messages[-1].content)["evidence_catalog"]
            return AIMessage(content="已取得满足该用例的证据。")
        if len(messages) == 2:
            assert tool_name in {item.name for item in tools}
            return AIMessage(content="", tool_calls=[{"id": "check", "name": tool_name, "args": arguments}])
        receipt = json.loads(messages[-1].content)
        selected = select_evidence(receipt.get("evidence", []))
        assert selected is not None, f"{tool_name} 未返回满足预期的证据，error={receipt.get('error')}"
        return AIMessage(content=json.dumps({"findings": [{
            "statement": question, "evidence_ids": [selected["evidence_id"]],
        }], "gaps": []}))

    graph = probe.build_graph(root, {source: None}, ask, trace, roomy=True, source_footer=True, **options)
    result = await graph.ainvoke({"question": question})
    assert result["results"]["check"]["status"] == "done"
    assert result["delivery_status"] == "completed"
    assert result["answer"].count("程序整理的本轮读取来源") == 1
    assert [row["tool"] for row in trace] == [tool_name]
    return {"tool": tool_name, "status": "passed", "seconds": round(perf_counter() - started, 2)}


async def main(root):
    first, second = probe.BOOKS
    actual_rerank = rag.rerank_candidates
    rerank_modes = []

    def capture_rerank(query, documents, **kwargs):
        assert all(doc.metadata["source"] == first for doc in documents)
        outcome = actual_rerank(query, documents, **kwargs)
        rerank_modes.append(kwargs["backend"])
        return outcome

    with patch.object(rag, "rerank_candidates", side_effect=capture_rerank):
        retrieval = await check(
            root, source=first, question="在《AI Agents in Depth》中找到 tool_call_id 关联工具请求和结果的正文。",
            tool_name="search_material",
            arguments={"queries": ["tool_call_id 工具结果通过请求 ID 关联对应工具调用"], "relative_path": first},
            select_evidence=lambda records: next((row for row in records if "tool_call_id" in row["text"]
                                                 and row.get("page")), None),
            options={"reuse_retrieval": True},
        )
    assert rerank_modes, "本次未完成有效重排"
    print(json.dumps({**retrieval, "rerank_backends": rerank_modes}, ensure_ascii=False), flush=True)

    outline = await check(
        root, source=second, question="列出《Hello-Agents》正文已核验的一级目录。",
        tool_name="inspect_document_structure", arguments={"relative_path": second},
        select_evidence=lambda records: next((row for row in records if row["mode"] == "verified_outline"
                                             and len(json.loads(row["text"])) > 1), None),
        options={"structure_lookup": True},
    )
    print(json.dumps(outline, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    asyncio.run(main(Path(sys.argv[1]).resolve()))
