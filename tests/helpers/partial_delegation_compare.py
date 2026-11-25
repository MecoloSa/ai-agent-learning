"""Replay one frozen delegation receipt through the main interactive loop."""

import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from pathlib import Path
from time import perf_counter

from langchain_core.messages import AIMessage, SystemMessage
from langchain_core.tools import tool

import ai_agent.main as app
from ai_agent.runtime.delegated_reading import prepare_partial_evidence_handoff


def with_excerpts(row):
    receipt = deepcopy(row["delegated_report"])
    store = {item["evidence_id"]: item for entry in row["trace"]
             for item in entry.get("result", {}).get("evidence", [])}
    # 保持冻结结果中的分支顺序与实验一致。
    tasks = {task["id"]: task for task in row["tasks"]}
    prepare_partial_evidence_handoff(receipt, [tasks[name] for name in row["results"]],
                                     row["results"], store)
    return receipt


def run(root, row, output, arm):
    @tool
    def delegate_learning_task(objective: str) -> str:
        """Return the captured delegation report."""
        return json.dumps({"ok": True, "data": receipt}, ensure_ascii=False)

    receipt = row["delegated_report"] if arm == "locations" else with_excerpts(row)
    read = app.build_read_material_lines_tool(root)
    retrieval = app.build_retrieve_learning_context_tool(root, structured_result=True)
    tools = [app.build_list_material_files_tool(root), app.build_read_pdf_pages_tool(root), delegate_learning_task]
    model = app.create_llm_with_tools([read, retrieval, *tools])

    class ReplayFirst:
        def invoke(self, messages, **kwargs):
            if not audit["model_responses"]:
                return AIMessage(content="", tool_calls=[{"id": "frozen-delegation", "name": "delegate_learning_task",
                    "args": {"objective": row["question"]}}])
            return model.invoke(messages, **kwargs)

    audit = {}
    start = perf_counter()
    app._invoke_interactive_turn(
        request_messages=[SystemMessage(content="你是学习助手，依据用户问题和可用资料回答。")],
        user_input=row["question"], plain_llm=app.create_llm(), tool_llm=ReplayFirst(),
        retrieval_tool=retrieval, read_tool=read, extra_tools=tools, audit_trace=audit,
    )
    audit["seconds"] = round(perf_counter() - start, 2)
    (output / f"{arm}.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf8")
    print(arm, audit["seconds"], [item["tool_name"] for item in audit["tool_results"]], flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("materials", type=Path)
    parser.add_argument("record", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    record = json.loads(args.record.read_text(encoding="utf8"))
    with ThreadPoolExecutor() as pool:
        list(pool.map(lambda arm: run(str(args.materials.resolve()), record, args.output, arm),
                      ("locations", "excerpts")))
