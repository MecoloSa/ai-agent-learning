"""固定取证历史的规模/外置对照，仅实验工具，不接入生产Agent循环。"""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import time

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool

from ai_agent.config import create_llm
from ai_agent.resources.tools import build_agent_tools
from ai_agent.runtime.agents import invoke_llm_response, stage_evidence
from ai_agent.runtime.request_context import prepare_model_messages
from ai_agent.evaluation.acquisition_pair import message_record
from ai_agent.evaluation.interactive_metrics import summarize_chat_usage
from tests.helpers.context_exploration import RawReceiptStore
from tests.helpers.receipt_view_compare import fingerprint

QUESTION = "只依据target.md原文，说明LARCH_CODE要求口头说明几次，并引用来源。不要把次数当作理解达标，不制定学习计划。"


def prepare_fixture(directory, count):
    directory.mkdir(parents=True, exist_ok=False)
    filler = "背景资料仅用于规模测试，不规定任何标记的次数。\n" * 80
    target = filler + "LARCH_CODE要求口头说明7次，完成次数不能证明已经理解。\n" + filler
    (directory / "target.md").write_text(target, encoding="utf-8")
    for i in range(count - 1):
        (directory / f"notes-{i}.md").write_text(filler * 2, encoding="utf-8")
    readers = {t.name: t for t in build_agent_tools(str(directory))}
    messages = [SystemMessage(content="依据实际取得的原文回答，区分原文、预览和推断，引用来源。"), HumanMessage(content=QUESTION)]
    for i, path in enumerate(sorted(directory.glob("*.md"))):
        args = {"relative_path": path.name, "max_chars": 5000}
        response = readers["read_learning_file"].invoke(args)
        assert json.loads(response)["data"]["truncated"] is False
        messages.extend([AIMessage(content="", tool_calls=[{"id": f"initial-{i}", "name": "read_learning_file", "args": args, "type": "tool_call"}]),
            ToolMessage(content=response, tool_call_id=f"initial-{i}")])
    return messages


def run(history, arm, store_root, material_dir=None):
    store = RawReceiptStore(store_root)
    references = []
    view = prepare_model_messages(history, tool_result_formatter=stage_evidence)
    if arm == "offload":
        for original, projected in zip(history, view):
            if isinstance(original, ToolMessage) and len(original.content) > 2000:
                ref = store.put(original.content)
                data = json.loads(original.content)["data"]
                ref.update(source=data["source"], citation=data["citation"])
                references.append(ref)
                projected.content = json.dumps(ref, ensure_ascii=False)

    reads = []
    @tool
    def read_stored_receipt(receipt_id: str) -> str:
        """按已返回的receipt_id读取外置的原始工具回执，返回完整证据。"""
        content = store.get(receipt_id)
        reads.append({"receipt_id": receipt_id, "source": json.loads(content)["data"]["source"]})
        return content

    # 两组都绑定同一个回读工具；只有候选组将大正文替换为预览引用。
    available = {"read_stored_receipt": read_stored_receipt}
    if material_dir is not None:
        available.update({t.name: t for t in build_agent_tools(str(material_dir))
            if t.name in {"read_learning_file", "read_material_lines"}})
    llm = create_llm().bind_tools(list(available.values()))
    row = {"arm": arm, "user_input": QUESTION, "requests": [], "model_responses": [],
        "offloaded_references": references, "readbacks": reads, "quality_status": "pending_manual_review"}
    started = time.perf_counter()
    try:
        for index in range(6):
            row["requests"].append([message_record(m) for m in view])
            response = invoke_llm_response(llm, view, stage="上下文压力与原文回读实验")
            row["model_responses"].append({"content": response.content, "tool_calls": response.tool_calls,
                "usage": response.usage_metadata, "invocation_audit": response.response_metadata.get("invocation_audit")})
            view.append(response)
            if not response.tool_calls:
                row.update(status="completed", final_answer=response.content)
                break
            for call in response.tool_calls:
                try:
                    if call["name"] not in available:
                        raise ValueError("未知工具")
                    result = available[call["name"]].invoke(call["args"])
                except Exception as exc:
                    result = json.dumps({"ok": False, "error_type": type(exc).__name__})
                view.append(ToolMessage(content=result, tool_call_id=call["id"]))
        else:
            row.update(status="failed", error_type="experiment_round_limit")
    except Exception as exc:
        row.update(status="failed", error_type=type(exc).__name__)
    row["latency_ms"] = round((time.perf_counter() - started) * 1000)
    row["usage"] = summarize_chat_usage(row)
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--llm", action="store_true")
    parser.add_argument("--large-only", action="store_true")
    args = parser.parse_args()
    root = args.output_dir.resolve()
    root.mkdir(parents=True, exist_ok=True)
    if (root / "manifest.json").exists():
        parser.error("不覆盖已有实验")
    before = fingerprint()
    (root / "manifest.json").write_text(json.dumps({"question": QUESTION, "source_sha256": before,
        "mode": "online" if args.llm else "preflight_only", "file_counts": [1, 4, 10],
        "candidate_threshold_chars": 2000, "scope": "固定完整读取历史，不测生产自主取证；阈值仅实验参数"}, ensure_ascii=False, indent=2), encoding="utf-8")
    if not args.llm:
        return 0
    with (root / "results.jsonl").open("x", encoding="utf-8") as stream:
        for count in ((10,) if args.large_only else (1, 4, 10)):
            history = prepare_fixture(root / f"materials-{count}", count)
            (root / f"input-{count}.json").write_text(json.dumps([message_record(m) for m in history], ensure_ascii=False, indent=2), encoding="utf-8")
            for arm in (("current", "offload") if count == 10 else ("current",)):
                row = run(history, arm, root / f"store-{count}-{arm}", root / f"materials-{count}")
                row.update(file_count=count, source_unchanged=fingerprint() == before,
                    initial_material_bytes=sum(p.stat().st_size for p in (root / f"materials-{count}").glob("*.md")))
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
                stream.flush()
                print(f"files={count} arm={arm} status={row['status']} readbacks={len(row['readbacks'])} usage={row['usage']}", flush=True)
    return 0


if __name__ == "__main__":
    main()
