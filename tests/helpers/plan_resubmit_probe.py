"""从漏调工具前的真实快照，隔离重放工具契约/结构化回执对照。"""
import argparse
from contextlib import nullcontext
import json
from pathlib import Path
import shutil
import subprocess
import sys
from unittest.mock import patch

import ai_agent.main as app
from ai_agent.learning.plan_proposals import PlanProposalInbox
from ai_agent.persistence.state_store import load_learning_state, save_learning_state
from tests.helpers.learning_closed_loop import CANDIDATE, DISCUSSION, formal, worker

DESCRIPTION = (
    "提交完整计划正文，交由程序展示并请求用户确认；返回pending表示待确认，尚未采用。"
    "用户要求生成或调整候选计划时使用，普通讨论不调用。保留未要求改变的部分，"
    "不修改目标、任务或反馈；正文不包含更新标记。"
)
PARAMETER = "完整的待确认计划正文，包含保留内容与本次调整；不是修改说明或片段。"


def run_worker(spec):
    original = PlanProposalInbox.__init__
    def init(self, *args, **kwargs):
        original(self, *args, **kwargs)
        self.tool.description = DESCRIPTION
        # 影子实验只替换给模型的schema，不改变候选校验或执行逻辑。
        schema = self.tool.args_schema.model_json_schema()
        schema["properties"]["plan_markdown"]["description"] = PARAMETER
        self.tool.args_schema = schema
    original_turn = app._invoke_interactive_turn
    def projected_turn(**kwargs):
        messages = []
        marker = "\n【程序确认结果数据】\n"
        for message in kwargs["request_messages"]:
            if message.type == "ai" and marker in message.content:
                _, payload = message.content.rsplit(marker, 1)
                details = json.loads(payload)
                if details.get("confirmation_reason") == "user_declined":
                    # 仅修改请求副本；原文仍在存档中，可由既有查询工具读取。
                    message = message.model_copy(update={"content": json.dumps({
                        "context_type": "historical_context_change_receipt",
                        "decision": "cancelled", "confirmation_reason": "user_declined",
                        "candidate_body_in_request": False,
                        "readback_tool": "read_confirmation_history",
                    }, ensure_ascii=False)})
            messages.append(message)
        return original_turn(**{**kwargs, "request_messages": messages})
    with patch.object(PlanProposalInbox, "__init__", init) if spec["variant"] == "contract" else nullcontext():
        with patch.object(app, "_invoke_interactive_turn", projected_turn) if spec["variant"] == "projection" else nullcontext():
            return worker(spec)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", type=Path)
    p.add_argument("--source-dir", type=Path, default=Path(".evaluation-results/closed-loop-fixed-20261103/literature"))
    p.add_argument("--worker", type=Path)
    p.add_argument("--repeats", type=int, default=2)
    p.add_argument("--variants", nargs="+", default=["baseline", "contract", "receipt"])
    p.add_argument("--boundary-variant", default="contract")
    args = p.parse_args()
    if args.worker:
        return run_worker(json.loads(args.worker.read_text(encoding="utf-8")))
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    source = args.source_dir.resolve()
    state = load_learning_state(source / "first.json")
    memory = state.conversation_memory
    # 去掉发生失败的最后一问一答，历史读取权限也退回对应前缀，避免偷看未来。
    archive = memory.history_archive
    archive = archive.model_copy(update={"sources": tuple(s.model_copy(update={"through_turn": 2}) for s in archive.sources)})
    memory = memory.model_copy(update={"messages": memory.messages[:-2], "history_archive": archive})
    state = state.model_copy(update={"conversation_memory": memory})
    workflow = json.loads((source / "initial-workflow.json").read_text(encoding="utf-8"))
    rows = []
    schedule = [(variant, "resubmit", i) for i in range(args.repeats) for variant in args.variants]
    schedule += [(args.boundary_variant, "discussion", 0), (args.boundary_variant, "keep_rejected", 0)]
    for variant, case, repeat in schedule:
        directory = output / f"{variant}-{case}-{repeat + 1}"
        directory.mkdir()
        shutil.copytree(source / "archives", directory / "archives")
        snapshot = state
        if variant == "receipt":
            messages = list(memory.messages)
            marker = "\n【程序确认结果数据】\n"
            prefix, payload = messages[-1].content.rsplit(marker, 1)
            details = json.loads(payload)
            details.update(decision="cancelled", context_revision=state.context_revision)
            messages[-1] = messages[-1].model_copy(update={"content": prefix + marker + json.dumps(details, ensure_ascii=False)})
            snapshot = state.model_copy(update={"conversation_memory": memory.model_copy(update={"messages": tuple(messages)})})
        save_learning_state(snapshot, directory / "source.json")
        query = {"resubmit": CANDIDATE, "discussion": DISCUSSION,
                 "keep_rejected": "上个候选仍然不采用，请解释它与当前正式计划的区别，不要生成新候选，不修改计划、任务或反馈。"}[case]
        spec = {"variant": variant, "observe_only": True, "probe_input": query, "events": [],
                "source": str(directory / "source.json"), "current": str(directory / "current.json"),
                "archives": str(directory / "archives"), "analysis": workflow["analysis"], "context": workflow["context"],
                "observation": str(directory / "observation.json")}
        (directory / "input.json").write_text(json.dumps(spec, ensure_ascii=False, indent=2), encoding="utf-8")
        with (directory / "console.log").open("w", encoding="utf-8") as stream:
            code = subprocess.run([sys.executable, "-X", "utf8", "-m", "tests.helpers.plan_resubmit_probe", "--worker", str(directory / "input.json")], stdout=stream, stderr=subprocess.STDOUT, timeout=900).returncode
        obs = json.loads((directory / "observation.json").read_text(encoding="utf-8"))
        row = {"variant": variant, "case": case, "repeat": repeat + 1, "process_code": code,
               "status": obs["status"], "reviews": len(obs["plan_reviews"]),
               "unchanged": formal(snapshot) == formal(load_learning_state(directory / "current.json")),
               "model_calls": len(obs["models"]), "tools": [call["name"] for m in obs["models"] for call in m.get("tool_calls", [])]}
        rows.append(row)
        (output / "results.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(row, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
