"""Actual workspace browser or online smoke, using isolated synthetic material."""

import argparse
import json
from pathlib import Path
import re
import subprocess
from time import sleep
from unittest.mock import patch
from urllib.request import Request, urlopen

from langchain_core.messages import AIMessageChunk

import ai_agent.main as app
from ai_agent.learning.learning_state import LearningState, LearningTask
from ai_agent.learning.workspace_service import WorkspaceService
from ai_agent.learning.workspace_store import WorkspaceStore
from ai_agent.persistence.session_materials import create_material_binding
from tests.test_web_workspace import command, serving

ONLINE_INPUT = "请实际调用 read_material_lines 读取 notes.md 第1至3行，再把现有计划调整为总计15分钟，只阅读和口头说明，请提交完整候选供我确认。"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--llm", action="store_true")
    mode.add_argument("--edge", type=Path)
    args = parser.parse_args()
    root = args.output_dir.resolve()
    root.mkdir(parents=True, exist_ok=False)
    materials = root / "materials"
    materials.mkdir()
    (materials / "notes.md").write_text(
        "通用工具适合组合与探索。\n专用工具适合受约束的业务动作。\n先阅读定义，再口头说明差异。\n", encoding="utf-8")
    state = LearningState(learning_goal="理解两类工具的使用边界", plan_markdown="阅读30分钟",
        tasks=(LearningTask(task_id="read-001", title="阅读工具定义"),),
        material_binding=None if args.edge else create_material_binding(str(materials)))
    store = WorkspaceStore(root / "workspace.sqlite3", state)
    service = WorkspaceService(store, root / "archives")
    if args.edge:
        calls = []

        class Model:
            def stream(self, messages):
                text = messages[-1].content
                calls.append(text)
                if "请先澄清" in text:
                    yield AIMessageChunk(content="请说明你希望")
                    sleep(1)
                    yield AIMessageChunk(content="保留哪项活动。")
                elif text == "如何区分通用工具和专用工具？":
                    yield AIMessageChunk(content="**通用工具**适合探索。")
                    sleep(3)
                    yield AIMessageChunk(content=(
                        "\n\n> 专用工具适合约束明确的业务动作。\n\n"
                        "| 名称 | 用途 |\n| --- | --- |\n| 通用 | 探索 |\n\n"
                        "```python\nprint('example')\n```\n\n"
                        "[说明](https://example.org) [危险](javascript:alert(1))\n\n"
                        "<img src=x onerror=alert(1)>"
                    ))
                elif "提取尚未登记" in text:
                    yield AIMessageChunk(content="", tool_calls=[{"name": "propose_plan_change", "id": "extraction",
                        "args": {"plan_markdown": "阅读5分钟，口头说明10分钟，**复述要点**",
                                 "task_changes": [{"operation": "add", "title": "整理学习要点"}]}}])
                else:
                    yield AIMessageChunk(content="", tool_call_chunks=[{"name": "propose_plan_change", "id": "candidate",
                        "index": 0, "args": '{"plan_markdown":"阅读5分钟，'}])
                    yield AIMessageChunk(content="", tool_call_chunks=[{"index": 0, "args":
                        '口头说明10分钟，**复述要点**","task_changes":[{"operation":"add","title":"口头说明工具区别","estimated_minutes":10}]}'}])

        with patch.object(app, "create_llm", lambda **_: Model()), patch.object(app, "create_llm_with_tools", lambda _, **kw: Model()), \
             serving(service) as url:
            subprocess.run(["node", str(Path(__file__).with_name("workspace_browser.mjs")),
                            url, str(root), str(args.edge)], check=True, timeout=90)
        assert len(calls) == 5
        assert store.load().state.context_revision == 2
        assert len(store.load().state.feedback_records) == 2
        (root / "result.json").write_text(json.dumps({"model_stub_calls": calls, "revision": 2},
                                          ensure_ascii=False, indent=2), encoding="utf-8")
        return

    audit = {}
    original = app._invoke_interactive_turn
    def traced(**kwargs):
        # Web now forwards an optional audit_trace itself. Reuse it when supplied.
        active_audit = kwargs.get("audit_trace")
        kwargs["audit_trace"] = audit if active_audit is None else active_audit
        result = original(**kwargs)
        if active_audit is not None:
            audit.update(active_audit)
        return result

    with patch.object(app, "_invoke_interactive_turn", traced), serving(service) as url:
        with urlopen(url, timeout=10) as response:
            token = re.search(r'name="workspace-token" content="([0-9a-f]+)"', response.read().decode("utf-8"))[1]

        def post(cmd):
            req = Request(url + "/api/action", data=cmd.model_dump_json().encode("utf-8"),
                headers={"Origin": url, "X-Workspace-Token": token, "Content-Type": "application/json"})
            with urlopen(req, timeout=300) as response:
                return json.load(response)

        try:
            drafted = post(command("draft", 1, mode="adjust", instructions=ONLINE_INPUT))
            (root / "draft-result.json").write_text(json.dumps(drafted, ensure_ascii=False, indent=2), encoding="utf-8")
            assert drafted["event"]["status"] == "pending"
            assert drafted["workspace"]["state"]["context_revision"] == 1
            candidate = drafted["event"]["proposal_id"]
            post(command("decide", 1, proposal_id=candidate, decision="reject"))
            assert store.load().state == state.model_copy(update={"conversation_memory": store.load().state.conversation_memory})
            resubmitted = post(command("resubmit", 1, proposal_id=candidate))["event"]["proposal_id"]
            adopted = post(command("decide", 1, proposal_id=resubmitted, decision="adopt"))
            assert adopted["workspace"]["state"]["context_revision"] == 2
            assert [r["tool_name"] for r in audit["tool_results"] if r["executed"]] == [
                "read_material_lines", "propose_plan_change"]
            (root / "adopt-result.json").write_text(json.dumps(adopted, ensure_ascii=False, indent=2), encoding="utf-8")
            print(json.dumps({"status": "passed", "revision": 2, "model_calls": len(audit["model_responses"])}))
        finally:
            (root / "audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
