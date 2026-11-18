"""Record reproducible workflow gaps against temporary workspace databases."""

import argparse
import json
from pathlib import Path
from uuid import uuid4

from ai_agent.learning.learning_state import LearningState, LearningTask
from ai_agent.learning.plan_change_request import PlanDraftResult
from ai_agent.learning.workspace_service import WorkspaceCommand, WorkspaceService, workspace_view
from ai_agent.learning.workspace_store import WorkspaceConflict, WorkspaceStore


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    store = WorkspaceStore(args.output / "study.sqlite3", LearningState(
        learning_goal="解释工具调用", plan_markdown="先读示例，再口头复述。",
        tasks=(LearningTask(task_id="read-001", title="阅读示例"),),
    ))
    service = WorkspaceService(store, args.output / "archives")

    def request(action, scope=None, **payload):
        return WorkspaceCommand(action=action, conversation_id=scope, version=store.load().scope(scope).version,
                                request_id=uuid4().hex, payload=payload)

    def execute(action, scope=None, **payload):
        return service.execute(request(action, scope, **payload))

    other = execute("new_conversation").conversation_id
    first = request("rename_conversation", title="工具调用入门")
    service.execute(first)
    execute("rename_conversation", title="工具调用复习")
    service.execute(first)
    replayed_title = store.load().original_title

    candidate = execute("manual_plan", plan_markdown="先口头复述，再回读工具示例。")
    execute("edit_task", task_id="read-001", activity_status="completed")
    stale_view = workspace_view(store.load())["proposals"][0]
    try:
        execute("decide", proposal_id=candidate.proposal_id, decision="reject")
        reject_error = None
    except WorkspaceConflict as exc:
        reject_error = str(exc)

    def slow_chat(snapshot, text, draft_request):
        # 在 A 生成与提交之间插入 B 已成功保存的独立操作。
        execute("create_task", other, title="阅读另一份资料")
        return snapshot.state, PlanDraftResult("工具结果回传后由模型继续回答。", None)

    service.draft_turn = slow_chat
    try:
        execute("chat", text="工具执行后结果怎么回传？")
        chat_error = None
    except WorkspaceConflict as exc:
        chat_error = str(exc)
    final = store.load()
    result = {
        "rename_replay": {"latest_expected": "工具调用复习", "actual": replayed_title},
        "reject_stale_candidate": {"visible_status": stale_view["status"], "error": reject_error,
                                   "stored_status": final.scope(None).proposals[0].status},
        "cross_conversation_write": {"error": chat_error, "other_tasks": [task.title for task in final.scope(other).state.tasks],
                                     "chat_saved": any(event.action == "chat" for event in final.scope(None).events)},
        "preserved_plan": final.state.plan_markdown,
    }
    (args.output / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    assert replayed_title == "工具调用复习"
    assert reject_error is None and final.scope(None).proposals[0].status == "rejected"
    assert chat_error is None and result["cross_conversation_write"]["chat_saved"]
    assert result["cross_conversation_write"]["other_tasks"] == ["阅读另一份资料"]


if __name__ == "__main__":
    main()
