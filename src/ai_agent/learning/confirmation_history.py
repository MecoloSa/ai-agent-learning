"""按需读取当前存档保留的程序确认回执，而不是把全部历史塞进模型窗口。

不扫描其他文件、不写状态、不采用候选。window隐藏的原消息仍可查询；
summary已经压掉的原回执不能从摘要补造，空结果只说明本查询范围没有匹配。
"""
from __future__ import annotations

import json
from pydantic import Field
from ai_agent.learning.learning_state import LearningState, StateModel


class ConfirmationHistoryArgs(StateModel):
    offset: int = Field(default=0, ge=0, strict=True)
    limit: int = Field(default=3, ge=1, le=5, strict=True)


def read_confirmation_history(state: LearningState, request: ConfirmationHistoryArgs) -> str:
    """只读AI消息尾部的既有程序回执；普通回答、用户文本和摘要不是确认事实。"""
    snapshot = state.conversation_memory
    messages = snapshot.messages if snapshot is not None else ()
    markers = {
        "\n【程序学习操作结果数据】\n": "learning_action",
        "\n【程序确认结果数据】\n": "context_change",
    }
    reasons = {"user_confirmed", "user_declined", "user_not_confirmed", "input_ended", "not_recorded"}
    records = []
    unreadable = 0
    for index in range(len(messages) - 1, -1, -1):
        message = messages[index]
        if message.role != "ai":
            continue
        present = [(message.content.rfind(marker), marker) for marker in markers if marker in message.content]
        if not present:
            continue
        position, marker = max(present)
        try:
            payload = json.loads(message.content[position + len(marker):].strip())
        except (ValueError, TypeError):
            unreadable += 1
            continue
        if not isinstance(payload, dict):
            unreadable += 1
            continue
        kind = markers[marker]
        if kind == "learning_action" and payload.get("context_type") != "learning_action_result":
            unreadable += 1
            continue

        # 不从自然语言猜测用户是否否决；旧回执没有该字段时保留未知。
        reason = payload.get("confirmation_reason", "not_recorded")
        if not isinstance(reason, str) or reason not in reasons:
            reason = "not_recorded"
        item = {
            "source_message_index": index,
            "receipt_kind": kind,
            "action": payload.get("action") if kind == "learning_action" else "context_change",
            "decision": payload.get("decision"),
            "confirmation_reason": reason,
            "decision_note": str(payload.get("decision_note") or "未保留具体确认原因，不推断。")[:500],
            "context_revision": payload.get("context_revision"),
        }
        proposal = payload.get("feedback_proposal")
        if isinstance(proposal, dict):
            text = str(proposal.get("text") or "")
            item["feedback_proposal"] = {
                "record_id": proposal.get("record_id"),
                "supersedes_record_id": proposal.get("supersedes_record_id"),
                "text": text[:1000],
                "text_truncated": bool(proposal.get("text_truncated")) or len(text) > 1000,
            }
        # 模型正文只帮助识别是哪份候选，不当作正式状态或确认原因的依据。
        preview = message.content[:position]
        item["assistant_candidate_excerpt"] = preview[:1000]
        item["assistant_candidate_excerpt_truncated"] = len(preview) > 1000
        records.append(item)

    page = records[request.offset:request.offset + request.limit]
    end = request.offset + len(page)
    has_more = end < len(records)
    # 动态评估：核对候选类型、确认原因及分页；不得把历史候选混成当前计划/正式反馈。
    return json.dumps({"ok": True, "data": {
        "search_scope": "current_loaded_state.conversation_memory.raw_ai_receipts",
        "memory_strategy": snapshot.strategy if snapshot is not None else None,
        "records": page,
        "total_matches": len(records),
        "has_more": has_more,
        "next_offset": end if has_more else None,
        "unreadable_receipt_count": unreadable,
        "note": (
            "仅查询当前存档保留的原始程序回执，摘要和其他存档不在范围内。"
            "空结果不代表从未发生；缺失字段不补造。"
            "assistant_candidate_excerpt仅作候选识别，不是正式计划或确认事实。"
            "当前生效值以最新学习状态为准；本次查询不执行操作。"
        ),
    }}, ensure_ascii=False)
