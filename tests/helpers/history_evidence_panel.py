"""从本轮成功回读生成独立原文展示块；不改写或认证助手论断。"""
import json


def render_history_evidence(tool_results):
    blocks, seen = [], set()
    for row in tool_results:
        if row.get("tool_name") != "read_conversation_turn" or not row.get("executed"):
            continue
        try:
            payload = json.loads(row["result"])
        except (ValueError, TypeError, KeyError):
            continue
        data = payload.get("data")
        if payload.get("ok") is not True or not isinstance(data, dict):
            continue
        # 展示不完整正文时不能标成完整用户原话。
        if data.get("truncated") is not False or data.get("evidence_kind") != "archived_conversation":
            continue
        citation, text = data.get("citation"), data.get("user_text")
        if not isinstance(citation, str) or not citation or not isinstance(text, str) or not text:
            continue
        if citation in seen:
            continue
        seen.add(citation)
        blocks.append("原始用户消息：\n" + text + "\n程序来源：" + citation)
    return "\n\n".join(blocks)
