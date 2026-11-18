"""从实际请求断言唯一状态消息；兼容对象与审计序列化格式。"""
import json


def current_state(messages):
    states = []
    for message in messages:
        role = message.get("type") if isinstance(message, dict) else message.type
        content = message.get("content") if isinstance(message, dict) else message.content
        if role != "human" or not isinstance(content, str):
            continue
        try:
            data = json.loads(content)
        except ValueError:
            continue
        if isinstance(data, dict) and data.get("context_type") == "current_learning_state":
            states.append(data)
    assert len(states) == 1, "每轮应包含唯一的程序状态消息"
    return states[0]
