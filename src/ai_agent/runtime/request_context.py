"""从完整历史构建请求副本，并按需整理已知读取回执。"""

from collections.abc import Callable, Sequence
from copy import deepcopy

from langchain_core.messages import AIMessage, BaseMessage, ToolMessage


def prepare_model_messages(
    history: Sequence[BaseMessage],
    *,
    tool_result_formatter: Callable[[str], str] | None = None,
) -> list[BaseMessage]:
    """只改变请求副本的读取正文表示，保留消息顺序和调用配对。"""
    messages = deepcopy(list(history))
    if tool_result_formatter is None:
        return messages

    # 以调用ID关联工具名，不依赖ToolMessage可能缺失的name字段。
    # 不删除任何调用或结果；未知工具与非文本内容保持原样。
    read_call_ids: set[str] = set()
    for message in messages:
        if isinstance(message, AIMessage):
            for call in message.tool_calls:
                if call["name"] in {"read_learning_file", "read_material_lines"}:
                    read_call_ids.add(call["id"])
        elif (
            isinstance(message, ToolMessage)
            and message.tool_call_id in read_call_ids
            and isinstance(message.content, str)
        ):
            # 格式识别与完整性校验由现有stage_evidence复用维护；
            # 失败、未知结构等回执由它原样返回，不用模型改写证据。
            message.content = tool_result_formatter(message.content)
    return messages