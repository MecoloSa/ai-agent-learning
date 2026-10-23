"""真实计划暴露的字段标题括号说明兼容测试。

只检查结构提取，不把标题识别当作内容质量或引用支持度通过。
业务片段尚未应用时跳过，保留已有严格空字段边界。
"""
import pytest

from ai_agent.runtime import plan_validation as validation


pytestmark = pytest.mark.skipif(
    validation._FIELD.match("具体行动（限时）：正文") is None,
    reason="字段括号说明兼容片段尚未应用",
)


@pytest.mark.parametrize("note", ["（限时20分钟，仅阅读与口头复述）", "(限时)", ""])
def test_field_annotation_keeps_action_recognizable(note):
    text = f"""任务1：比较
学习内容：两类停止
具体行动{note}：阅读案例并口头比较
完成标志：说出停止决定来自谁
证据位置：[source=notes.md, lines=1-4]
"""
    assert validation.inspect_plan_format(text)["warnings"] == []


def test_empty_annotated_action_does_not_borrow_next_field():
    text = """任务1：比较
学习内容：两类停止
具体行动（限时）：
完成标志：能比较
证据位置：[source=notes.md, lines=1-4]
"""
    warnings = validation.inspect_plan_format(text)["warnings"]
    assert any("具体行动" in item for item in warnings)
