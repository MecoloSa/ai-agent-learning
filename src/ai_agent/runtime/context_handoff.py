"""跨阶段传递资料正文与获取限制；不把程序记录混入可引用证据。"""
from __future__ import annotations

import json


def pack_material_context(evidence: str, facts: list[dict]) -> str:
    """保持现有字符串/检查点契约；无获取限制时不增加包装。"""
    if not facts:
        return evidence
    return json.dumps({
        "ai_agent_material_context": 1,
        "evidence": evidence,
        "acquisition_facts": facts,
    }, ensure_ascii=False)


def unpack_material_context(context: str) -> tuple[str, list[dict]]:
    """旧文本和旧JSON读取结果原样兼容；已识别的损坏包装明确报错。"""
    try:
        data = json.loads(context)
    except json.JSONDecodeError:
        return context, []
    if not isinstance(data, dict) or "ai_agent_material_context" not in data:
        return context, []
    if (
        type(data["ai_agent_material_context"]) is not int
        or data["ai_agent_material_context"] != 1
        or not isinstance(data.get("evidence"), str)
        or not isinstance(data.get("acquisition_facts"), list)
        or not all(isinstance(item, dict) for item in data["acquisition_facts"])
    ):
        raise ValueError("不支持或损坏的资料交接格式")
    return data["evidence"], data["acquisition_facts"]


def format_acquisition_facts(facts: list[dict]) -> str:
    """仅描述实际获取边界，不指示模型必须补查或拒答。"""
    if not facts:
        return ""
    return (
        "\n\n初始资料获取范围与限制（程序记录，非原文证据）：\n"
        + json.dumps(facts, ensure_ascii=False)
    )
