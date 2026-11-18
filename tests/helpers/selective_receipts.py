"""方案一影子实现：复用正式函数源码，只增加不完整回执保留分支。"""
import inspect
import json

import ai_agent.runtime.agents as agents


def candidate_source():
    source = inspect.getsource(agents.stage_evidence)
    if 'if data["truncated"]:' in source and 'if any(row["partial"] for row in rows):' in source:
        return source
    first = '    prefix_keys = {"source", "citation", "scope", "returned_chars", "truncated", "text"}'
    second = '    if (rows[0]["line"] != data["returned_start_line"]'
    if source.count(first) != 1 or source.count(second) != 1:
        raise ValueError("读取整理函数锚点变化，拒绝加载候选")
    source = source.replace(first,
        '    # 不完整读取保留原始结构；不强制模型继续读取。\n'
        '    if data["truncated"]:\n        return block\n' + first, 1)
    return source.replace(second,
        '    # 部分行也不转换成普通完整正文表示。\n'
        '    if any(row["partial"] for row in rows):\n        return block\n' + second, 1)


def candidate_formatter():
    namespace = {"json": json}
    exec(compile(candidate_source(), "selective_receipts_candidate", "exec"), namespace)
    return namespace["stage_evidence"]
