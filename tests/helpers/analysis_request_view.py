"""分析请求视图的恒等候选：只写替换文件，不写正式src。"""
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import sys
import types
import argparse
import re

ROOT = Path(__file__).resolve().parents[2]
SOURCE = '''"""从完整消息历史构建模型请求副本；首版不筛选、不压缩。"""
from collections.abc import Sequence
from copy import deepcopy

from langchain_core.messages import BaseMessage


def prepare_model_messages(history: Sequence[BaseMessage], *, tool_result_formatter=None) -> list[BaseMessage]:
    """保持消息内容、顺序和调用ID，返回与原始历史隔离的请求副本。"""
    # 仅list(history)仍共享消息及嵌套参数；深复制隔离后续视图修改。
    # 字符串正文不被改写；本版不判断哪些历史已无用。
    # 恒等基线明确忽略整理器，以兼容正式入口新增的可选参数。
    return deepcopy(list(history))
'''


@contextmanager
def candidate(output):
    import ai_agent.runtime.agents as agents
    path = ROOT / "src/ai_agent/runtime/agents.py"
    before = path.read_bytes()
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    helper_path = output / "request_context.py"
    helper_path.write_text(SOURCE, encoding="utf-8")
    name = "ai_agent.runtime.request_context"
    previous = sys.modules.get(name)
    module = types.ModuleType(name)
    module.__file__ = str(helper_path)
    exec(compile(SOURCE, str(helper_path), "exec"), module.__dict__)
    original = agents.__dict__.copy()
    source = before.decode("utf-8")
    old = '            llm, messages, stage=f"资料分析-第{round_idx + 1}轮"'
    new = '            llm, prepare_model_messages(messages), stage=f"资料分析-第{round_idx + 1}轮"'
    if source.count(old) == 1:
        source = source.replace(old, new, 1)
        source = source.replace(
            "from pathlib import Path",
            "from pathlib import Path\nfrom ai_agent.runtime.request_context import prepare_model_messages", 1,
        )
    elif len(re.findall(
        r'llm,\s*prepare_model_messages\(\s*messages\s*(?:,\s*tool_result_formatter=stage_evidence\s*,?\s*)?\),\s*stage=f"资料分析-第\{round_idx \+ 1\}轮"',
        source,
    )) != 1:
        raise ValueError("分析调用锚点不唯一，拒绝加载候选")
    candidate_path = output / "agents.py"
    candidate_path.write_text(source, encoding="utf-8")
    sys.modules[name] = module
    try:
        exec(compile(source, str(candidate_path), "exec"), agents.__dict__)
        yield agents, module.prepare_model_messages
    finally:
        agents.__dict__.clear()
        agents.__dict__.update(original)
        if previous is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = previous
        if path.read_bytes() != before:
            raise RuntimeError("正式源码变化，停止实验")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("pytest_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = args.output_dir / "manifest.json"
    if manifest.exists():
        parser.error("不覆盖已有实验")
    path = ROOT / "src/ai_agent/runtime/agents.py"
    before = hashlib.sha256(path.read_bytes()).hexdigest()
    with candidate(args.output_dir / "replacement-files"):
        import pytest
        test_args = args.pytest_args
        if test_args[:1] == ["--"]:
            test_args = test_args[1:]
        code = pytest.main(test_args)
    manifest.write_text(json.dumps({"mode": "offline_identity_request_view",
        "exit_code": int(code), "source_sha256": before,
        "source_unchanged": before == hashlib.sha256(path.read_bytes()).hexdigest(),
    }, indent=2), encoding="utf-8")
    raise SystemExit(code)


if __name__ == "__main__":
    main()
