"""跨进程回归的离线入口；保留真实CLI、确认、记忆与磁盘保存。"""
import builtins
import json
import sys
from pathlib import Path

import ai_agent.main as app


def main():
    spec = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    remaining = iter(spec["inputs"])
    observations = []

    def read_input(prompt=""):
        # 意外增加输入入口时直接失败，不能补默认确认来制造通过。
        return next(remaining)

    def unexpected(*args, **kwargs):
        raise AssertionError("恢复路径不应重新生成初始计划或降级为状态管理")

    def invoke(**kwargs):
        observations.append({
            "user_input": kwargs["user_input"],
            "request_messages": [
                {"type": message.type, "content": message.content}
                for message in kwargs["request_messages"]
            ],
        })
        return spec["response"]

    # 替换外部模型边界，不替换会话、确认、序列化及恢复逻辑。
    builtins.input = read_input
    app.create_llm = lambda **kwargs: object()
    app.create_llm_with_tools = lambda *args, **kwargs: object()
    app.needs_rag = lambda _: False
    app._invoke_interactive_turn = invoke
    app._run_initial_workflow = unexpected
    app.run_state_recovery_session = unexpected
    # 只隔离本次写入目标，防止pytest子进程使用个人current目录。
    app.new_current_state_path = lambda: Path(spec["current"])
    if spec.get("history_root") is not None:
        original_resolve = app.resolve_project_path
        app.resolve_project_path = lambda value: (
            Path(spec["history_root"]) if value == ".agents_data/history_archives"
            else original_resolve(value)
        )
    sys.argv = ["agent", "--resume-state", spec["source"], "--interactive"]
    if spec.get("archive_history") is not None:
        sys.argv.append("--archive-history" if spec["archive_history"] else "--no-archive-history")
    app.main()
    assert next(remaining, None) is None, "存在未消费的测试输入"
    Path(spec["observation"]).write_text(
        json.dumps(observations, ensure_ascii=False, indent=2), encoding="utf-8",
    )


if __name__ == "__main__":
    main()
