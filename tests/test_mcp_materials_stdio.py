"""真实stdio协议冒烟测试，不请求模型、不访问个人资料。

新业务模块未应用时跳过；验证发现、读取等价、错误与会话关闭。
本测试不表示主聊天已通过MCP调用，也不评价回答质量。
"""
import asyncio
import json
from datetime import timedelta
from pathlib import Path
import sys

import pytest
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from ai_agent.resources.tools import build_read_material_lines_tool


ROOT = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.skipif(
    not (ROOT / "src/ai_agent/integrations/mcp_materials.py").exists(),
    reason="MCP只读服务方案尚未应用",
)


def test_mcp_stdio_read_and_boundaries(tmp_path):
    root = tmp_path / "materials"
    root.mkdir()
    (root / "notes.md").write_text("第一行\n第二行\n", encoding="utf-8")
    (tmp_path / "outside.md").write_text("不可读取", encoding="utf-8")
    arguments = {"relative_path": "notes.md", "start_line": 1, "end_line": 2}
    expected = json.loads(build_read_material_lines_tool(str(root)).invoke(arguments))

    async def exercise():
        params = StdioServerParameters(
            command=sys.executable,
            args=["-m", "ai_agent.integrations.mcp_materials", "--material-dir", str(root)],
            env={"PYTHONPATH": str(ROOT / "src"), "PYTHONIOENCODING": "utf-8"},
        )
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write, read_timeout_seconds=timedelta(seconds=20)) as session:
                await session.initialize()
                discovered = await session.list_tools()
                assert [tool.name for tool in discovered.tools] == ["read_material_lines"]
                assert "root_dir" not in discovered.tools[0].inputSchema["properties"]
                print("[MCP] 初始化与工具发现通过，schema不提供根目录参数")
                result = await session.call_tool("read_material_lines", arguments)
                assert result.isError is False
                payload = json.loads("".join(block.text for block in result.content if block.type == "text"))
                assert payload == expected
                print("[MCP] 读取结果与本地Function Calling一致")
                for label, bad in [
                    ("越界", {**arguments, "relative_path": "../outside.md"}),
                    ("非法行号", {**arguments, "start_line": "1"}),
                    ("不存在文件", {**arguments, "relative_path": "missing.md"}),
                ]:
                    failed = await session.call_tool("read_material_lines", bad)
                    assert failed.isError is True
                    print(f"[MCP] {label}拒绝，isError=True")
        print("[MCP] 客户端退出，stdio会话关闭")

    asyncio.run(exercise())
