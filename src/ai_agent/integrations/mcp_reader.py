"""同步工具接口到持久stdio会话的适配。

不改读取业务；连接在with内复用，退出时关闭，不静默回退本地执行。
供现有同步工具循环使用，不自行计数、确认或保存学习状态。
"""

from contextlib import asynccontextmanager, contextmanager
from datetime import timedelta
import json
from pathlib import Path
import sys

from anyio import fail_after
from anyio.from_thread import start_blocking_portal
from langchain_core.tools import StructuredTool
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from pydantic import BaseModel, StrictInt

from ai_agent.resources.materials import resolve_material_root


class McpLineReadInput(BaseModel):
    relative_path: str
    start_line: StrictInt
    end_line: StrictInt


@asynccontextmanager
async def _reader_session(material_dir: str):
    import ai_agent

    source_dir = Path(ai_agent.__file__).resolve().parent.parent
    params = StdioServerParameters(
        command=sys.executable,
        args=[
            "-m", "ai_agent.integrations.mcp_materials",
            "--material-dir", material_dir,
        ],
        env={
            "PYTHONPATH": str(source_dir),
            "PYTHONIOENCODING": "utf-8",
        },
    )

    async with stdio_client(params) as (read, write):
        async with ClientSession(
            read,
            write,
            read_timeout_seconds=timedelta(seconds=20),
        ) as session:
            with fail_after(20):
                await session.initialize()
                tools = await session.list_tools()

            if [tool.name for tool in tools.tools] != [
                "read_material_lines"
            ]:
                raise RuntimeError("MCP服务工具集合与预期不符")

            yield session


@contextmanager
def open_mcp_reader(material_dir: str):
    """with期间提供可invoke的工具；退出后不允许继续调用。"""
    root = str(resolve_material_root(material_dir))

    # Portal维护异步事件循环，让同步.invoke复用同一连接。
    # 不在每次调用时重新启动服务，也不迁移整个聊天为异步。
    with start_blocking_portal() as portal:
        with portal.wrap_async_context_manager(
            _reader_session(root)
        ) as session:
            active = True

            async def remote_call(arguments):
                with fail_after(20):
                    result = await session.call_tool(
                        "read_material_lines", arguments
                    )

                if result.isError:
                    raise RuntimeError("MCP资料工具返回错误")

                if (
                    len(result.content) != 1
                    or result.content[0].type != "text"
                ):
                    raise RuntimeError(
                        "MCP资料结果不是预期的单段文本"
                    )

                text = result.content[0].text
                payload = json.loads(text)
                if (
                    not isinstance(payload, dict)
                    or payload.get("ok") is not True
                ):
                    raise RuntimeError(
                        "MCP资料结果缺少业务成功标记"
                    )
                return text

            def invoke(
                relative_path: str,
                start_line: int,
                end_line: int,
            ) -> str:
                if not active:
                    raise RuntimeError("MCP读取会话已关闭")

                return portal.call(remote_call, {
                    "relative_path": relative_path,
                    "start_line": start_line,
                    "end_line": end_line,
                })

            adapter = StructuredTool.from_function(
                func=invoke,
                name="read_material_lines",
                description="读取已关联资料目录内 TXT/MD/PY 的指定行。相对路径；行号从1开始，包含两端，每次最多200行。",
                args_schema=McpLineReadInput,
            )

            try:
                yield adapter
            finally:
                # 错误向调用方报告，不自动重读或绕过原预算。
                # 动态评估：核对错误、跨轮计数和连接开销。
                active = False
