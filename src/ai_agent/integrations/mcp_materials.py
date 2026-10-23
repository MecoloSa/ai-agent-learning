"""本地资料只读MCP服务。

复用现有按行读取工具，不另写路径、字符预算或引用逻辑。
资料目录由启动者固定，不交给模型或客户端选择。
stdio的stdout专供协议消息，不能向其中print调试内容。
"""

from __future__ import annotations

import argparse

from mcp.server.fastmcp import FastMCP
from pydantic import StrictInt

from ai_agent.resources.materials import resolve_material_root
from ai_agent.resources.tools import build_read_material_lines_tool


def create_server(material_dir: str) -> FastMCP:
    """构造服务和绑定工具；不启动进程、不调用模型。"""
    root = resolve_material_root(material_dir)
    local_reader = build_read_material_lines_tool(str(root))
    server = FastMCP("learning-materials")
    # FastMCP 是 mcp SDK 提供的轻量服务框架，"learning-materials" 是服务名，会出现在客户端工具列表中

    @server.tool()
    def read_material_lines(
        relative_path: str,
        start_line: StrictInt,
        end_line: StrictInt,
    ) -> str:
        """读取绑定目录内TXT/MD/PY的指定行，不执行文件。

        行号从1开始且包含两端；返回JSON和实际范围引用。
        不支持PDF；截断结果不能作为完整原文。
        """
        try:
            # 只增加协议入口，业务结果仍由原工具生成。
            return local_reader.invoke({
                "relative_path": relative_path,
                "start_line": start_line,
                "end_line": end_line,
            })
        except (ValueError, OSError) as exc:
            # SDK将异常转为工具错误结果。
            # 不把异常中的绝对路径、输入内容等直接发给客户端。
            raise ValueError(
                f"资料读取失败：{type(exc).__name__}"
            ) from None

    # 动态评估：比较本地/MCP结果、错误标记及额外进程开销。
    # 协议正常返回不等于资料足够，更不代表最终回答正确。
    return server


def main() -> None:
    parser = argparse.ArgumentParser(description="本地资料只读MCP服务")
    parser.add_argument("--material-dir", required=True)
    args = parser.parse_args()

    server = create_server(args.material_dir)
    server.run(transport="stdio")


if __name__ == "__main__":
    main()