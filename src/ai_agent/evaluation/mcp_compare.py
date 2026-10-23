"""本地与MCP按行读取的小规模配对对照。

不调用模型、不修改学习状态，不保存资料正文。
一次连接内重复读取，只观察结果等价和工具层开销。
"""

import argparse
import json
from importlib.metadata import version
from pathlib import Path
from statistics import median
from time import perf_counter

from ai_agent.integrations.mcp_reader import open_mcp_reader
from ai_agent.resources.materials import resolve_material_root
from ai_agent.resources.tools import build_read_material_lines_tool


def compare_readers(
    local, remote, arguments: dict, repeat: int
) -> dict:
    if not 2 <= repeat <= 20:
        raise ValueError("repeat必须在2到20之间")

    rows = []
    for index in range(repeat):
        values, timings = {}, {}

        # 交替先后顺序，减轻固定顺序偏差；不等于控制了所有噪声。
        order = [("local", local), ("mcp", remote)]
        if index % 2:
            order.reverse()

        for name, tool in order:
            start = perf_counter()
            value = json.loads(tool.invoke(dict(arguments)))
            timings[name] = round(
                (perf_counter() - start) * 1000, 3
            )

            if (
                not isinstance(value, dict)
                or value.get("ok") is not True
            ):
                raise ValueError("读取没有返回业务成功结果")
            values[name] = value

        rows.append({
            "pair": index + 1,
            "equal": values["local"] == values["mcp"],
            "local_ms": timings["local"],
            "mcp_ms": timings["mcp"],
        })

    return {
        "pair_count": repeat,
        "all_results_equal": all(row["equal"] for row in rows),
        "median_local_ms": median(
            row["local_ms"] for row in rows
        ),
        "median_mcp_ms": median(
            row["mcp_ms"] for row in rows
        ),
        "pairs": rows,
    }


def main():
    parser = argparse.ArgumentParser(
        description="本地与MCP读取对照，不调用LLM"
    )
    parser.add_argument("--material-dir", required=True)
    parser.add_argument("--relative-path", required=True)
    parser.add_argument("--start-line", type=int, required=True)
    parser.add_argument("--end-line", type=int, required=True)
    parser.add_argument(
        "--repeat", type=int, default=5, choices=range(2, 21)
    )
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    root = str(resolve_material_root(args.material_dir))
    arguments = {
        "relative_path": args.relative_path,
        "start_line": args.start_line,
        "end_line": args.end_line,
    }
    local = build_read_material_lines_tool(root)

    started = perf_counter()
    with open_mcp_reader(root) as remote:
        open_ms = round(
            (perf_counter() - started) * 1000, 3
        )
        result = compare_readers(
            local, remote, arguments, args.repeat
        )

    result.update({
        "mcp_open_ms": open_ms,
        "arguments": arguments,
        "mcp_sdk_version": version("mcp"),
        "note": (
            "小样本工具层对照，不含模型调用；"
            "不证明稳定提速或回答质量提升。"
        ),
    })

    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))

    if not result["all_results_equal"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()