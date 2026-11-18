"""静态盘点当前包的模型消息、规则常量、工具说明；不创建模型。"""
import argparse
import ast
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def inventory():
    rows, files = [], []
    for path in sorted((ROOT / "src/ai_agent").rglob("*.py")):
        relative = path.relative_to(ROOT)
        if any(part in {"eval", "架构优化", "experiments", "__pycache__"} for part in relative.parts):
            continue
        files.append(relative.as_posix())
        text = path.read_text(encoding="utf-8")
        tree = ast.parse(text)
        parents = {}
        for parent in ast.walk(tree):
            for child in ast.iter_child_nodes(parent):
                parents[child] = parent

        def owner(node):
            names = []
            while node in parents:
                node = parents[node]
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    names.append(node.name)
            return ".".join(reversed(names)) or "<module>"

        def add(node, kind, expression):
            rows.append({"file": relative.as_posix(), "line": node.lineno,
                         "owner": owner(node), "kind": kind,
                         "expression": expression,
                         "literal_chars": sum(len(n.value) for n in ast.walk(node)
                                              if isinstance(n, ast.Constant) and isinstance(n.value, str))})

        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                name = ast.unparse(node.func)
                if name.split(".")[-1] in {"SystemMessage", "HumanMessage", "AIMessage", "ToolMessage"}:
                    add(node, "message", ast.unparse(node))
                for keyword in node.keywords:
                    if keyword.arg == "description" and not name.endswith("ArgumentParser"):
                        add(keyword.value, "description", ast.unparse(keyword.value))
                if name.endswith("StructuredTool.from_function"):
                    functions = [kw.value.id for kw in node.keywords
                                 if kw.arg == "func" and isinstance(kw.value, ast.Name)]
                    for function in ast.walk(tree):
                        if isinstance(function, ast.FunctionDef) and function.name in functions:
                            doc = ast.get_docstring(function)
                            if doc:
                                add(function, "factory_tool_docstring", doc)
            if isinstance(node, (ast.Assign, ast.AnnAssign)):
                targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                for target in targets:
                    if isinstance(target, ast.Name) and any(key in target.id.upper() for key in ("PROMPT", "RULES")):
                        if node.value is not None:
                            add(node, "prompt_or_rule_assignment", f"{target.id} = {ast.unparse(node.value)}")
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if any(ast.unparse(d).startswith("tool") for d in node.decorator_list):
                    add(node, "tool_docstring", ast.get_docstring(node) or "")
            if isinstance(node, ast.Dict):
                pairs = {k.value: v for k, v in zip(node.keys, node.values) if isinstance(k, ast.Constant) and isinstance(k.value, str)}
                role = pairs.get("role")
                if isinstance(role, ast.Constant) and role.value in {"system", "developer", "user"} and "content" in pairs:
                    add(node, "role_dictionary", ast.unparse(node))
            if isinstance(node, ast.Return) and node.value is not None:
                if "render_" in owner(node) and any(
                    isinstance(n, ast.Constant) and isinstance(n.value, str)
                    for n in ast.walk(node.value)
                ):
                    add(node, "rendered_receipt_candidate", ast.unparse(node.value))
    return {"scope": "current src/ai_agent Python files; excludes legacy eval/architecture notes/experiments",
            "files_scanned": files, "entries": sorted(rows, key=lambda r: (r["file"], r["line"], r["kind"]))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    if (output / "inventory.json").exists():
        parser.error("拒绝覆盖已有盘点")
    data = inventory()
    (output / "inventory.json").write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    table = ["# 提示词静态入口清单", "", "逐项表达式见同目录inventory.json。literal_chars不是实际请求字符，更不是token；项目间存在引用，不能相加为请求长度。", "", "|文件|行|所属函数/类|类型|", "|---|---:|---|---|"]
    table.extend(f"|{r['file']}|{r['line']}|{r['owner']}|{r['kind']}|" for r in data["entries"])
    (output / "inventory.md").write_text("\n".join(table) + "\n", encoding="utf-8")
    print(json.dumps({"files": len(data["files_scanned"]), "entries": len(data["entries"]),
                      "files_with_entries": len({r['file'] for r in data['entries']})}))


if __name__ == "__main__":
    main()
