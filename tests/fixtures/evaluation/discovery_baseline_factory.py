def build_agent_tools(
    root_dir: str,
    *,
    evidence_blocks: list[str] | None = None,
    acquisition_facts: list[dict] | None = None,
) -> list[BaseTool]:
    """创建绑定资料目录的分析工具，模型不能选择或替换根目录。"""
    root = Path(root_dir).expanduser().resolve()
    if not root.is_dir():
        raise ValueError("资料目录不存在")

    current_settings = get_settings()

    class NoExtraArgs(BaseModel):
        model_config = ConfigDict(extra="forbid")

    class ReadArgs(NoExtraArgs):
        relative_path: str = Field(min_length=1)
        max_chars: StrictInt = Field(
            default=MAX_CHARS_PER_FILE,
            ge=1,
            le=MAX_CHARS_PER_FILE,
        )

    class SearchArgs(NoExtraArgs):
        keyword: str = Field(min_length=1)

    def safe_files():
        # 固定根目录还不够：目录中的文件链接也可能指向外部。
        # 复用解析后的边界检查，列举与搜索都不接受这类文件。
        for path in iter_material_files(root):
            relative_path = path.relative_to(root).as_posix()
            try:
                checked_path = safe_material_path(
                    str(root), relative_path
                )
            except (ValueError, OSError):
                continue
            yield path, checked_path

    @tool(
        "list_learning_files",
        args_schema=NoExtraArgs.model_json_schema(),
    )
    def bound_list_files(**arguments: object) -> str:
        """列出当前绑定目录中的 TXT、MD、PY、PDF 文件，无需目录参数。"""
        # 当前 LangChain 会跳过空 Pydantic schema 的参数校验。
        # 显式提供 JSON schema，让原始参数进入函数，再在扫描前校验。
        # 动态评估：{} 应成功，任何额外字段都必须在读取目录前被拒绝。
        NoExtraArgs.model_validate(arguments)

        files = list(safe_files())
        if not files:
            return "没有找到可分析的资料文件。"

        return "\n".join(
            f"{path.relative_to(root).as_posix()} "
            f"({checked_path.stat().st_size} bytes)"
            for path, checked_path in (
                files[:current_settings.tool_file_list_limit]
            )
        )

    @tool("read_learning_file", args_schema=ReadArgs)
    def bound_read_file(
        relative_path: str,
        max_chars: int = MAX_CHARS_PER_FILE,
    ) -> str:
        """读取绑定目录内 TXT、MD、PY 文件的开头，使用清单中的相对路径。"""
        if Path(relative_path).is_absolute():
            raise ValueError("请使用资料目录内的相对路径")

        data = read_material_excerpt(
            str(root),
            relative_path,
            max_chars,
        )
        # 模型观察与下游证据复用同一读取结果，
        # 保留真实读取范围、来源和截断信息。
        result = json.dumps(
            {"ok": True, "data": data},
            ensure_ascii=False,
        )

        if evidence_blocks is not None and data["text"].strip():
            evidence_blocks.append(result)

        if acquisition_facts is not None and data["truncated"]:
            acquisition_facts.append({
                "tool": "read_learning_file", "status": "truncated",
                "source": data["source"], "scope": data["scope"],
                "returned_chars": data["returned_chars"],
            })

        return result
    @tool("search_learning_files", args_schema=SearchArgs)
    def bound_search_files(keyword: str) -> str:
        """在绑定目录的 TXT、MD、PY 文件中查找关键词，返回文件名和行号。"""
        results: list[str] = []
        for path, checked_path in safe_files():
            if checked_path.suffix.lower() == ".pdf":
                continue

            try:
                lines = checked_path.read_text(
                    encoding="utf-8", errors="replace"
                ).splitlines()
            except OSError:
                continue

            for line_number, line in enumerate(lines, start=1):
                if keyword.casefold() in line.casefold():
                    results.append(
                        f"{path.relative_to(root).as_posix()}:"
                        f"{line_number}: {line.strip()[:160]}"
                    )
                    if (
                        len(results)
                        >= current_settings.tool_search_result_limit
                    ):
                        return "\n".join(results)

        return "\n".join(results) or f"未找到关键词：{keyword}"

    tools: list[BaseTool] = [
        bound_list_files,
        bound_read_file,
        bound_search_files,
        summarize_text_statistics,
        build_read_material_lines_tool(
            str(root),
            evidence_blocks=evidence_blocks,
            acquisition_facts=acquisition_facts,
        ),
    ]
    if needs_rag(str(root)):
        tools.append(build_retrieve_learning_context_tool(
            str(root),
            evidence_blocks=evidence_blocks,
            acquisition_facts=acquisition_facts,
        ))
    # 动态评估：工具 schema 中不应再有 root_dir；
    # 正常读取应成功，额外目录参数及越界路径应被拒绝。
    return tools
