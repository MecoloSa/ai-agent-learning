"""交互PDF回答的局部证据检查。

只检查本轮审计中可观察的事实，不调用LLM、不修改回答。
引用位置受到支持，不代表引用对应的具体陈述正确。
未发现规则问题也不代表回答整体通过，需要人工或后续语义评审。
"""

from __future__ import annotations

import json
import re


SOURCE_TAG = re.compile(r"\[source=[^\]\n]+\]")
PDF_PAGE_TAG = re.compile(
    r"\[source=(?P<source>[^,\]\n]+),\s*page=(?P<page>[0-9]+)\]"
)


def check_pdf_answer(
    trace: dict,
    *,
    answer: str | None = None,
) -> dict:
    """检查最终回答，或调用方明确指定的模型原始回答。"""
    if trace.get("schema_version") != 1:
        raise ValueError("暂不支持该审计轨迹版本")

    if trace.get("finished") is not True:
        # 未正常结束的轨迹不能当作完整实验给出通过/失败结论。
        return {
            "answer_nonempty": None,
            "pdf_read_succeeded": None,
            "page_citation_supported": None,
            "unsupported_citations": [],
            "needs_human_review": True,
            "review_notes": ["轨迹未正常结束，暂不判定"],
        }

    text = trace.get("final_answer") if answer is None else answer
    if not isinstance(text, str):
        raise ValueError("待检查回答必须是字符串")

    supported_pages: set[tuple[str, int]] = set()
    succeeded = False
    evidence_incomplete = False
    notes: list[str] = []

    for record in trace.get("tool_results", []):
        if record.get("tool_name") != "read_pdf_pages":
            continue
        if record.get("executed") is not True:
            # 被预算或重复保护拒绝的请求，不构成读取证据。
            continue

        try:
            payload = json.loads(record["result"])
        except (KeyError, TypeError, json.JSONDecodeError):
            evidence_incomplete = True
            continue

        if not isinstance(payload, dict):
            evidence_incomplete = True
            continue

        if payload.get("ok") is False:
            continue
        if payload.get("ok") is not True:
            evidence_incomplete = True
            continue

        data = payload.get("data")
        if not isinstance(data, dict):
            evidence_incomplete = True
            continue

        source = data.get("source")
        pages = data.get("pages")
        if (
            not isinstance(source, str)
            or not source
            or not isinstance(pages, list)
            or not pages
        ):
            evidence_incomplete = True
            continue

        valid_pages = []
        for page in pages:
            if not isinstance(page, dict):
                evidence_incomplete = True
                continue

            number = page.get("page")
            page_text = page.get("text")
            if (
                type(number) is not int
                or number < 1
                or not isinstance(page_text, str)
            ):
                evidence_incomplete = True
                continue

            valid_pages.append((source, number))

            if not page_text.strip():
                notes.append(
                    "存在无可提取文字页；不能据此判断页面空白"
                )
            if page.get("partial") is True:
                notes.append("存在只返回部分文字的页面")

        if valid_pages:
            succeeded = True
            supported_pages.update(valid_pages)

        if data.get("truncated") is True:
            notes.append("工具返回发生截断，需检查回答是否说明限制")

    tags = list(dict.fromkeys(SOURCE_TAG.findall(text)))
    unsupported = []
    unknown_format = False

    for tag in tags:
        match = PDF_PAGE_TAG.fullmatch(tag)
        if match is None:
            # lines或page/chunk引用不属于本检查器的页级检查范围。
            unknown_format = True
            continue

        location = (
            match.group("source").strip(),
            int(match.group("page")),
        )
        if location not in supported_pages:
            unsupported.append(tag)

    if unsupported and not evidence_incomplete:
        citation_supported = False
    elif not tags or unknown_format or evidence_incomplete:
        citation_supported = None
    else:
        citation_supported = True

    if not tags:
        notes.append("回答没有位置引用，无法进行页级引用检查")
    if unknown_format:
        notes.append("存在本检查器暂不支持的引用格式")
    if evidence_incomplete:
        notes.append("部分工具证据无法解析，不能据此断言引用错误")
    if unsupported:
        notes.append("部分页级引用没有本轮成功PDF读取结果支持")

    # 这里只发出复核提示，不凭关键词直接判错。
    # “不代表内容完整”也可能匹配，因此不能把命中当作语义结论。
    if re.search(r"(?:内容|页面信息|页面内容)\s*完整", text):
        notes.append(
            "出现完整性措辞：人工确认是否把未截断误说成页面信息完整"
        )

    # 动态评估：先人工核对误报与漏报，再扩大规则覆盖。
    # 本轮不合成整体正确率，避免把位置一致冒充内容忠实。
    return {
        "answer_nonempty": bool(text.strip()),
        "pdf_read_succeeded": (
            None if evidence_incomplete else succeeded
        ),
        "page_citation_supported": citation_supported,
        "unsupported_citations": unsupported,
        "needs_human_review": True,
        "review_notes": list(dict.fromkeys(notes + [
            "尚未检查具体陈述、引文内容与PDF正文是否一致"
        ])),
    }

def check_retrieval_to_pdf_handoff(
    trace: dict,
    *,
    source: str,
    page: int,
) -> bool | None:
    """检查目标位置是否先被检索返回，再由后续模型响应请求读取。

    True仅表示位置与可观察顺序吻合，不表示读取成功或内容正确。
    None表示轨迹未完成，或缺少关联调用所需的审计信息。
    """
    if trace.get("schema_version") != 1:
        raise ValueError("暂不支持该审计轨迹版本")
    if not isinstance(source, str) or not source:
        raise ValueError("source必须是非空字符串")
    if type(page) is not int or page < 1:
        raise ValueError("page必须是从1开始的整数")
    if trace.get("finished") is not True:
        return None

    # 调用ID把工具结果关联回生成它的模型响应。
    # 不能只看工具执行顺序：同一响应中的两个调用并不存在结果反馈。
    response_index_by_call_id = {
        call["id"]: index
        for index, response in enumerate(trace.get("model_responses", []))
        for call in response.get("tool_calls", [])
        if isinstance(call.get("id"), str)
    }

    # 按当前format_documents的证据头格式识别目标位置。
    # re.escape避免文件名中的点号、括号被当作正则表达式语法。
    header_pattern = re.compile(
        r"\[source="
        + re.escape(source)
        + rf", page={page}, chunk=[^,\]\n]+, role=(?:hit|neighbor)"
        + r"(?:, [^\]\n]+)?\]"
    )
    expected_args = {
        "relative_path": source,
        "start_page": page,
        "end_page": page,
    }

    prior_retrievals = []
    unknown = False

    for record in trace.get("tool_results", []):
        if record.get("executed") is not True:
            continue

        if record.get("tool_name") == "retrieve_learning_context":
            prior_retrievals.append(record)
            continue

        if (
            record.get("tool_name") != "read_pdf_pages"
            or record.get("args") != expected_args
        ):
            continue

        read_index = response_index_by_call_id.get(record.get("tool_call_id"))
        if read_index is None:
            unknown = True
            continue

        for retrieval in prior_retrievals:
            retrieval_index = response_index_by_call_id.get(
                retrieval.get("tool_call_id")
            )
            result = retrieval.get("result")
            if retrieval_index is None or not isinstance(result, str):
                unknown = True
                continue

            if retrieval_index >= read_index:
                # 同一模型响应里的并列调用，也不能算根据观察结果继续行动。
                continue

            # 只检查证据块首行，不扫描正文中的普通引用。
            # 这是当前文本协议的匹配，不是防伪或语义真实性验证。
            headers = [
                block.split("\n", 1)[0].strip()
                for block in result.split("\n\n---\n\n")
            ]
            if any(header_pattern.fullmatch(header) for header in headers):
                return True

    # 动态评估：格式变更时需同步修改这里和测试；
    # False表示未找到符合当前协议的链路，不直接等同回答错误。
    return None if unknown else False