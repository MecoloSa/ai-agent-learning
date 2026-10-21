"""用户原话的最小排版等价匹配。

只兼容中英文单双引号差异；不改字词、数字、否定词、大小写或空白。
返回真实原文片段，而不是把模型改写后的文本保存为用户原话。
"""

from __future__ import annotations


class SourceQuoteError(ValueError):
    """无法从指定用户消息定位引用，或排版等价匹配存在歧义。"""


_QUOTE_TRANSLATION = str.maketrans({
    "“": '"',
    "”": '"',
    "‘": "'",
    "’": "'",
})


def resolve_user_quote(quote: str, user_message: str) -> str:
    """优先逐字匹配；仅排版不同且匹配唯一时，返回对应原文。"""
    if not isinstance(quote, str) or not quote.strip():
        raise SourceQuoteError("用户原话引用不能为空")

    if not isinstance(user_message, str) or not user_message.strip():
        raise SourceQuoteError("指定用户消息不能为空")

    if quote in user_message:
        return quote

    folded_quote = quote.translate(_QUOTE_TRANSLATION)
    folded_message = user_message.translate(_QUOTE_TRANSLATION)
    start = folded_message.find(folded_quote)

    if start < 0:
        raise SourceQuoteError(
            "引用必须是指定用户原话中的连续片段；"
            "请从来源表复制较短原句，不改写字词、数字或空白。"
        )

    if folded_message.find(folded_quote, start + 1) >= 0:
        raise SourceQuoteError(
            "引号等价匹配对应多处原文，请逐字引用并补充可区分的上下文。"
        )

    # 映射只做单字符到单字符替换，位置和长度不变，才能直接切回原文。
    # 不使用NFKC、模糊匹配或语义相似度，以免放过数字、否定词等变化。
    # 动态评估：兼容排版后仍须拒绝20→40、未完成→完成及伪造引用。
    return user_message[start:start + len(quote)]
