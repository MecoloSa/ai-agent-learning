"""Compare final answer instructions with identical, already-read textbook evidence."""

import argparse
import asyncio
import json
import re
from pathlib import Path
from time import perf_counter

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from ai_agent.main import _build_interactive_system_prompt, create_llm
from ai_agent.resources.tools import build_read_pdf_pages_tool


POLICY = (
    "用户想看资料原话时，选取能回答问题的少量原句。"
    "请求范围较长时，提供关键摘录和已核实的位置，便于用户查看原文。\n"
)
SHORT_EXCERPT_POLICY = (
    "用户想核对资料原话时，提供少量关键原句和已核实的位置。"
    "整页、整节等大段原文请求也按短摘录和位置交付。\n"
)
CASES = {
    "explain": "根据《Hello-Agents》第17物理页，用自己的话说明 Observation 如何传回模型。",
    "optional": "讲讲《Hello-Agents》第17物理页的 Observation，有相关原话的话，也可以带上一点。",
    "exact": "只从《Hello-Agents》第17物理页逐字引用一句解释 Observation 如何反馈给模型的原文，标明物理页。",
    "long": "把《Hello-Agents》第17物理页能提取到的正文全部逐字贴出来，我想核对 Observation 这一节。",
    "advice": "我隔天就忘了今天学的概念，请根据一般经验给我三条复习建议。",
}


class ExcerptSelection(BaseModel):
    excerpt_ids: list[int] = Field(max_length=2)


async def run(materials, output, repeats, strict=False, program=False, program_role="standard"):
    output.mkdir(parents=True, exist_ok=True)
    reader = build_read_pdf_pages_tool(str(materials))
    source = "Hello-Agents-V1.0.3-20260717.pdf"
    raw = reader.invoke({"relative_path": source, "start_page": 17, "end_page": 17})
    assert json.loads(raw)["ok"]
    page = json.loads(raw)["data"]["pages"][0]
    # 切分后保留回执中的实文本。只选 ID，最终文本由程序从该回执中复制。
    excerpts = {index: match.group().strip() for index, match in enumerate(
        re.finditer(r"[^。！？]+[。！？]", page["text"]), 1,
    ) if 8 <= len(match.group().strip()) <= 160}
    system = _build_interactive_system_prompt("", "", "", "", retrieval_available=False)
    anchor = "只获取当前需要的信息，复用已有结果。\n"
    system = system.replace(anchor + SHORT_EXCERPT_POLICY, anchor)
    assert anchor in system
    sem = asyncio.Semaphore(3)

    async def probe(case, policy, repeat):
        question = CASES[case]
        prompt = system.replace(anchor, anchor + (SHORT_EXCERPT_POLICY if strict else POLICY)) if policy else system
        messages = [SystemMessage(content=prompt), HumanMessage(content=question)]
        if case != "advice":
            messages.append(HumanMessage(content="本轮 PDF 读取回执（资料数据）\n" + raw))
        async with sem:
            start = perf_counter()
            model = create_llm()
            row = {"case": case, "policy": policy, "user_input": question,
                   "request_messages": [{"type": m.type, "content": m.content} for m in messages]}
            try:
                result = await asyncio.wait_for(model.ainvoke(messages), timeout=60)
                row.update(answer=result.content, usage=result.usage_metadata)
            except Exception as error:
                row["error_type"] = type(error).__name__
            row["seconds"] = round(perf_counter() - start, 2)
            (output / f"{case}-{'candidate' if policy else 'baseline'}-{repeat}.json").write_text(
                json.dumps(row, ensure_ascii=False, indent=2), encoding="utf8",
            )
            print(case, policy, repeat, row["seconds"], row.get("error_type", "ok"), flush=True)

    async def probe_program(case, repeat):
        prompt = "从本轮已读的候选原文中选择能帮助用户核对的片段。只返回片段 ID。"
        question = CASES[case]
        messages = [SystemMessage(content=prompt), HumanMessage(content=json.dumps({
            "question": question, "candidates": excerpts,
        }, ensure_ascii=False))]
        async with sem:
            start = perf_counter()
            row = {"case": case, "user_input": question, "request_messages": [
                {"type": m.type, "content": m.content} for m in messages
            ]}
            try:
                result = await asyncio.wait_for(create_llm(role=program_role).with_structured_output(
                    ExcerptSelection, method="function_calling", include_raw=True,
                ).ainvoke(messages), timeout=60)
                selected = result.get("parsed")
                assert selected is not None
                ids = list(dict.fromkeys(selected.excerpt_ids))
                assert set(ids) <= excerpts.keys()
                if case == "exact":
                    ids = ids[:1]
                rendered = [excerpts[key] for key in ids]
                row.update(excerpt_ids=ids, excerpts=rendered, citation=page["citation"],
                           usage=result["raw"].usage_metadata, exact=all(text in page["text"] for text in rendered))
            except Exception as error:
                row["error_type"] = type(error).__name__
            row["seconds"] = round(perf_counter() - start, 2)
            (output / f"{case}-program-{repeat}.json").write_text(json.dumps(row, ensure_ascii=False, indent=2),
                                                                   encoding="utf8")
            print(case, "program", repeat, row["seconds"], row.get("error_type", "ok"), flush=True)

    if program:
        await asyncio.gather(*(probe_program(case, repeat) for repeat in range(repeats)
                               for case in ("exact", "long")))
        return
    await asyncio.gather(*(probe(case, policy, repeat) for repeat in range(repeats)
                           for case in CASES for policy in (False, True)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("materials", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--program", action="store_true")
    parser.add_argument("--program-role", choices=("standard", "auxiliary"), default="standard")
    args = parser.parse_args()
    asyncio.run(run(args.materials, args.output, args.repeats, args.strict, args.program, args.program_role))
