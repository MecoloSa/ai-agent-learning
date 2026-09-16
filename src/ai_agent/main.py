"""命令行入口：运行两个智能体并把结果保存为 Markdown。"""

from __future__ import annotations

import argparse
import logging
import re
from datetime import date
from pathlib import Path

from .agents import (
    run_learning_assistant,
    LLMInvocationError,
    log_citation_summary,
    invoke_llm_response,
)
from .rag import RAGError
from .config import get_settings, resolve_project_path

logger = logging.getLogger("ai_agent")

PLAN_START_TAG = "===NEW_PLAN_START==="
PLAN_END_TAG = "===NEW_PLAN_END==="
GOAL_START_TAG = "===NEW_GOAL_START==="
GOAL_END_TAG = "===NEW_GOAL_END==="

def _build_interactive_system_prompt(analysis: str, plan: str, material_context: str, learning_goal: str) -> str:
    """把资料分析结果、（可能已被用户要求更新过的）计划和资料上下文拼成 system_prompt。

    这里的四类信息更新方式并不相同，函数内的措辞需要对应体现：
    - plan、learning_goal：决策类信息，用户可能随时要求调整，通过下面约定的
      两组标记由模型在回复末尾整体重写，被 main.py 抽取后经
      memory.update_system_message 写回——本函数因此被复用于每次重建整段
      system_prompt。
    - analysis、material_context：证据类信息，是资料分析阶段一次性产出的，
      不应该让对话模型去猜测式地重写；
    """
    return (
        "你是学习计划的对话助手。用户已经拿到以下资料分析结果和学习计划，"
        "接下来可能会要求调整计划、调整学习目标、解释某天任务、或补充细节。"
        "回答时优先基于下面给出的资料内容、分析结果和计划；"
        "所有需要的资料内容已经在下面给出，不需要再查看或读取任何文件。\n\n"
        "如果用户明确要求调整/修改学习计划，你必须在本轮回答的最后，"
        f"用 {PLAN_START_TAG} 和 {PLAN_END_TAG} 包裹一份【完整】的新计划"
        "（必须包含未修改部分+已修改部分的全部内容，不能只给增量或片段）；"
        "如果用户明确要求调整学习目标本身（不是调整计划），你必须在本轮回答的最后，"
        f"用 {GOAL_START_TAG} 和 {GOAL_END_TAG} 包裹更新后的完整学习目标（一句话即可）。\n"
        "以上两组标记互不影响，一轮回答中可以同时出现、只出现一个，或都不出现——"
        "本轮回答不涉及某一项变更时，就完全不要输出对应的标记。\n\n"
        f"当前学习目标：{learning_goal}\n\n"
        f"资料分析结果：\n{analysis}\n\n"
        f"当前学习计划：\n{plan}\n\n"
        f"初始资料证据：\n{material_context}\n\n"
        "当用户询问资料来源、章节内容或需要补充资料证据时，"
        "系统会在当前轮次额外提供按需检索结果；"
        "不要假设初始分析覆盖了整份资料。"
    )

def _extract_tagged_block(response_text: str, start_tag: str, end_tag: str) -> str | None:
    """从模型回复中提取被指定标记包裹的内容；一轮回复里出现多次时取最后一次；未找到时返回 None。

    plan 和 learning_goal 的更新抽取共用这一个通用函数：两者本质上都是模型在回复末尾用一对标记包裹整体重写后的新内容
    """
    matches = re.findall(
        rf"{re.escape(start_tag)}(.*?){re.escape(end_tag)}",
        response_text,
        re.S,
    )
    return matches[-1].strip() if matches else None

def _extract_updated_plan(response_text: str) -> str | None:
    """从模型回复中提取被 PLAN 标记包裹的新计划。"""
    return _extract_tagged_block(response_text, PLAN_START_TAG, PLAN_END_TAG)

def _extract_updated_goal(response_text: str) -> str | None:
    """从模型回复中提取被 GOAL 标记包裹的新学习目标。"""
    return _extract_tagged_block(response_text, GOAL_START_TAG, GOAL_END_TAG)

def run_interactive_session(
        analysis: str, plan: str, material_context: str, learning_goal: str, strategy: str, material_dir: str,
) -> tuple[str, str]:
    """运行多轮对话；返回（最终计划，最终学学习目标）

    若对话期间被模型用 PLAN / GOAL 标记更新过，则返回更新后的版本，否则原样返回传入的值。
    """
    from .memory import ConversationMemory
    from .config import create_llm

    current_plan = plan
    current_goal = learning_goal
    system_prompt = _build_interactive_system_prompt(analysis, current_plan, material_context, learning_goal)
    logger.info("交互式 system_prompt 总长度：%d 字符（其中资料上下文 %d 字符）\n",
                len(system_prompt), len(material_context))
    memory = ConversationMemory(system_prompt=system_prompt, strategy=strategy)
    llm = create_llm()

    from langchain_core.messages import HumanMessage
    from .rag import LearningRAG, needs_rag
    from .config import create_llm, get_settings

    settings = get_settings()
    interactive_rag = (
        LearningRAG(material_dir, data_dir=settings.rag_data_dir)
        if needs_rag(material_dir)
        else None
    )

    print(f"（多轮对话模式，记忆策略={strategy}，输入 exit/quit/q 退出）")
    while True:
        user_input = input("你：").strip()
        if user_input.lower() in {"exit", "quit", "q"}:
            break

        request_messages = memory.get_messages_for_llm()

        if interactive_rag is not None:
            turn_evidence = interactive_rag.retrieve_text(
                query=user_input,
                k=settings.rag_retrieval_k,
                min_relevance=settings.rag_min_relevance,
                relative_margin=settings.rag_relative_margin,
                neighbor_window=settings.rag_neighbor_window,
                max_context_chars=max(
                    1,
                    settings.rag_max_context_chars // 2,
                ),
            )

            turn_prompt = (
                f"用户本轮问题：{user_input}\n\n"
                "以下是针对本轮问题重新检索到的资料证据。"
                "回答时优先使用这些证据；如果与初始分析冲突，"
                "以本轮更直接的正文证据为准。\n\n"
                f"{turn_evidence}"
            )
        else:
            turn_prompt = user_input

        response = invoke_llm_response(
            llm,
            request_messages
            + [HumanMessage(content=turn_prompt)],
            stage="交互回答",
        )
        response_text = str(response.content)
        log_citation_summary("交互回答", response_text,)

        # 记忆中只保存原始问题，不保存大段临时检索上下文。
        memory.add_user(user_input)
        memory.add_ai(response_text)

        print(f"助手：{response_text}\n")

        updated_plan = _extract_updated_plan(response_text)
        updated_goal = _extract_updated_goal(response_text)
        if updated_plan or updated_goal:
            if updated_plan:
                current_plan = updated_plan
                logger.info("检测到模型返回了更新后的学习计划（%d 字符），同步进 system_prompt。", len(current_plan))
            if updated_goal:
                current_goal = updated_goal
                logger.info("检测到模型返回了更新后的学习目标（%d 字符），同步进 system_prompt。", len(current_goal))
            # 检测到更新后，替换 system_message[0] 为最新
            new_system_prompt = _build_interactive_system_prompt(analysis, current_plan, material_context, current_goal)
            memory.update_system_message(
                new_system_prompt,
                plan_changed=updated_plan is not None,
                goal_changed=updated_goal is not None,
            )

    return current_plan, current_goal

def main() -> None:
    parser = argparse.ArgumentParser(description="本地学习资料分析与学习计划助手")
    parser.add_argument("--materials", default="../bailian", help="存放学习资料的目录")
    parser.add_argument("--goal", required=True, help="本次学习目标，例如：掌握项目中的 MCP 基础")
    parser.add_argument("--days", type=int, default=3, help="计划天数，默认为 3")
    parser.add_argument(
        "--output",
        default="plans/study_plan.md",
        help="结果 Markdown 文件路径；相对路径固定基于项目根目录解析",
    )
    parser.add_argument("--mode", choices=["pipeline", "agent"], default="agent",
                        help="运行模式：agent（默认）或 pipeline（固定流程）")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="输出检索任务、命中证据、临界淘汰项和请求规模等调试信息；不输出完整向量、资料正文或模型 messages")
    parser.add_argument("--interactive", action="store_true", help="生成计划后进入多轮对话模式")
    parser.add_argument("--memory-strategy", choices=["full", "window", "summary"], default="window")
    args = parser.parse_args()
    if args.days < 1 or args.days > 30:
        parser.error("--days 必须在 1 到 30 之间。")

    settings = get_settings()
    configured_level = getattr(logging, settings.log_level, logging.INFO)
    level = logging.DEBUG if args.verbose else configured_level

    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    # 即使应用开启 DEBUG，也避免 HTTP、Chroma 等依赖刷屏。
    for noisy_logger in (
            "httpx",
            "httpcore",
            "urllib3",
            "chromadb",
            "openai",
            "dashscope",
    ):
        logging.getLogger(noisy_logger).setLevel(logging.WARNING)
    logger.info("启动参数：materials=%s, goal=%s, days=%s, output=%s",
                args.materials, args.goal, args.days, args.output)
    logger.info("====== 开始运行学习助手 ======")
    try:
        analysis, plan, material_context = run_learning_assistant(
            args.materials, args.goal, args.days, mode=args.mode
        )
    except (LLMInvocationError, RAGError, RuntimeError) as exc:
        logger.error("运行失败：%s", exc)
        print("运行失败，请根据日志中的具体原因检查配置或资料。")
        return

    if not plan:
        # 资料目录为空或指向不存在的路径：第一步已返回提示，直接退出，不再进入文件生成步骤
        logger.warning("资料目录校验未通过，跳过文件生成。")
        print(analysis)
        return

    goal = args.goal
    if args.interactive:
        print(f"\n模型生成的初始学习计划：\n{plan}\n")
        logger.info("====== 开始运行交互修改模块 ======")
        plan, goal = run_interactive_session(analysis, plan, material_context, args.goal, args.memory_strategy, args.materials,)

    document = f"# 学习资料分析与计划\n\n生成日期：{date.today()}\n\n目标：{goal}\n\n## 资料分析智能体\n\n{analysis}\n\n"
    document += (
        f"## 学习计划智能体\n\n{plan}\n"
        if plan
        else "## 学习计划智能体\n\n（资料目录校验未通过，未生成学习计划，请参考上方资料分析结果中的指导信息）\n"
    )

    # 不直接使用 Path(args.output)：它会受 IDE/终端当前工作目录影响，
    # 导致同一命令有时写到项目根目录、有时写到 src/ai_agent。
    output = resolve_project_path(args.output)
    try:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(document, encoding="utf-8")
    except OSError as exc:
        logger.error(
            "保存结果失败：path=%s error=%s",
            output,
            exc,
        )
        print(f"生成成功，但无法保存结果：{exc}")
        return
    print(document)
    print(f"\n已保存到：{output.resolve()}")


if __name__ == "__main__":
    main()
