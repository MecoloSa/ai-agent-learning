"""Run an isolated, real-model study conversation and retain each receipt."""

import argparse
import json
from pathlib import Path
from time import perf_counter
from uuid import uuid4

from ai_agent.learning.learning_state import LearningState, LearningTask
from ai_agent.learning.workspace_service import WorkspaceCommand, WorkspaceService
from ai_agent.learning.workspace_store import WorkspaceStore
from ai_agent.persistence.session_materials import create_material_binding


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("scenario", choices=("reading", "textbook", "resume", "evidence", "review", "tutoring"))
    parser.add_argument("materials", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    resume = args.scenario in {"resume", "review"}
    if not resume:
        args.output.mkdir(parents=True, exist_ok=False)
    reading = args.scenario != "textbook"
    state = LearningState(
        learning_goal="解释重读、回想与核对的作用" if reading else "解释工具调用从发起到返回的过程",
        plan_markdown="阅读20分钟，口头解释10分钟。" if reading else "阅读教材并练习解释工具调用，共40分钟。",
        tasks=(LearningTask(task_id="read-001", title="阅读短文", activity_status="completed"),
               LearningTask(task_id="explain-002", title="口头解释")) if reading else (),
        material_binding=create_material_binding(str(args.materials.resolve())),
    )
    db = args.output / "study.sqlite3"
    service = WorkspaceService(WorkspaceStore(db, None if resume else state), args.output / "archives")
    receipt = args.output / "turns.jsonl"

    def run(action, **payload):
        before = service.store.load()
        request = WorkspaceCommand(action=action, version=before.version, request_id=uuid4().hex, payload=payload)
        started = perf_counter()
        stream = []
        row = {"request": request.model_dump(mode="json"), "before": before.state.model_dump(mode="json", exclude={"conversation_memory"})}
        try:
            event = service.execute(request, stream_observer=lambda kind, data: (
                stream.append({"type": kind, "data": data}) if kind in {"tool_activity", "phase"} else None))
            row["event"] = event.model_dump(mode="json")
        except Exception as exc:
            row["error"] = type(exc).__name__
            event = None
        after = service.store.load()
        row.update(seconds=round(perf_counter() - started, 3), stream=stream,
                   after=after.state.model_dump(mode="json", exclude={"conversation_memory"}), version=after.version)
        with receipt.open("a", encoding="utf-8") as file:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")
        return event

    def chat(text):
        return run("chat", text=text)

    if args.scenario == "tutoring":
        chat("我刚读完《从重读到回想》，以为背得流利就等于理解。根据第二节给我一道短题，先别揭晓答案，我想练习怎么判断是否真的懂。")
        chat("这题先搁置。我准备做组会汇报，请先明确回答：第四节能证明这种方法对所有人都有效吗？给我原文依据。")
        chat("回到第一道题。我的答案是：默背时能说对字词就是理解。我现在只听你指出最大的问题，再给一次提示，我想自己改。")
        run("feedback", task_id="read-001", text="文章读完了，能解释为什么要核对原文，但还不会用新例子说明。")
        first = chat("今天只剩8分钟。依据我刚记录的反馈起草复习候选，安排一次口头练习和一次核对。练习还没开始，不要更改登记的任务或反馈，等我确认。")
        if first and first.proposal_id:
            run("decide", proposal_id=first.proposal_id, decision="reject")
        second = chat("改为5分钟。我先自己说一次，再核对理由和适用范围。重提完整候选，仍然等我确认。")
        if second and second.proposal_id:
            run("decide", proposal_id=second.proposal_id, decision="adopt")
    elif args.scenario == "review":
        chat("今天先到这里。根据我卡住的地方，给我一份下次 15 分钟的复习候选，只安排一项回顾和一项练习。我还没汇报完成哪些阅读、也还没回答上一道题，请不要推断进度。")
    elif resume:
        chat("请告诉我目前已生效的练习安排、剩余时间和阅读进度。先不要改变任何记录。")
    elif args.scenario == "evidence":
        chat("请依据《从重读到回想》第四节回答：文章里的方法是不是所有材料、所有人都有效？作者实际观察到了什么？")
    elif reading:
        chat("我今天要学完《从重读到回想》。先用不超过四句话说清主旨，再告诉我哪两段最值得精读，并解释原因。")
        chat("我的理解是，只要合上文章能顺畅复述，就等于我已经理解了。哪里有问题？先提示我，再出一道能检查这个区别的题，先别给答案。")
        chat("我的答案是：先把文章背下来，再看看和原文用词是不是一样。")
        first = chat("文章已经读完了，我还不太会用新例子解释。今天只剩 10 分钟，只想口头练习，不想重读全文。请给我一份调整后的候选，等我确认。")
        if first and first.proposal_id:
            run("decide", proposal_id=first.proposal_id, decision="reject")
        second = chat("刚才那份先不采用。现在只剩 6 分钟，我只想完成一次口头解释和一次核对。请重新给我一份候选。")
        if second and second.proposal_id:
            run("decide", proposal_id=second.proposal_id, decision="adopt")
        service = WorkspaceService(WorkspaceStore(db), args.output / "archives")
        chat("我上次最没弄懂什么？先根据那一点问我一道题，等我答了再评。")
        chat("文章里的方法是不是所有材料、所有人都有效？")
    else:
        chat("这两本书太长了。我已经会基本 Python，今天有 40 分钟，想先学会讲清楚一次工具调用从发起到返回的过程。帮我选最值得读的两三处，说明先后顺序、各自用处和具体位置。")
        chat("我还分不清工具执行成功和 Agent 回答正确的区别。请从资料中找一段能帮助我理解的内容，再用一个学完后能自己复述的例子解释。")
        chat("假设工具成功读到了第 27 页，但它其实应该读 PDF 物理第 35 页。我应该检查哪几个环节？先让我自己回答一遍。")
        chat("今天先到这里。根据我卡住的地方，给我一份下次 15 分钟的复习候选，只安排一项回顾和一项练习。")


if __name__ == "__main__":
    main()
