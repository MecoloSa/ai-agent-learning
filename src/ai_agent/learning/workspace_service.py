"""Application commands for the local workspace, independent of HTTP and UI."""

from difflib import unified_diff
import json
import logging
from pathlib import Path
import re
from threading import Lock, Thread
from typing import Literal
from uuid import uuid4

from pydantic import Field

from ai_agent.config import get_settings
from ai_agent.learning.feedback_runtime import commit_user_task_feedback
from ai_agent.learning.learning_state import (
    ActivityStatus, LearningState, LearningTask, StateModel, UnderstandingStatus, UserFeedback, update_learning_context,
)
from ai_agent.learning.plan_change_request import PlanChangeRequest
from ai_agent.learning.plan_tasks import apply_task_changes, validate_split_changes, validate_task_changes
from ai_agent.learning.task_cancellation import TaskCancellation, cancel_learning_task
from ai_agent.learning.task_creation import TaskCreationArgs, commit_task_creation, prepare_task_creation
from ai_agent.learning.workspace_store import (
    INITIAL_PLAN, PlanCandidate, Workspace, WorkspaceConflict, WorkspaceConversation, WorkspaceEvent, WorkspaceStore,
    empty_learning_state,
)
from ai_agent.persistence.session_materials import resolve_resume_materials
from ai_agent.resources.materials import iter_material_files, safe_material_path
from ai_agent.resources.tools import build_list_material_files_tool


class WorkspaceCommand(StateModel):
    action: Literal["chat", "draft", "manual_plan", "feedback", "create_task", "cancel_task", "decide", "resubmit",
                    "bind_materials", "edit_task", "new_conversation", "rename_conversation", "delete_conversation", "clear_history", "remove_material"]
    request_id: str = Field(pattern=r"^[0-9a-f]{32}$", strict=True)
    version: int = Field(ge=1, strict=True)
    payload: dict[str, object]
    conversation_id: str | None = Field(default=None, pattern=r"^[0-9a-f]{32}$")


class ChatInput(StateModel):
    text: str = Field(min_length=1, max_length=12000, strict=True)
    deep_mode: bool = Field(default=False, strict=True)
    worker_thinking: bool = Field(default=True, strict=True)


class CandidateSelection(StateModel):
    proposal_id: str = Field(pattern=r"^[0-9a-f]{32}$")


class CandidateDecision(CandidateSelection):
    decision: Literal["adopt", "reject"]


class TaskTitle(StateModel):
    title: str = Field(min_length=1, max_length=200, strict=True)


class MaterialDirectory(StateModel):
    directory: str = Field(min_length=1, max_length=4096, strict=True)


class MaterialRemoval(StateModel):
    relative_path: str = Field(min_length=1, max_length=4096, strict=True)
    confirm: Literal[True]


class TaskEdit(StateModel):
    task_id: str = Field(min_length=1)
    title: str | None = Field(default=None, min_length=1, max_length=200)
    estimated_minutes: int | None = Field(default=None, ge=1, le=1440, strict=True)
    instructions: str | None = Field(default=None, max_length=4000)
    activity_status: ActivityStatus | None = None
    understanding_status: UnderstandingStatus | None = None
    text: str | None = Field(default=None, min_length=1, max_length=4000)


def _task_deltas(before: LearningState, after: LearningState) -> tuple[dict, ...]:
    previous = {task.task_id: task.model_dump(exclude={"task_id"}) for task in before.tasks}
    deltas = []
    for task in after.tasks:
        old = previous.get(task.task_id, {})
        changes = {key: {"before": old.get(key), "after": value}
                   for key, value in task.model_dump(exclude={"task_id"}).items() if old.get(key) != value}
        if changes:
            deltas.append({"task_id": task.task_id, "changes": changes})
    return tuple(deltas)


def setup_stage(state: LearningState) -> Literal["materials", "plan", "ready"]:
    binding = state.material_binding
    if binding is None or not Path(binding.root_dir).is_dir():
        return "materials"
    return "plan" if state.plan_markdown.strip() == INITIAL_PLAN else "ready"


def workspace_view(workspace: Workspace, conversation_id: str | None = None) -> dict:
    """Return display data without local paths, archive IDs or storage internals."""
    if conversation_id is None and workspace.original_deleted:
        conversation_id = workspace.conversations[0].id
    workspace = workspace.scope(conversation_id)
    state = workspace.state
    proposals = []
    tasks = {task.task_id: task for task in state.tasks}
    for candidate in workspace.proposals:
        item = candidate.model_dump(mode="json", exclude={"base_plan", "task_baseline"})
        item["diff"] = "\n".join(unified_diff(candidate.base_plan.splitlines(),
            candidate.markdown.splitlines(), fromfile="起草时的正式计划", tofile="候选计划", lineterm=""))
        if candidate.status == "pending" and candidate.base_version != workspace.version:
            item["status"] = "stale"
        item["can_resubmit"] = candidate.status == "rejected" and candidate.base_version == workspace.version
        for change in item["task_changes"]:
            task = candidate.task_baseline.get(change["task_id"])
            if task is None and candidate.base_version == workspace.version:
                task = tasks.get(change["task_id"])
            change["current_title"] = task.title if task else None
            if task:
                change["previous_minutes"] = task.estimated_minutes
                change["previous_instructions"] = task.instructions
        proposals.append(item)
    memory = state.conversation_memory
    binding = state.material_binding
    materials_available = bool(binding and Path(binding.root_dir).is_dir())
    stage = setup_stage(state)
    return {
        "version": workspace.version,
        "conversation_id": conversation_id,
        "materials": {
            "folder": Path(binding.root_dir).name if binding else None,
            "available": materials_available,
        },
        "setup_stage": stage,
        "needs_initial_plan": state.plan_markdown.strip() == INITIAL_PLAN,
        "deep_mode_available": stage == "ready" and materials_available and get_settings().interactive_task_delegation_enabled,
        "state": state.model_dump(mode="json", exclude={"material_binding", "conversation_memory"}),
        "proposals": proposals,
        "conversations": ([{"id": None, "title": workspace.original_title}] if not workspace.original_deleted else []) + [
            {"id": session.id, "title": session.title} for session in workspace.conversations
        ],
        "events": [event.model_dump(mode="json") for event in workspace.events],
        "recent_chat": [message.model_dump(mode="json") for message in memory.messages[-12:]] if memory else [],
    }


class WorkspaceService:
    def __init__(self, store: WorkspaceStore, archive_root: Path, *, draft_turn=None):
        self.store = store
        self.archive_root = archive_root
        self._title_lock = Lock()
        self._title_jobs: set[str | None] = set()
        self.draft_turn = draft_turn or self._draft_turn

    def material_catalog(self, conversation_id: str | None = None, *, offset: int = 0) -> dict:
        binding = self.store.load().scope(conversation_id).state.material_binding
        if binding is None or not Path(binding.root_dir).is_dir():
            raise ValueError("请先关联可访问的本地资料目录。")
        return json.loads(build_list_material_files_tool(binding.root_dir).invoke({"offset": offset}))

    @staticmethod
    def _validate_setup_action(scoped: Workspace, command: WorkspaceCommand) -> None:
        stage = setup_stage(scoped.state)
        if stage == "ready" or command.action in {"bind_materials", "new_conversation", "rename_conversation", "delete_conversation"}:
            return
        if stage == "materials":
            raise ValueError("请先关联可访问的本地资料，再生成首份学习计划。")
        if command.action == "draft" and command.payload.get("mode") == "replace":
            return
        if command.action == "decide" and any(
            candidate.id == command.payload.get("proposal_id") and candidate.mode == "replace"
            for candidate in scoped.proposals
        ):
            return
        raise ValueError("请先生成并采用首份学习计划，再开始聊天。")

    def execute(self, command: WorkspaceCommand, *, stream_observer=None, enforce_setup=False) -> WorkspaceEvent:
        excluded = {"request_id", "conversation_id"} if command.conversation_id is None else {"request_id"}
        fingerprint = json.dumps(command.model_dump(mode="json", exclude=excluded), sort_keys=True)
        if command.action == "rename_conversation":
            if enforce_setup:
                self._validate_setup_action(self.store.load().scope(command.conversation_id), command)
            title = command.payload.get("title")
            if set(command.payload) != {"title"} or not isinstance(title, str) or not 1 <= len(title.strip()) <= 60:
                raise ValueError("请输入 1 到 60 个字符的标题。")
            event = WorkspaceEvent(action=command.action, status="saved", message="已重命名对话。",
                                   conversation_id=command.conversation_id)
            return self.store.rename(" ".join(title.split()), command.request_id, fingerprint, event)
        previous = self.store.replay(command.request_id, fingerprint)
        if previous is not None:
            return previous
        root = self.store.load()
        before = root.scope(command.conversation_id)
        if enforce_setup:
            self._validate_setup_action(before, command)
        if command.version != before.version:
            raise WorkspaceConflict("当前学习状态已更新，请核对后再提交。")

        if command.action == "new_conversation":
            session = WorkspaceConversation(id=uuid4().hex, state=empty_learning_state())
            after = root.model_copy(update={"conversations": (*root.conversations, session)})
            event = WorkspaceEvent(action=command.action, status="saved", message="已新建对话。",
                                   conversation_id=session.id)
            return self.store.commit(root, after, command.request_id, fingerprint, event)
        if command.action in {"delete_conversation", "clear_history"}:
            return self._remove_conversation(root, before, command, fingerprint)
        if command.action == "remove_material":
            return self._remove_material(root, before, command, fingerprint)
        if command.action in {"chat", "draft"}:
            after, event = self._generate(before, command, stream_observer)
        elif command.action == "manual_plan":
            after, event = self._manual_plan(before, command)
        elif command.action == "bind_materials":
            directory = MaterialDirectory.model_validate(command.payload).directory
            if not Path(directory).is_absolute():
                raise ValueError("请输入本机资料文件夹的绝对路径。")
            state, notice = resolve_resume_materials(before.state, directory)
            # 资料变更使基于旧资料起草的候选失效，不递增正式计划的修订号。
            after = before.model_copy(update={"state": state, "version": before.version + int(state != before.state)})
            event = WorkspaceEvent(action=command.action, status="saved", message=notice)
        elif command.action in {"decide", "resubmit"}:
            after, event = self._candidate_command(before, command)
        else:
            after, event = self._task_command(before, command)
        event = event.model_copy(update={"conversation_id": command.conversation_id})
        merged = self._merge_scope(root, before, after, command.conversation_id, title_from=event)
        if command.action in {"chat", "draft"}:
            def rebase(latest):
                current = latest.scope(command.conversation_id)
                # 同会话的新对话也会改变生成依据，即使正式计划版本未增加。
                if (current.version, current.state, current.events, current.proposals) != (
                    before.version, before.state, before.events, before.proposals
                ):
                    raise WorkspaceConflict("操作期间当前对话已变化，请核对最新状态后重试。")
                return self._merge_scope(latest, current, after, command.conversation_id, title_from=event)

            return self.store.commit(root, merged, command.request_id, fingerprint, event, rebase=rebase)
        return self.store.commit(root, merged, command.request_id, fingerprint, event)

    def schedule_title(self, command: WorkspaceCommand, event: WorkspaceEvent) -> bool:
        if not event.title_seed or self.store.has_title(command.conversation_id):
            return False
        conversation_id = command.conversation_id
        with self._title_lock:
            if conversation_id in self._title_jobs:
                return True
            if len(self._title_jobs) >= 2:
                return False
            self._title_jobs.add(conversation_id)

        def generate():
            try:
                from ai_agent.config import create_llm
                from langchain_core.messages import HumanMessage, SystemMessage

                answer = create_llm(role="workspace_title").invoke([
                    SystemMessage(content="为下面的资料总结写一个简短的会话标题。只输出标题。"),
                    HumanMessage(content=event.message[:6000]),
                ])
                if isinstance(answer.content, str):
                    title = " ".join(answer.content.split())[:32].strip()
                    if title:
                        self.store.set_title(conversation_id, title, source_request=command.request_id)
            except Exception:
                logging.getLogger(__name__).warning("后台会话命名未完成。")
            finally:
                with self._title_lock:
                    self._title_jobs.discard(conversation_id)

        Thread(target=generate, name="workspace-title", daemon=True).start()
        return True

    @staticmethod
    def _merge_scope(root, before, after, conversation_id, *, title_from=None):
        owned = {item.id for item in before.proposals}
        proposals = tuple(item for item in root.proposals if item.id not in owned) + tuple(
            item.model_copy(update={"conversation_id": conversation_id}) for item in after.proposals
        )
        if conversation_id is None:
            return root.model_copy(update={"state": after.state, "version": after.version, "proposals": proposals})
        sessions = []
        for session in root.conversations:
            if session.id == conversation_id:
                title = session.title
                if title == "新对话" and title_from and title_from.action == "chat":
                    title = " ".join(title_from.input_text.split())[:36]
                session = session.model_copy(update={
                    "title": title, "version": after.version, "memory": after.state.conversation_memory,
                    "state": after.state.model_copy(update={"conversation_memory": None}),
                })
            sessions.append(session)
        return root.model_copy(update={"conversations": tuple(sessions), "proposals": proposals})

    def _remove_conversation(self, root, before, command, fingerprint):
        conversation_id = command.conversation_id
        delete = command.action == "delete_conversation"
        if set(command.payload) != {"confirm"} or command.payload["confirm"] is not True:
            raise ValueError("请确认删除范围。")
        state = empty_learning_state() if delete else before.state.model_copy(update={"conversation_memory": None})
        cleared = before.model_copy(update={"state": state, "version": before.version + 1, "proposals": ()})
        after = self._merge_scope(root, before, cleared, conversation_id)
        events = tuple(event for event in root.events if event.conversation_id != conversation_id or (
            not delete and event.action not in {"chat", "draft", "manual_plan", "resubmit", "decide"}
        ))
        changes = {"events": events}
        if delete:
            sessions = tuple(session for session in after.conversations if session.id != conversation_id)
            original_deleted = after.original_deleted or conversation_id is None
            if original_deleted and not sessions:
                sessions = (WorkspaceConversation(id=uuid4().hex, state=empty_learning_state()),)
            changes.update(conversations=sessions, original_deleted=original_deleted)
        event = WorkspaceEvent(action=command.action, status="saved", conversation_id=conversation_id,
                               message="已删除对话。" if delete else "已清空聊天和候选历史。")
        return self.store.commit(root, after.model_copy(update=changes), command.request_id, fingerprint, event,
                                 purge="all" if delete else "history")

    def _remove_material(self, root, before, command, fingerprint):
        removal = MaterialRemoval.model_validate(command.payload)
        if command.payload["confirm"] is not True:
            raise ValueError("请确认移除该文件。")
        binding = before.state.material_binding
        relative = Path(removal.relative_path)
        if (binding is None or relative.anchor or ".." in relative.parts
                or any(part.casefold() == ".yanxi-trash" for part in relative.parts)):
            raise ValueError("只能移除本对话已关联目录内的资料文件。")
        directory = Path(binding.root_dir)
        source = safe_material_path(binding.root_dir, removal.relative_path)
        if source != directory / relative or not source.is_file():
            raise ValueError("文件不存在，或路径经过链接。请刷新文件清单。")
        trash = directory / ".yanxi-trash"
        if trash.resolve() != trash:
            raise ValueError("资料回收目录不可用。")
        destination = trash / uuid4().hex / relative.parent / (relative.name + ".removed")
        destination.parent.mkdir(parents=True, exist_ok=False)
        source.rename(destination)
        try:
            updated_bindings = {}

            def rebind(state):
                if state is None or state.material_binding is None:
                    return state
                material_root = Path(state.material_binding.root_dir)
                if not source.is_relative_to(material_root):
                    return state
                if material_root not in updated_bindings:
                    updated_bindings[material_root] = (
                        resolve_resume_materials(state)[0].material_binding
                        if next(iter_material_files(material_root), None) is not None else None
                    )
                return state.model_copy(update={"material_binding": updated_bindings[material_root]})

            # 关联目录可能重叠。所有实际受影响的对话都更新资料指纹并淘汰旧候选。
            sessions = []
            for session in root.conversations:
                state = rebind(session.state)
                sessions.append(session.model_copy(update={"state": state, "version": session.version + 1})
                                if state != session.state else session)
            state = rebind(root.state)
            changed = {"conversations": tuple(sessions), "state": state,
                       "version": root.version + int(state != root.state)}
            event = WorkspaceEvent(action="remove_material", status="saved", conversation_id=command.conversation_id,
                message=f"已移入资料目录的回收区：{destination.relative_to(directory).as_posix()}")
            return self.store.commit(root, root.model_copy(update=changed), command.request_id, fingerprint, event)
        except Exception:
            # 文件移动不属于 SQLite 事务，数据库提交失败时把文件放回原位。
            destination.rename(source)
            raise

    @staticmethod
    def _manual_plan(before: Workspace, command: WorkspaceCommand):
        from ai_agent import main as app
        from ai_agent.learning.plan_proposals import PlanProposalArgs, PlanProposalInbox

        request = PlanProposalArgs.model_validate(command.payload)
        inbox = PlanProposalInbox(app.PLAN_START_TAG, app.PLAN_END_TAG,
                                  (app.GOAL_START_TAG, app.GOAL_END_TAG))
        inbox.start_turn(before.state)
        receipt = json.loads(inbox.tool.invoke(request.model_dump()))
        if not receipt["ok"]:
            raise ValueError(receipt["error"]["message"])
        # 手动编辑与模型起草共享版本校验和采用流程。
        candidate = PlanCandidate(id=uuid4().hex, base_version=before.version, mode="replace",
            instructions="用户手动编辑计划", base_plan=before.state.plan_markdown,
            markdown=inbox.pending.plan_markdown)
        after = before.model_copy(update={"proposals": (*before.proposals, candidate)})
        return after, WorkspaceEvent(action="manual_plan", status="pending",
            message="手动编辑已生成候选，请核对差异后确认。", proposal_id=candidate.id)

    def _generate(self, before: Workspace, command: WorkspaceCommand, stream_observer):
        request = PlanChangeRequest.model_validate(command.payload) if command.action == "draft" else None
        chat = ChatInput.model_validate(command.payload) if request is None else None
        text = chat.text if chat is not None else request.instructions
        if request and request.mode == "split_task" and not any(
            task.task_id == request.task_id and not task.is_cancelled for task in before.state.tasks
        ):
            raise ValueError("请选择仍在执行的任务。")
        task_only = request is not None and request.mode in {"extract_tasks", "split_task"}
        generation = before
        if task_only:
            generation = before.model_copy(update={"state": before.state.model_copy(update={
                "conversation_memory": None,
            })})
        title_audit = {} if request is None and re.search(r"总结|概括|概述|梳理|归纳|讲了什么|summari[sz]e|overview", text, re.I) else None
        if self.draft_turn != self._draft_turn:
            state, result = self.draft_turn(generation, text, request)
        else:
            state, result = self._draft_turn(generation, text, request, stream_observer=stream_observer,
                                            audit_trace=title_audit, chat=chat)
        if state.model_dump(exclude={"conversation_memory"}) != before.state.model_dump(exclude={"conversation_memory"}):
            raise ValueError("起草过程中发现正式状态变化，已拒绝交付。")
        proposal_id = None
        proposals = before.proposals
        if task_only:
            state = before.state
        # 生成适配器即使返回候选，也不能绕过入口授权进入待确认记录。
        if request is None and result.confirmation_plan is not None:
            raise ValueError("当前为学习对话，请通过“调整计划”发起候选。")
        if result.confirmation_plan is not None:
            changes = result.task_changes
            if task_only:
                issue = None
                if request.mode == "extract_tasks":
                    adds = tuple(change for change in changes if change.operation == "add")
                    existing = {task.title.casefold() for task in before.state.tasks if not task.is_cancelled}
                    titles = {change.title.casefold() for change in adds}
                    if not adds or len(adds) != len(changes) or len(adds) != len(titles) or existing & titles:
                        issue = "提取任务只能新增尚未登记的任务。"
                else:
                    try:
                        validate_split_changes(changes, request.task_id)
                    except ValueError as exc:
                        issue = str(exc)
                if issue is not None:
                    return before.model_copy(update={"state": state}), WorkspaceEvent(
                        action="draft", status="not_submitted", input_text=text,
                        message=issue,
                        conversation_id=command.conversation_id)
            if changes:
                validate_task_changes(before.state, changes)
            changed_ids = {change.task_id for change in changes if change.task_id}
            candidate = PlanCandidate(id=uuid4().hex, base_version=before.version,
                mode=request.mode if request else "chat", instructions=text,
                base_plan=before.state.plan_markdown,
                markdown=before.state.plan_markdown if task_only else result.confirmation_plan,
                task_changes=changes, task_baseline={
                    task.task_id: task for task in before.state.tasks if task.task_id in changed_ids
                })
            proposal_id = candidate.id
            proposals = (*proposals, candidate)

        status = result.status if request or proposal_id else "answered"
        message = "任务候选待确认，正式计划正文保持原样。" if proposal_id and task_only else result.message
        from ai_agent.main import tool_result_succeeded
        title_seed = bool(title_audit and message and any(
            item["executed"] and item["tool_name"] in {
                "read_material_lines", "read_pdf_pages", "retrieve_learning_context",
            } and tool_result_succeeded(item["tool_name"], item["result"])
            for item in title_audit["tool_results"]
        ))
        event = WorkspaceEvent(action=command.action, status=status, message=message, title_seed=title_seed,
                               input_text=text, proposal_id=proposal_id, conversation_id=command.conversation_id)
        return before.model_copy(update={"state": state, "proposals": proposals}), event

    @staticmethod
    def _candidate_command(before: Workspace, command: WorkspaceCommand):
        selection = (CandidateDecision.model_validate(command.payload) if command.action == "decide"
                     else CandidateSelection.model_validate(command.payload))
        candidate = next((item for item in before.proposals if item.id == selection.proposal_id), None)
        if candidate is None:
            raise ValueError("找不到指定候选。")
        rejecting = isinstance(selection, CandidateDecision) and selection.decision == "reject"
        if candidate.base_version != before.version and not rejecting:
            raise WorkspaceConflict("候选依据的正式状态已变化，请基于当前状态重新起草。")

        if command.action == "resubmit":
            if candidate.status != "rejected":
                raise WorkspaceConflict("只有已拒绝的候选可以原样重新提交。")
            submitted = candidate.model_copy(update={"id": uuid4().hex, "status": "pending",
                                                       "previous_id": candidate.id})
            after = before.model_copy(update={"proposals": (*before.proposals, submitted)})
            return after, WorkspaceEvent(action="resubmit", status="pending",
                                         message="原方案已重新提交，请核对。", proposal_id=submitted.id)

        if candidate.status != "pending":
            raise WorkspaceConflict("该候选已经处理，请查看当前结果。")
        adopt = selection.decision == "adopt"
        state = update_learning_context(before.state, plan_markdown=candidate.markdown) if adopt else before.state
        if adopt and (candidate.task_changes or candidate.mode == "split_task"):
            split_from = next((change.task_id for change in candidate.task_changes
                               if change.operation == "archive"), None) if candidate.mode == "split_task" else None
            if candidate.mode == "split_task" and split_from is None:
                raise ValueError("细化任务候选缺少原任务。")
            state = apply_task_changes(state, candidate.task_changes, split_from_task_id=split_from).model_copy(
                update={"tasks_need_review": before.state.tasks_need_review})
        if adopt and state is before.state:
            raise WorkspaceConflict("候选与当前正式计划相同。")
        status = "adopted" if adopt else "rejected"
        proposals = tuple(item.model_copy(update={"status": status}) if item.id == candidate.id else item
                          for item in before.proposals)
        after = before.model_copy(update={"state": state, "proposals": proposals,
                                           "version": before.version + int(adopt)})
        event = WorkspaceEvent(action="decide", status=status, proposal_id=candidate.id,
            task_delta=_task_deltas(before.state, state) if adopt and candidate.task_changes else (),
            message=f"已采用并保存，计划修订号 {state.context_revision}。" if adopt else "已拒绝，当前正式计划继续生效。")
        return after, event

    @staticmethod
    def _task_command(before: Workspace, command: WorkspaceCommand):
        if command.action == "feedback":
            feedback = UserFeedback.model_validate(command.payload)
            state = commit_user_task_feedback(before.state, feedback)
            message = "反馈与任务状态已保存。"
            input_text = feedback.text
        elif command.action == "edit_task":
            edit = TaskEdit.model_validate(command.payload)
            task = next((item for item in before.state.tasks if item.task_id == edit.task_id), None)
            if task is None or task.is_cancelled:
                raise ValueError("只有当前任务可以编辑。")
            fields = edit.model_dump(exclude_unset=True, exclude={"task_id", "text"})
            fields = {key: value for key, value in fields.items() if value is not None or key == "estimated_minutes"}
            updated = LearningTask.model_validate({**task.model_dump(), **fields})
            state = before.state.model_copy(update={"tasks": tuple(updated if item.task_id == task.task_id else item
                                                                   for item in before.state.tasks)})
            if edit.text:
                state = commit_user_task_feedback(state, UserFeedback(task_id=task.task_id, text=edit.text,
                    activity_status=edit.activity_status, understanding_status=edit.understanding_status))
            if state == before.state:
                raise ValueError("任务没有变更。")
            message, input_text = "任务已保存。", edit.text or ""
        elif command.action == "create_task":
            title = TaskTitle.model_validate(command.payload).title
            pending = prepare_task_creation(before.state, TaskCreationArgs(title=title), user_message=title)
            state = commit_task_creation(before.state, pending)
            message = f"已登记并保存任务「{title}」。"
            input_text = title
        else:
            request = TaskCancellation.model_validate(command.payload)
            state = cancel_learning_task(before.state, request)
            message = "任务已取消，已有进度和反馈已保留。"
            input_text = request.text
        after = before.model_copy(update={"state": state, "version": before.version + 1})
        return after, WorkspaceEvent(action=command.action, status="saved", message=message, input_text=input_text,
                                     task_delta=_task_deltas(before.state, state))

    def _draft_turn(self, workspace: Workspace, text: str, request: PlanChangeRequest | None, *,
                    stream_observer=None, audit_trace=None, chat: ChatInput | None = None):
        # Clients are created only in the two model-backed actions.
        from ai_agent import main as app
        from ai_agent.learning.feedback_runtime import LearningActionInbox
        from ai_agent.learning.learning_context import build_learning_state_message
        from ai_agent.learning.plan_change_request import collect_plan_draft
        from ai_agent.learning.plan_proposals import PlanProposalInbox
        from ai_agent.persistence.history_readback import HistoryReadAccess
        from ai_agent.persistence.memory import ConversationMemory
        from ai_agent.resources.document_structure import build_inspect_document_structure_tool
        from langchain_core.messages import HumanMessage

        state = workspace.state
        task_only = request is not None and request.mode in {"extract_tasks", "split_task"}
        directory = state.material_binding.root_dir if state.material_binding and not task_only else None
        if directory is not None and not Path(directory).is_dir():
            raise ValueError("已关联的资料目录不可用。请从右栏重新关联目录。")
        settings = app.get_settings()
        use_rag = directory is not None and app.needs_rag(directory)
        # 无资料关联时仍支持对话和计划候选，资料工具只绑定到已确认的本地目录。
        prompt = app._build_interactive_system_prompt(
            "", state.plan_markdown, "", state.learning_goal,
            retrieval_available=use_rag,
            on_demand_eligible=directory is not None and app._supports_on_demand_materials(directory),
            plan_only=True,
            allow_plan_proposals=request is not None,
        ) + app.RESUMED_SESSION_RULES
        memory = (ConversationMemory.from_snapshot(prompt, state.conversation_memory)
                  if state.conversation_memory else ConversationMemory(system_prompt=prompt, strategy="window"))
        messages = [*memory.get_messages_for_llm(), build_learning_state_message(state)]
        feedback = LearningActionInbox()
        feedback.start_turn(state, text, history_messages=memory.get_messages_for_llm())
        markers = (app.PLAN_START_TAG, app.PLAN_END_TAG, app.GOAL_START_TAG, app.GOAL_END_TAG)
        plans = PlanProposalInbox(*markers[:2], markers[2:], workspace_tasks=True)
        plans.start_turn(state)
        read = app.build_read_material_lines_tool(directory) if directory is not None else None
        retrieval = app.build_retrieve_learning_context_tool(
            directory, max_queries_per_call=settings.interactive_max_queries_per_tool_call,
            max_context_chars=settings.interactive_retrieval_max_context_chars, structured_result=True,
        ) if use_rag else None
        extra = [feedback.history_tool, feedback.confirmation_history_tool]
        # 普通对话可以读取反馈和确认历史，显式起草才绑定候选提交工具。
        if request is not None:
            extra.append(plans.tool)
        if directory is not None:
            extra[:0] = [
                app.build_list_material_files_tool(directory), app.build_read_pdf_pages_tool(directory),
                build_inspect_document_structure_tool(directory, observer=stream_observer),
            ]
        if memory.history_archive is not None:
            history = HistoryReadAccess(self.archive_root, strategy=settings.history_retrieval_strategy,
                min_similarity=settings.history_semantic_min_similarity, embedding_factory=app.create_embeddings)
            history.start_turn(memory.history_archive)
            messages.append(history.source_message())
            extra.extend(history.tools)
        if workspace.proposals:
            latest = workspace.proposals[-1]
            status = "stale" if latest.status == "pending" and latest.base_version != workspace.version else latest.status
            messages.append(HumanMessage(content=json.dumps({"context_type": "workspace_latest_candidate",
                "id": latest.id, "status": status}, ensure_ascii=False)))
        if request is not None:
            messages.append(request.mode_message())
        last_changes = []
        # 只给下一轮有限的已提交差异，完整记录仍在 SQLite。
        for event in reversed(workspace.events):
            for delta in event.task_delta:
                changes = {key: {side: value[:240] if isinstance(value, str) else value
                                 for side, value in pair.items()} for key, pair in delta["changes"].items()}
                last_changes.append({"task_id": delta["task_id"], "changes": changes})
                if len(last_changes) == 6:
                    break
            if len(last_changes) == 6:
                break
        if last_changes:
            messages.append(HumanMessage(content=json.dumps({
                "context_type": "recent_committed_task_changes", "changes_newest_first": last_changes,
            }, ensure_ascii=False)))

        model_role = {}
        if request is not None:
            model_role = {"role": "workspace_tasks" if task_only else "auxiliary"}
        elif chat is not None and chat.deep_mode and directory is not None and settings.interactive_task_delegation_enabled:
            from ai_agent.runtime.task_delegation import build_learning_delegation_tool

            extra.append(build_learning_delegation_tool(
                directory, text, main_plans=settings.interactive_main_plans_delegation,
                worker_thinking=chat.worker_thinking,
            ))
        reply = app._invoke_interactive_turn(
            request_messages=messages, user_input=text, plain_llm=app.create_llm(**model_role),
            tool_llm=app.create_llm_with_tools([tool for tool in (retrieval, read, *extra) if tool is not None],
                                             **model_role),
            retrieval_tool=retrieval, read_tool=read, extra_tools=tuple(extra),
            plan_inbox=plans if request is not None else None,
            stream_observer=stream_observer,
            audit_trace=audit_trace,
        )
        result = collect_plan_draft(reply, plans, markers)
        memory.add_user(text)
        memory.add_ai(result.message, compress=True)
        return LearningState.model_validate({**state.model_dump(),
            "conversation_memory": memory.export_state().model_dump()}), result
