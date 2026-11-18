"use strict";

const byId = id => document.getElementById(id);
const token = document.querySelector('meta[name="workspace-token"]').content;
const auth = {"X-Workspace-Token": token};
const activity = {not_started: "未开始", in_progress: "进行中", completed: "已完成"};
const understanding = {unknown: "未填写", self_reported_unclear: "还没理解", self_reported_partial: "大致理解", self_reported_clear: "已理解"};
const statuses = {pending: "待确认", adopted: "已采用", rejected: "已拒绝", stale: "已失效"};
let view;
let busy = false;
let chatStreaming = false;
const waitingForReply = "回复生成中，完成后可提交。";
let retry;
let editorKey;
let activeView = "all";
let searchQuery = "";
let conversationId = sessionStorage.getItem("workspace-conversation") || null;
const conversationDrafts = new Map();

function selectConversation(id) {
    const input = byId("chat-input");
    conversationDrafts.set(conversationId, input.value);
    conversationId = id;
    input.value = conversationDrafts.get(id) || "";
    if (id) sessionStorage.setItem("workspace-conversation", id);
    else sessionStorage.removeItem("workspace-conversation");
    searchQuery = "";
}

function scopedUrl(path, id = conversationId) {
    return id ? `${path}?conversation=${encodeURIComponent(id)}` : path;
}

async function switchConversation(id) {
    if (busy) return false;
    busy = true;
    try {
        const response = await fetch(scopedUrl("/api/state", id), {headers: auth});
        const workspace = await response.json();
        if (!response.ok) throw new Error(workspace.error);
        selectConversation(workspace.conversation_id);
        byId("plan-details").open = false;
        showInspectorDetails(false);
        selectView("all");
        render(workspace);
        notice("");
        return true;
    } catch (error) {
        notice(error.message, true);
        return false;
    } finally {
        busy = false;
    }
}

function renderConversations() {
    const list = byId("conversation-list");
    list.replaceChildren();
    for (const session of [...view.conversations].reverse()) {
        const row = element("div", "", "conversation-entry");
        const control = button(session.title, () => void switchConversation(session.id));
        control.className = "nav-button conversation-link";
        control.title = session.title;
        if (session.id === conversationId) control.setAttribute("aria-current", "true");
        const menu = element("div", "", "conversation-menu");
        const trigger = element("button", "⋯", "conversation-menu-trigger");
        trigger.type = "button";
        trigger.setAttribute("aria-label", `${session.title}的更多操作`);
        const actions = element("div", "", "conversation-menu-actions");
        actions.popover = "auto";
        trigger.popoverTargetElement = actions;
        actions.addEventListener("toggle", event => {
            if (event.newState !== "open") return;
            const anchor = trigger.getBoundingClientRect();
            actions.style.left = `${Math.max(8, anchor.right - 160)}px`;
            actions.style.top = `${Math.max(8, Math.min(anchor.bottom + 4, innerHeight - actions.offsetHeight - 8))}px`;
        });
        for (const [mode, label] of [["rename_conversation", "重命名"], ["json_edit", "编辑受控 JSON"], ["export", "导出学习状态 JSON"],
            ["clear_history", "清空聊天历史"], ["delete_conversation", "删除对话"]]) {
            actions.append(button(label, async () => {
                actions.hidePopover();
                if (mode === "export") {
                    await exportState(session.id);
                    return;
                }
                if (session.id !== conversationId && !await switchConversation(session.id)) return;
                openEditor(mode);
            }));
        }
        menu.append(trigger, actions);
        row.append(control, menu);
        list.append(row);
    }
}

const markdown = window.markdownit({html: false, breaks: true}).disable("image");
// 只允许用户主动打开普通网页，阻止本机文件、脚本及自定义协议。
markdown.validateLink = url => /^https?:\/\//i.test(url);
markdown.renderer.rules.link_open = (tokens, index, options, env, renderer) => {
    tokens[index].attrSet("target", "_blank");
    tokens[index].attrSet("rel", "noopener noreferrer");
    return renderer.renderToken(tokens, index, options);
};

function element(tag, text = "", className = "") {
    const node = document.createElement(tag);
    node.textContent = text;
    node.className = className;
    return node;
}

function button(text, action, primary = false) {
    const node = element("button", text, primary ? "primary" : "");
    node.type = "button";
    node.addEventListener("click", action);
    return node;
}

function notice(text, error = false) {
    const dialog = byId("editor").open;
    byId("notice").textContent = dialog ? "" : text;
    byId("notice-bar").hidden = dialog || !text;
    const target = byId(dialog ? "editor-notice" : "notice");
    target.textContent = text;
    target.classList.toggle("error", error);
}

function readableText(text) {
    const content = element("div", "", "rich-text");
    content.innerHTML = markdown.render(text);
    return content;
}

function selectView(name) {
    activeView = name;
    document.querySelectorAll("[data-view]").forEach(node => {
        const selected = node.dataset.view === name;
        node.classList.toggle("active", selected);
        node.setAttribute("aria-pressed", String(selected));
    });
    const session = view?.conversations.find(item => item.id === conversationId);
    byId("chat-title").textContent = {all: session?.title ?? "学习对话", plans: "计划候选", tasks: "任务与反馈"}[name];
    byId("view-caption").textContent = {
        all: "提问、核对资料，或讨论你的学习安排。",
        plans: "检查变更内容，决定是否纳入正式计划。",
        tasks: "查看任务登记、进度和理解反馈的记录。",
    }[name];
    document.querySelector(".composer").hidden = name !== "all";
    if (chatStreaming) {
        byId("pending-message").hidden = name !== "all" || Boolean(searchQuery);
        if (byId("stream-preview").childElementCount) byId("stream-preview").hidden = byId("pending-message").hidden;
    }
    updateReplyJump();
    if (view) renderTimeline();
}

function nearLatestReply() {
    const page = document.scrollingElement;
    return page.scrollHeight - page.clientHeight - page.scrollTop <= 80;
}

function updateReplyJump() {
    byId("jump-to-latest").hidden = activeView !== "all" || Boolean(searchQuery) ||
        !byId("timeline").querySelector(".turn") && !chatStreaming || nearLatestReply();
}

function showLatestReply() {
    const page = document.scrollingElement;
    page.scrollTop = page.scrollHeight;
    updateReplyJump();
}

function setChatWriteControls(waiting) {
    for (const control of document.querySelectorAll(
        "#chat-form button[type=submit], #editor button[type=submit], #feedback-reload, .task-quick select, #new-conversation, #refresh, .conversation-link, .candidate .actions button"
    )) {
        control.disabled = waiting;
        if (waiting && !("previousTitle" in control.dataset)) {
            control.dataset.previousTitle = control.title;
            control.title = waitingForReply;
        } else if (!waiting && "previousTitle" in control.dataset) {
            control.title = control.dataset.previousTitle;
            delete control.dataset.previousTitle;
        }
    }
    const note = byId("feedback-wait");
    if (note) note.hidden = !waiting;
}

async function refresh() {
    let response = await fetch(scopedUrl("/api/state"), {headers: auth});
    if (response.status === 409 && conversationId) response = await fetch("/api/state", {headers: auth});
    const body = await response.json();
    if (!response.ok) throw new Error(body.error);
    render(body);
}

async function watchConversationTitle(id, oldTitle) {
    for (let attempt = 0; attempt < 22; attempt++) {
        await new Promise(resolve => setTimeout(resolve, 1000));
        if (!view?.conversations.some(item => item.id === id)) return;
        try {
            const response = await fetch(scopedUrl("/api/state", id), {headers: auth});
            if (!response.ok) return;
            const {conversations} = await response.json();
            const title = conversations.find(item => item.id === id)?.title;
            if (!title || title === oldTitle) continue;
            // 只更新标题，避免旧快照覆盖用户正在浏览或编辑的状态。
            const current = view.conversations.find(item => item.id === id);
            if (current?.title === oldTitle) {
                current.title = title;
                renderConversations();
                if (id === conversationId && activeView === "all") byId("chat-title").textContent = title;
            }
            return;
        } catch {
            // 短暂断网时继续等待，常规刷新也能读取已保存的标题。
        }
    }
}

async function readAction(response, preview) {
    if (!response.headers.get("Content-Type")?.startsWith("application/x-ndjson")) {
        return {...await response.json(), status: response.status};
    }
    const reader = response.body.pipeThrough(new TextDecoderStream()).getReader();
    const label = element("div", "", "section-label");
    const text = element("div", "", "stream-text rich-text");
    const tools = element("ol", "", "stream-tools");
    tools.setAttribute("aria-label", "本轮工具调用");
    tools.hidden = true;
    const toolRows = new Map();
    preview.replaceChildren(label, tools, text);
    const isChat = preview.id === "stream-preview";
    const follow = isChat && nearLatestReply();
    preview.hidden = isChat && (activeView !== "all" || Boolean(searchQuery));
    if (follow && !preview.hidden) showLatestReply();
    let pending = "";
    let draft = "";
    let frame = null;
    function updatePreview() {
        if (frame !== null) return;
        frame = requestAnimationFrame(() => {
            frame = null;
            // 判断使用本批文字写入前的位置，避免新内容增长改变判断结果。
            const follow = isChat && !preview.hidden && nearLatestReply();
            text.innerHTML = markdown.render(draft);
            if (follow) showLatestReply();
            else if (isChat) updateReplyJump();
        });
    }
    try {
        while (true) {
            const {value, done} = await reader.read();
            if (done) break;
            pending += value;
            const lines = pending.split("\n");
            pending = lines.pop();
            for (const line of lines) {
                const {type, data} = JSON.parse(line);
                if (type === "result") return data;
                if (type === "phase") {
                    label.textContent = `${data} · 进行中`;
                    draft = "";
                    updatePreview();
                } else if (type === "tool_activity") {
                    let row = toolRows.get(data.id);
                    if (!row) {
                        row = element("li");
                        toolRows.set(data.id, row);
                        tools.append(row);
                    }
                    const status = {running: "执行中", completed: "已完成", failed: "执行失败", reused: "复用结果"};
                    row.textContent = `${data.name} · ${status[data.status]}`;
                    tools.hidden = false;
                    updatePreview();
                } else if (type === "delta") {
                    label.textContent = "生成预览 · 正在核验";
                    draft += data;
                    updatePreview();
                }
            }
        }
    } finally {
        if (frame !== null) cancelAnimationFrame(frame);
        await reader.cancel().catch(() => {});
    }
    throw new Error("连接中断，尚未收到保存回执。可以保留原输入重试。");
}

async function send(action, payload, expectedVersion = view?.version) {
    if (busy || !view) return null;
    const targetConversation = conversationId;
    const key = JSON.stringify([action, payload, expectedVersion, targetConversation]);
    if (retry?.key !== key) {
        retry = {key, command: {action, payload, version: expectedVersion, conversation_id: targetConversation,
            request_id: crypto.randomUUID().replaceAll("-", "")}};
    }
    busy = true;
    chatStreaming = action === "chat";
    const submittingEditor = byId("editor").open && !chatStreaming;
    if (submittingEditor) byId("editor").setAttribute("aria-busy", "true");
    const submitters = Array.from(document.querySelectorAll(
        '#chat-form button[type="submit"], #editor button[type="submit"]'
    ));
    submitters.forEach(node => { node.disabled = true; });
    const streaming = action === "chat" || action === "draft";
    const preview = byId(action === "draft" ? "editor-stream" : "stream-preview");
    byId("close-editor").disabled = submittingEditor;
    if (chatStreaming) setChatWriteControls(true);
    byId("composer-status").textContent = streaming ? "正在生成回复…" : "正在保存…";
    notice(byId("editor").open ? (streaming ? "正在准备方案…" : "正在保存…") : "");
    try {
        const response = await fetch("/api/action", {method: "POST",
            headers: {...auth, "Content-Type": "application/json",
                "Accept": streaming ? "application/x-ndjson" : "application/json"}, body: JSON.stringify(retry.command)});
        const body = await readAction(response, preview);
        if (body.status >= 400) {
            if (body.status < 500) retry = null;
            if (body.status === 409) await refresh();
            throw new Error(body.error);
        }
        retry = null;
        if (["new_conversation", "delete_conversation"].includes(action)) {
            if (action === "delete_conversation") byId("chat-input").value = "";
            selectConversation(body.workspace.conversation_id);
            byId("plan-details").open = false;
            showInspectorDetails(false);
            selectView("all");
        } else if (action === "clear_history") {
            byId("chat-input").value = "";
            conversationDrafts.delete(conversationId);
            byId("pending-message").hidden = true;
            byId("pending-message").replaceChildren();
            searchQuery = "";
            selectView("all");
        } else if (body.event.status === "pending" && action !== "chat") {
            // 方案保留在当前会话的时间线，候选页可由用户主动打开。
            searchQuery = "";
            selectView("all");
        } else if (!["chat", "draft", "bind_materials", "edit_task", "remove_material", "rename_conversation"].includes(action)) {
            selectView(body.event.proposal_id || action === "draft" ? "plans" : "tasks");
            searchQuery = "";
        }
        const followReply = chatStreaming && activeView === "all" && !searchQuery && nearLatestReply();
        if (chatStreaming) byId("pending-message").hidden = true;
        render(body.workspace);
        if (body.title_pending) {
            const title = body.workspace.conversations.find(item => item.id === targetConversation)?.title;
            if (title) void watchConversationTitle(targetConversation, title);
        }
        if (submittingEditor && byId("editor").open && body.event.status !== "not_submitted" && action !== "bind_materials") byId("editor").close();
        notice(body.event.status === "pending" ? "方案已保存为待确认候选。" :
            (action === "chat" ? "" : body.event.message));
        if (chatStreaming) {
            preview.hidden = true;
            if (followReply) showLatestReply();
        } else if (body.event.proposal_id) byId(`proposal-${body.event.proposal_id}`)?.scrollIntoView({block: "nearest"});
        return body.event;
    } catch (error) {
        notice(error.message || "未取得操作结果，请保留输入后重试。", true);
        return null;
    } finally {
        busy = false;
        if (chatStreaming) setChatWriteControls(false);
        chatStreaming = false;
        preview.hidden = true;
        preview.replaceChildren();
        updateReplyJump();
        byId("close-editor").disabled = false;
        byId("composer-status").textContent = "";
        byId("editor").removeAttribute("aria-busy");
        submitters.forEach(node => { node.disabled = false; });
    }
}

function render(workspace) {
    view = workspace;
    if (workspace.conversation_id !== conversationId) selectConversation(workspace.conversation_id);
    renderConversations();
    if (activeView === "all") {
        byId("chat-title").textContent = workspace.conversations.find(item => item.id === conversationId)?.title ?? "学习对话";
    }
    const state = workspace.state;
    byId("goal").innerHTML = markdown.renderInline(state.learning_goal);
    const plan = readableText(state.plan_markdown);
    byId("formal-plan").replaceChildren(plan);
    byId("plan-preview").replaceChildren(plan.cloneNode(true));
    const active = state.tasks.filter(task => !task.is_cancelled);
    const completed = active.filter(task => task.activity_status === "completed").length;
    const pending = workspace.proposals.filter(item => item.status === "pending").length;
    byId("nav-pending").textContent = pending ? String(pending) : "";
    byId("progress").max = active.length || 1;
    byId("progress").value = completed;
    byId("status").textContent = `计划修订 ${state.context_revision} · 已完成 ${completed}/${active.length} 项登记任务 · ${pending} 份待确认`;
    byId("review-note").textContent = state.tasks_need_review ? "计划已调整，请核对登记任务与新安排的关系。" : "";
    const materials = workspace.materials;
    byId("materials-status").textContent = materials.folder
        ? `${materials.available ? "资料已关联" : "目录不可用"} · ${materials.folder}`
        : "尚未关联资料";
    const tasks = byId("tasks");
    const historyOpen = tasks.querySelector(".task-history")?.open ?? false;
    tasks.replaceChildren();
    for (const [label, group] of [
        ["", state.tasks.filter(task => task.from_plan && !task.is_cancelled)],
        ["独立登记", state.tasks.filter(task => !task.from_plan && !task.is_cancelled)],
        ["历史任务", state.tasks.filter(task => task.is_cancelled)],
    ]) {
        if (!group.length) continue;
        if (label === "历史任务") {
            const history = element("details", "", "task-history");
            history.open = historyOpen;
            const entries = element("div", "", "task-history-entries");
            entries.append(...group.map(taskCard));
            history.append(element("summary", `${label}（${group.length}）`), entries);
            tasks.append(history);
        } else {
            if (label) tasks.append(element("p", label, "section-label"));
            tasks.append(...group.map(taskCard));
        }
    }
    if (!state.tasks.some(task => task.from_plan && !task.is_cancelled)) {
        tasks.prepend(element("p", "采用含任务的计划候选，或从当前计划提取任务。", "muted"));
    }
    renderTimeline();
}

function renderTimeline() {
    const workspace = view;
    const query = searchQuery.toLocaleLowerCase();
    byId("search-filter").hidden = !query;
    byId("search-description").textContent = query ? `搜索结果 · ${searchQuery}` : "";
    const timeline = byId("timeline");
    timeline.replaceChildren();
    const originalTimeline = new Set(workspace.events.filter(event =>
        ["chat", "draft"].includes(event.action)).flatMap(event => [event.input_text, event.message]));
    const imported = workspace.recent_chat.filter(item => !originalTimeline.has(item.content));
    if (activeView === "all" && imported.length && !query) {
        const history = element("details", "", "card");
        history.append(element("summary", "从存档恢复的近期对话"));
        for (const item of imported) {
            history.append(element("p", item.role === "human" ? "你" : "研习助手", "speaker"), readableText(item.content));
        }
        timeline.append(history);
    }
    if (activeView === "tasks" && !query) {
        const overview = element("section", "", "task-overview");
        overview.append(element("h2", "任务总览"));
        if (workspace.state.tasks.length) {
            overview.append(...workspace.state.tasks.map(task => {
                const card = taskCard(task);
                if (task.latest_feedback) card.append(element("p", "最近反馈", "section-label"), readableText(task.latest_feedback));
                return card;
            }));
        } else {
            overview.append(element("p", "还没有任务。可从当前计划提取，也可以添加临时任务。", "muted"),
                button("从计划提取", () => openEditor("extract_tasks"), true),
                button("添加临时任务", () => openEditor("create_task")));
        }
        timeline.append(overview);
    }
    const proposals = new Map(workspace.proposals.map(item => [item.id, item]));
    for (const event of workspace.events) {
        const isPlan = Boolean(event.proposal_id) || ["draft", "manual_plan"].includes(event.action);
        const isTask = ["feedback", "create_task", "cancel_task", "edit_task"].includes(event.action);
        if ((activeView === "plans" && !isPlan) || (activeView === "tasks" && !isTask)) continue;
        if (activeView === "all" && (event.action === "new_conversation" ||
            (!query && (event.conversation_id ?? null) !== conversationId))) continue;
        const candidate = proposals.get(event.proposal_id);
        if (query && ![event.input_text, event.message, candidate?.markdown].some(text => text?.toLocaleLowerCase().includes(query))) continue;
        const isDialogue = ["chat", "draft"].includes(event.action);
        const card = element("article", "", isDialogue ? "turn" : "operation");
        if (event.input_text && isDialogue) {
            const message = readableText(event.input_text);
            message.classList.add("user-message");
            card.append(element("div", "你", "speaker user-speaker"), message,
                element("div", "研习助手", "speaker"));
        } else if (event.input_text) card.append(element("p", event.input_text, "muted"));
        if (event.status === "not_submitted") card.append(element("h3", "需要补充信息"));
        if (candidate && event.action !== "decide") {
            card.append(candidateCard(candidate));
            if (event.message && !["extract_tasks", "split_task"].includes(candidate.mode)) {
                const details = element("details");
                details.append(element("summary", "查看本轮说明与来源"), readableText(event.message));
                card.append(details);
            }
        } else if (event.message) card.append(readableText(event.message));
        timeline.append(card);
    }
    if (query && activeView === "all") {
        for (const message of imported.filter(item => item.content.toLocaleLowerCase().includes(query))) {
            const card = element("article", "", "card");
            card.append(element("p", message.role === "human" ? "你 · 近期对话" : "助手 · 近期对话", "muted"), readableText(message.content));
            timeline.append(card);
        }
    }
    if (!timeline.childElementCount) {
        const empty = element("section", "", "empty-state");
        empty.append(element("div", "学", "brand-mark"), element("h2", query ? "没有匹配的记录" : "从一个学习问题开始"),
            element("p", query ? "试试另一个关键词，或切换记录分类。" : "你可以询问资料、调整安排，或从右侧记录任务进度。", "muted"));
        timeline.append(empty);
    }
    if (chatStreaming) setChatWriteControls(true);
    updateReplyJump();
}

function taskCard(task) {
    const row = element("section", "", "task");
    const meta = element("p",
        `${task.is_cancelled ? "已取消 · " : ""}${activity[task.activity_status]} · ${understanding[task.understanding_status]}`, "muted task-meta");
    if (task.estimated_minutes) meta.append(` · 预计 ${task.estimated_minutes} 分钟`);
    row.append(element("h3", task.title), meta);
    if (task.instructions) {
        const details = element("details", "", "task-instructions");
        details.append(element("summary", "任务说明"), readableText(task.instructions));
        row.append(details);
    }
    if (!task.is_cancelled) {
        const quick = element("div", "", "task-quick");
        for (const [key, label, choices] of [
            ["activity_status", "任务进度", activity], ["understanding_status", "理解程度", understanding],
        ]) {
            const field = select(choices);
            field.options[0].remove();
            field.value = task[key];
            field.addEventListener("change", async () => {
                const result = await send("edit_task", {task_id: task.task_id, [key]: field.value});
                if (!result) field.value = task[key];
            });
            labeled(quick, label, field);
        }
        row.append(quick);
    }
    const controls = element("div", "", "actions task-actions");
    controls.append(button("查看／反馈", () => openEditor("feedback", task.task_id)));
    if (!task.is_cancelled) controls.append(
        button("编辑", () => openEditor("edit_task", task.task_id)),
        button("细化", () => openEditor("split_task", task.task_id)),
        button("取消任务", () => openEditor("cancel_task", task.task_id)));
    for (const action of controls.children) action.classList.add("text-action");
    row.append(controls);
    return row;
}

function candidateCard(candidate) {
    const card = element("section", "", "candidate");
    card.id = `proposal-${candidate.id}`;
    const header = element("div", "", "candidate-header");
    const taskOnly = ["extract_tasks", "split_task"].includes(candidate.mode);
    const title = {extract_tasks: "提取的任务", split_task: "任务细化方案"}[candidate.mode] || "计划方案";
    header.append(element("h3", title), element("span", statuses[candidate.status], `badge ${candidate.status}`));
    card.append(header);
    if (!taskOnly && candidate.diff) {
        const comparison = element("details");
        comparison.open = candidate.status === "pending";
        const diff = element("div", "", "plan-diff");
        for (const line of candidate.diff.split("\n").slice(2)) {
            if (line.startsWith("@@")) continue;
            const kind = line.startsWith("+") ? "add" : line.startsWith("-") ? "archive" : "context";
            diff.append(element("div", line, `diff-${kind}`));
        }
        comparison.append(element("summary", "计划正文变更"), diff);
        card.append(comparison);
        const full = element("details");
        full.append(element("summary", "查看完整候选计划"), readableText(candidate.markdown));
        card.append(full);
    }
    if (candidate.task_changes.length) {
        const list = element("ul", "", "candidate-task-list");
        for (const change of candidate.task_changes) {
            const operation = {add: "新增", update: "修改 · 保留进度", archive: "移除 · 保留历史"}[change.operation];
            const title = change.title || change.current_title || change.task_id;
            const rename = change.title && change.current_title && change.title !== change.current_title
                ? `${change.current_title} → ` : "";
            const previous = change.previous_minutes != null ? `${change.previous_minutes} → ` : "";
            const estimate = change.estimated_minutes ? ` · ${previous}${change.estimated_minutes} 分钟` : "";
            const item = element("li", `${operation} · ${rename}${title}${estimate}`, `diff-${change.operation}`);
            if (change.instructions !== null && change.previous_instructions &&
                change.previous_instructions !== change.instructions) {
                const previous = readableText(change.previous_instructions);
                previous.classList.add("diff-archive");
                item.append(previous);
            }
            if (change.instructions) {
                const next = readableText(change.instructions);
                next.classList.add("diff-add");
                item.append(next);
            }
            list.append(item);
        }
        card.append(element("h4", "任务变更"), list);
    } else if (!taskOnly) {
        card.append(element("p", "此候选没有任务变更，采用后会保留现有任务和反馈。", "muted"));
    }
    const controls = element("div", "", "actions");
    if (candidate.status === "pending") {
        for (const [decision, title] of [["adopt", "采用并保存"], ["reject", "拒绝"]]) {
            controls.append(button(title, () => send("decide", {proposal_id: candidate.id, decision}), decision === "adopt"));
        }
    } else if (candidate.status === "stale") {
        controls.append(button("拒绝", () => send("decide", {proposal_id: candidate.id, decision: "reject"})));
    } else if (candidate.can_resubmit) {
        controls.append(button("原样重新提交", () => send("resubmit", {proposal_id: candidate.id})));
    }
    card.append(controls);
    return card;
}

function labeled(form, title, input) {
    const label = element("label", title);
    label.append(input);
    form.append(label);
    return input;
}

function select(choices) {
    const field = element("select");
    for (const [value, text] of [["", "保持原值"], ...Object.entries(choices)]) {
        const option = element("option", text);
        option.value = value;
        field.append(option);
    }
    return field;
}

async function showMaterialCatalog(target, offset = 0) {
    target.replaceChildren();
    if (!view.materials.available) {
        target.textContent = "关联后会在此显示可访问的文件清单。";
        return;
    }
    target.textContent = "正在检查目录…";
    try {
        const url = new URL(scopedUrl("/api/materials"), location.origin);
        url.searchParams.set("offset", offset);
        const response = await fetch(url, {headers: auth});
        const body = await response.json();
        if (!response.ok) throw new Error(body.error);
        const {files, total_files, next_offset, limit} = body.data;
        const list = element("ul", "", "material-list");
        for (const file of files) {
            const row = element("li", "", "material-entry");
            const remove = button("移除", () => openEditor("remove_material", file.relative_path));
            remove.className = "text-action";
            remove.setAttribute("aria-label", `移除文件 ${file.relative_path}`);
            row.append(element("span", file.relative_path), remove);
            list.append(row);
        }
        const pages = element("div", "", "actions");
        if (offset) pages.append(button("上一页", () => void showMaterialCatalog(target, Math.max(0, offset - limit))));
        if (next_offset !== null) pages.append(button("下一页", () => void showMaterialCatalog(target, next_offset)));
        target.replaceChildren(element("p", total_files
            ? `共 ${total_files} 个文件，显示第 ${offset + 1}–${offset + files.length} 个。` : "当前目录没有可用文件。"),
            list, pages);
    } catch (error) {
        target.textContent = error.message;
    }
}

function openEditor(mode, taskId = null) {
    if (!view || (busy && !(chatStreaming && mode === "feedback"))) return;
    const key = `${mode}:${taskId ?? ""}`;
    const editor = byId("editor");
    if (editorKey === key && editor.open) return;
    editorKey = key;
    const editorConversationId = conversationId;
    let editVersion = view.version;
    const task = view.state.tasks.find(item => item.task_id === taskId);
    const form = element("form");
    const input = element(["create_task", "bind_materials"].includes(mode) ? "input" : "textarea");
    input.required = true;
    input.maxLength = mode === "create_task" ? 200 : 12000;
    input.rows = 4;
    let payload;
    let catalog;
    let reloadTaskFeedback;
    let action = mode;
    let title;
    let submitLabel = "提交并保存";
    if (mode === "rename_conversation") {
        title = "重命名对话";
        input.rows = 1;
        input.maxLength = 60;
        input.value = view.conversations.find(item => item.id === conversationId)?.title ?? "";
        labeled(form, "新标题", input);
        payload = () => ({title: input.value});
    } else if (["delete_conversation", "clear_history", "remove_material"].includes(mode)) {
        const session = view.conversations.find(item => item.id === conversationId);
        if (mode === "remove_material") {
            title = "移除资料文件";
            form.append(element("p", `「${taskId}」会移入资料目录中的 .yanxi-trash 回收区。其他使用该文件的对话也会失去访问。`));
            submitLabel = "移入回收区";
            payload = () => ({relative_path: taskId, confirm: true});
        } else {
            const remove = mode === "delete_conversation";
            title = remove ? "删除对话" : "清空聊天历史";
            form.append(element("p", `对话：${session.title}`), element("p", remove
                ? "将删除这条对话的聊天、计划、任务及候选记录。关联的本地文件会保留。"
                : "将清空这条对话的消息和计划候选历史，保留正式计划、任务、反馈与资料关联。"));
            submitLabel = remove ? "删除对话" : "清空历史";
            payload = () => ({confirm: true});
        }
    } else if (mode === "bind_materials") {
        title = "关联本地资料";
        form.append(element("p", "复制资料所在文件夹的绝对路径。支持 PDF、Markdown、TXT 和 Python 文件。", "muted"));
        input.placeholder = "粘贴文件夹路径";
        input.autocomplete = "off";
        input.maxLength = 4096;
        labeled(form, "本机资料文件夹", input);
        catalog = element("section", "", "muted");
        catalog.setAttribute("aria-live", "polite");
        form.append(catalog);
        submitLabel = "检查并关联";
        payload = () => ({directory: input.value.trim().replace(/^"(.*)"$/, "$1")});
    } else if (mode === "json_edit") {
        title = "编辑受控 JSON";
        const kind = labeled(form, "修改对象", select({plan: "正式计划（生成候选）", task: "任务进度与反馈"}));
        kind.options[0].remove();
        const tasks = labeled(form, "选择登记任务", select(Object.fromEntries(view.state.tasks.map(item => [item.task_id, item.title]))));
        tasks.options[0].remove();
        const hint = element("p", "", "muted");
        form.append(hint);
        labeled(form, "可编辑字段", input);
        input.rows = 10;
        input.spellcheck = false;
        input.className = "json-input";
        function template() {
            tasks.disabled = kind.value === "plan";
            const task = view.state.tasks.find(item => item.task_id === tasks.value);
            const data = kind.value === "plan" ? {plan_markdown: view.state.plan_markdown} : task ? {
                text: "", activity_status: task.activity_status, understanding_status: task.understanding_status,
            } : {};
            input.value = JSON.stringify(data, null, 2);
            hint.textContent = kind.value === "plan"
                ? "提交后会生成候选。请在主页面核对差异并采用。"
                : "反馈正文必填。进度或理解填 null 表示保留原值。任务以当前选择为准。";
        }
        kind.addEventListener("change", template);
        tasks.addEventListener("change", template);
        template();
        submitLabel = "校验并提交";
        payload = () => {
            const data = JSON.parse(input.value);
            if (!data || Array.isArray(data) || typeof data !== "object") throw new Error("请输入一个 JSON 对象。");
            action = kind.value === "plan" ? "manual_plan" : "feedback";
            if (action === "manual_plan") return data;
            if (!tasks.value) throw new Error("请先登记任务。");
            if (Object.hasOwn(data, "task_id")) throw new Error("任务由上方选择，请移除 JSON 中的 task_id。");
            return {...data, task_id: tasks.value};
        };
    } else if (mode === "feedback") {
        title = `任务反馈 · ${task.title}`;
        const progress = labeled(form, "活动进度", select(activity));
        const level = labeled(form, "理解程度", select(understanding));
        progress.options[0].remove();
        level.options[0].remove();
        const previous = element("section", "", "previous-feedback");
        form.append(previous);
        labeled(form, "本次反馈原文", input).closest("label").classList.add("feedback-input");
        let baseline = task;
        reloadTaskFeedback = current => {
            baseline = current;
            progress.value = current.activity_status;
            progress.disabled = current.is_cancelled;
            level.value = current.understanding_status;
            previous.replaceChildren(element("h3", "最近反馈"),
                current.latest_feedback ? readableText(current.latest_feedback) : element("p", "暂无反馈", "muted"));
        };
        reloadTaskFeedback(task);
        payload = () => ({
            task_id: taskId, text: input.value,
            activity_status: progress.disabled || progress.value === baseline.activity_status ? null : progress.value,
            understanding_status: level.value === baseline.understanding_status ? null : level.value,
        });
    } else if (mode === "cancel_task") {
        title = `取消任务 · ${task.title}`;
        form.append(element("p", "取消会保留此前的进度与反馈。"));
        labeled(form, "取消原因", input);
        submitLabel = "确认取消并保存";
        payload = () => ({task_id: taskId, text: input.value});
    } else if (mode === "create_task") {
        title = "登记任务";
        labeled(form, "任务名称", input);
        payload = () => ({title: input.value});
    } else if (mode === "edit_task") {
        title = `编辑任务 · ${task.title}`;
        const name = labeled(form, "任务标题", element("input"));
        name.value = task.title;
        name.required = true;
        name.maxLength = 200;
        const duration = labeled(form, "预计分钟（可留空）", element("input"));
        duration.type = "number";
        duration.min = 1;
        duration.max = 1440;
        duration.value = task.estimated_minutes ?? "";
        input.value = task.instructions;
        input.maxLength = 4000;
        input.required = false;
        labeled(form, "任务说明", input);
        payload = () => ({task_id: taskId, title: name.value,
            estimated_minutes: duration.value ? duration.valueAsNumber : null, instructions: input.value});
    } else if (mode === "extract_tasks" || mode === "split_task") {
        action = "draft";
        title = mode === "extract_tasks" ? "从正式计划提取任务" : `细化任务 · ${task.title}`;
        if (mode === "extract_tasks" && view.state.tasks.some(task => task.from_plan && !task.is_cancelled)) {
            form.append(element("p", "当前计划已有任务记录。采用新提取的任务前，请对照已有任务和进度，避免重复登记。", "task-extraction-note"));
        }
        input.value = mode === "extract_tasks" ? "从当前计划提取尚未登记的可执行任务。"
            : "将所选任务拆成两个或更多可执行的子任务。";
        labeled(form, "要求", input);
        submitLabel = "生成待确认任务";
        payload = () => ({mode, task_id: taskId, instructions: input.value});
    } else {
        action = "draft";
        title = mode === "adjust" ? "调整现有计划" : "重新制定并替换";
        labeled(form, "请填写调整要求", input);
        submitLabel = "生成待确认方案";
        payload = () => ({mode, instructions: input.value});
    }
    const submit = element("button", submitLabel, "primary");
    submit.type = "submit";
    const controls = element("div", "", "actions");
    const versionNote = element("p", `表单基于状态版本 ${editVersion}。`, "muted");
    const reload = button("核对最新状态", async () => {
        if (busy) return;
        try {
            await refresh();
            if (editorConversationId !== conversationId) throw new Error("原对话已不可用，请重新打开表单。");
            const currentTask = view.state.tasks.find(item => item.task_id === taskId);
            if (reloadTaskFeedback && currentTask) {
                reloadTaskFeedback(currentTask);
                editVersion = view.version;
                versionNote.textContent = "已加载最新状态并保留反馈原文。如需修改状态，请重新选择。";
            } else if (editVersion !== view.version) {
                versionNote.textContent = "状态已更新。请保留所填内容并重新打开表单后提交。";
            } else {
                versionNote.textContent = `表单已核对，当前状态版本 ${editVersion}。`;
            }
        } catch (error) { notice(error.message, true); }
    });
    reload.id = "feedback-reload";
    controls.append(submit, reload, button("取消", () => { if (!editor.hasAttribute("aria-busy")) editor.close(); }));
    form.append(controls);
    form.addEventListener("submit", async event => {
        event.preventDefault();
        try {
            if (editorConversationId !== conversationId) throw new Error("对话已切换，请重新打开表单。");
            const data = payload();
            const result = await send(action, data, editVersion);
            if (result && catalog) {
                editVersion = view.version;
                versionNote.textContent = `表单基于状态版本 ${editVersion}。`;
                await showMaterialCatalog(catalog);
            }
        } catch (error) { notice(error.message, true); }
    });
    byId("editor-title").textContent = title;
    const waitNote = element("p", waitingForReply, "muted");
    waitNote.id = "feedback-wait";
    waitNote.hidden = !chatStreaming;
    byId("editor-body").replaceChildren(versionNote, waitNote, form);
    if (chatStreaming) setChatWriteControls(true);
    byId("editor-notice").replaceChildren();
    showInspectorDetails(false);
    if (!editor.open) editor.showModal();
    (input.isConnected ? input : byId("close-editor")).focus({preventScroll: true});
    if (catalog) void showMaterialCatalog(catalog);
}

byId("close-editor").addEventListener("click", () => {
    if (!byId("editor").hasAttribute("aria-busy")) byId("editor").close();
});
byId("editor").addEventListener("cancel", event => {
    if (event.currentTarget.hasAttribute("aria-busy")) event.preventDefault();
});
byId("dismiss-notice").addEventListener("click", () => notice(""));
document.querySelectorAll("[data-mode]").forEach(node => node.addEventListener("click", () => openEditor(node.dataset.mode)));
document.querySelectorAll("[data-view]").forEach(node => node.addEventListener("click", () => selectView(node.dataset.view)));
function openSearch() {
    if (!view || byId("editor").open || byId("search-dialog").open) return;
    const input = byId("timeline-search");
    input.value = searchQuery;
    byId("search-dialog").showModal();
    input.select();
}
byId("open-search").addEventListener("click", openSearch);
byId("new-conversation").addEventListener("click", () => send("new_conversation", {}));
byId("close-search").addEventListener("click", () => byId("search-dialog").close());
byId("search-form").addEventListener("submit", event => {
    event.preventDefault();
    searchQuery = byId("timeline-search").value.trim();
    byId("search-dialog").close();
    selectView("all");
    byId("search-filter").scrollIntoView({block: "nearest"});
});
byId("clear-search").addEventListener("click", () => {
    searchQuery = "";
    selectView("all");
    byId("open-search").focus();
});
document.addEventListener("keydown", event => {
    if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === "k" && !event.isComposing) {
        if (byId("editor").open) return;
        event.preventDefault();
        openSearch();
    }
});
function showInspectorDetails(expanded) {
    document.querySelector(".inspector").dataset.mobileExpanded = String(expanded);
    byId("toggle-inspector").setAttribute("aria-expanded", String(expanded));
}
byId("toggle-inspector").addEventListener("click", event => {
    showInspectorDetails(event.currentTarget.getAttribute("aria-expanded") !== "true");
});
document.querySelectorAll("[data-inspector-target]").forEach(control => {
    control.addEventListener("click", () => {
        const target = byId(control.dataset.inspectorTarget);
        if (target.id !== "inspector-status" && byId("toggle-inspector").getClientRects().length) {
            showInspectorDetails(true);
        }
        target.focus({preventScroll: true});
        const panel = document.querySelector(".inspector-scroll");
        panel.scrollTop += target.getBoundingClientRect().top - panel.getBoundingClientRect().top;
    });
});
byId("chat-input").addEventListener("keydown", event => {
    if (event.key === "Enter" && (event.ctrlKey || event.metaKey) && !event.isComposing) {
        event.preventDefault();
        byId("chat-form").requestSubmit();
    }
});
byId("chat-form").addEventListener("submit", async event => {
    event.preventDefault();
    if (busy || !view) return;
    const input = byId("chat-input");
    const original = input.value;
    if (!original.trim()) return;
    if (searchQuery) {
        searchQuery = "";
        selectView("all");
    }
    const pending = byId("pending-message");
    const message = element("div", original, "user-message pending-text");
    pending.replaceChildren(element("div", "你", "speaker user-speaker"), message);
    const follow = nearLatestReply();
    pending.hidden = Boolean(searchQuery);
    input.value = "";
    if (follow && !pending.hidden) showLatestReply();
    const result = await send("chat", {text: original});
    pending.hidden = true;
    if (!result && !input.value) input.value = original;
    if (!result && input.value !== original) {
        pending.firstElementChild.textContent = "你 · 未收到保存确认";
        pending.hidden = activeView !== "all" || Boolean(searchQuery);
    } else pending.replaceChildren();
});
byId("refresh").addEventListener("click", () => { if (!busy) refresh().catch(error => notice(error.message, true)); });
function download(blob, name) {
    const url = URL.createObjectURL(blob);
    const link = element("a");
    link.href = url;
    link.download = name;
    link.click();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
}
byId("export-plan").addEventListener("click", () => {
    if (!view) return;
    download(new Blob([view.state.plan_markdown], {type: "text/markdown;charset=utf-8"}), "learning-plan.md");
});
byId("jump-to-latest").addEventListener("click", showLatestReply);
window.addEventListener("scroll", updateReplyJump, {passive: true});
async function exportState(id) {
    try {
        const response = await fetch(scopedUrl("/api/export", id), {headers: auth});
        if (!response.ok) throw new Error("导出失败，请刷新后重试。");
        download(await response.blob(), "learning-state.json");
    } catch (error) { notice(error.message, true); }
}
refresh().catch(error => notice(error.message, true));
