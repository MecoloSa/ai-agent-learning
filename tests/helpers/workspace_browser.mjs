// Run the real browser and page against a temporary local workspace.
import {spawn} from "node:child_process";
import {writeFile, open} from "node:fs/promises";
import {join, resolve} from "node:path";
import {createServer} from "node:net";

const [url, output, edge] = process.argv.slice(2);
const profile = resolve(output, "edge-profile");
const log = await open(join(output, "edge-stderr.log"), "w");
const listener = createServer();
await new Promise((done, fail) => listener.once("error", fail).listen(0, "127.0.0.1", done));
const port = listener.address().port;
await new Promise(done => listener.close(done));
const browser = spawn(edge, ["--headless=new", "--disable-gpu", "--no-first-run",
    "--no-default-browser-check", "--disable-background-networking", `--remote-debugging-port=${port}`,
    "--enable-logging=stderr", `--user-data-dir=${profile}`, "about:blank"],
    {windowsHide: true, stdio: ["ignore", "ignore", log.fd]});
let socket;
const pause = ms => new Promise(done => setTimeout(done, ms));
async function until(check, label, timeout = 20000) {
    const end = Date.now() + timeout;
    while (Date.now() < end) {
        if (browser.exitCode !== null) throw new Error(`Edge exited: ${browser.exitCode}`);
        if (await check()) return;
        await pause(100);
    }
    throw new Error(`Timed out: ${label}`);
}

try {
    await until(async () => {
        try { return (await fetch(`http://127.0.0.1:${port}/json/version`, {signal: AbortSignal.timeout(300)})).ok; }
        catch { return false; }
    }, "Edge DevTools port");
    const targets = await (await fetch(`http://127.0.0.1:${port}/json/list`)).json();
    socket = new WebSocket(targets.find(item => item.type === "page").webSocketDebuggerUrl);
    await new Promise((done, fail) => {
        socket.addEventListener("open", done, {once: true});
        socket.addEventListener("error", fail, {once: true});
    });
    const pending = new Map();
    const errors = [];
    let id = 0;
    socket.addEventListener("message", event => {
        const result = JSON.parse(event.data);
        if (result.method === "Runtime.exceptionThrown") errors.push(result.params.exceptionDetails.text);
        const handler = pending.get(result.id);
        if (handler) {
            pending.delete(result.id);
            result.error ? handler.fail(new Error(result.error.message)) : handler.done(result.result);
        }
    });
    function call(method, params = {}) {
        return new Promise((done, fail) => {
            pending.set(++id, {done, fail});
            socket.send(JSON.stringify({id, method, params}));
        });
    }
    async function evaluate(expression) {
        const result = await call("Runtime.evaluate", {expression, returnByValue: true, awaitPromise: true});
        if (result.exceptionDetails) throw new Error(result.exceptionDetails.exception?.description || result.exceptionDetails.text);
        return result.result.value;
    }
    const statusHas = text => evaluate(`document.querySelector('#status')?.textContent.includes(${JSON.stringify(text)})`);
    const clickText = text => evaluate(`Array.from(document.querySelectorAll('button')).find(b => b.textContent.endsWith(${JSON.stringify(text)})).click()`);
    const fill = text => evaluate(`document.querySelector('#editor textarea').value = ${JSON.stringify(text)}; document.querySelector('#editor form').requestSubmit();`);
    await call("Runtime.enable");
    await call("Page.enable");
    await call("Emulation.setDeviceMetricsOverride", {width: 1280, height: 900, deviceScaleFactor: 1, mobile: false});
    await call("Page.navigate", {url});
    await until(() => evaluate("!document.querySelector('#setup-materials').hidden && !document.querySelector('.rail').hidden"), "default conversation without materials");
    const bind = directory => evaluate(`document.querySelector('#setup-directory').value = ${JSON.stringify(directory)}; document.querySelector('#setup-materials').requestSubmit();`);
    await bind(join(output, "missing"));
    await until(() => evaluate("document.querySelector('#setup-status').classList.contains('error')"), "invalid directory");
    await bind(join(output, "materials"));
    await until(() => statusHas("计划修订 1"), "material binding preserves the existing plan");
    await clickText("本地资料");
    await until(() => evaluate("document.querySelector('.material-list')?.textContent.includes('notes.md')"), "real local files listed");
    await evaluate("document.querySelector('#close-editor').click(); document.querySelector('#dismiss-notice').click()");
    if (!await evaluate("document.querySelector('#materials-status').textContent.includes('资料已关联')")) throw new Error("Missing bound material status");
    if (!await evaluate("document.querySelector('#deep-mode').getAttribute('aria-checked') === 'false' && document.querySelector('#thinking-option').hidden")) throw new Error("Chat must default to fast mode");
    await evaluate("document.querySelector('#deep-mode').click()");
    if (!await evaluate("document.querySelector('#deep-mode').getAttribute('aria-checked') === 'true' && !document.querySelector('#thinking-option').hidden && document.querySelector('#worker-thinking').checked")) throw new Error("Deep mode must show thinking enabled by default");
    await evaluate("window.scrollTo(0, 0)");
    const modes = await call("Page.captureScreenshot", {format: "png"});
    await writeFile(join(output, "workspace-chat-modes.png"), Buffer.from(modes.data, "base64"));
    await evaluate("document.querySelector('#deep-mode').click()");

    await clickText("查看／反馈");
    if (!await evaluate("document.querySelector('#editor').matches(':modal') && document.querySelector('#notice').textContent === ''")) {
        throw new Error("Sidebar selection should only open the editor");
    }
    await evaluate("document.querySelector('#close-editor').click()");
    if (await evaluate("document.querySelector('#editor').open")) throw new Error("Modal close button failed");
    await clickText("查看／反馈");
    await evaluate("document.querySelectorAll('#editor select')[0].value='completed'; document.querySelectorAll('#editor select')[1].value='self_reported_unclear';");
    await fill("阅读完成，但还分不清两类工具。");
    await until(() => statusHas("已完成 1/1"), "feedback refresh");
    await clickText("查看／反馈");
    if (!await evaluate("document.querySelectorAll('#editor select')[0].value === 'completed' && document.querySelectorAll('#editor select')[1].value === 'self_reported_unclear' && document.querySelector('.previous-feedback').textContent.includes('还分不清') && !!document.querySelector('.feedback-input textarea')")) throw new Error("Feedback should show saved values and separate the original note");
    await evaluate(`(async () => {
        const headers = {"X-Workspace-Token": document.querySelector('meta[name="workspace-token"]').content, "Content-Type": "application/json"};
        const workspace = await (await fetch('/api/state', {headers})).json();
        const response = await fetch('/api/action', {method: 'POST', headers,
            body: JSON.stringify({action: 'edit_task', version: workspace.version, request_id: crypto.randomUUID().replaceAll('-', ''), payload: {task_id: workspace.state.tasks[0].task_id, understanding_status: 'self_reported_partial'}})});
        if (!response.ok) throw new Error('Concurrent task edit failed');
    })()`);
    await fill("这段尚未提交的反馈应保留。");
    await until(() => evaluate("document.querySelector('#editor-notice').classList.contains('error')"), "stale feedback rejected");
    await clickText("核对最新状态");
    await until(() => evaluate("document.querySelectorAll('#editor select')[1].value === 'self_reported_partial' && document.querySelector('#editor textarea').value === '这段尚未提交的反馈应保留。'"), "feedback rebase retains draft and latest status");
    await evaluate("document.querySelector('#close-editor').click()");
    await clickText("调整计划");
    await fill("请先澄清我需要保留哪一项活动。");
    await until(() => evaluate("!document.querySelector('#editor-stream').hidden && document.querySelector('#editor-stream .stream-text')?.textContent.trim() === '请说明你希望'"), "live delta before completion");
    await until(() => evaluate("document.querySelector('#editor-notice').textContent.includes('保留哪项')"), "clarification");
    if (!await evaluate("document.querySelector('#editor-notice').firstChild.textContent === '本轮未生成待确认候选。'")) {
        throw new Error("Clarification must show the authoritative draft status");
    }
    if (!await evaluate("document.querySelector('#editor').open && document.querySelector('#editor textarea').value.includes('请先澄清')")) {
        throw new Error("Clarification should retain the user's form");
    }
    await fill("改为阅读5分钟、口头说明10分钟，提交完整候选。");
    await until(() => statusHas("1 份待确认"), "pending candidate");
    if (!await evaluate("!!document.querySelector('.plan-diff .diff-add') && !!document.querySelector('.plan-diff .diff-archive')")) throw new Error("Plan diff should show colored additions and removals");
    const screenshot = await call("Page.captureScreenshot", {format: "png"});
    await writeFile(join(output, "workspace.png"), Buffer.from(screenshot.data, "base64"));
    await clickText("拒绝");
    await until(() => statusHas("0 份待确认"), "rejection");
    await clickText("原样重新提交");
    await until(() => statusHas("1 份待确认"), "resubmission");
    await clickText("采用并保存");
    await until(() => statusHas("计划修订 2"), "adoption");
    if (!await evaluate("document.querySelector('#tasks').textContent.includes('口头说明工具区别')")) throw new Error("Adoption should create the plan task");
    await evaluate("const task = Array.from(document.querySelectorAll('.task')).find(row => row.querySelector('h3').textContent === '口头说明工具区别'); const level = task.querySelectorAll('select')[1]; level.value = 'self_reported_partial'; level.dispatchEvent(new Event('change'));");
    await until(() => evaluate("document.querySelector('#notice').textContent.includes('任务已保存') && document.querySelector('#tasks').textContent.includes('未开始 · 大致理解')"), "quick self-reported understanding");
    await call("Page.reload");
    await until(() => statusHas("计划修订 2"), "reload persisted state");
    if (!await evaluate("document.querySelector('#materials-status').textContent.includes('资料已关联')")) throw new Error("Material binding should survive reload");
    await evaluate("document.querySelector('#plan-details summary').click()");
    if (!await evaluate("document.querySelector('#plan-details').open && document.querySelector('#formal-plan').textContent.includes('口头说明10分钟') && document.querySelector('#formal-plan strong')?.textContent === '复述要点'")) {
        throw new Error("The sidebar should show the adopted plan");
    }
    await evaluate("const makeUrl = URL.createObjectURL; const click = HTMLAnchorElement.prototype.click; URL.createObjectURL = blob => { blob.text().then(text => window.downloadedPlan = text); return makeUrl(blob); }; HTMLAnchorElement.prototype.click = function () { window.downloadedFilename = this.download; }; document.querySelector('#export-plan').click(); URL.createObjectURL = makeUrl; HTMLAnchorElement.prototype.click = click;");
    await until(() => evaluate("window.downloadedFilename === 'learning-plan.md' && window.downloadedPlan?.includes('**复述要点**')"), "formal plan Markdown export");
    await evaluate("document.querySelector('.conversation-menu-trigger').click()");
    await clickText("编辑受控 JSON");
    await fill('{"plan_markdown":"阅读10分钟并复述","context_revision":99}');
    await until(() => evaluate("document.querySelector('#editor-notice').classList.contains('error')"), "protected JSON fields rejected");
    await fill('{"plan_markdown":"阅读10分钟并复述"}');
    await until(() => statusHas("1 份待确认"), "manual JSON candidate");
    if (!await statusHas("计划修订 2")) throw new Error("JSON draft must not update the formal plan");
    await clickText("拒绝");
    await until(() => statusHas("0 份待确认"), "manual JSON rejection");
    await evaluate("document.querySelector('#open-search').click()");
    if (!await evaluate("document.querySelector('#search-dialog').matches(':modal')")) throw new Error("Search should open a dialog");
    const search = await call("Page.captureScreenshot", {format: "png"});
    await writeFile(join(output, "workspace-search.png"), Buffer.from(search.data, "base64"));
    await evaluate("document.querySelector('#timeline-search').value='阅读10分钟并复述'; document.querySelector('#search-form').requestSubmit();");
    if (!await evaluate("document.querySelector('#timeline').textContent.includes('阅读10分钟并复述') && !document.querySelector('#timeline').textContent.includes('请先澄清')")) throw new Error("Timeline filter failed");
    await evaluate("document.querySelector('#clear-search').click()");
    if (!await evaluate("document.querySelector('#search-filter').hidden && document.querySelector('#timeline').textContent.includes('请先澄清')")) throw new Error("Clear filter failed");
    await evaluate("document.querySelector('.conversation-menu-trigger').click()");
    await clickText("编辑受控 JSON");
    await evaluate("const kind = document.querySelector('#editor select'); kind.value = 'task'; kind.dispatchEvent(new Event('change'));");
    await fill('{"text":"JSON 补充反馈：现在可以区分两类工具。","understanding_status":"self_reported_clear"}');
    await until(() => evaluate("!document.querySelector('#editor').open && !document.querySelector('#notice-bar').hidden"), "task JSON saved");
    if (!await statusHas("已完成 1/2")) throw new Error("Feedback-only JSON should preserve activity progress");
    await call("Page.navigate", {url});
    await until(() => statusHas("计划修订 2"), "desktop screenshot state");
    await evaluate(`window.sentModes = null;
        const originalFetch = window.fetch;
        window.fetch = (...args) => {
            const body = args[1]?.body && JSON.parse(args[1].body);
            if (body?.action === "chat") window.sentModes = body.payload;
            return originalFetch(...args);
        };
        document.querySelector('#deep-mode').click();
        document.querySelector('#worker-thinking').click();
        document.querySelector('#chat-input').value = '如何区分通用工具和专用工具？';
        document.querySelector('#chat-form').requestSubmit();`);
    if (!await evaluate("window.sentModes?.deep_mode === true && window.sentModes?.worker_thinking === false")) throw new Error("Chat must send the selected mode and thinking state");
    if (!await evaluate("!document.querySelector('#pending-message').hidden && document.querySelector('#pending-message').textContent.includes('如何区分通用工具和专用工具？') && !document.querySelector('#chat-input').value")) throw new Error("Sent message should appear before the model returns");
    await until(() => evaluate("!document.querySelector('#stream-preview').hidden && document.querySelector('#stream-preview strong')?.textContent === '通用工具'"), "Markdown chat delta before completion");
    if (!await evaluate(`(() => {
        document.querySelector('#open-search').click();
        const search = document.querySelector('#search-dialog');
        const searchAvailable = search.open && !search.querySelector('button[type=submit]').disabled;
        search.close();
        document.querySelector('#plan-details').open = true;
        const planVisible = document.querySelector('#formal-plan').getBoundingClientRect().height > 0;
        let exported = false;
        const click = HTMLAnchorElement.prototype.click;
        try {
            HTMLAnchorElement.prototype.click = function () { exported = this.download === 'learning-plan.md'; };
            document.querySelector('#export-plan').click();
        } finally { HTMLAnchorElement.prototype.click = click; }
        document.querySelector('#tasks .task-actions button').click();
        const editor = document.querySelector('#editor');
        if (!editor.open) return false;
        editor.querySelector('textarea').value = '生成期间填写的反馈草稿';
        document.querySelector('#timeline').style.minHeight = '200vh';
        window.scrollTo(0, 0);
        return searchAvailable && planVisible && exported &&
            editor.querySelector('button[type=submit]').disabled &&
            !document.querySelector('#feedback-wait').hidden;
    })()`)) throw new Error("Read-only controls and feedback draft must remain available during chat");
    await until(() => evaluate("document.querySelector('#stream-preview').hidden && document.querySelector('#timeline').textContent.includes('专用工具适合约束明确的业务动作。')"), "chat committed");
    if (!await evaluate(`(() => {
        const editor = document.querySelector('#editor');
        const preserved = editor.open && editor.querySelector('textarea').value === '生成期间填写的反馈草稿' &&
            !editor.querySelector('button[type=submit]').disabled;
        const stayedUp = window.scrollY < 80 && !document.querySelector('#jump-to-latest').hidden;
        document.querySelector('#close-editor').click();
        document.querySelector('#jump-to-latest').click();
        const page = document.scrollingElement;
        const returned = page.scrollHeight - page.clientHeight - page.scrollTop <= 80;
        document.querySelector('#timeline').style.minHeight = '';
        return preserved && stayedUp && returned;
    })()`)) throw new Error("Chat completion should preserve and unlock the open feedback draft");
    if (!await evaluate("!!document.querySelector('#timeline table') && !!document.querySelector('#timeline pre code') && !!document.querySelector('#timeline blockquote') && !!document.querySelector('#timeline a[href=\"https://example.org\"]') && !document.querySelector('#timeline img, #timeline a[href^=javascript]')")) throw new Error("Markdown formatting or URL safety failed");
    if (!await evaluate("document.querySelector('.composer').getBoundingClientRect().height <= 64")) throw new Error("Empty composer should occupy one compact row");
    await evaluate("window.scrollTo(0, document.documentElement.scrollHeight)");
    const markdown = await call("Page.captureScreenshot", {format: "png"});
    await writeFile(join(output, "workspace-markdown.png"), Buffer.from(markdown.data, "base64"));
    await clickText("任务与反馈");
    if (!await evaluate("document.querySelector('.task-overview').textContent.includes('口头说明工具区别') && document.querySelector('.task-overview').textContent.includes('JSON 补充反馈')")) throw new Error("Task view should show live tasks and feedback");
    await clickText("从计划提取");
    if (!await evaluate("document.querySelector('#editor .task-extraction-note')?.textContent.includes('对照已有任务和进度')")) throw new Error("Existing plan tasks should trigger a progress review reminder");
    await fill("从当前计划提取尚未登记的可执行任务。");
    await until(() => evaluate("Array.from(document.querySelectorAll('.candidate')).some(card => card.textContent.includes('提取的任务') && card.textContent.includes('整理学习要点') && !card.textContent.includes('复述要点'))"), "task-only review");
    if (!await evaluate("document.querySelector('[data-view=all]').getAttribute('aria-pressed') === 'true' && document.querySelector('#pending-message').hidden")) throw new Error("Pending proposal should be visible in the conversation");
    await evaluate("document.querySelector('[data-inspector-target=inspector-tasks]').click()");
    if (!await evaluate("document.activeElement.id === 'inspector-tasks'")) throw new Error("Inspector navigation should focus the matching section");
    await evaluate("document.querySelector('#chat-input').value = '原始对话的草稿'; document.querySelector('#new-conversation').click()");
    await until(() => evaluate("!document.querySelector('#setup-materials').hidden && document.querySelector('.conversation').hidden && document.querySelector('.conversation-link[aria-current=true]').textContent === '新对话'"), "new conversation onboarding");
    await evaluate("document.querySelector('#setup-directory').value='未保存的目录'");
    await clickText("原始对话");
    await until(() => evaluate("document.querySelector('#setup').hidden && document.querySelector('#chat-input').value === '原始对话的草稿'"), "switch away during setup");
    await clickText("新对话");
    await until(() => evaluate("!document.querySelector('#setup').hidden && document.querySelector('#setup-directory').value === '未保存的目录'"), "restore setup draft for this conversation");
    if (!await evaluate(`(() => {
        document.dispatchEvent(new KeyboardEvent("keydown", {key: "k", ctrlKey: true}));
        return !document.querySelector("#search-dialog").open;
    })()`)) throw new Error("The search shortcut must respect onboarding");
    const setup = await call("Page.captureScreenshot", {format: "png"});
    await writeFile(join(output, "workspace-onboarding.png"), Buffer.from(setup.data, "base64"));
    await evaluate(`document.querySelector('#setup-directory').value = ${JSON.stringify(join(output, "materials"))}; document.querySelector('#setup-materials').requestSubmit()`);
    await until(() => evaluate("!document.querySelector('#setup-plan').hidden && document.querySelector('.conversation').hidden"), "materials unlock only the plan form");
    await evaluate("document.querySelector('#setup-instructions').value='参考资料生成第一份学习计划'; document.querySelector('#setup-plan').requestSubmit()");
    await until(() => evaluate("!document.querySelector('#setup-candidate').hidden && document.querySelector('.conversation').hidden"), "candidate is visible before chat");
    await clickText("采用计划并开始聊天");
    await until(() => evaluate("document.querySelector('#setup').hidden && !document.querySelector('.conversation').hidden && document.querySelector('#chat-title').textContent === '新对话'"), "adoption unlocks chat");
    if (!await evaluate("!document.querySelector('#timeline').textContent.includes('专用工具适合约束') && !document.querySelector('#chat-input').value")) throw new Error("New conversation should not display original messages or drafts");
    await clickText("原始对话");
    await until(() => evaluate("document.querySelector('#chat-input').value === '原始对话的草稿'"), "switch restores original draft");
    await clickText("新对话");
    await until(() => evaluate("document.querySelector('#chat-title').textContent === '新对话'"), "switch to new conversation");
    await call("Page.reload");
    await until(() => evaluate("document.querySelector('#chat-title').textContent === '新对话'"), "new conversation reload");
    await evaluate(`(() => {
        const makeUrl = URL.createObjectURL;
        const click = HTMLAnchorElement.prototype.click;
        URL.createObjectURL = blob => { blob.text().then(text => window.downloadedState = JSON.parse(text)); return makeUrl(blob); };
        HTMLAnchorElement.prototype.click = function () {
            window.downloadedFilename = this.download;
            URL.createObjectURL = makeUrl;
            HTMLAnchorElement.prototype.click = click;
        };
        const row = Array.from(document.querySelectorAll('.conversation-entry')).find(item => item.querySelector('.conversation-link').textContent === '原始对话');
        row.querySelector('.conversation-menu-trigger').click();
        Array.from(row.querySelectorAll('.conversation-menu-actions button'))
            .find(button => button.textContent === '导出学习状态 JSON').click();
    })()`);
    await until(() => evaluate("window.downloadedFilename === 'learning-state.json' && window.downloadedState?.state.plan_markdown.includes('复述要点') && document.querySelector('#chat-title').textContent === '新对话'"), "export another conversation without switching");
    await clickText("原始对话");
    await until(() => evaluate("document.querySelector('#timeline').textContent.includes('专用工具适合约束')"), "original history");
    if (!await evaluate("document.querySelector('#formal-plan').textContent.includes('复述要点')")) throw new Error("Original plan should return with its conversation");
    await evaluate("const row = Array.from(document.querySelectorAll('.conversation-entry')).find(item => item.querySelector('.conversation-link').textContent === '新对话'); row.querySelector('.conversation-menu-trigger').click(); row.querySelector('.conversation-menu-actions button:last-child').click();");
    await until(() => evaluate("document.querySelector('#editor-title').textContent === '删除对话' && document.querySelector('#editor').open"), "delete selected conversation confirmation");
    await evaluate("document.querySelector('#editor form').requestSubmit()");
    await until(() => evaluate("!document.querySelector('#editor').open && document.querySelectorAll('.conversation-entry').length === 1"), "delete selected conversation");
    await clickText("本地资料");
    await until(() => evaluate("!!document.querySelector('.material-entry button')"), "material catalog after switching");
    await evaluate("document.querySelector('.material-entry button').click()");
    await until(() => evaluate("document.querySelector('#editor-title').textContent === '移除资料文件'"), "file removal confirmation");
    await evaluate("document.querySelector('#editor form').requestSubmit()");
    await until(() => evaluate("!document.querySelector('#editor').open && !document.querySelector('#setup-materials').hidden"), "file removal opens the material guide for this conversation");
    await writeFile(join(output, "materials", "replacement.md"), "后续学习资料。", "utf8");
    await bind(join(output, "materials"));
    await until(() => evaluate("document.querySelector('#setup').hidden && document.querySelector('#formal-plan').textContent.includes('复述要点')"), "rebinding preserves the existing plan");
    async function pointerClick(selector) {
        const {x, y} = await evaluate(`(() => {
            const node = document.querySelector(${JSON.stringify(selector)});
            node.scrollIntoView({block: 'nearest'});
            const box = node.getBoundingClientRect();
            return {x: box.left + box.width / 2, y: box.top + box.height / 2};
        })()`);
        for (const type of ["mousePressed", "mouseReleased"]) {
            await call("Input.dispatchMouseEvent", {type, x, y, button: "left", clickCount: 1});
        }
    }
    const menuOpen = () => evaluate("!!document.querySelector('.conversation-menu-actions:popover-open')");
    await pointerClick(".conversation-menu-trigger");
    if (!await menuOpen()) throw new Error("Pointer should open the conversation menu");
    await pointerClick(".conversation-menu-trigger");
    if (await menuOpen()) throw new Error("Second pointer click should close the conversation menu");
    await pointerClick(".conversation-menu-trigger");
    await pointerClick("#chat-title");
    if (await menuOpen()) throw new Error("Outside click should close the conversation menu");
    await pointerClick(".conversation-menu-trigger");
    await call("Input.dispatchKeyEvent", {type: "keyDown", key: "Escape", code: "Escape", windowsVirtualKeyCode: 27});
    await call("Input.dispatchKeyEvent", {type: "keyUp", key: "Escape", code: "Escape", windowsVirtualKeyCode: 27});
    if (await menuOpen()) throw new Error("Escape should close the conversation menu");
    if (!await evaluate(`(() => {
        const row = document.querySelector('.conversation-entry');
        const height = row.getBoundingClientRect().height;
        row.querySelector('.conversation-menu-trigger').click();
        const opened = row.querySelector(':popover-open') !== null;
        const sameHeight = row.getBoundingClientRect().height === height;
        row.querySelector('.conversation-menu-actions button').click();
        return opened && sameHeight;
    })()`)) throw new Error("Opening the conversation menu should not change row height");
    await fill("工具学习笔记");
    await until(() => evaluate("document.querySelector('#chat-title').textContent === '工具学习笔记'"), "renamed conversation");
    await evaluate(`(() => {
        const row = document.querySelector('.conversation-entry');
        row.querySelector('.conversation-menu-trigger').click();
        Array.from(row.querySelectorAll('.conversation-menu-actions button'))
            .find(button => button.textContent === '清空聊天历史').click();
        document.querySelector('#editor form').requestSubmit();
    })()`);
    await until(() => evaluate("!document.querySelector('#editor').open && !document.querySelector('#timeline').textContent.includes('专用工具适合约束') && document.querySelector('#formal-plan').textContent.includes('复述要点')"), "cleared chat with retained plan");
    if (!await evaluate(`(() => {
        const panel = document.querySelector('.inspector-scroll');
        const header = document.querySelector('.inspector-fixed');
        const tasks = document.querySelector('#inspector-tasks');
        const top = header.getBoundingClientRect().top;
        const pageY = window.scrollY;
        tasks.style.minHeight = '160vh';
        const jump = id => {
            document.querySelector('[data-inspector-target=' + id + ']').click();
            return Math.abs(document.getElementById(id).getBoundingClientRect().top -
                header.getBoundingClientRect().bottom) < 2;
        };
        const aligned = jump('inspector-plan') && jump('inspector-tasks');
        const goalScrolledAway = document.querySelector('#inspector-status').getBoundingClientRect().bottom <=
            panel.getBoundingClientRect().top;
        const fixed = panel.scrollTop > 0 && header.getBoundingClientRect().top === top &&
            Math.abs(header.getBoundingClientRect().bottom - panel.getBoundingClientRect().top) < 2 &&
            window.scrollY === pageY && getComputedStyle(panel).overflowY === 'auto';
        const returned = jump('inspector-status');
        tasks.style.minHeight = '';
        return aligned && goalScrolledAway && fixed && returned;
    })()`)) throw new Error("Inspector navigation should align all three sections within its own scroll area");
    await evaluate("window.scrollTo(0, 0)");
    const desktop = await call("Page.captureScreenshot", {format: "png"});
    await writeFile(join(output, "workspace-desktop.png"), Buffer.from(desktop.data, "base64"));
    await call("Emulation.setDeviceMetricsOverride", {width: 390, height: 844, deviceScaleFactor: 1, mobile: true});
    if (!await evaluate("document.documentElement.scrollWidth <= innerWidth && getComputedStyle(document.querySelector('.inspector')).position === 'sticky'")) throw new Error("Narrow layout should fit and retain sticky state");
    const mobile = await call("Page.captureScreenshot", {format: "png"});
    await writeFile(join(output, "workspace-mobile.png"), Buffer.from(mobile.data, "base64"));
    if (!await evaluate("getComputedStyle(document.querySelector('#inspector-details')).display === 'none'")) throw new Error("Mobile details should start collapsed");
    await clickText("计划与任务");
    if (!await evaluate("getComputedStyle(document.querySelector('#inspector-details')).display !== 'none'")) throw new Error("Mobile status panel should expand");
    await clickText("计划与任务");
    if (!await evaluate("getComputedStyle(document.querySelector('#inspector-details')).display === 'none'")) throw new Error("Mobile status panel should collapse on the second click");
    if (errors.length) throw new Error(errors.join("; "));
    console.log(JSON.stringify({browser: "Edge headless", state: "passed", checks: ["sidebar opens main form",
        "feedback refresh", "clarification retains input", "candidate reject and resubmit", "adoption survives reload",
        "bind local directory and list files", "modal close", "streamed chat and draft", "Markdown rendering and URL safety",
        "sidebar plan query", "JSON protected fields and candidate", "search and clear filter", "task JSON update",
        "task overview", "colored plan diff and task-only review", "isolated chat history and inspector",
        "Markdown export", "conversation and file removal", "compact composer and narrow layout"], errors}));
    await Promise.race([call("Browser.close").catch(() => {}), pause(500)]);
} finally {
    socket?.close();
    browser.kill();
    await log.close();
}
