# 研习台 Java 后端

这一阶段迁移任务查询入口。Spring Boot 接收对话 ID，调用 Python 查询该对话的任务摘要。SQLite 的读写权限由 Python 统一管理。网页日常操作继续使用 Python 服务。

## 本地启动

用 IntelliJ IDEA 打开 `F:\LLM\projects\agent\java-backend\pom.xml`，选择项目 JDK 21 或 25。Maven 按 Java 21 编译。Maven Wrapper 会在首次运行时下载固定版本的 Maven 和项目依赖。

在第一个 PowerShell 窗口启动 Python。示例凭据只用于本地演示，可自行替换为私有随机值。后续查询使用同一个值。

```powershell
Set-Location 'F:\LLM\projects\agent'
$env:YANXI_TASK_API_TOKEN = 'local-dev-task-token-change-me'
.\.venv\Scripts\python.exe -X utf8 -m ai_agent.web_workspace
```

在第二个窗口启动 Java。

```powershell
Set-Location 'F:\LLM\projects\agent\java-backend'
.\mvnw.cmd spring-boot:run
```

在第三个窗口查询原始对话。

```powershell
Invoke-RestMethod 'http://127.0.0.1:8766/api/conversations/original/tasks' `
    -Headers @{'X-Yanxi-Task-Token' = 'local-dev-task-token-change-me'} |
    ConvertTo-Json -Depth 6
```

返回 `conversation`、正式状态 `version` 和 `tasks`。没有任务时 `tasks` 是空数组。查询其他对话时把 `original` 替换成该对话的 32 位 ID。Python 若使用其他端口，在 Java 窗口启动前设置 `YANXI_PYTHON_BASE_URL`。

## 断点路线

1. 在 `TaskController.listTasks` 查看 `conversation`。Spring MVC 从 URL 取出这个值，Controller 核对其格式和请求是否携带凭据。
2. 到 `TaskQueryService.findTasks` 查看 `RestClient` 请求。Java 调用 Python 的 `/api/integration/tasks`，传递同一个对话 ID 和只读凭据。Python 校验凭据、定位对话并返回任务摘要。
3. 在 Service 的 `return result` 查看 `TaskList`。Jackson 将上游 JSON 转成 Java record，Spring MVC 再把返回值转成 HTTP 响应。两端的 JSON 字段沿用 Python 命名。
4. 停掉 Python 后重试。Java 返回 503。缺凭据或凭据错误返回 403，合法但不存在的对话返回 404，ID 格式错误返回 400，上游数据异常返回 502。

## 复验

```powershell
Set-Location 'F:\LLM\projects\agent\java-backend'
.\mvnw.cmd verify
Set-Location 'F:\LLM\projects\agent'
.\.venv\Scripts\python.exe -X utf8 -m tests.helpers.java_task_bridge_smoke
```

跨进程复验使用临时 SQLite，分别创建「阅读工具定义」和「第二个对话的任务」，查询两次并核对隔离，再验证错误凭据、未知 ID 和 Python 停止时的响应。
