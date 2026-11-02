# 两个独立进程的计划恢复回归

测试入口：tests/test_feedback_plan_process_resume.py。只检查程序接线，不评价模型生成能力。

## 初始条件

每组独立复制adoption_state.json：目标为理解正常结束与保护性停止的区别；
旧计划60分钟，修订号1；阅读已完成但仍不理解，口头解释进行中，绘图已取消；
3项任务、1条用户反馈。关联独立临时资料目录，不接触个人存档。

## 完整输入与替身输出

第一进程发送：

> 请依据已有学习状态，为还没理解的部分调整本次后续安排，总计不超过20分钟。只给完整计划候选供确认，不改变学习目标，不更改或重建已登记任务，不恢复已取消任务，不新增或改写反馈。

模型替身固定返回以下正文，并使用真实NEW_PLAN标记包围：

```text
# 后续学习安排
活动1：对比两类停止的触发条件，5分钟。
活动2：口头解释区别并举例，10分钟。
总计15分钟。
```

自动输入随后为v、y、exit；拒绝对照组仅将y改成n。
第一进程完全退出后，第二个Python进程读取其实际current.json，经真实--resume-state入口恢复，发送：

> 只复述当前安排和已确认记录，不修改状态

替身固定回复“这里只读取当前记录，不产生候选或反馈。”，随后自动exit。
第二进程继承的是磁盘存档，不是第一进程的Python对象。

## 预期与边界

采用组：新计划生效，修订号2，tasks_need_review=true；拒绝组：原计划、修订号1、标记false。
两组：原目标、初始版本、任务、反馈和资料关联保留；恢复后的请求使用生效计划，包含任务状态、
前一轮提问及程序confirmed/cancelled结论；记忆沿用window，问答追加2条，不覆盖来源存档。
不要求替身回答复述正确，因为回答本来就是固定的；不证明真实模型理解、知识正确性或工具证据恢复。

## 运行与可见记录

在项目根目录运行：

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_feedback_plan_process_resume.py -q -p no:cacheprovider --basetemp tmp/feedback-plan-process-manual
```

预期2 passed。临时目录每组的first/second下包含input.json（完整脚本及固定回答）、
request.json（实际user_input和request_messages）、current.json与console.txt。
再次运行相同basetemp会由pytest清理旧临时数据，要保留记录时换新目录名。
