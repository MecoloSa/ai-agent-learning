# 反馈计划离线回放与人工评审辅助

本阶段验证“固定回复 → 真实交互控制流 → 观察 → 独立人工标注 → 标注汇总”的接线。
没有新增在线模型调用，不衡量真实模型的候选生成质量。3个案例均为开发案例，不是留出集。

## 已验证范围

2026-10-24实际运行：3个案例全部completed，3项control为true，3项quality和1项evidence待评审。
关联八文件136项测试通过；其中7项覆盖观察，31项覆盖评审导出/校验/汇总。
31项测试里的标签是测试资料，不作为实验人工标签发布。

固定坏候选在第二例中仍为60分钟，违反30分钟上限；用户拒绝后存档保留原计划。
因此控制通过不意味着候选质量通过，也不能拿current.json中的旧计划评价被拒绝的候选。

## 从项目根目录运行

以下命令适用于Windows PowerShell。输出文件已存在时换用新版本名，禁止覆盖已有标注。

```powershell
Set-Location F:\LLM\projects\agent

.\.venv\Scripts\python.exe -m ai_agent.evaluation.feedback_plan_runner `
    --cases tests/fixtures/evaluation/feedback_plan_cases.jsonl `
    --responses tests/fixtures/evaluation/feedback_plan/offline_responses.json `
    --output .evaluation-results/feedback-plan-offline-v3.jsonl

.\.venv\Scripts\python.exe -m ai_agent.evaluation.feedback_plan_review prepare `
    --results .evaluation-results/feedback-plan-offline-v3.jsonl `
    --output .evaluation-results/feedback-plan-manual-v1.jsonl

.\.venv\Scripts\python.exe -m ai_agent.evaluation.feedback_plan_review report `
    --results .evaluation-results/feedback-plan-offline-v3.jsonl `
    --reviews .evaluation-results/feedback-plan-manual-v1.jsonl `
    --output .evaluation-results/feedback-plan-manual-pending-v1.json
```

初始报告预期：runs=3，annotation_status=pending，quality.pending=3，evidence.pending=1，
model_quality_evaluated=false。未评审不计入unknown，也不当成false或true。
旧版v2没有完整观察，prepare会拒绝，不从固定回复或最终存档补猜经过。

读取JSONL应逐行解析，并显式指定UTF8：

```powershell
$rows = @(
    Get-Content .evaluation-results/feedback-plan-offline-v3.jsonl -Encoding UTF8 |
        ForEach-Object { $_ | ConvertFrom-Json }
)
$case = $rows | Where-Object case_id -eq 'feedback-plan-dev-002'
$case.turns | Format-List
$case.plan_changes | Format-List
```

预期第二例proposed_plan包含60分钟，decision=cancelled，confirmation_reason=user_declined。
proposed_plan只是送入校验入口的候选；raised表示抛出异常，不能声称已校验或已展示。

## 标注约定

保留原始结果；复制导出的评审表作为自己的工作副本，以UTF-8保存。
每行对应一次运行，只允许修改reviewer和checks中status、value、reason、evidence_refs四个字段。
不要更改案例判据、执行观察、顺序、来源哈希；report会与原结果重新比对。

- pending：尚未评审，value必须为null，reason为空，evidence_refs为空列表。
- reviewed且value为true/false：填写实际评审人、具体理由，以及至少一个指向观察的引用。
- reviewed且value为null：已经查看但无法判断，需要填写评审人和无法判断的原因，不强制引用。
- 未完成回放：允许导出排查，但本入口不接受其内容结论。

例如，人工核对第二例后，其检查项可填为以下内容。此处仅是填写示例，不是程序生成或已提交的实验标签：

```json
{
  "check_id": "candidate_time_alignment",
  "category": "quality",
  "criterion": "候选的活动分项与说明一致，整体不超过30分钟；缺少必要估时则无法确认。",
  "basis": "用户明确要求总计不超过30分钟，不是每任务30分钟。",
  "status": "reviewed",
  "value": false,
  "reason": "用户要求总计不超过30分钟，候选安排阅读30分钟、口述30分钟，共60分钟。",
  "evidence_refs": ["/turns/0/user_input", "/plan_changes/0/proposed_plan"]
}
```

同时填写该行顶层reviewer。引用使用JSON Pointer：0表示列表第一项；只可指向initial_context、turns、plan_changes。
引用可定位不代表它足以支持结论；校验器不判断理由是否正确，不能拿其通过作为语义通过。

填写后重新report，使用新的output名。报告区分pending/true/false/unknown，只汇总录入标签，
不改变原始结果中的assessment，也不把固定回复成绩当成模型成绩。
即使annotation_status=complete，也只表示每项已填写，不表示内容质量通过。

source_sha256绑定原结果字节，连空行变化也会触发不一致；它不是签名或真实性认证。
原结果变化时使用匹配的旧文件，或重新导出并重新审阅，不手改哈希绕过校验。

## 后续核心决策边界

当前source_evidence_status=not_collected：资料目录路径不是工具读取证据，也没有冻结原文内容。
因此本阶段不能证明回答中的事实或引用受资料支持；判不了时保留pending或明确reviewed/null。

下一阶段需要决定真实交互评测的范围：模型如何接入、如何记录实际工具结果及资料版本、
哪些质量项采用人工基准，以及如何校准自动评审。它涉及主实验方法与效果验证，应先给设计方案。
本次没有改变主流程、模型配置、检索策略、现有评审器或旧实验。

## 回归命令

```powershell
.\.venv\Scripts\python.exe -m pytest `
    tests/test_feedback_plan_cases.py `
    tests/test_feedback_plan_input.py `
    tests/test_feedback_plan_runner.py `
    tests/test_feedback_plan_controls.py `
    tests/test_feedback_plan_observations.py `
    tests/test_feedback_plan_review.py `
    tests/test_feedback_plan_loop.py `
    tests/test_plan_confirmation_preview.py `
    -q -p no:cacheprovider
```

预期136 passed。这是相关范围的离线回归，不声称运行过整个仓库或在线效果实验。
