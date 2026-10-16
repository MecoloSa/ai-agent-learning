# GitHub发布范围检查（2026-10-16）

项目主题是证据驱动的学习Agent，评测也是核心工程能力。因此 `src/ai_agent/evaluation/`、实验代码/说明/汇总、`tests/` 与测试金标准、README、设计和复盘文档、依赖与锁文件应保留。

## 本地保留、Git忽略

- 密钥与本地环境配置；`.env.example` 作为无凭证模板继续公开。
- 虚拟环境、缓存、构建产物、日志、IDE配置、操作系统元数据。
- `.agents_data/` 的向量索引/学习状态、`.evaluation-results/` 的原始运行结果、`tmp/` 的临时验证快照。
- `plans/` 的个人生成计划；未来 `plans/README.md` 可以保留。
- `src/ai_agent/eval/` 的历史程序/本地教材目录与 `架构优化/` 的历史内容，沿用原忽略设置。
- `docs/proposals/` 的未落地实现草稿，它包含维护代码的重复副本；正式实现和设计复盘继续保留。
- 实验目录下原始JSONL、冻结 `inputs*.json`、JSON轨迹、单次 `integration_smoke.json`，以及evaluation下的本地outputs/results目录。

## evaluation敏感性结论

冻结 `inputs.local.json` 约588KB，包含288个教材片段，不能作为普通源码一并上传。原始轨迹含请求ID、证据位置与执行细节，当前请求ID本身不等于API Key，但统一作为本地审计产物处理。单次接入烟测含请求ID，也不公开。汇总指标、失败计数、排序索引诊断和文字实验报告没有资料正文或凭证，可保留作为实习项目证据。

对当前待上传文本检查了常见API Key/GitHub令牌/私钥/URL内嵌凭证，并与 `.env` 中实际密钥逐项对照；扫描只输出路径和风险类别，不输出凭证。发现的本机绝对路径已通过以下方式处理：

- 4组在线案例的本地副本忽略，新增不含本机盘符与用户目录的 `*.example.jsonl`。
- 项目评估文档中的个人桌面目录改为资料文件名，技术内容保留。
- 本地教材、已有本地案例、原始实验数据均未删除。

处理后扫描未命中当前实际密钥、常见凭证模式或个人绝对路径。这不是对所有秘密类型或整个Git历史的绝对安全保证。当前历史文件名检查未发现提交过 `.env` / 私钥文件；4份旧生成计划已经被Git跟踪，新增忽略规则对它们无效，需要在首次develop提交前执行 `git rm --cached -r -- plans`。这只停止当前版本跟踪，磁盘文件仍在，历史提交中的旧内容也仍在。

GitHub后续新增实验时，把可公开题目/标注放入 `tests/fixtures/evaluation/`；原始片段、回答、工具messages和用户反馈写到已忽略目录。提交前仍需检查 `git diff --cached`。不要用 `git add -f` 绕过本次保护。

详细分支操作见 [develop/product工作流](git-workflow-develop-product.md)。本次未创建分支、未修改暂存区、未提交、未推送。
