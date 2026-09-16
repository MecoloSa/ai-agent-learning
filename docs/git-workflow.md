# Git 与 GitHub 分支工作流操作手册

本文档适用于本项目当前的单人开发阶段，也为以后通过 Pull Request（PR）协作预留了流程。

## 1. 推荐的分支模型

建议直接把现有的 `main` 当作生产分支，不再额外创建含义重复的 `production`：

| 分支 | 用途 | 是否直接开发 | 生命周期 |
| --- | --- | --- | --- |
| `main` | 可发布、可部署的生产版本 | 不建议 | 长期 |
| `develop` | 已完成能力的集成与测试 | 不建议直接写大功能 | 长期 |
| `feature/<名称>` | 单个功能或实验 | 是 | 合并后删除 |
| `fix/<名称>` | 普通缺陷修复 | 是 | 合并后删除 |
| `hotfix/<名称>` | 生产分支紧急修复 | 是 | 合并后删除 |

推荐流向：

```text
feature/* ──PR──> develop ──验证/PR──> main（生产）
                    ↑                    │
                    └──── hotfix 回合并 ─┘
```

为什么不推荐同时保留 `main` 和 `production`：两者如果都代表稳定版本，就容易出现“哪个才是当前生产代码”的歧义。只有部署平台明确要求 `production`，或者团队有独立发布审批阶段时，才增加该分支，参见第 12 节。

## 2. 先理解四个位置

```text
工作区 ──git add──> 暂存区 ──git commit──> 本地仓库 ──git push──> GitHub
```

- 工作区：正在编辑的真实文件。
- 暂存区：下一次提交准备包含的内容。
- 本地仓库：已经形成提交但不一定上传的历史。
- GitHub：名为 `origin` 的远程仓库。

任何操作前先运行：

```bash
git status
git branch -vv
```

它们分别回答“现在有哪些改动”和“我在哪个分支、它跟踪哪个远程分支”。

## 3. 首次创建 `develop`

只需要执行一次：

```bash
git switch main
git pull --ff-only origin main
git switch -c develop
git push -u origin develop
```

解释：

- `git switch main`：切换到本地生产分支。
- `git pull --ff-only origin main`：只接受快进更新，避免一次普通拉取意外生成合并提交。
- `git switch -c develop`：从当前最新的 `main` 创建并切换到 `develop`。
- `git push -u origin develop`：首次上传，并建立 `develop` 与 `origin/develop` 的跟踪关系。

验证：

```bash
git branch -vv
git branch -r
```

应能看到本地 `main`、`develop`，以及远程 `origin/main`、`origin/develop`。

## 4. 开发一个新功能（推荐日常流程）

假设要建立 RAG 自动评测基线：

### 4.1 从最新 `develop` 创建功能分支

```bash
git switch develop
git pull --ff-only origin develop
git switch -c feature/rag-eval-baseline
```

分支名建议使用小写英文和连字符，例如：

```text
feature/long-term-memory
feature/langgraph-orchestration
fix/rag-empty-result
hotfix/startup-crash
```

### 4.2 开发并提交

```bash
git status
git diff
git add <本次相关的文件>
git diff --cached
git commit -m "feat: add RAG evaluation baseline"
```

优先明确列出文件，而不是习惯性执行 `git add .`。这样能降低把日志、密钥或无关实验文件一起提交的风险。

如果确认当前全部改动都属于同一件事，也可以使用：

```bash
git add .
```

提交信息建议：

| 前缀 | 用途 | 示例 |
| --- | --- | --- |
| `feat` | 新功能 | `feat: add evaluation dataset loader` |
| `fix` | 修复缺陷 | `fix: handle empty retrieval results` |
| `test` | 测试 | `test: cover tool retry limit` |
| `docs` | 文档 | `docs: add Git workflow guide` |
| `refactor` | 不改变外部行为的重构 | `refactor: extract retrieval policy` |
| `chore` | 构建、依赖、杂务 | `chore: update uv lockfile` |

### 4.3 首次上传功能分支

```bash
git push -u origin feature/rag-eval-baseline
```

以后继续提交后只需：

```bash
git push
```

## 5. 将功能分支合并到 `develop`

推荐在 GitHub 创建 PR：

```text
base: develop  <-  compare: feature/rag-eval-baseline
```

PR 合并前至少检查：

- 自动测试是否通过。
- `.env`、日志和本地数据是否未提交。
- README 或实验记录是否与行为变化同步。
- 关键 Agent 行为是否有可观察日志或评测结果。

PR 在 GitHub 合并后，同步本地 `develop`：

```bash
git switch develop
git pull --ff-only origin develop
```

如果暂时不用 PR，也可以本地合并：

```bash
git switch develop
git pull --ff-only origin develop
git merge --no-ff feature/rag-eval-baseline
git push origin develop
```

`--no-ff` 会保留一次明确的功能合并记录，便于学习和回顾分支边界。单人、小提交较多时也可以使用 GitHub 的 Squash and merge，让一个功能在 `develop` 上只留下一个提交。

## 6. 将 `develop` 发布到 `main`

先确保 `develop` 已完成测试，再在 GitHub 创建 PR：

```text
base: main  <-  compare: develop
```

合并后更新本地分支：

```bash
git switch main
git pull --ff-only origin main
```

可选：为可复现版本创建标签：

```bash
git tag -a v0.1.0 -m "release: v0.1.0"
git push origin v0.1.0
```

标签适合对应实验报告、演示版本或简历中描述的里程碑。

## 7. 切换分支与“转移改动”

### 7.1 普通切换

```bash
git switch develop
git switch feature/rag-eval-baseline
git switch -
```

`git switch -` 返回上一个分支。

### 7.2 改动写在了错误分支，但还没有提交

先临时保存，包括未跟踪文件：

```bash
git stash push -u -m "move work to feature branch"
git switch develop
git pull --ff-only origin develop
git switch -c feature/correct-branch
git stash pop
```

然后运行 `git status` 和测试，确认改动恢复正确。

### 7.3 提交写在了错误分支

先找到提交哈希：

```bash
git log --oneline --decorate -5
```

切换到目标分支并复制该提交：

```bash
git switch <目标分支>
git cherry-pick <提交哈希>
```

`cherry-pick` 会在目标分支创建内容相同的新提交。确认新分支无误后，再处理错误分支上的原提交；如果该提交已经推送，优先使用 `git revert`，不要随意重写远程历史。

## 8. 更新本地和远程分支

### 8.1 查看远程最新状态，不修改工作区

```bash
git fetch --prune origin
git branch -vv
git branch -r
```

- `fetch`：下载远程分支和提交信息，但不自动合并。
- `--prune`：清理已经从远程删除的过期分支引用。

### 8.2 更新当前分支

```bash
git pull --ff-only
```

如果提示无法快进，说明本地和远程都有新提交。先运行：

```bash
git status
git log --oneline --graph --decorate --all -15
```

观察分叉原因，不要立刻使用强制推送。

### 8.3 让功能分支吸收最新 `develop`

对初学阶段，推荐合并方式，不重写已有提交：

```bash
git switch develop
git pull --ff-only origin develop
git switch feature/rag-eval-baseline
git merge develop
```

解决冲突并测试后：

```bash
git push
```

熟悉 Git 后，可以在尚未共享的个人分支使用 `rebase` 整理历史，但不要对多人使用的分支随意 rebase。

## 9. 合并冲突处理

出现冲突时先查看：

```bash
git status
```

冲突文件中可能出现：

```text
<<<<<<< HEAD
当前分支内容
=======
待合并分支内容
>>>>>>> develop
```

人工决定最终内容，并删除三类标记。然后：

```bash
git add <已解决的文件>
git status
git commit
```

如果不想继续本次合并：

```bash
git merge --abort
```

如果冲突发生在 `cherry-pick`：

```bash
git cherry-pick --continue
# 或放弃
git cherry-pick --abort
```

冲突解决后必须重新运行相关测试；“Git 已允许提交”不等于业务逻辑正确。

## 10. 撤销操作：按影响范围选择

执行撤销前先运行 `git status`。下列命令会丢弃或改变内容时，要确认目标文件和提交哈希。

### 10.1 取消暂存，保留文件改动

```bash
git restore --staged <文件>
```

### 10.2 丢弃尚未暂存的文件改动

```bash
git restore <文件>
```

这会丢失该文件的未提交改动，执行前建议先看 `git diff <文件>`。

### 10.3 修改最近一次尚未推送的提交

```bash
git add <遗漏的文件>
git commit --amend
```

### 10.4 撤销已经推送的提交

```bash
git revert <提交哈希>
git push
```

`revert` 会新增一个反向提交，保留可审计历史，适合共享分支。不要在 `main`、`develop` 上随意使用 `git reset --hard` 或 `git push --force`。

## 11. 删除和重命名分支

功能合并后删除本地分支：

```bash
git switch develop
git branch -d feature/rag-eval-baseline
```

删除远程分支：

```bash
git push origin --delete feature/rag-eval-baseline
git fetch --prune origin
```

`-d` 会在分支尚未合并时拒绝删除，是安全保护。不要因为拒绝就立即改用 `-D`，先确认是否仍有需要保留的提交。

重命名当前分支：

```bash
git branch -m <新名称>
```

如果旧名称已经推送，还需要推送新名称、建立跟踪关系，再确认后删除旧远程分支。

## 12. 如果确实需要独立的 `production`

优先方案仍是让 `main` 直接代表生产。如果部署工具明确只监听 `production`，可以从最新 `main` 创建：

```bash
git switch main
git pull --ff-only origin main
git switch -c production
git push -u origin production
```

此后固定流向：

```text
feature/* -> develop -> main -> production
```

`production` 只接收经过审批的 `main` 发布提交，不直接开发。额外分支会增加同步成本，因此需要在 README 或部署配置中明确：

- `main` 是候选发布版本还是正式稳定版本。
- `production` 由谁、在什么检查通过后更新。
- 回滚通过标签、`revert` 还是部署平台完成。

## 13. 生产紧急修复（hotfix）

从生产代码 `main` 创建：

```bash
git switch main
git pull --ff-only origin main
git switch -c hotfix/<问题名称>
```

完成修复、测试、提交并推送：

```bash
git add <相关文件>
git commit -m "fix: describe production issue"
git push -u origin hotfix/<问题名称>
```

先通过 PR 合并到 `main`，发布后再把同一个修复合并回 `develop`，避免下一次发布重新引入旧问题。

## 14. GitHub 侧建议

为 `main` 设置分支保护或 Ruleset：

- 禁止直接强制推送。
- 通过 PR 合并。
- 合并前要求测试通过。
- 将 `main` 保持为默认分支。

单人学习阶段可以允许自己合并 PR，但仍保留 PR 描述、测试结果和关键评测指标，作为后续复盘和面试展示的工程证据。

## 15. 本项目网络代理说明

当前仓库曾使用本地代理：

```bash
git config --local http.proxy http://127.0.0.1:7890
```

查看配置：

```bash
git config --local --get http.proxy
```

代理软件关闭后，如果 GitHub 可以直接访问，可删除当前仓库的代理配置：

```bash
git config --local --unset http.proxy
```

代理设置保存在 `.git/config`，不会提交到 GitHub。

## 16. 常用检查清单

开始开发：

```bash
git switch develop
git pull --ff-only origin develop
git switch -c feature/<功能名称>
```

准备提交：

```bash
git status
git diff
git add <相关文件>
git diff --cached
git commit -m "类型: 清楚描述改动"
```

准备上传：

```bash
git status
git push -u origin <首次上传的新分支>
# 已建立跟踪关系后只需 git push
```

查看全局历史：

```bash
git log --oneline --graph --decorate --all -20
```

遇到问题时先保存以下输出，再决定操作：

```bash
git status
git branch -vv
git remote -v
git log --oneline --graph --decorate --all -15
```

不要在不理解影响时直接使用 `reset --hard`、`clean -fd` 或 `push --force`。
