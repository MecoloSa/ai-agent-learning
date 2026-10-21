# 当前项目的develop/product工作流

检查时（2026-10-16）本地当前分支是 `main`，GitHub远端仅有 `main`；没有 `master`、`develop` 或 `product`。远端为 `https://github.com/MecoloSa/ai-agent-learning.git`。已有大量未提交的开发改动，首次先把它们提交到develop，不在main先提交，也不在脏工作区先pull。

| 分支 | 当前约定 |
|---|---|
| main | 保留既有分支，本轮不删除、不重命名 |
| develop | 日常实验、开发、测试与提交 |
| product | 通过验证的演示/发布版本，从develop更新 |

## 1. 从当前main创建develop

在PowerShell中逐条执行；某一步报错先停下，不继续后面的推送：

```powershell
git status
git branch -a
git switch -c develop
```

新分支从当前提交创建；目前的工作区与暂存区改动会继续保留，尚未形成新提交。这里分支名称按用户要求使用 `develop` 和 `product`。如果操作时同名分支已存在，不重复 `-c`，先检查来源和状态再切换。

## 2. 整理并提交当前项目

已确认4份旧生成计划被Git跟踪。忽略规则不会自动停止跟踪，先执行：

```powershell
git rm --cached -r -- plans
git add .gitignore .python-version pyproject.toml uv.lock src tests docs
git status
git diff --cached --stat
git diff --cached
git ls-files --cached --ignored --exclude-standard
```

`--cached` 只从Git暂存快照中移除plans，保留本机文件；提交后它们从当前版本移除，但旧提交里的计划仍存在。最后一个检查应不再列出需要忽略的已跟踪文件。`.env.example` 是公开模板，应保留；真正 `.env`、教材、原始轨迹和向量库应不在暂存列表中。

检查和测试通过后：

```powershell
.venv/Scripts/python.exe -X utf8 -m pytest -q
git commit -m "feat: publish learning agent baseline with evaluation"
git branch product develop
git branch -vv
```

`product` 在提交之后创建，与这次develop提交指向同一版本。`git branch product develop` 只创建分支，不切换；当前仍在develop。测试或人工检查发现问题时先修复，不提交/推送未确认的内容。

## 3. 首次推送两个分支

已有origin，不需要重新 `git remote add`：

```powershell
git push -u origin develop
git push -u origin product
git branch -vv
git branch -r
```

`-u` 建立每个本地分支与同名远端分支的跟踪关系，后续在develop里执行 `git push` 会上传到origin/develop。GitHub应出现main、develop、product三个分支；本地当前仍为develop。若远端已有新提交导致推送拒绝，先fetch并检查差异，不使用force。

## 4. 日常实验默认在develop

每次开始编辑之前，确保工作区没有尚未处理的其他分支改动：

```powershell
git switch develop
git pull --ff-only origin develop
git branch --show-current
```

最后一行应为 `develop`。分支切换状态会保留，不需要把 `init.defaultBranch` 当作自动切换开关。可设置本仓库默认推送规则：

```powershell
git config --local push.default simple
```

修改后先查看差异、运行相关测试，再只暂存本次相关文件。例如修改了RAG与对应测试：

```powershell
git status
git diff
.venv/Scripts/python.exe -X utf8 -m pytest tests/test_rag.py -q
git add src/ai_agent/retrieval/rag.py tests/test_rag.py
git diff --cached
git commit -m "experiment: compare retrieval behavior"
git push
```

`git add` 中的文件和测试命令按实际改动调整；新实验代码与可公开报告可以提交，原始模型输出和个人资料继续保留本地。未提交的编辑不专属于某个分支，可能随切换带到其他分支；切换到product前先完成提交或妥善保存，不能只凭分支名判断改动归属。

## 5. 验证后发布到product

develop完成测试、工作区干净后：

```powershell
git switch product
git pull --ff-only origin product
git merge --ff-only develop
git push origin product
git switch develop
```

遵守“不直接在product写代码”，上述更新通常可以快进。如果提示不能快进，先检查分叉原因，不强推。协作或启用分支保护后可改用PR：base为product，compare为develop。操作最后切回develop，继续实验。

## 6. GitHub默认分支是另一件事

本机停留在develop，只影响本机工作目录；不会改变GitHub网页默认展示或别人clone默认检出的分支。需要新clone默认develop时，可在GitHub仓库的默认分支设置中选择develop；若希望访客首先看到稳定演示版本，则将网页默认分支设为product，同时本地仍在develop开发。不要为此删除main。

官方参考：[gitignore](https://git-scm.com/docs/gitignore)、[git switch](https://git-scm.com/docs/git-switch)、[git push](https://git-scm.com/docs/git-push)。此次助手只整理忽略规则/公开模板/文档，没有执行这里的创建、暂存、提交和推送命令。
