# 评测案例与本地资料

此目录保留可公开的金标准、合成边界案例和在线案例模板。教材PDF、用户资料、原始模型回答与调用轨迹不随仓库发布。

`end_to_end_smoke`、`end_to_end_citation_check`、`learning_goal_alignment`、`rag_cases` 的无后缀本地JSONL包含机器专属路径，已忽略。对应 `*.example.jsonl` 保留原题目和标注，使用项目相对资料路径。模板文件不会自动触发在线请求。

第一次使用时，可在项目根目录复制模板，再将本地副本中的 `material_dir` 改成你有权使用的实际资料目录，例如：

```powershell
Copy-Item tests/fixtures/evaluation/end_to_end_smoke.example.jsonl tests/fixtures/evaluation/end_to_end_smoke.jsonl
```

若本地副本已经存在，直接编辑它，不重复复制覆盖。书籍页码金标准只适用于对应书籍版本，资料不同需重新标注，不能把没有教材导致的失败解释成模型退化。原本机案例保留在磁盘，此次未改写其运行路径。
