# 空最终响应：重复实验与返回层定位（2026-11-03）

## 结论和边界

本次24次HTTP请求复现3次同型空响应。复现时HTTP原始JSON已经content=""、无tool_calls、completion_tokens=0、reasoning_tokens=0、finish_reason=stop；SDK和LangChain没有丢弃非空正文。直接HTTP发送相同JSON、绕过两层SDK调用/响应解析，也复现。

可以定位到：本次复现的空输出在到达项目响应解析之前已由上游接口返回。不能进一步断言服务端是模型直接结束、推理调度、输出后处理或兼容层的哪一个内部机制；这些需要提供方日志。历史最初失败没有HTTP原件，不能假称已经追溯其内部原因。

## 输入与条件

- 请求A：复用discovery-limit-online-20261103旧版第2链的最后真实请求，目标资料已成功取得；恢复消息及工具schema参数核对一致，输入1413token。原模型qwen3.7-plus，temperature=0.2。
- 请求B：复用receipt-ab-online-20261103的failure_then_success/compact/2失败请求；同样是取得资料后的回答阶段，原实验未绑定工具，输入225token。
- 均连接当前阿里云域名下的兼容接口，端点仅保存SHA256，不公开凭据。HTTP参数包括model/messages/stream/temperature，以及A的tools；未传stop、max_tokens或max_completion_tokens。
- 非流式；SDK max_retries=0；不经过生产网络重试包装、不执行工具、不添加提示词、不增加语义重试。每组各请求JSON参数一致。
- langchain-openai=1.4.1、langchain-core=1.5.3、openai=2.53.0、httpx=0.28.1。

## 逐层观测方法

HTTPX request hook保存实际请求JSON（不含认证头）；response hook在SDK解析前读取JSON，保存公开回答、结束状态、usage、请求ID；LangChain _create_chat_result边界观察SDK对象；invoke返回后记录AIMessage。
推理正文不存储，仅记录字段是否存在和长度，避免将私有思维链写入实验产物。直接HTTP组保留相同网络策略与请求JSON，使用HTTPX post跳过SDK的调用和响应解析。
每个模型请求独立观察；同次HTTP/SDK/LC互相比，而非拿不同响应比内容。

## 实验矩阵

| 组 | 输入 | 调用路径 | temperature | 实际次数 | 空响应 |
|---|---|---|---:|---:|---:|
| A原条件 | 已绑定工具的历史失败请求 | LangChain→OpenAI SDK→HTTP |0.2|16|0|
| A温度对照 | 与A相同，仅temperature改0 | 同上 |0|3|1（第3次）|
| B原条件 | 未绑定工具的另一历史失败请求 | 同上 |0.2|2|1（第2次）|
| B直接HTTP | 与上一组请求JSON一致 | HTTPX直接POST |0.2|3|1（第3次）|

全部HTTP200。A/B每组在首次捕获空响应后停止，A原条件到预设16次停止。次数不是固定样本量的成功率实验，不能把3/24当服务失败率或比较温度策略。
有SDK的21次请求中，HTTP/SDK/LC正文没有观测到不一致。直接HTTP的3次不具备SDK层，不能把它们算成SDK一致性样本。

空响应公开示例：

```json
{
  "choices": [{"message": {"role": "assistant", "content": ""}, "finish_reason": "stop"}],
  "usage": {"prompt_tokens": 225, "completion_tokens": 0, "total_tokens": 225}
}
```

同组相同请求前次能够正常回答，不能解释为该输入必然非法。B两次空响应cached_tokens=0，因此不能把提示缓存视为发生空输出的必要条件；不能由此排除其他服务端缓存问题。

## 已缩小的原因范围

- 不是本次复现中LangChain/OpenAI SDK将非空正文解析丢失：HTTP原件已空，直接HTTP亦复现。
- 不是只有绑定工具才发生：请求B未发送tools。
- 不是只有temperature=0.2才发生：temperature=0亦复现；不意味着温度与概率绝对无关。
- 不是本次请求显式stop或零输出额度设置：实际请求没有这些字段；也没有超工具预算或重跑工具。
- 不是答案藏在reasoning_content：三个捕获空响应该字段长度0，推理token也为0。
- 没有HTTP错误或断线证据：状态200、JSON完整；这不代表上游任务语义成功。

尚不能确认：提供方内部模型解码、路由/调度、输出后处理或兼容层中的具体环节。仅凭finish_reason=stop不能判定是哪个内部原因。

## 提供方排查用请求ID（未发送外部消息）

| 样本 | x-request-id |
|---|---|
| A温度0空响应 |6e5619a5-dca2-470c-ac30-d43b9b3f73ef|
| B SDK空响应 |35fb4f03-3eb2-408f-9b6f-11715df66495|
| B直接HTTP空响应 |b19931ae-ae27-40ff-8132-201686c277a7|

对应完整公开证据见`.evaluation-results/empty-diagnosis-layer-report-20261103.json`；各组保存input.json、manifest.json、results.jsonl，未记录密钥/Authorization/私有推理正文。

## 官方说明与代码定位

- [阿里云OpenAI兼容Chat接口](https://help.aliyun.com/en/model-studio/qwen-api-via-openai-chat-completions)：stop表示自然结束或指定停止条件；不能据此推导有有效回答。长度限制通常用length区分；仍以实际参数和响应为准。
- 本地实际安装的BaseChatOpenAI._generate先解析SDK对象再调用_create_chat_result；本次在该边界捕获SDK对象，与HTTP及AIMessage对照。
- 辅助入口`tests/helpers/empty_response_diagnosis.py`，汇总`empty_response_report.py`；没有修改生产调用、提示词、模型配置或恢复策略。
- 新增3项离线测试通过：捕获不泄露推理正文、None推理长度为0、上游空响应的本地解析保留为空。没有重复全量业务测试。

## 后续决策

本地继续无假设刷请求不会暴露提供方内部日志。若要确认服务内部根因，需要用上述request_id由提供方定位；本轮未自动联系任何人。
最多重发一次仍只是可选容错，不是根因修复；应先讨论额外成本、真实请求预算和错误记录，不能把本次定位直接当作实施授权。更改模型、流式、关闭思考等都未证明能根治，不据此修改默认配置。
