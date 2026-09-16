# 学习资料分析与计划

生成日期：2026-09-15

目标：深入理解提供资料并提供针对新手的学习计划

## 资料分析智能体

# 📚 《深入理解 AI Agent：设计原理与工程实践》学习指南

> **作者**：李博杰（Pine AI 首席科学家） | **版本**：v1.4 · 2026 年 8 月  
> **配套代码**：https://github.com/bojieli/ai-agent-book

---

## 一、这本书讲什么？

**核心公式**：**Agent = LLM（大脑）+ 上下文（眼睛）+ 工具（手脚）** [source=PDF, page=15, chunk=54]

本书系统讲解如何**设计、构建、评估和持续进化**生产级 AI Agent 系统。作者基于在 Pine AI 打造能自主打电话、处理金钱交易的长程 Agent 的实战经验，提炼出**穿越模型迭代周期**的架构原则 [source=PDF, page=9-10, chunk=41-43]。

---

## 二、全书结构（四个层次，共 10 章）

| 层次 | 章节 | 核心主题 |
|------|------|---------|
| **🏗️ 构建 Agent** | 第1章 Agent 入门 | 核心公式、ReAct 循环、Harness 工程 |
| | 第2章 上下文工程 ⭐ | 提示工程、KV Cache、Skills、上下文压缩 |
| | 第3章 记忆与知识库 | 用户记忆、RAG、GraphRAG、Agentic RAG |
| | 第4章 工具 | MCP、工具安全、异步事件架构 |
| | 第5章 Coding Agent | 代码生成、Agent 自举 |
| **📊 评估与进化** | 第6章 评估 | 环境、数据集、自动化判断 |
| | 第7章 模型后训练 | SFT、强化学习 |
| | 第8章 持续进化 | 学习信号、四种更新、回滚 |
| **🤝 交互与协作** | 第9章 多模态 | 语音 Agent、Computer Use、机器人 |
| | 第10章 多 Agent 协作 | 管理者模式、消息总线、Agent 社会 |

[source=PDF, page=11-12, chunk=45-49]

---

## 三、前置知识自检

### ✅ 必需（全书基础）
- **Python 编程**：基本语法、pip、数据结构
- **LLM 使用经验**：用过 ChatGPT/Claude，理解 Prompt → 回复
- **AI 辅助编程工具**：Claude Code、Cursor、TRAE 等（本身就是 Coding Agent）
- **软件工程常识**：命令行、Git、JSON、REST API

### 📌 推荐（提升特定章节体验）
- 机器学习基础（第7章）、线性代数直觉（第2-3、7章）
- Web 开发基础（第4、9章）、Transformer 基本了解（第2、7章）

> 💡 **好消息**：除第7章外，全书对数学和 ML 要求很低，**完全可以作为起点** [source=PDF, page=13, chunk=52]

---

## 四、🎯 新手学习计划（8 周路线图）

### 📅 第一阶段：建立认知（第 1-2 周）

**Week 1：第 1 章 AI Agent 入门** ⭐ 全书概念地图
- **核心目标**：理解 Agent = LLM + 上下文 + 工具，掌握 ReAct 循环（思考→行动→观察）
- **关键概念**：
  - 大脑/眼睛/手脚 的直觉类比
  - 从"提示工程"到"Loop 工程"的范式演进
  - 工作流 vs 自主 Agent 的编排模式
- **动手实验**：实验 1-1 ★（体验现代 Agent 能力）
- **学习建议**：初读不必记住所有概念，先建立整体印象 [source=PDF, page=15, chunk=54]

**Week 2：第 2 章 上下文工程** ⭐ **全书最关键一章**
- **核心目标**：掌握 Agent 的"眼睛"——上下文管理
- **关键概念**：
  - 上下文 = 消息列表
  - KV Cache 三条核心结论（底层原理可先跳过）
  - 提示工程流程化设计、Agent Skills 按需加载
  - 上下文压缩策略
- **动手实验**：实验 2-1 ★、2-2 ★

> 💡 **时间紧张？** 优先读第1章（全局认知）+ 第2章（最关键） [source=PDF, page=12, chunk=49]

---

### 📅 第二阶段：核心能力构建（第 3-5 周）

**Week 3：第 3 章 记忆与知识库**
- **核心目标**：让 Agent 跨会话记住知识
- **关键概念**：
  - 用户记忆四种渐进式策略
  - RAG 完整技术栈（分块→嵌入→检索→重排序）
  - 混合检索（稠密 + 稀疏）
  - Agentic RAG（让 Agent 自主决定何时检索）
- **动手实验**：实验 3-1 ★（搭建简单 RAG）

**Week 4：第 4 章 工具**
- **核心目标**：掌握 Agent 的"手脚"设计
- **关键概念**：
  - 五类工具：感知、执行、协作、事件触发、用户沟通
  - 工具粒度权衡：整合 vs 分离
  - MCP（Model Context Protocol）标准
  - 工具安全与权限控制
- **动手实验**：实验 4-1 ★（实现简单工具调用）

**Week 5：第 5 章 Coding Agent**
- **核心目标**：理解"代码即思考"与 Agent 自举
- **关键概念**：
  - 代码生成作为元能力
  - Agent 通过生成代码创造新能力
  - 文件系统 + 代码执行架构范式
- **动手实验**：实验 5-1 ★★

---

### 📅 第三阶段：评估与进阶（第 6-7 周）

**Week 6：第 6 章 Agent 的评估** ⭐ **方法论地基**
- **核心目标**：建立"没有评估就没有进步"的科学方法论
- **关键概念**：
  - 评估环境设计
  - 数据集构建
  - 自动化判断（LLM-as-Judge）
  - 区分"真变好"还是"运气"
- **动手实验**：实验 6-1 ★★

**Week 7：第 8 章 Agent 的持续进化**
- **核心目标**：让 Agent 从运行经验中学习
- **关键概念**：
  - 四种更新载体（提示、工具、Skills、参数）
  - 学习信号与回滚机制
  - 经验学习 vs 简单记录的区别
- **动手实验**：实验 8-1 ★★

> 💡 **关注模型训练？** 直接读第7章（后训练），但建议先读第6章（评估是训练前提） [source=PDF, page=12, chunk=49]

---

### 📅 第四阶段：拓展视野（第 8 周，选读）

**Week 8：第 9-10 章（按兴趣选读）**

- **第 9 章 多模态与实时交互**：语音 Agent、Computer Use、机器人 VLA
- **第 10 章 多 Agent 协作**：
  - 上下文共享/独立 × 对等/管理者/去中心化的分类框架
  - 翻译 Agent 案例（Glossary + Translation + Proofreading + Manager）
  - 消息总线（Redis Pub/Sub、RabbitMQ）
  - Agent 社会与 Agent 经济前沿

---

## 五、🛠️ 学习方法建议

### 1. **必须动手跑实验**
> "AI Agent 是一个实践性极强的领域，很多设计上的直觉需要在动手调试的过程中才能真正建立起来。" [source=PDF, page=13, chunk=50]

```bash
git clone https://github.com/bojieli/ai-agent-book.git
cd ai-agent-book
```
实验按 `chapter1/` 至 `chapter10/` 组织，先看 `README.md` 找项目目录。

### 2. **善用 AI 辅助编程工具**
使用 Claude Code、Cursor 等工具做实验——它们本身就是成熟的 Coding Agent，你能** firsthand 体验 ReAct 循环、工具调用、上下文管理** [source=PDF, page=13, chunk=51]

### 3. **理解术语约定**
- **思考**（reasoning）= 模型展开中间推导、"想"的过程
- **推理**（inference）= 模型的前向计算与部署运行

[source=PDF, page=13, chunk=50]

### 4. **按难度递进做实验**
- ★ 入门级：适合所有读者
- ★★ 中等难度：需要一定工程基础
- ★★★ 进阶挑战：开放性问题或复杂系统设计

---

## 六、📖 核心概念速查表

| 概念 | 一句话解释 | 章节 |
|------|-----------|------|
| **ReAct 循环** | 思考→行动→观察 的迭代过程 | 第1章 |
| **Harness 工程** | 模型之外的工程竞争力，解决模型做不到的事 | 第1章 |
| **上下文工程** | 管理 Agent 在每个决策点能看到的全部信息 | 第2章 |
| **KV Cache** | 推理过程中复用历史计算结果的机制 | 第2章 |
| **Skills** | 按需加载的知识文档与脚本，解决提示词膨胀 | 第2章 |
| **RAG** | 先检索相关文档再让模型生成回答 | 第3章 |
| **Agentic RAG** | 让 Agent 自主决定何时检索、检索什么 | 第3章 |
| **MCP** | Model Context Protocol，工具接入标准 | 第4章 |
| **提议者-审核者** | 解决模型过早认为任务完成的方法 | 引言 |
| **管理者模式** | 多 Agent 协作中 Manager 调度 Sub-Agent | 第10章 |

---

## 七、⚠️ 新手常见陷阱

1. **不要跳过第1章**：它是全书的概念地图，后续每章都会展开其中的某个方面
2. **第2章 KV Cache 原理较技术化**：初读可先跳过底层原理，只记住三条核心结论 [source=PDF, page=12, chunk=49]
3. **不要只读不练**：Agent 是实践性极强的领域
4. **不要忽视评估**：没有评估，就没有进步，迭代方向只能依赖直觉
5. **不要等名词流行才实践**：实践在前，命名在后——头部公司早已把问题趟过一遍了 [source=PDF, page=9, chunk=42]

---

## 八、🎓 学习成果检验

完成本计划后，你应该能够：

- [ ] 解释 Agent = LLM + 上下文 + 工具 的核心公式
- [ ] 设计一个包含 ReAct 循环的简单 Agent
- [ ] 实现基础的 RAG 系统
- [ ] 设计合理的工具接口与安全边界
- [ ] 建立 Agent 评估体系
- [ ] 理解多 Agent 协作的架构模式

---

> 💬 **作者寄语**：  
> "Agent 技术仍在快速演进，但好的架构设计原则具有穿越时间的力量。掌握了'为什么要这样设计'，你就能在技术浪潮的变化中保持清醒的判断力。" [source=PDF, page=14, chunk=53]

祝你学习顺利！🚀

## 学习计划智能体

# 📚 《深入理解 AI Agent》3 天新手学习计划

> **适用对象**：有 Python 基础和 LLM 使用经验（用过 ChatGPT/Claude）的初学者
> **配套代码**：`git clone https://github.com/bojieli/ai-agent-book.git` [source=AI-Agents-in-Depth-zh-CN.pdf, page=12, chunk=49]
> **核心公式**：**Agent = LLM（大脑）+ 上下文（眼睛）+ 工具（手脚）** [source=AI-Agents-in-Depth-zh-CN.pdf, page=15, chunk=54]

---

## 📅 Day 1：建立全局认知——Agent 是什么、怎么运转

> **今日目标**：理解 Agent 的核心公式与运行机制，建立"大脑+眼睛+手脚"的直觉框架

---

### 任务 1：掌握 Agent 核心公式与三大组件（约 2 小时）

**学习内容**：第 1 章 §1.1——现代 Agent = LLM + 上下文 + 工具

**具体步骤**：

1. **阅读 §1.1 全文**，重点理解三个类比的广义含义 [source=AI-Agents-in-Depth-zh-CN.pdf, page=15, chunk=55]：
   - **LLM = 大脑**：不只是一组参数，而是整个决策内核——理解意图、思考规划、做出判断
   - **上下文 = 眼睛**：不只是输入文本，而是 Agent 在每个决策点能看到的全部信息——环境信息、用户记忆、领域知识、自身状态和任务进展
   - **工具 = 手脚**：不只是 API 函数，而是 Agent 能做的所有事情的集合

2. **对照三层映射表**，理解同一个对象的三种说法 [source=AI-Agents-in-Depth-zh-CN.pdf, page=15, chunk=55]：

   | 直觉理解 | 实现组件 | 学术概念 |
   |---------|---------|---------|
   | 大脑 | LLM | 策略（Policy） |
   | 眼睛 | 上下文 | 观察空间（Observation Space） |
   | 手脚 | 工具 | 动作空间（Action Space） |

3. **阅读 §1.1.2 工具五类分类** [source=AI-Agents-in-Depth-zh-CN.pdf, page=17, chunk=59]：感知工具、执行工具、协作工具、事件触发工具、用户沟通工具——先建立整体印象，不必记住细节。

4. **动手实验 1-1 ★**（上下文消融实验）：运行配套代码 `chapter1/` 目录下的实验 1-1，逐一移除上下文组件（工具定义、思考过程、历史记录、工具结果），观察 Agent 行为如何退化 [source=AI-Agents-in-Depth-zh-CN.pdf, page=21, chunk=69]。

**可验证产出**：
- ✅ 用自己的话写一段 50 字以内的 Agent 定义（包含三大组件）
- ✅ 完成一张表格，记录消融实验中每种组件缺失后 Agent 的具体表现（参照原文的"完整基线/无工具定义/无思考过程/无历史记录/无工具结果"五行对照 [source=AI-Agents-in-Depth-zh-CN.pdf, page=21, chunk=69]）

---

### 任务 2：理解 ReAct 循环与 Harness 工程（约 2 小时）

**学习内容**：第 1 章 §1.1.5 ReAct 循环 + §1.2 Harness 工程

**具体步骤**：

1. **精读 ReAct 循环**，理解"思考→行动→观察"的三步迭代 [source=AI-Agents-in-Depth-zh-CN.pdf, page=21, chunk=70]：
   - 模型先**思考**当前应该做什么
   - 然后调用工具**行动**
   - 再**观察**工具返回的结果，继续思考下一步
   - 循环直到任务完成

2. **跟踪多币种汇总案例** [source=AI-Agents-in-Depth-zh-CN.pdf, page=22, chunk=71]：理解 Agent 轨迹（trajectory）的结构——用户消息、模型回复（思考过程 + 工具调用）、工具执行结果。关键事实：**Agent 的上下文 = 静态前缀（系统提示词 + 工具定义）+ 轨迹（动态消息历史）** [source=AI-Agents-in-Depth-zh-CN.pdf, page=21, chunk=70]。

3. **阅读 §1.2 Harness 工程**，理解从 Demo 到生产的跨越 [source=AI-Agents-in-Depth-zh-CN.pdf, page=25, chunk=80]：
   - **Agent = Model + Harness**，Harness = 上下文 + 工具 + 约束 + 验证 + 纠正
   - 记住五个功能的闭环：上下文与工具支撑决策，约束预防错误，验证发现偏差，纠正闭合循环 [source=AI-Agents-in-Depth-zh-CN.pdf, page=27, chunk=84]

4. **理解工程范式演进** [source=AI-Agents-in-Depth-zh-CN.pdf, page=26, chunk=82]：提示工程 → 上下文工程 → Harness 工程 → Loop 工程，层层包含而非替代。

5. **（计划设计）绘制概念图**：在纸上画出 Agent 运行全景图——LLM 在中心，ReAct 循环串联上下文和工具，Harness 五功能包裹在外层。

**可验证产出**：
- ✅ 用伪代码或流程图描述 ReAct 循环的三步迭代（参照原文的多币种汇总轨迹 [source=AI-Agents-in-Depth-zh-CN.pdf, page=22, chunk=72]）
- ✅ 列出 Harness 五个功能（上下文、工具、约束、验证、纠正）各一句话职责 [source=AI-Agents-in-Depth-zh-CN.pdf, page=26, chunk=81]

---

## 📅 Day 2：全书最关键一章——上下文工程

> **今日目标**：掌握 Agent 的"眼睛"——上下文管理，理解 KV Cache 三条核心结论、Skills 按需加载和上下文压缩策略
> **作者提示**：如果你时间有限，优先阅读第一章（建立全局认知）和第二章（掌握最关键的上下文工程）[source=AI-Agents-in-Depth-zh-CN.pdf, page=12, chunk=49]

---

### 任务 1：KV Cache 三条核心结论与提示工程（约 2 小时）

**学习内容**：第 2 章 §2.3 KV Cache + §2.4 提示工程

**具体步骤**：

1. **先记住三条核心结论**（底层原理可先跳过）[source=AI-Agents-in-Depth-zh-CN.pdf, page=47, chunk=125]：
   - **结论一**：系统提示词和工具定义一旦确定就不要改。任何改动，哪怕多一个空格，都可能使缓存失效；改动越靠前，延迟和成本影响越大
   - **结论二**：动态信息永远追加到末尾——时间戳、用户状态等变化的内容，作为新消息追加到对话末尾，而不是修改已有的系统提示词
   - **结论三**：使用标准 API 格式，不要自行拼接消息

2. **理解为什么**：阅读"时间戳导致延迟暴涨"的案例——某团队在系统提示词加了 `Current time: {{now}}`，首 token 延迟从 0.5 秒涨到 3-5 秒，月度推理账单翻倍 [source=AI-Agents-in-Depth-zh-CN.pdf, page=47, chunk=126]。

3. **了解常见错误模式**（§2.3.3）[source=AI-Agents-in-Depth-zh-CN.pdf, page=52, chunk=137]：
   - 动态系统提示词（嵌入时间戳）
   - 工具定义的动态排序
   - 滑动窗口对话历史（会丢失关键工具调用结果）
   - 文本格式化（偏离模型训练格式）

4. **动手实验 2-2 ★**（KV Cache 与注意力直觉实验）：通过配套代码直观感受 Key/Value 向量的作用。

**可验证产出**：
- ✅ 能解释"为什么在系统提示词里加时间戳会导致延迟暴涨"（用前缀不变性原理解释）
- ✅ 列出四种常见错误上下文管理模式及其危害 [source=AI-Agents-in-Depth-zh-CN.pdf, page=52, chunk=137]

---

### 任务 2：Agent Skills 按需加载与上下文压缩（约 2 小时）

**学习内容**：第 2 章 §2.5 Agent Skills + §2.7 上下文压缩策略

**具体步骤**：

1. **理解 Skills 的核心思想** [source=AI-Agents-in-Depth-zh-CN.pdf, page=59, chunk=158]：
   - 不是把所有知识一次性塞给 Agent，而是**按需加载**
   - **渐进式披露三层机制** [source=AI-Agents-in-Depth-zh-CN.pdf, page=59, chunk=157]：
     - 第一层：元数据（name + description，启动时加载，约 300 tokens）
     - 第二层：SKILL.md 核心流程（任务触发时按需加载，约 2K tokens）
     - 第三层：子文档（选择性深入，按需加载）
   - 解决两个问题：浪费 token（大部分内容与当前任务无关）和注意力被稀释 [source=AI-Agents-in-Depth-zh-CN.pdf, page=59, chunk=157]

2. **理解上下文压缩的六种策略**（§2.7）[source=AI-Agents-in-Depth-zh-CN.pdf, page=70, chunk=186]：
   - 无压缩（失败）、个体摘要、组合摘要、**上下文感知压缩**（最优：节省 76% token）、感知+引用、自适应窗口
   - 关键洞察：上下文感知压缩将查询意图和已有信息纳入压缩决策 [source=AI-Agents-in-Depth-zh-CN.pdf, page=70, chunk=187]

3. **记住生产级分层压缩的五层机制** [source=AI-Agents-in-Depth-zh-CN.pdf, page=71, chunk=189]：工具结果预算控制 → 噪声直接删除 → API 层微压缩 → 归档式摘要 → 全量压缩（配熔断器）

4. **理解"隔离优于压缩"** [source=AI-Agents-in-Depth-zh-CN.pdf, page=72, chunk=191]：子 Agent 上下文隔离——让大体积中间信息根本不进入主上下文，只回传结论性摘要。

5. **（计划设计）对比思考**：比较 Skills 渐进式披露与上下文压缩——两者都在解决"上下文膨胀"问题，但一个从源头控制（不加载不需要的），一个在事后处理（压缩已加载的）。

**可验证产出**：
- ✅ 画出 Skills 渐进式披露的三层结构图（标注每层的加载时机和 token 量级）[source=AI-Agents-in-Depth-zh-CN.pdf, page=59, chunk=157]
- ✅ 用一句话解释"上下文感知压缩"为什么比"个体摘要"效果好（答案：它将当前查询意图和已积累信息纳入压缩决策，实现有针对性的信息保留 [source=AI-Agents-in-Depth-zh-CN.pdf, page=70, chunk=187]）

---

## 📅 Day 3：延伸视野——记忆、工具与多 Agent 协作

> **今日目标**：了解 Agent 的跨会话记忆、工具设计原则和多 Agent 协作模式，形成完整的知识地图

---

### 任务 1：记忆与知识库——让 Agent 跨会话记住知识（约 2 小时）

**学习内容**：第 3 章核心概念——用户记忆 + RAG + Agentic RAG

**具体步骤**：

1. **理解 RAG 完整技术栈**（§3.2）[source=AI-Agents-in-Depth-zh-CN.pdf, page=3, chunk=10]：
   - **分块（Chunking）**→ **嵌入（Embedding）**→ **检索（Retrieval）**→ **重排序（Reranking）**
   - 稠密嵌入（语义理解）vs 稀疏嵌入（精确匹配关键词）[source=AI-Agents-in-Depth-zh-CN.pdf, page=3, chunk=10]
   - 混合检索 = 稠密 + 稀疏，两全其美

2. **理解 Agentic RAG 的范式转变**（§3.3.4）[source=AI-Agents-in-Depth-zh-CN.pdf, page=97, chunk=251]：
   - 非智能体化 RAG：固定前置步骤，一次性检索
   - **智能体化 RAG**：Agent 用 ReAct 循环主导检索——思考→行动→观察→判断信息是否充分→不够则再次检索
   - 关键价值：从"被动管道"到"主动探索者"的转变 [source=AI-Agents-in-Depth-zh-CN.pdf, page=98, chunk=253]

3. **了解 RAG 的安全风险** [source=AI-Agents-in-Depth-zh-CN.pdf, page=97, chunk=251]：间接提示注入（攻击者把恶意指令藏进会被检索的文档）和知识库投毒。防御两层：指令与数据分离 + 不让检索内容直接触发高风险操作。

4. **动手实验 3-1 ★**（搭建简单 RAG）：按配套代码 `chapter3/` 的 README 运行基础 RAG 系统。

**可验证产出**：
- ✅ 画出 RAG 技术栈的四步流程图（分块→嵌入→检索→重排序）
- ✅ 用对比图说明智能体化 RAG 与非智能体化 RAG 的区别（参照原文图 3-12 [source=AI-Agents-in-Depth-zh-CN.pdf, page=97, chunk=251]）

---

### 任务 2：工具设计与多 Agent 协作概览（约 2 小时）

**学习内容**：第 4 章工具核心概念 + 第 10 章多 Agent 协作入门

**具体步骤**：

1. **理解工具调用的四步流程**（§1.1.2）[source=AI-Agents-in-Depth-zh-CN.pdf, page=17, chunk=60]：
   - ① 在上下文中声明可用工具 → ② 模型自主判断是否调用、调用哪个、传什么参数 → ③ 工具执行结果追加到上下文 → ④ 模型据此决定下一步

2. **了解 MCP（Model Context Protocol）**：工具接入标准，正在让工具接入变得更容易 [source=AI-Agents-in-Depth-zh-CN.pdf, page=17, chunk=60]。

3. **理解工具安全的核心原则**：Harness 的约束功能——"故障安全默认值：所有能力默认关闭，必须显式开放"（类似手机 App 权限管理）[source=AI-Agents-in-Depth-zh-CN.pdf, page=27, chunk=83]。

4. **阅读多 Agent 协作的管理者模式**（§10.4）[source=AI-Agents-in-Depth-zh-CN.pdf, page=295, chunk=745]：
   - 以书籍翻译 Agent 为例：Glossary Agent（术语表）→ Translation Agent（章节翻译）→ Proofreading Agent（全文审校）→ Manager Agent（调度）
   - **核心优势：上下文隔离**——每个 Agent 在精简、专注的上下文中工作，避免信息过载 [source=AI-Agents-in-Depth-zh-CN.pdf, page=296, chunk=746]

5. **（计划设计）回顾与串联**：用 Day 1 学到的核心公式审视三天所学——
   - Day 1 建立了 Agent = LLM + 上下文 + 工具 的全局框架
   - Day 2 深入了"眼睛"（上下文工程）的设计细节
   - Day 3 延伸了"眼睛"的跨会话能力（记忆/RAG）和"手脚"的协作能力（工具/多 Agent）

**可验证产出**：
- ✅ 用管理者模式的翻译 Agent 案例，解释"上下文隔离"为什么能解决单 Agent 上下文膨胀问题 [source=AI-Agents-in-Depth-zh-CN.pdf, page=295, chunk=745]
- ✅ 完成一份三天学习总结思维导图，以"Agent = LLM + 上下文 + 工具"为中心，串联所有已学概念

---

## 📋 学习成果自检清单

完成 3 天学习后，检验自己是否能够：

- [ ] 解释 Agent = LLM + 上下文 + 工具 的核心公式及其三层映射（直觉/实现/学术）
- [ ] 描述 ReAct 循环的"思考→行动→观察"迭代过程
- [ ] 说出 KV Cache 三条核心结论，并解释为什么系统提示词不能动态修改
- [ ] 解释 Skills 渐进式披露如何解决提示词膨胀问题
- [ ] 区分智能体化 RAG 与非智能体化 RAG
- [ ] 用管理者模式案例解释多 Agent 上下文隔离的优势

> 💡 **后续建议**：本计划覆盖了全书最核心的第 1-2 章和第 3、4、10 章的关键概念。完成后可按原书 8 周路线图继续深入——第 5 章（Coding Agent）、第 6 章（评估）、第 8 章（持续进化）是下一阶段的优先章节 [source=AI-Agents-in-Depth-zh-CN.pdf, page=12, chunk=48]。记住作者的核心主张：**"没有评估，就没有进步"** [source=AI-Agents-in-Depth-zh-CN.pdf, page=10, chunk=43]。
