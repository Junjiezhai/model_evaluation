# DeepEval 框架使用手册（小白入门版）

> 适用版本：deepeval 4.3.1（本仓库实际安装版本）
> 本手册结合本仓库 `eval/` 目录下的真实评测工程编写，所有代码示例均经过实际运行验证。

---

## 目录

1. [DeepEval 框架提供的核心能力](#1-deepeval-框架提供的核心能力)
2. [核心参数及说明](#2-核心参数及说明)
3. [标准使用方式](#3-应用于模型评测的标准使用方式)
4. [可扩展性与自定义开发](#4-可扩展性如何支持自定义开发)
5. [与 pytest 框架的优劣分析](#5-deepeval-与-pytest-在模型评测领域的优劣分析)

---

## 1. DeepEval 框架提供的核心能力

### 1.1 一句话理解 DeepEval

DeepEval 是一个**开源的 LLM 输出质量评测框架**（GitHub: confident-ai/deepeval）。它要解决的问题是：传统软件测试有明确的"对/错"断言（`assert 1+1==2`），但大模型的输出是自然语言，没有标准答案可言——**DeepEval 用"另一个大模型当裁判（LLM-as-a-Judge）+ 成熟的多阶段评分算法"把这个主观判断过程标准化、可量化、可复现**。

在本仓库的工程中，它的角色是"评测运行时"：不管评测算法、不做调度之外的事，裁判力由接入的模型（我们用的是 Kimi）提供。

### 1.2 五大核心能力

#### ① 标准化评测指标（Metrics）—— 框架的核心资产

4.3.1 版本内置 **55 个指标类**，按用途分几大类：

| 类别 | 代表指标 | 评测什么 |
|---|---|---|
| **RAG / 知识库问答**（本仓库使用） | `FaithfulnessMetric`（忠实度）、`AnswerRelevancyMetric`（答案相关性）、`ContextualRecallMetric`（上下文召回）、`ContextualPrecisionMetric`、`ContextualRelevancyMetric` | 检索增强问答链路各环节质量 |
| **通用质量** | `BiasMetric`（偏见）、`ToxicityMetric`（毒性）、`PIILeakageMetric`（隐私泄露）、`HallucinationMetric`（幻觉） | 输出内容安全性 |
| **任务达成** | `TaskCompletionMetric`、`GoalAccuracyMetric`、`PromptAlignmentMetric`、`JsonCorrectnessMetric` | Agent/工具类应用 |
| **无 LLM 指标** | `ExactMatchMetric`、`PatternMatchMetric` | 有确定答案场景的硬匹配 |
| **自定义模板** | `GEval` | 用户自定义任何评分标准 |

**关键认知**：指标 = 评测算法 + 内置 prompt 模板。以忠实度为例，它内部是"先让裁判从答案抽取原子声明 → 逐条对照检索内容判定 → 统计比例"的多阶段流程，这套模板经过大量验证，这是用框架而不是自己写 prompt 的最大价值。

#### ② 统一测试数据结构（TestCase）

`LLMTestCase` 是评测的通用货币，所有指标都消费它：

```python
LLMTestCase(
    input="用户问题",                # 必填
    actual_output="模型实际回答",     # 必填
    expected_output="期望/标准答案",  # ContextualRecall、GEval 等需要
    retrieval_context=["知识片段1"],  # Faithfulness、ContextualRecall 需要
    context=["背景信息"],            # 部分指标使用
    metadata={"category": "提额降息"}, # 自定义元数据，原样透传
)
```

字段语义由指标约定，缺失必填字段时指标会显式报错而非静默给分。

#### ③ 评测执行引擎（evaluate）

```python
from deepeval import evaluate
results = evaluate(test_cases, metrics)
```

引擎负责：把"用例 × 指标"拆成独立异步任务并发执行（默认最多 20 并发）、单项任务超时预算（默认 180s，可配置）、失败重试（指数退避）、**故障隔离**（单个任务超时只标记 error，不拖垮全局）、结果聚合为 `EvaluationResult`。

#### ④ 测试用例合成器（Synthesizer）

```python
from deepeval.synthesizer import Synthesizer
synthesizer = Synthesizer(model=你的LLM)
goldens = synthesizer.generate_goldens_from_docs(document_paths=["知识库.pdf"])
```

从文档/知识库自动生成带标准答案的测试集，支持问题进化（改写、模糊化、对抗化）和过滤。解决"没有测试数据"的冷启动问题。（本仓库为了完全控制出题风格，自行用 Kimi 实现了简化版，见 `eval/synthesize_cases.py`。）

#### ⑤ 模型接入抽象层

`DeepEvalBaseLLM` 协议让**任何模型**（OpenAI、Claude、Kimi、本地 vLLM……）都能成为裁判或被测系统，这是"一个 Kimi 分饰三角"方案的技术基础（见第 4 节）。

#### ⑥ 多轮对话评测（ConversationalTestCase）—— 1.3 节详解

4.3.1 原生支持多轮对话评测：用 `Turn` 组装一段对话记录，由对话类指标（`BaseConversationalMetric` 子类，实测 25 个）打分，`evaluate()` 同一引擎执行。但它只管"事后给一段已有对话打分"，对话怎么跑出来的、业务门禁、状态机这些都不在内——能力边界与本仓库的自研方案详见 **1.3 节**。

### 1.3 多轮对话评测能力（ConversationalTestCase）

#### 1.3.1 数据结构与执行入口

单轮用 `LLMTestCase`，多轮用 `ConversationalTestCase`——**一段静态对话记录 + case 级元数据**：

```python
from deepeval.test_case import ConversationalTestCase, Turn

convo = ConversationalTestCase(
    turns=[
        Turn(role="user",      content="怎么提额"),
        Turn(role="assistant", content="您可以在提额页面选择提额项。"),
        Turn(role="user",      content="提额后多久能借款"),
        Turn(role="assistant", content="提额成功只代表额度更新，以借款页审核为准。"),
    ],
    scenario="用户咨询提额流程",          # 可选：场景描述，注入裁判 prompt
    expected_outcome="用户了解提额入口与借款条件",  # 可选：GoalAccuracy/TurnContextual* 需要
    chatbot_role="信贷客服助手",          # 可选：RoleAdherence 需要
)
```

`Turn` 的可用字段（4.3.1 实测）：`role`、`content`（均必填）、`retrieval_context`（逐轮 RAG 指标需要，**Turn 级而非 case 级**）、`latency_ms`（仅存储，框架不做性能统计）、`tools_called`、`audio`、`metadata` 等。

执行入口与单轮完全相同，`evaluate()` 自动识别用例类型：

```python
from deepeval import evaluate
from deepeval.metrics import ConversationCompletenessMetric, KnowledgeRetentionMetric

results = evaluate([convo], [
    ConversationCompletenessMetric(threshold=0.5, model=judge),
    KnowledgeRetentionMetric(threshold=0.5, model=judge),
])
```

#### 1.3.2 内置多轮指标清单（4.3.1 实测 25 个 `BaseConversationalMetric` 子类）

| 类别 | 指标 | 评测什么 | 必填材料 |
|---|---|---|---|
| **对话整体** | `ConversationCompletenessMetric` | 用户目标是否被完整达成（内部按 `window_size=3` 滑窗逐段核实） | role+content |
| | `KnowledgeRetentionMetric` | 跨轮信息记忆保持度 | role+content |
| | `RoleAdherenceMetric` / `RoleViolationMetric` | 角色/人设保持 | role+content（+chatbot_role） |
| | `TopicAdherenceMetric` | 是否跑题 | role+content |
| | `GoalAccuracyMetric` | 目标达成度 | role+content+expected_outcome |
| **RAG 逐轮** | `TurnFaithfulnessMetric`、`TurnRelevancyMetric` | 逐轮的忠实度/相关性（单轮指标的 Turn 版） | role+content（+**每轮** retrieval_context） |
| | `TurnContextualRecall/Precision/RelevancyMetric` | 逐轮检索质量 | +expected_outcome（Precision/Recall） |
| **Agent/工具** | `ToolUseMetric`、`MCPTaskCompletionMetric`、`MultiTurnMCPUseMetric` | 工具调用轨迹与任务完成 | role+content（+tools_called） |
| **语音/音频** | `AudioGlitch/Integrity`、`VoiceConsistency/Naturalness/Reliability`、`SpeechIntelligibility`、`TurnTakingNaturalness`、`TranscriptionAccuracy` 共 9 个 | 呼叫中心/语音助手质检（Turn 支持 audio/provider_transcript） | role+content（+audio 等） |
| **自定义** | **`ConversationalGEval`** | GEval 的多轮版：`criteria` + `evaluation_params`（15 种材料：scenario/expected_outcome/retrieval_context/chatbot_role…），自定义任何对话评分维度 | 按声明 |
| | `ConversationalJevEval`、`ConversationalDAGMetric` |  judge 变异测试 / DAG 工作流对话 | — |

#### 1.3.3 适用场景

原生多轮能力适合的场景有这些共性：**期望值是 case 级、无业务红线、judge 成本可接受、对话记录已有现成来源**。

| 场景 | 推荐指标组合 |
|---|---|
| 通用对话助手质量回归（闲聊、任务型客服） | ConversationCompleteness + KnowledgeRetention + TopicAdherence |
| RAG 多轮问答（知识库客服） | TurnFaithfulness + TurnContextualRecall + TurnContextualRelevancy |
| 角色扮演 / 人设 Bot | RoleAdherence + RoleViolation |
| 语音/呼叫中心质检 | Audio* / Voice* / Speech* / TranscriptionAccuracy |
| Agent / 工具调用对话 | ToolUse / MCPTaskCompletion / MultiTurnMCPUse |
| 快速验证一个新的对话维度 | ConversationalGEval 写 criteria，30 分钟上线 |

#### 1.3.4 能力边界：原生多轮做不到什么（5 条硬边界）

1. **只做"事后裁判"，不管对话怎么跑出来**。`Turn` 是既成记录：不维护 history、不逐轮注入检索上下文、不串行等待上轮返回。对话执行层（历史累积 + 检索注入 + 计时 + 缓存）需要自建。
2. **指标是通用质量维度，无业务语义**。全部是 0~1 分 + 单 threshold 判定，表达不了：一票否决门禁（任一红线 FAIL → 综合分清零）、多维度加权合成（Q1×0.25+Q2×0.25+…）、跨轮业务规则（状态机转移、数值跨轮一致性）。`ConversationalGEval` 也只是又一条独立 threshold 指标。
3. **judge 调用不可合并降本**。每个指标的 `measure()` 内部自行发起 1~N 次裁判调用（滑窗逐段、逐轮抽声明核实），且无法合并成一次请求——多指标 × 多 case × 多轮时调用次数不可控。
4. **逐轮指标有严格的材料齐备要求**。`TurnFaithfulness/TurnContextual*` 要求**每一轮**都填 `retrieval_context`；`expected_outcome` 是 case 级，表达不了"每轮各自的期望值"。
5. **不管性能与报告**。`latency_ms` 只是可存字段，不做平均/P90 等统计；无意图混淆矩阵、漂移率概念；多轮报告需自行渲染。

#### 1.3.5 本仓库的对策：自研多轮流水线 + 与 DeepEval 保持接口兼容

本仓库的多轮评测（`--mode multi`）落在上述边界之外，因此采用"**自研执行引擎与业务判定，复用 DeepEval 的模型桥接与指标设计哲学**"的混合方案：

| 多轮需求 | DeepEval 原生 | 本仓库实现 |
|---|---|---|
| 对话执行（历史注入/检索/计时/指纹缓存） | ❌ 无 | `eval/qa_system.py::run_multi_turn_dialog` |
| G1/G2 一票否决 + Q1~Q5 加权综合分 | ❌ 只有独立 threshold | `eval/metrics/multi_turn_judge.py`（单次结构化调用出全维度，支持批量合并降本） |
| 评分细则外置（YAML 注入 judge prompt） | ❌ criteria 硬编码 | `base_knowledge/multi_turn_eval_dimensions.yaml` |
| 会话状态机 / 链式状态验证 | ❌ 无 | `eval/metrics/state_validator.py`（YAML 矩阵驱动，strict/lenient/category） |
| 逐轮独立期望值（轮级 intent/state 标注） | ❌ Turn 仅 role/content 语义 | 13 列 CSV：`round_intent`/`state_intent` 每轮独立 |
| 数值一致性本地断言 / 意图混淆矩阵 / 雷达图报告 | ❌ 无 | `eval/metrics/local_asserts.py`、`eval/report.py::generate_multi_turn_report` |
| 自定义多轮指标扩展 | ✅ ConversationalGEval | 同一条路：GEval/BaseMetric 契约兼容（`eval/metrics/custom_example.py`） |

> 经验法则：**对话需要现跑、评测含业务规则与门禁、judge 成本敏感**——三者命中任一，就走自研流水线；
> 只是给已有对话记录按通用维度打分，用 1.3.3 的原生组合即可。

---

## 2. 核心参数及说明

### 2.1 指标构造参数（所有 LLM 指标通用）

| 参数 | 类型/默认 | 说明 |
|---|---|---|
| `threshold` | float，多数指标 0.5 | **及格线**。score ≥ threshold 才算通过 |
| `model` | str 或 DeepEvalBaseLLM，默认 GPT-4o | **最重要的参数**。指定谁来当裁判。不传会用默认 OpenAI 模型，需要 OPENAI_API_KEY；接入自定义模型后彻底摆脱这一依赖（本仓库做法） |
| `include_reason` | bool，默认 True | 是否要求裁判输出评分理由。排障和人工复核必备 |
| `strict_mode` | bool，默认 False | True 时任何指标失败直接抛异常中断评测；False（默认）只记录失败，跑完全部用例 |
| `async_mode` | bool，默认 True | 是否异步执行。保持默认即可 |
| `flaky` | bool，默认 False | 标记为不稳定的指标（如随机性大的），失败不计入总结果 |
| `evaluation_template` | Prompt，可选 | 替换指标内置的裁判 prompt 模板（高级用法，见 4.3） |

各指标特有参数（4.3.1 实测）：

| 指标 | 特有参数 | 说明 |
|---|---|---|
| `FaithfulnessMetric` | `truths_extraction_limit` | 最多抽取多少条声明参与判定 |
| `FaithfulnessMetric` | `penalize_ambiguous_claims` | 是否惩罚语义模糊的可疑声明 |
| `GEval` | `name`、`criteria`、`evaluation_params`、`evaluation_steps`、`rubric` | 见 2.2 |
| `ContextualRecallMetric` | — | 仅需 threshold/model 等通用参数 |

### 2.2 GEval 专用参数（自定义指标的瑞士军刀）

```python
GEval(
    name="correctness",                 # 指标名，会出现在结果里
    criteria="对照标准答案评估事实正确性……",  # 评分标准，自然语言描述
    evaluation_params=[                 # 裁判 prompt 中注入哪些用例字段
        SingleTurnParams.INPUT,
        SingleTurnParams.ACTUAL_OUTPUT,
        SingleTurnParams.EXPECTED_OUTPUT,
    ],
    evaluation_steps=[...],             # 可选：把评分拆成有序步骤，比 criteria 更精细
    rubric={...},                       # 可选：自定义评分细则
    threshold=0.7,
    model=judge_llm,
)
```

- `criteria` 写"评什么"，`evaluation_steps` 写"按什么顺序评"，`rubric` 写"每档分什么标准"；三者至少给一个。
- `evaluation_params` 决定裁判"看得到什么材料"。给错字段组合是 GEval 最常见的踩坑点。

### 2.3 evaluate() 执行参数（4.3.1 实测签名）

```python
evaluate(
    test_cases,                 # List[LLMTestCase] 或 List[ConversationalTestCase]，必填
    metrics=None,               # 指标列表；None 时用各指标的默认配置
    classifiers=None,           # 分类器（按内容打标签，用于结果分组）
    metric_collection=None,     # 预定义指标集合名
    hyperparameters=None,       # Dict，记录实验超参（温度、模型版本等），写入结果便于追溯
    identifier=None,            # 本次评测的标识名
    async_config=AsyncConfig(run_async=True, throttle_value=0, max_concurrent=20),
    display_config=DisplayConfig(show_indicator=True, print_results=True, ...),
    cache_config=CacheConfig(write_cache=True, use_cache=False),
    error_config=ErrorConfig(ignore_errors=False, skip_on_missing_params=False),
)
```

| 参数 | 说明 |
|---|---|
| `async_config.max_concurrent` | **并发度，默认 20**。裁判 API 有速率限制时要调低；想更快可调高 |
| `async_config.throttle_value` | 每个任务启动间隔（秒），用于平滑突发流量 |
| `error_config.ignore_errors` | True 时任务异常不中断整体评测 |
| `error_config.skip_on_missing_params` | 用例缺字段时跳过该指标而非报错 |
| `display_config.print_results` | 是否在终端打印逐条结果 |

### 2.4 环境变量（必须在 import deepeval 之前设置）

| 环境变量 | 作用 |
|---|---|
| `DEEPEVAL_PER_TASK_TIMEOUT_SECONDS_OVERRIDE` | 单任务超时预算（秒），**默认 180**。慢模型/高并发必调大（本仓库设 900，第一轮评测就死在默认值上） |
| `DEEPEVAL_MAX_CONCURRENT_...` 系列 | 各类任务并发上限 |
| `DEEPEVAL_RETRY_MAX_ATTEMPTS` | 任务内重试次数，默认 2 |
| `DEEPEVAL_TELEMETRY_OPT_OUT=YES` | 关闭匿名遥测 |

> ⚠️ 这些变量在 deepeval **首次 import 时读取**，之后修改无效。务必放在文件顶部 `import deepeval` 之前。

---

## 3. 应用于模型评测的标准使用方式

### 3.1 最小可运行示例（5 分钟跑通）

```python
import os
os.environ["DEEPEVAL_TELEMETRY_OPT_OUT"] = "YES"

from deepeval import evaluate
from deepeval.test_case import LLMTestCase
from deepeval.metrics import FaithfulnessMetric, AnswerRelevancyMetric

# 1. 准备测试用例：input + actual_output 是必填
test_case = LLMTestCase(
    input="提额成功是不是马上就能借款？",
    actual_output="提额成功只代表额度可能更新，实际能否借款以借款页审核为准。",
    retrieval_context=["提额成功只代表可用额度可能更新，不等于借款一定成功……"],
)

# 2. 选择指标并设定及格线（model 参数见第 4 节，这里先用默认）
metrics = [
    FaithfulnessMetric(threshold=0.7, include_reason=True),
    AnswerRelevancyMetric(threshold=0.7, include_reason=True),
]

# 3. 执行评测
results = evaluate([test_case], metrics)

# 4. 读取结果
for r in results.test_results:
    print(r.input, "通过" if r.success else "失败")
    for m in r.metrics_data:
        print(f"  {m.name}: {m.score:.2f} | {m.reason}")
```

输出中每个指标都有 `score`（0~1）、`threshold`、`success`（布尔）、`reason`（裁判理由）——**这就是"模型评测"的最终产物：可量化的分数 + 可追溯的理由**。

### 3.2 完整生产级流程（本仓库采用的标准范式）

```text
┌──────────────┐   ┌──────────────┐   ┌──────────────┐   ┌──────────────┐
│ 1. 构建测试集 │ → │ 2. 被测系统  │ → │ 3. Judge 打分│ → │ 4. 结果分析  │
│              │   │    作答      │   │              │   │              │
│ 知识库/人工  │   │ 生成actual_  │   │ 多指标并发   │   │ JSON落盘+    │
│ 合成测试用例 │   │   output     │   │  打分+判定   │   │ 可视化报告   │
└──────────────┘   └──────────────┘   └──────────────┘   └──────────────┘
```

对应代码（摘自本仓库 `eval/run_evaluation.py`，已脱敏精简）：

```python
# ---- 阶段 1：测试集（本仓库用 Kimi + 技能文档自行合成，存 CSV 复用）----
from eval.case_loader import load_single_turn
cases_data = load_single_turn(limit=None)

# ---- 阶段 2：被测系统作答 → 组装 LLMTestCase ----
test_cases = []
for c in cases_data:
    contexts = retriever.retrieve(c.input, top_k=3)      # 模拟 RAG 检索
    answer = answer_llm.generate(build_prompt(contexts, c.input))
    test_cases.append(LLMTestCase(
        input=c.input,
        actual_output=answer,
        retrieval_context=[x.text for x in contexts],
        expected_output=c.expected_output,
        metadata={"category": c.category},               # 透传，报告按分类聚合用
    ))

# ---- 阶段 3：Judge 打分 ----
judge = build_judge_llm()                                    # 自定义 Kimi 裁判（见 4.1）
metrics = [
    FaithfulnessMetric(threshold=0.7, model=judge),
    AnswerRelevancyMetric(threshold=0.7, model=judge),
    ContextualRecallMetric(threshold=0.7, model=judge),
    GEval(name="correctness", criteria="对照标准答案评估事实正确性……",
          evaluation_params=[SingleTurnParams.INPUT,
                             SingleTurnParams.ACTUAL_OUTPUT,
                             SingleTurnParams.EXPECTED_OUTPUT],
          threshold=0.7, model=judge),
]
results = evaluate(test_cases, metrics)

# ---- 阶段 4：结果落盘 + 报告 ----
data = serialize_results(results)                            # 提取 score/success/reason
json.dump(data, open("results/evaluation_results.json", "w", encoding="utf-8"),
          ensure_ascii=False, indent=2)
generate_html_report(data)                                   # 自包含 HTML 可视化
```

### 3.3 指标选择与数据字段的对应关系（最常见的配置错误来源）

| 指标 | 需要 LLMTestCase 的哪些字段 | 裁判实际比对什么 |
|---|---|---|
| FaithfulnessMetric | `actual_output` + `retrieval_context` | 答案 vs 检索知识：有没有编造 |
| AnswerRelevancyMetric | `input` + `actual_output` | 问题 vs 答案：答非所问检测 |
| ContextualRecallMetric | `expected_output` + `retrieval_context` | 标准答案 vs 检索知识：检索漏没漏（**不依赖模型答案**） |
| ContextualRelevancyMetric | `input` + `retrieval_context` | 问题 vs 检索知识：检回来的内容相关吗 |
| GEval | 由 `evaluation_params` 显式声明 | 你声明什么就比什么 |

**口诀**：忠实度管"编没编"，相关性管"跑没跑题"，召回率管"检索全不全"——三个指标合在一起才能定位 RAG 链路的问题出在哪一环。

### 3.4 另一种入口：pytest 风格（框架官方推荐）

```python
import pytest
from deepeval import assert_test
from deepeval.test_case import LLMTestCase
from deepeval.metrics import FaithfulnessMetric

def test_faithfulness():
    test_case = LLMTestCase(
        input="...", actual_output="...",
        retrieval_context=["..."],
    )
    assert_test(test_case, [FaithfulnessMetric(threshold=0.7)])

# 运行：deepeval test test_eval.py   （不是 pytest！见第 5 节）
```

适合把评测接入 CI 的场景，与 `evaluate()` 是同一套底层，二选一即可。

---

## 4. 可扩展性：如何支持用户进行自定义开发

DeepEval 的可扩展性围绕三个抽象展开：**换裁判模型、换指标、换评测逻辑**。

### 4.1 自定义模型接入（必学，一切扩展的基础）

场景：不用 OpenAI，用自己的模型（Kimi、Claude、本地部署……）当裁判或被测系统。

实现：继承 `DeepEvalBaseLLM`，实现 4 个方法：

```python
from deepeval.models import DeepEvalBaseLLM

class MyCustomLLM(DeepEvalBaseLLM):
    def __init__(self, model_name="my-model", system_prompt=None, temperature=0.0):
        self._model_name = model_name
        self.system_prompt = system_prompt
        self.temperature = temperature
        self._client = MyApiClient()          # 你自己的 HTTP/SDK 客户端
        super().__init__(model_name=model_name)

    def load_model(self):                      # 框架初始化时调用
        return self._client

    def generate(self, prompt: str, schema=None) -> str:   # 同步调用，指标打分时走这里
        return self._client.chat(
            prompt=prompt,
            system_prompt=self.system_prompt,
            temperature=self.temperature,
        )

    async def a_generate(self, prompt: str, schema=None) -> str:
        # 并发评测时走这里；若底层是同步阻塞客户端，
        # 必须 asyncio.to_thread 转线程，否则并发退化为串行！
        import asyncio
        return await asyncio.to_thread(self.generate, prompt, schema)

    def get_model_name(self) -> str:           # 写入结果元数据，用于报告展示
        return self._model_name
```

使用：`FaithfulnessMetric(model=MyCustomLLM(...))` —— 之后所有 Judge 调用全部走你的模型。

**本仓库的进阶玩法**：同一个底层模型实例化三个角色（出题人 t=0.7 / 问答助手 t=0.3 / 裁判 t=0 + 不同系统提示词），实现"一个 Kimi 分饰三角"。代码见 `eval/kimi_llm.py`，只有约 60 行。

**真实踩坑记录**：
- `a_generate` 里直接调同步方法会卡死事件循环 → 必须 `asyncio.to_thread`；
- 裁判角色建议 `temperature=0` 保证打分可复现；
- 评测场景建议关闭模型的"思考模式"降低延迟。

### 4.2 自定义指标（两种路线）

**路线 A：GEval 拼装（推荐，90% 场景够用）** —— 不写代码，只写评分标准：

```python
GEval(
    name="客服礼貌度",
    criteria="评估客服回答是否礼貌、共情，是否在用户抱怨时先安抚再给方案",
    evaluation_params=[SingleTurnParams.INPUT, SingleTurnParams.ACTUAL_OUTPUT],
    evaluation_steps=[                       # 可选：分步指引比一句话标准更稳
        "识别用户情绪（平静/抱怨/愤怒）",
        "检查回答是否包含与情绪匹配的安抚",
        "检查是否先安抚后给方案",
    ],
    threshold=0.7,
    model=judge,
)
```

**路线 B：继承 BaseMetric 完全自定义（评测逻辑也要自己控制时）**：

```python
from deepeval.metrics import BaseMetric
from deepeval.test_case import LLMTestCase

class MyMetric(BaseMetric):
    def __init__(self, threshold=0.5, model=None):
        self.threshold = threshold
        self.model = model
        # 必需属性：score / reason / success / error
        self.score = None

    def measure(self, test_case: LLMTestCase) -> float:
        # 完全自由：调几次模型、怎么算分、加权……都由你定
        raw = self.model.generate(f"评分：{test_case.input} / {test_case.actual_output}，只输出0-100")
        self.score = int(raw) / 100
        self.reason = "基于自定义算法"
        self.success = self.score >= self.threshold
        return self.score

    def is_successful(self) -> bool:
        return self.success
```

然后 `evaluate(test_cases, [MyMetric(model=judge)])`，与其他内置指标混用。

### 4.3 其他扩展点一览

| 扩展点 | 方式 | 场景 |
|---|---|---|
| 替换指标内置 prompt | 构造参数 `evaluation_template=Prompt(...)` | 想微调忠实度的判定口径但不改算法 |
| 自定义分类器 | 继承 `BaseClassifier` | 评测同时按"问法类型"自动打标签 |
| 合成测试集 | `Synthesizer(model=...)` + `generate_goldens_from_docs` | 从 PDF/Word 知识库自动造测试数据 |
| 自定义评分维度 | GEval 的 `rubric` 参数 | 替换 GEval 默认的 0~1 打分档 |
| 结果后处理 | `EvaluationResult.test_results` 是标准 pydantic 对象 | 接自己的 BI/报表系统 |

---

## 5. DeepEval 与 pytest 在模型评测领域的优劣分析

先说结论：**两者不是竞品，是不同层面的工具**——pytest 是通用测试运行器，DeepEval 是 LLM 评测框架（它内部甚至支持 pytest 风格）。真正的对比是"用 DeepEval 评模型 vs 用 pytest + 手写断言评模型"。

### 5.1 对比总表

| 维度 | DeepEval | pytest + 手写评测 |
|---|---|---|
| **断言能力** | 自然语言级：忠实度、相关性、正确性…55 个开箱即用指标 | 只有代码级断言：字符串相等、正则包含。面对"回答得对不对"基本无能为力 |
| **裁判机制** | 内置 LLM-as-a-Judge 多阶段算法（抽声明→逐条判定），有研究背书 | 需自己设计 prompt、解析分数、处理裁判不稳定，工作量大且难以做好 |
| **评测执行** | 原生异步并发、超时预算、重试、故障隔离，为"几百次 API 调用"场景而生 | pytest-xdist 并发与异步支持面向单元测试，管理 LLM 调用的超时/限流要全部自己写 |
| **结果产出** | 每个指标带分数+理由，结果即评分卡，天然适合做质量报告 | 只有 pass/fail，理由要自己存 |
| **数据构造** | 自带 Synthesizer 从文档合成测试集 | 手工写 |
| **学习成本** | 需理解指标语义、字段契约、模型接入 | pytest 本身简单，但"如何评 LLM"这个难题被转移给了你 |
| **成熟度** | 领域专用，4.x 版本仍在快速迭代（本仓库就遇到 3 处 API 变更：弃用参数名、返回结构、默认超时） | pytest 极度稳定，API 多年不变 |
| **生态/CI** | `deepeval test` 命令可跑 pytest 文件，但生态小 | 测试生态最大，插件千余个，CI 集成文档遍地 |
| **离线/确定性** | 依赖外部 LLM 当裁判，分数有随机性（可 t=0 缓解），有 API 成本 | 纯本地断言，零成本、完全确定 |
| **适用场景** | LLM 应用的质量评测（RAG、Agent、客服…） | 传统代码逻辑、确定性的数据处理管线 |

### 5.2 各自的最佳适用场景

**选 DeepEval 当：**
- 你在做 RAG / Agent / 对话类应用，需要回答"模型答得好不好"；
- 需要向团队/老板交付**带分数和理由**的质量报告；
- 评测要跑几十上百条用例，需要并发、容错、重试。

**选 pytest（手写断言）当：**
- 测的是 LLM 应用中的**确定性部分**：输出 JSON 的 schema 校验、工具调用的参数正确性、代码执行结果；
- 需要零成本、可离线、100% 确定性的回归测试（可用 `ExactMatchMetric` / `PatternMatchMetric` 这类免 LLM 指标补足）；
- 团队已深度使用 pytest/CI，想保持工具链统一。

**本仓库的混合实践（推荐范式）**：
- 评测编排放在 DeepEval 里（`evaluate()` + 4 个 LLM 指标），产出质量分；
- 结果落盘为标准 JSON，报告生成是普通 Python（不依赖任何框架）；
- 传统单元测试思维用于框架自身：知识库解析、检索器这些确定性模块完全可以用 pytest 测——两者各司其职。

### 5.3 已知的 DeepEval 短板（客观陈述）

1. **版本迭代快、API 变动多**：本仓库升级到 4.3.1 时遇到 `LLMTestCaseParams→SingleTurnParams`、`additional_metadata→metadata`、`evaluate(print_results=)` 参数移除三处变更；锁定版本（`requirements.txt` 钉死 `deepeval==4.3.1`）是必备操作。
2. **默认依赖 OpenAI**：不传 `model` 就走 GPT-4o，国内环境必须做自定义模型接入（第 4.1 节）。
3. **默认超时偏紧**：180s/任务在慢模型下不够，需显式调大。
4. **自评偏差是方法论问题**：用被测模型自评会偏宽松，框架管不了，需要第二模型交叉裁判或人工抽检兜底。

---

## 附录：本仓库相关文件索引

| 文件 | 内容 |
|---|---|
| `eval/kimi_llm.py` | 4.1 自定义模型接入的完整实现（三角色） |
| `eval/run_evaluation.py` | 3.2 标准评测流程 + 多轮自研流水线的完整实现（CLI 入口） |
| `eval/case_loader.py` / `eval/synthesize_cases.py` | 阶段 1：CSV 测试集读取校验 / 技能文档驱动生成 |
| `eval/qa_system.py` | 阶段 2：单轮与多轮作答链路（检索注入、批量、缓存、计时） |
| `eval/metrics/single_turn.py` | 单轮 4 指标的构建与注册（含自定义指标接入点） |
| `eval/metrics/multi_turn_judge.py` | 1.3.5 多轮 G1/G2 门禁 + Q1~Q5 加权评估器 |
| `eval/metrics/state_validator.py` | 1.3.5 会话状态机验证（YAML 矩阵驱动） |
| `eval/metrics/local_asserts.py` | 1.3.5 数值一致性 + 意图分类本地断言 |
| `eval/report.py` | 结果可视化：单轮性能区块 + 多轮雷达图（框架外，纯 Python） |
| `config.py` | 阈值、超时、角色提示词、批量/缓存/多轮权重等配置 |
| `results/evaluation_report.html` | 一次真实评测的完整产出样例 |
