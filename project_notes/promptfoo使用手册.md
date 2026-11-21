# promptfoo 使用手册（小白入门版）

> 适用版本：promptfoo 0.121.x（2026 年下半年当前版本系列，基于官方文档编写）
> 说明：promptfoo 已在 `../promptfoo_eval` 落地为 **独立的公共评测集评测轨**（GSM8K / C-Eval，见该目录 README；与 DeepEval 链路零代码耦合，模型接入复用同一 Kimi for Coding 端点与凭据）。本手册是独立学习文档，结构与 `project_notes/deepeval使用手册.md` 平行，便于对照阅读。
> 背景速览：promptfoo 是 MIT 协议的开源 LLM 评测/红队测试工具，CLI 优先、配置驱动；2026 年 3 月 OpenAI 宣布收购，官方承诺保持开源和模型中立。

> **新手阅读路径建议**（约 1 小时）：
> ① 第 0 节环境准备 → ② 第 1.1~1.2 节建立认知 → ③ 第 3.1 节跑通最小示例 → ④ 第 2 节理解配置 → ⑤ 第 4.1 节接自家模型 → ⑥ 已熟悉 DeepEval 的读者重点看第 5 节对比。
> 第 1.4 节（红队）、第 4 节其余部分按需求选读。

---

## 目录

0. [安装与环境准备](#0-安装与环境准备)
1. [promptfoo 框架提供的核心能力](#1-promptfoo-框架提供的核心能力)
2. [核心配置及参数说明](#2-核心配置及参数说明)
3. [应用于模型评测的标准使用方式](#3-应用于模型评测的标准使用方式)
4. [可扩展性：如何支持自定义开发](#4-可扩展性如何支持自定义开发)
5. [promptfoo 与 DeepEval 对比](#5-promptfoo-与-deepeval-对比)
6. [常见问题排查（FAQ）](#6-常见问题排查faq)

---

## 0. 安装与环境准备

### 0.1 安装

```bash
npm install -g promptfoo        # 方式一：全局安装（Node.js 18+）
npx promptfoo@latest eval ...   # 方式二：免安装，npx 直接跑
promptfoo --version             # 验证
promptfoo init                  # 在当前目录生成示例 promptfooconfig.yaml
```

要求 Node.js ≥ 18。与 DeepEval（Python 包）不同，promptfoo 是 **Node CLI 工具**——Python 技术栈的团队把它当"外部命令"用即可，无需混入 Python 依赖。

### 0.2 第一次运行前必须知道的三件事

1. **模型通过 Provider 接入，API Key 走环境变量**。约定式命名：`OPENAI_API_KEY`、`ANTHROPIC_API_KEY`、`GOOGLE_API_KEY`……国内模型走 OpenAI 兼容端点也能配（见 4.1 节 Kimi 示例）。
2. **一切评测定义都在 YAML/CSV 里**，CLI 只负责执行：`promptfoo eval -c promptfooconfig.yaml`。没有"先写 Python 脚本"这一步。
3. **默认上报匿名使用统计**（不含 prompt 内容）。介意的话设环境变量 `PROMPTFOO_DISABLE_TELEMETRY=1`。

### 0.3 与 DeepEval 的本质差异（先建立直觉）

| | DeepEval | promptfoo |
|---|---|---|
| 形态 | Python 库（`import deepeval`，写代码） | CLI 工具 + YAML 配置（不写代码也能评测） |
| 评测单元 | Metric（含多阶段算法 + prompt 模板） | Assertion（对单次输出的一条判定规则） |
| 强项 | 学术指标丰富、Python 生态、结果对象可编程 | 多模型横向对比、提示词回归、CI 门禁、红队测试 |

一句话：**DeepEval 像"评测框架"，promptfoo 像"评测流水线工具"**——前者你写代码调用它，后者你写配置它来跑。

---

## 1. promptfoo 框架提供的核心能力

### 1.1 一句话理解 promptfoo

promptfoo 要解决的问题是：大模型应用的质量依赖"提示词 × 模型 × 参数"的组合，而这个组合会频繁变动——**promptfoo 把"用哪些 prompt、调哪些模型、跑哪些测试用例、按什么标准判分"全部声明在一个 YAML 配置里，一条命令并发执行并产出可对比的评分表**。它是 LLM 时代的"矩阵化回归测试工具"。

### 1.2 四大核心能力

#### ① 配置即评测定义（prompts × providers × tests 三维矩阵）

这是 promptfoo 的心脏。一次评测 = **N 个提示词 × M 个模型 × T 条测试用例** 的全组合，自动并发执行：

```yaml
prompts:
  - prompts/customer_service_v1.txt    # 提示词文件
  - prompts/customer_service_v2.txt    # 换个提示词，立刻对比
providers:
  - openai:gpt-4o
  - anthropic:claude-sonnet-5-5        # 换个模型，立刻对比
tests:
  - vars: { question: "提额成功是不是马上就能借款？" }
  - vars: { question: "利息怎么算？" }
```

**关键认知**：DeepEval 里"换模型对比"要自己写 for 循环；promptfoo 里改一行 YAML 就是一次完整的矩阵实验。这是它最独特的能力。

#### ② 断言库（Assertions）—— 判分标准

每条测试用例挂一组断言，全部通过才算 PASS。两类：

- **确定性断言**（零成本、本地判定）：字符串包含/相等/正则、JSON 格式校验、SQL 合法性等
- **模型裁判断言**（LLM-as-a-Judge）：`llm-rubric`（自定义评分标准）、`model-graded-factuality`（事实性）、`model-graded-correctness`（正确性）等，由 `--grader` 指定的模型打分

与 DeepEval 指标的区别：DeepEval 的指标内部是"抽声明→逐条判定→聚合"的多阶段算法；**promptfoo 的裁判断言基本是"一次调用出分"**，更轻量但更依赖你写的 rubric 质量。

#### ③ 红队测试（Red Teaming）

```bash
promptfoo redteam init    # 生成 redteam.yaml（插件/策略库）
promptfoo redteam run     # 执行攻击探测，产出安全报告
```

内置 50+ 漏洞插件库（越狱、提示注入、PII 泄露、不当内容等，对齐 OWASP LLM Top 10），对目标应用自动发起对抗探测。**这是 DeepEval 完全没有的能力维度**，也是 promptfoo 在社区出圈的主因。

#### ④ 结果查看与工程集成

- `promptfoo view`：本地 Web UI（表格视图、逐条查看 prompt/输出/断言详情）
- 结果可导出 JSON/CSV/HTML（`outputPath` 配置），可上传分享（`promptfoo share`）
- CI 友好：退出码非零即门禁失败，可直接接 GitHub Actions

### 1.3 能力边界：promptfoo 不擅长什么

对照本仓库 DeepEval 手册 1.3.4 的写法，如实列出边界：

1. **无内置 RAG 指标体系**。没有忠实度/上下文召回这类学术指标；RAG 场景要么用 `llm-rubric` 手写标准凑，要么在自定义 Provider 里内嵌检索逻辑（见 4.1），评测语义精细度不如 DeepEval。
2. **裁判调用同样不可合并降本**。每条断言独立发起 1 次裁判调用；N 条用例 × M 个断言就是 N×M 次调用，与 DeepEval 同病。
3. **配置驱动的另一面**：复杂业务逻辑（加权综合分、状态机验证、跨轮规则）在 YAML 里表达不了，得靠 `javascript`/`python` 断言文件或自定义 Provider 写代码——那为什么不直接用 DeepEval/Python？
4. **多轮对话支持弱**。没有 ConversationalTestCase 那样的对话级指标，对话记录只能塞进单个 var 里当文本处理。
5. **测试数据合成能力弱**。没有 Synthesizer 等价物（网页版有"生成测试用例"功能，依赖云端，可控性不如自己写）。

> 经验法则：**要做多模型/prompt 矩阵对比、提示词回归门禁、红队安全测试** → promptfoo；
> **要评 RAG/对话质量、要学术指标、要在 Python 里编排复杂判定** → DeepEval（本仓库的选择）。

---

## 2. 核心配置及参数说明

### 2.1 promptfooconfig.yaml 完整结构

```yaml
description: 客服问答评测                          # 展示用描述
prompts:
  - prompts/system_v1.txt                          # 提示词：文件引用或内联字符串
providers:
  - openai:gpt-4o                                  # 短语法
  - id: anthropic:claude-sonnet-5-5               # 长语法（需要附加配置时用）
    config:
      temperature: 0
tests:                                            # 测试用例：内联列表
  - vars:
      question: "提额成功是不是马上就能借款？"
    assert:
      - type: llm-rubric
        value: "回答应明确说明提额≠可借款，以借款页审核为准"
        threshold: 0.7
defaultTest:                                      # 所有用例共享的默认断言/变量
  assert:
    - type: not-contains
      value: "对不起，我不知道"
outputPath: results/output.json                   # 结果落盘（json/csv/html）
```

| 配置键 | 说明 |
|---|---|
| `prompts` | 提示词列表，支持文件路径、内联字符串、Nunjucks 模板。列表里放多个 = 自动做提示词对比 |
| `providers` | 模型列表，短语法 `厂商:模型名` 或长语法 `id + config`（温度、自定义端点等） |
| `tests` | 测试用例列表；也可用 `tests: tests.csv` 直接引用 CSV 文件 |
| `vars` | 注入提示词的变量，在提示词里用 `{{question}}` 占位（Nunjucks 模板语法） |
| `assert` | 该用例的断言列表；搭配 `defaultTest.assert` 实现"公共断言 + 个性断言" |
| `outputPath` | 结果导出路径，便于二次处理（接 BI/报告系统） |

### 2.2 断言类型速查（按用途分类）

| 类别 | 断言类型 | 判定什么 |
|---|---|---|
| **文本匹配**（免 LLM） | `equals` / `contains` / `icontains` / `not-contains` / `regex` / `starts-with` / `contains-all` / `contains-any` 及 `not-*` 系列 | 输出是否包含/匹配期望内容，禁语检测特别好用 |
| **结构校验**（免 LLM） | `is-json` / `contains-json` / `is-sql` / `is-valid-openai-function-call` | 输出格式合法性（JSON schema 类校验） |
| **代码断言** | `javascript`（JS 代码/文件）/ `python`（Python 文件） | 任意自定义判定逻辑，拿 var 和输出做计算 |
| **语义相似** | `similarity` | 输出与参考答案的 embedding 余弦相似度，超阈值即过 |
| **模型裁判** | `llm-rubric`（自定义标准）、`model-graded-factuality`、`model-graded-correctness`、`model-graded-closedqa` | LLM 按 rubric 打 0~1 分，对应 DeepEval 的 GEval 定位 |
| **成本性能** | `cost` / `latency` | 单次调用成本/延迟上限，超了即 FAIL |
| **安全护栏** | `guardrails` | PII 检测等护栏规则 |

断言的通用字段：

| 字段 | 说明 |
|---|---|
| `type` / `value` | 断言类型 / 期望值（`llm-rubric` 的 value 就是评分标准文本） |
| `threshold` | 裁判类断言的及格分（0~1），相当于 DeepEval 的 `threshold` |
| `weight` | 断言权重，用于多条断言加权合成 |
| `rubricPrompt` | `llm-rubric` 专用：整体替换裁判 prompt 模板（高级用法，见 4.3） |

### 2.3 测试用例的两种写法

**写法一：YAML 内联**（适合用例少、断言复杂的场景，如上例）。

**写法二：CSV 文件**（适合用例多、团队维护测试集的场景）——列名即变量名，特殊列 `__expected` 自动转为断言：

```csv
question,__expected
提额成功是不是马上就能借款？,contains: 以借款页审核为准
利息怎么算？,regex: 年利率|日利率
```

配置里只需一行 `tests: tests.csv`。CSV 与 YAML 断言可混用。本仓库 DeepEval 工程里 13 列的 CSV 测试集，迁移到 promptfoo 时把列名对上、加 `__expected` 列即可复用。

### 2.4 CLI 常用参数

```bash
promptfoo eval -c promptfooconfig.yaml \
  --grader openai:gpt-4o      # 指定裁判模型（所有裁判类断言统一走它）
  --num-retries 2             # 失败重试次数
  --delay 1000                # 请求间隔毫秒（API 限流时调大）
  --repeat 3                  # 每条用例重复跑 N 次（测稳定性）
  --no-cache                  # 关闭缓存强制重跑
  --filter-sample 50          # 随机抽 50 条用例（调试期省成本，可用 --filter-sample-seed 固定抽样）
  --filter-first-n 20         # 只跑前 20 条（等价于 --filter-range 0:20）
  --filter-pattern 提额       # 只跑描述匹配正则的用例
  -j 8                        # 最大并发 API 调用数（默认 4）
promptfoo view                # 本地 Web UI 查看最近一次结果
```

---

## 3. 应用于模型评测的标准使用方式

### 3.1 最小可运行示例（10 分钟跑通）

**第一步**：装好后初始化并替换配置

```bash
npm install -g promptfoo
mkdir my-eval && cd my-eval && promptfoo init
```

**第二步**：写 `promptfooconfig.yaml`

```yaml
prompts:
  - |
    你是信贷客服助手。请回答用户问题。
    问题：{{question}}
providers:
  - openai:gpt-4o
tests:
  - vars:
      question: "提额成功是不是马上就能借款？"
    assert:
      - type: llm-rubric
        value: "回答应明确说明提额成功不等于可以借款，实际以借款页审核为准"
        threshold: 0.7
  - vars:
      question: "你们利息怎么算？"
    assert:
      - type: not-contains
        value: "我不知道"
      - type: llm-rubric
        value: "回答应说明利率计算方式或引导至利率说明页，不得编造具体数字"
        threshold: 0.7
```

**第三步**：跑评测、看结果

```bash
promptfoo eval -c promptfooconfig.yaml --grader openai:gpt-4o
promptfoo view    # 表格视图：每行=用例×模型，每列=断言 PASS/FAIL + 分数
```

产出物：每条组合一行记录，含输出全文、每个断言的通过情况和裁判分数——**与 DeepEval 的 score + reason 等价，但以表格形态呈现，天然适合横向对比**。

### 3.2 生产级流程（提示词回归的典型范式）

```text
┌──────────────┐   ┌──────────────┐   ┌──────────────┐   ┌──────────────┐
│ 1. 维护测试集 │ → │ 2. 声明矩阵  │ → │ 3. CLI 执行  │ → │ 4. 门禁/对比 │
│              │   │              │   │              │   │              │
│ tests.csv    │   │ prompts×     │   │ promptfoo    │   │ CI 失败即    │
│ （分类/难度） │   │ providers    │   │ eval 并发跑  │   │ 阻断；view   │
│              │   │ × asserts    │   │              │   │ 人工复核     │
└──────────────┘   └──────────────┘   └──────────────┘   └──────────────┘
```

典型迭代场景：**提示词 v1 → v2 改了措辞，担心质量回退**——把两个版本的 txt 都放进 `prompts:`，跑一次 `promptfoo eval`，表格里直接看哪些用例 v2 挂了、裁判理由是什么。这类"变更前对比"正是 DeepEval 做起来别扭、promptfoo 做起来自然的事。

### 3.3 断言选择速查（对应 DeepEval 手册 3.3 的字段契约表）

| 评测意图 | DeepEval 指标 | promptfoo 等价做法 |
|---|---|---|
| 答案不得编造（有检索上下文） | `FaithfulnessMetric`（需 `retrieval_context`） | 把上下文拼进 var，`llm-rubric` 写"逐条核对回答与【知识】是否一致"；精细度不如 DeepEval |
| 答非所问检测 | `AnswerRelevancyMetric` | `model-graded-relevance` 或 `llm-rubric` |
| 对照标准答案的正确性 | `ContextualRecallMetric` / `GEval` | `model-graded-correctness`（有参考答案时）/`llm-rubric` |
| 格式合法性 | — | `is-json` / `is-sql`（DeepEval 需另写） |
| 禁语/必须包含 | — | `not-contains` / `contains`（零成本，比 LLM 判定更可靠） |
| 延迟/成本上限 | —（`latency_ms` 仅存储） | `latency` / `cost` 断言直接判 FAIL |

**口诀**：能用确定性断言（contains/regex/is-json）判定的，**绝不用 LLM 裁判**——零成本且 100% 稳定；LLM 裁判只留给"语义质量"这类无法用规则表达的维度。这条法则在 DeepEval 里同样成立（`ExactMatchMetric`/`PatternMatchMetric` 的存在理由）。

### 3.4 与 CI 集成（promptfoo 的主场场景）

```yaml
# .github/workflows/eval.yml 片段
- run: npx promptfoo eval -c promptfooconfig.yaml --grader ${{ secrets.GRADER_MODEL }}
  env:
    OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
```

任何断言 FAIL → 退出码非零 → 构建失败。对"prompt 改动必须过质量门禁"的团队，这是 promptfoo 最值钱的用法。

---

## 4. 可扩展性：如何支持自定义开发

promptfoo 的扩展围绕三条路：**换模型（Provider）、换判定（断言文件/自定义 rubric）、接进代码（Node API / 自定义 Provider）**。

### 4.1 自定义模型接入（含国内 OpenAI 兼容端点）

**场景一：接 Kimi/Moonshot 等 OpenAI 兼容端点**——不用写代码，配置即可：

```yaml
providers:
  - id: openai:gpt-4o                    # 复用 openai provider，只改端点和 key
    config:
      apiBaseUrl: https://api.moonshot.cn/v1
      apiKey: '{{env.MOONSHOT_API_KEY}}'
      model: kimi-k2-0905-preview        # 以实际模型名为准
      temperature: 0
```

**场景二：任意自定义端点 / 签名**——写一个 provider YAML（请求模板 + 响应解析），`id` 指向该文件：

```yaml
# providers/my_api.yaml
id: my-custom-api
config:
  url: https://my-api.example.com/chat
  method: POST
  headers:
    Authorization: Bearer {{env.MY_API_KEY}}
  body:
    message: '{{prompt}}'
  responseParser: 'json.choices[0].message.content'   # 从响应里取输出的路径
```

对应 DeepEval 手册 4.1 的 `DeepEvalBaseLLM`——**概念完全等价**（都是"让任意模型接入框架"的桥），只是 promptfoo 用声明式 YAML 而非继承式 Python。

### 4.2 自定义判定：python/javascript 断言

断言里跑自己的代码，拿到 `output`（模型输出）和 `context`（含 `vars` 等用例变量），返回 bool 或 `{"pass": bool, "reason": str, "score": 0~1}`：

```python
# assert_numeric_consistent.py
# promptfoo 约定：python 断言脚本导出 get_assert(output, context)
import json, sys, os

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '../..', '..'))


def get_assert(output, context):
    vars_ = (context or {}).get('vars') or {}
    expected = vars_.get('expected_answer', '') or ''
    # 例：检查输出中的数字是否覆盖标准答案中的关键数值
    from assertion.e2e.deterministic import check_numeric_consistency
    result = check_numeric_consistency(expected, output or '')
    passed = bool(result.passed)
    # ⚠️ 返回 dict 时 score 必填（有限数值），否则断言报"必须返回 boolean/number/{pass,score,reason}"
    return {'pass': passed, 'score': 1.0 if passed else 0.0, 'reason': result.details}
```

```yaml
assert:
  - type: python
    value: file://assert_numeric_consistent.py
```

> ⚠️ 注意：python 断言的约定是**脚本内定义 `get_assert(output, context)` 函数**（由 promptfoo 内嵌 Python 桥调用），不是"stdin 读 JSON / stdout 写 JSON"——后者是自定义 python provider 的约定，两者不要混淆。

**这是 promptfoo 回应"能力边界 3"（YAML 表达不了复杂逻辑）的官方通道**——本仓库 DeepEval 工程里的数值一致性本地断言（`assertion/e2e/deterministic.py`）迁过来就是这个形态。

### 4.3 其他扩展点一览

| 扩展点 | 方式 | 场景 |
|---|---|---|
| 自定义裁判 prompt | `llm-rubric` + `rubricPrompt` 整体替换模板 | 对裁判口径有精细要求时（对应 DeepEval 的 `evaluation_template`） |
| 独立裁判模型 | 断言级 `provider` 字段或全局 `--grader` | 不同断言用不同裁判；或统一指定避免每个断言重复写 |
| 提示词模板逻辑 | Nunjucks 模板（`{% if %}`、`{{var}}`） | 一个提示词文件适配多类用例（如按品类切换细则） |
| 程序化调用 | Node.js API（`promptfoo` 作为库 import） | 要把评测嵌进自有 Python/Node 流水线时（注意：无官方 Python API） |
| 红队插件定制 | redteam.yaml 的 `plugins` 列表 | 只跑关注的攻击面（如只测提示注入 + PII） |
| 结果二次处理 | `outputPath` 导出 JSON | 接自有报告系统（对应 DeepEval 手册的"结果后处理"） |

---

## 5. promptfoo 与 DeepEval 对比

你在本仓库已经熟悉 DeepEval，这一节直接回答"为什么还要学 promptfoo"和"什么时候该用哪个"。

### 5.1 对比总表

| 维度 | promptfoo | DeepEval |
|---|---|---|
| **形态** | CLI + YAML/CSV 配置，零代码起评 | Python 库，写代码编排 |
| **判分单元** | Assertion（单条判定规则，可挂多条） | Metric（含多阶段算法的完整指标） |
| **内置指标深度** | 无学术指标，裁判靠 rubric 手写 | 55 个指标类，RAG/对话/安全开箱即用 |
| **多模型对比** | 原生支持，改一行 YAML 即矩阵 | 需自己写循环，逐个构建 |
| **提示词回归** | 主场能力（prompts 列表天然 A/B） | 可做但绕 |
| **红队测试** | ✅ 50+ 插件，OWASP 对齐 | ❌ 无 |
| **CI 门禁** | 退出码 + GitHub Actions 模板，最顺手 | pytest 风格可用，但非其设计重心 |
| **复杂业务判定** | YAML 表达不了，靠断言脚本绕行 | Python 自由编排（加权/状态机/门禁） |
| **结果形态** | 表格 + Web UI，横向对比友好 | 结果对象（pydantic），编程处理友好 |
| **多轮对话** | 弱（无对话级指标） | 较强（ConversationalTestCase + 25 个对话指标） |
| **生态** | Node 生态，社区大、迭代快 | Python 生态，评测学术派 |

### 5.2 选型决策

**选 promptfoo 当：**
- 需要频繁对比**多个 prompt 版本 / 多个模型**的效果（矩阵是它的主场）；
- 要把评测挂进 CI，做提示词改动的回归门禁；
- 有**安全红队**需求（越狱、注入、PII 探测）；
- 团队偏前端/Node 技术栈，不想写 Python。

**选 DeepEval 当：**
- 评 RAG / 多轮对话质量，需要忠实度、上下文召回这类**有研究背书的指标**；
- 判定逻辑复杂（加权综合分、一票否决门禁、状态机），要在 Python 里自由编排；
- 评测 pipeline 本身是 Python 工程的一部分（如本仓库：出题→作答→打分→报告全链路 Python 化）。

**两者混用也成立**：用 promptfoo 做 prompt/模型的矩阵预筛和 CI 门禁，用 DeepEval 做深度质量评估——同一套 CSV 测试集两边都能吃。

---

## 6. 常见问题排查（FAQ）

### 6.1 报错速查表

| 症状 | 原因 | 解法 |
|---|---|---|
| `Provider API key not found` | 环境变量没设或名字不对 | 按 provider 厂商核对约定命名（`OPENAI_API_KEY` 等）；自定义端点用 `apiKey: '{{env.XXX}}'` 显式引用 |
| 裁判断言全挂/超时 | `--grader` 没指定，或裁判端点不可达 | 显式 `--grader 厂商:模型`；国内环境把 grader 也指向可达端点（见 4.1） |
| API 限流大量 429 | 并发过高 | CLI 加 `--delay` 调大间隔；`--num-retries` 加大重试 |
| 改了 YAML 但行为没变 | 跑的不是这个配置文件 | `promptfoo eval -c` 显式指定路径；检查 `prompts:` 里文件引用是否指向旧文件 |
| CSV 用例没生效 | `tests:` 路径错误或列名与模板变量对不上 | 列名必须与 `{{var}}` 占位符完全一致（区分大小写） |
| 结果想复跑但分数没变 | 缓存命中 | `--no-cache` 强制重跑（裁判类断言缓存尤其容易"旧分复用"） |
| 跑大批量时大量 429 限流 | 裁判+被测双模型流量叠加压垮端点 | `-j N` 调并发、`--delay` 加间隔毫秒、`--num-retries` 加重试 |
| rubric 断言分数忽高忽低 | 裁判模型温度/随机性 | grader 选确定性强的模型；rubric 写具体可判定的标准，少写"通顺、合理"这类主观词 |

### 6.2 概念性疑问

**Q1：promptfoo 的 `llm-rubric` 和 DeepEval 的 GEval 是一回事吗？**
定位相同（自定义 LLM 评分标准），实现深度不同：GEval 是"框架帮你组织 evaluation_params + 内置成熟裁判模板"；`llm-rubric` 更朴素——你的 `value` 文本基本就是裁判看到的全部要求。迁移时把 GEval 的 criteria 搬过来当 value，通常要补几句"输出格式：只给 0~1 分"之类的指令才稳定。

**Q2：一条用例挂几条断言合适？**
与 DeepEval 指标同理：1 条确定性禁语断言 + 2~3 条裁判断言是甜区。断言越多裁判调用越多，成本线性上升。

**Q3：阈值（threshold）怎么定？**
同样靠标定：先跑一批看分数分布，人工复核 FAIL 样本定阈值。不同的是 promptfoo 结果天然按断言分列，标定时看表格即可，不用写聚合脚本。

**Q4：测试集从哪来？**
promptfoo 没有 Synthesizer 等价物，但 CSV 格式简单，DeepEval 工程里技能文档生成的测试集（本仓库 `eval/synthesize_cases.py`）导出 CSV 后可直接复用；也可用任意工具生成后对齐列名。

**Q5：promptfoo 的缓存和 DeepEval 的缓存有何异同？**
思路一致（按输入指纹复用结果，本仓库 README 明确借鉴了此设计）：命中即不再调 API。区别在于 promptfoo 缓存默认开启、跨"配置未变"的全部组合；正式出分前记得 `--no-cache`，或至少换输出目录避免误用旧缓存。

---

## 附录：学习资源索引

| 资源 | 内容 |
|---|---|
| 官方文档 `promptfoo.dev/docs` | 配置参考、断言大全、Provider 接入、红队指南（权威来源） |
| GitHub `promptfoo/promptfoo` | 源码、issue 检索报错、示例配置 |
| 本仓库 `project_notes/deepeval使用手册.md` | 对照阅读：同一评测问题的 DeepEval 解法 |
| 本仓库 `eval/synthesize_cases.py` | 测试集生成逻辑（产出 CSV 可迁移到 promptfoo） |
| 本仓库 `README.md` 缓存机制一节 | promptfoo 缓存设计思路在本仓库的落地 |

> 说明：本手册基于 promptfoo 0.121.x 官方文档编写，本仓库未实际安装运行该工具；落地前请以官方文档最新版本为准，重点核对断言字段名和 Provider 配置语法（两者历史上均有调整）。
