# 模型评测工作区：信贷问答评测（DeepEval）+ 公共评测集体验（promptfoo）+ LLM 可观测性（Langfuse）

本仓库包含**两套互相独立、零代码耦合的评测轨**，外加一个横跨两轨的观测层：

| 轨道 | 工具 | 职责 |
|---|---|---|
| **信贷问答评测**（主） | DeepEval + 自研多轮流水线 | 信贷客服模型的正式出分：单轮 4 指标 / 多轮门禁加权、报告体系、缓存降本 |
| **公共评测集体验**（学习） | promptfoo | GSM8K / C-Eval benchmark 跑分，学习 promptfoo 用法；不依赖、不调用信贷链路任何代码 |
| **LLM 可观测性**（横切） | Langfuse | 每次模型调用的 prompt/response/token 明细、agent 链路 span 回放、评测 score 下钻；埋点在 `common/modelHttpAction.py`，对两条评测轨的所有调用自动生效 |

信贷轨基于 DeepEval + 自研多轮流水线的信贷问答评测工程。参考 making_decison_test（pytest 框架）重构，
支持**单轮 / 多轮双流水线**、**技能文档驱动测试集生成**、**知识库注入作答**、**批量合并请求降本**、
**指纹缓存**、**可插拔指标**、**性能指标 + 雷达图报告**。

promptfoo 轨见 `promptfoo_eval/`（[其 README](promptfoo_eval/README.md) 为该轨权威文档），
与信贷轨仅共享根目录 `.env` 的 Kimi 凭据，模型调用走**同一 Kimi for Coding 端点**，
其余零共享（不 import `config` / `common` / `eval`，不读 `results/` / `test_cases/`）。

## 目录结构

```
├── config.py                        # 全部配置：路径 / 阈值 / 批量 / 缓存 / 权重
├── debug_evaluation_by_deepeval.py  # ★ DeepEval IDE 调试入口（改 DebugConfig 后直接 Run）
├── common/modelHttpAction.py        # Kimi HTTP 客户端（两项目共用）★ 已内建 Langfuse generation 埋点
├── utils/handle_log.py              # loguru 日志
├── observability/                   # LLM 可观测性轨（Langfuse，详见 project_notes/langfuse使用手册.md）
│   ├── langfuse_obs.py              #   统一观测层：上报 / 本地渲染 / 轨迹捕获三模式（.env 加载、代理白名单）
│   ├── simulate_agent_traces.py     #   trace 模拟器：6 种 agent 场景（含故意埋 bug 的护栏绕过演示）
│   ├── import_eval_results.py       #   评测结果导入：case→trace、judge→score（打通评测↔观测）
├── project_notes/                   # ★ 学习手册（各轨权威文档之外的独立学习文档）
│   ├── deepeval使用手册.md            #   DeepEval 框架入门：核心能力 / 参数 / 标准用法 / 自定义扩展 / FAQ
│   ├── promptfoo使用手册.md           #   promptfoo 入门：与 deepeval 手册平行，便于对照阅读
│   ├── langfuse使用手册.md            #   Langfuse 权威文档：配置 / UI 指南 / 排障 / 三层接入 / 轨迹断言 / 迁移路径
│   └── rc-credit-agent架构设计.md     #   目标 agent（LangGraph 信贷）架构设计，观测字段对齐依据
├── promptfoo_eval/                  # promptfoo 独立 benchmark 轨（与信贷链路零耦合）
│   ├── README.md                    #   该轨权威文档（快速开始/模型接入/判分口径/成本控制）
│   ├── debug_benchmark_by_promptfoo.py  # ★ promptfoo IDE 调试入口：建库→评测→汇总→起查看器
│   ├── benchmark/                   #   gsm8k.yaml / ceval.yaml + prompts + scripts + tests + outputs
│   └── benchmark_data/              #   GSM8K + C-Eval 原始数据（gitignore，复现方式见其 README.md）
├── base_knowledge/                  # 基础知识素材
│   ├── credit_standard_QA_intent.md     # 标准问答知识库（检索注入用）
│   ├── intent_taxonomy.md               # 意图分类体系
│   ├── session_state_transition.md      # 会话状态转移规范（人读）
│   ├── session_state_matrix.yaml        # 状态转移矩阵（程序读，驱动状态机验证）
│   └── multi_turn_eval_dimensions.yaml  # 多轮 G/Q 维度评分细则（注入 judge prompt）
├── skills/                          # 用例生成技能文档（Claude Skill 格式）
│   ├── gen_single_turn_cases.md         # 单轮 10 列 CSV 生成规范
│   └── gen_multi_turn_cases.md          # 多轮 13 列 CSV 生成规范
├── test_cases/                      # 测试集（统一 CSV，utf-8-sig）
│   ├── single_turn_cases.csv            # 单轮：case_id/case_type/test_intent/test_dimension/
│   │                                    #   question/expected_answer/category/priority/expect_refuse/remark
│   └── multi_turn_cases.csv             # 多轮：每行一轮，13 列（含 state_intent 状态机标注）
├── eval/
│   ├── run_evaluation.py            # ★ 评测主入口（CLI）
│   ├── case_loader.py               # CSV 读取 / 校验 / 归一化（错误带行号聚合抛出）
│   ├── synthesize_cases.py          # --generate-testset 时按技能文档生成 CSV
│   ├── knowledge_base.py            # bigram 检索器（markdown 知识库 → 问答对）
│   ├── qa_system.py                 # 单/多轮作答链路：检索注入 + 批量 + 指纹缓存 + 耗时采集
│   ├── batch_runner.py              # 批量框架：合并请求 → 解析拆分 → 失败整组回退
│   ├── kimi_llm.py                  # Kimi × DeepEval 桥接（answer/judge/synthesizer 三角色）
│   ├── report.py                    # 单轮报告（含性能区块）+ 多轮报告（内联 SVG 雷达图）
├── assertion/                       # ★ 断言体系（按标准评测三层组织，eval/metrics 实现已迁入）
│   ├── core.py                      #   内核：AssertionResult / ASSERTION_REGISTRY / 报告渲染 / Langfuse sink
│   ├── e2e/                         #   端到端层：最终回答对不对
│   │   ├── single_turn.py           #     DeepEval 4 指标（阈值 config.METRIC_THRESHOLDS）
│   │   ├── multi_turn_judge.py      #     多轮 G1/G2 门禁 + Q1~Q5 加权 + aggregate_results
│   │   ├── deterministic.py         #     数值一致性（本地零成本，原 local_asserts 拆分）
│   │   └── custom_example.py        #     ★ 自定义指标注释样例（GEval / BaseMetric 两条路线）
│   ├── trajectory/                  #   轨迹层：过程/路径合不合法
│   │   ├── state_machine.py         #     状态机转移验证（原 state_validator.py，YAML 驱动）
│   │   ├── intent.py                #     逐轮意图断言 + 7 类规则分类器（原 local_asserts 拆分）
│   │   ├── engine.py + rules.yaml   #     YAML 规则引擎 + 8 条默认规则（结构/红线/预算/幂等/异步闭环）
│   │   └── __main__.py              #     CLI：python -m assertion.trajectory（CI 退出码语义）
│   └── component/                   #   组件层：单次调用/检索部件质量（骨架，随 agent 接入加深）
│       ├── generation.py            #     结构化输出合法率断言（已注册 ASSERTION_REGISTRY）
│       └── retrieval.py             #     检索命中率断言（已注册 ASSERTION_REGISTRY）
├── results/                         # 评测产物（结果 JSON / HTML 报告 / 作答缓存）
└── logs/                            # 执行日志（按日 txt，自动清理 30 天前的日志）
```

## 快速开始

> 框架本身的学习手册见 [project_notes/deepeval使用手册.md](project_notes/deepeval使用手册.md)
> （核心能力 / 参数说明 / 标准用法 / 自定义扩展 / pytest 对比 / FAQ）。

```bash
pip install -r requirements.txt

# 单轮评测（默认：读已有 CSV，非批量，缓存开启）
python -m eval.run_evaluation

# 多轮评测（前 5 个 case）
python -m eval.run_evaluation --mode multi --limit 5

# 单轮 + 多轮全量
python -m eval.run_evaluation --mode all

# 先重新生成测试集再评测（按 skills/*.md 技能文档，synthesizer 模型生成）
python -m eval.run_evaluation --generate-testset

# 批量模式：作答/judge 合并请求降本
python -m eval.run_evaluation --batch

# 其他参数
python -m eval.run_evaluation --force-answer     # 绕过作答缓存
python -m eval.run_evaluation --intent-llm       # 多轮意图分类走 LLM（默认本地规则）
```

IDE 调试：直接在 PyCharm 里 Run `debug_evaluation_by_deepeval.py`，
修改文件内 `DebugConfig(...)` 的参数即可，无需敲命令行。

## promptfoo 公共评测集轨（独立，详见 `promptfoo_eval/README.md`；框架学习手册见 [project_notes/promptfoo使用手册.md](project_notes/promptfoo使用手册.md)）

用公共 benchmark 数据集体验 promptfoo：GSM8K（数学推理，0-shot CoT）+ C-Eval（中文知识选择题，5-shot），
判分为本地 Python 断言（零裁判成本），分数口径**不可对标官方榜单**，价值在横向对比模型/提示词/参数。

```bash
pip install -r requirements.txt   # 含 pandas/pyarrow（仅 promptfoo 数据转换脚本使用）

# 方式一：IDE 一键（推荐）——建库 → 评测 → 汇总 → 自动起结果查看器并打印链接
python promptfoo_eval/debug_benchmark_by_promptfoo.py    # 改文件内 DebugConfig 后直接 Run

# 方式二：命令行分步
python promptfoo_eval/benchmark/scripts/build_gsm8k_tests.py --limit 300
promptfoo eval -c promptfoo_eval/benchmark/gsm8k.yaml --filter-first-n 20   # 须在仓库根目录执行
promptfoo view                                                              # Web UI 看明细
```

模型接入：promptfoo 内置 `anthropic:messages:kimi-for-coding` provider，与 `common/modelHttpAction.py`
**同一端点（api.kimi.com/coding）、同一模型、同一 `KIMI_API_KEY`**；数据集原始文件在
`promptfoo_eval/benchmark_data/`（gitignore，复现下载命令见其 README.md）。

## LLM 可观测性轨（Langfuse，权威文档见 [langfuse使用手册.md](project_notes/langfuse使用手册.md)）

三层数据接入，**未配置凭据时自动降级为控制台渲染 span 树**，学习零依赖：

```bash
pip install langfuse   # v4 SDK（需在 .env 配置 LANGFUSE_PUBLIC_KEY/SECRET_KEY，见手册第二节）

python -m observability.simulate_agent_traces      # ① trace 模拟器：5 种 agent 场景灌入平台
python -m observability.import_eval_results        # ② 评测结果导入：case→trace、judge→score
# ③ 模型调用埋点已内建于 common/modelHttpAction.py，跑任意评测即自动产生 generation 明细
```

Cloud UI 上：Traces 看 span 树回放（含每次调用的 prompt/response/token）、Sessions 按一次评测运行聚合、
Scores 下钻到产生该分数的完整调用链。查询 API 注意 v1 已于 2026-11-16 下线（traces→`v2/observations`、
scores→`v3/scores`，示例见手册 5.2）。观测模型字段对齐 rc-credit-agent 真实实现，将来接真实 agent 时代码原样复用。

**评测链路内建观测**（2026-11-21 起）：跑 `eval/run_evaluation.py` 时自动生成结构化 trace——
每个 case 一条 `eval_case.{case_id}` chain（多轮含逐轮 `round_N` span，缓存命中也留痕）、
judge 独立 span 嵌真实 generation、judge 分数按 `e2e.*` 命名空间回挂 case trace，
整次运行聚合在 session `eval-run-{run_id}` 下（手册 6.4）。UI → Sessions 即看本次评测全貌。

## 两个评测流水线

### 单轮（DeepEval 标准指标）

```
CSV 用例 → 知识库检索注入 → 被测模型作答 → 4 指标 LLM 评分 → 报告
```

默认 4 指标（阈值见 `config.METRIC_THRESHOLDS`）：Faithfulness 忠实度 /
AnswerRelevancy 答案相关性 / ContextualRecall 上下文召回 / GEval correctness 正确性。
DeepEval 内部对每条声明多阶段逐条核实（抑制 halo effect），**不参与批量合并**。

### 多轮（自研流水线，不经 deepeval.evaluate）

```
CSV 用例 → 逐轮对话执行（历史累积 + 逐轮计时）
        → 本地断言（数值一致性 + 逐轮意图分类）
        → 会话状态机验证（state_intent 标注的轮次）
        → LLM Judge（G1/G2 底线门禁一票否决 + Q1~Q5 加权综合分）
        → 雷达图报告
```

- **G1 业务合规 / G2 客诉控制**：任一 FAIL 综合分记 0；评分细则从
  `base_knowledge/multi_turn_eval_dimensions.yaml` 动态注入 judge prompt
- **Q1 上下文记忆 .25 / Q2 意图响应度 .25 / Q3 上下文污染度 .20 / Q4 对话连贯性 .15 /
  Q5 信息一致性 .15**（1-5 分映射 0-100，权重见 `config.MULTI_TURN_DIMENSION_WEIGHTS`）
- 状态机验证模式 `config.STATE_VALIDATION_MODE`：strict / lenient / category

## 缓存机制（业界主流，默认开启）

双层缓存，行为与 RAGAS / promptfoo / OpenAI evals 一致：

1. **作答缓存**（`results/qa_answers.json`）：key = `sha1(question|模型|prompt_version|top_k|temperature)`
   前 16 位指纹；多轮 key 额外含历史指纹（历史变化自动失效）。`--force-answer` 或改
   `config.PROMPT_VERSION` 即失效重答。
2. **DeepEval 指标缓存**：DeepEval 内置（`~/.deepeval`），同输入同指标不重复调用 judge。

## 自定义新指标（业务指标扩展）

参照 `assertion/e2e/custom_example.py` 两条路线：

- **路线一（GEval）**：自定义评分标准，复用 DeepEval 裁判调用框架，如"拒答恰当性"
- **路线二（BaseMetric）**：完全掌控 `measure()`，可纯本地规则零成本，如"话术合规红线"

接入两步：① 注册进 `assertion/e2e/single_turn.py` 的 `SINGLE_METRIC_REGISTRY`；
② 环境变量 `EVAL_SINGLE_METRICS=faithfulness,refusal_appropriateness` 或
`build_selected_metrics(judge, names)` 选用。批量降本用 `batch_runner.SingleTurnJudgeStrategy`。

本地规则类断言（非 DeepEval 指标）可注册进 `assertion/core.py` 的 `ASSERTION_REGISTRY`
（见 `assertion/component/generation.py` 的 `@register` 用法），自动获得统一报告与 Langfuse 出口。

## 报告产物（results/）

每次评测运行生成一个历史快照文件夹 `history/{YYYYMMDD_HHMMSS}/`（结果 JSON + HTML 报告 +
run_meta.json 配置快照），同时更新 `results/` 根下的 latest 副本；满 30 天的快照自动清理
（`config.RESULTS_HISTORY_RETENTION_DAYS` 可调）。

| 文件 | 内容 |
|---|---|
| `index.html` | **历史索引页（类 Allure 首页）**：单轮通过率/多轮均分/平均耗时趋势折线图 + 历史运行表（点击跳转各次快照报告与 JSON），每次运行结束自动重建 |
| `evaluation_results.json` | 统一 schema：mode/model/answer_model/single_turn{cases,perf}/multi_turn{cases,aggregate,perf} |
| `evaluation_report.html` | 单轮：总览 / 指标分布 / 分类聚合 / **性能区块（平均·P90·P95·最大·最小）** / 失败置顶 / 明细耗时 |
| `multi_turn_report.html` | 多轮：总览 / **Q1~Q5 雷达图（内联 SVG）** / G1-G2 门禁表 / 维度得分 / 状态转移合规 / 意图混淆矩阵 / 性能 / 失败 case 折叠明细 |
| `history/{run_id}/run_meta.json` | 当次运行的配置快照：CLI 参数 + 阈值/权重/模式等 11 项 config + 用例数 + 通过率/聚合分 + 耗时 |

## 模块自测

所有模块支持离线自测（不调 LLM）：

```bash
python -m eval.case_loader && python -m eval.batch_runner && python -m eval.qa_system \
  && python -m eval.synthesize_cases && python -m eval.report \
  && python -m assertion.e2e.single_turn && python -m assertion.e2e.custom_example \
  && python -m assertion.e2e.deterministic && python -m assertion.trajectory.intent \
  && python -m assertion.trajectory.state_machine \
  && python -m assertion.component.generation && python -m assertion.component.retrieval
```
