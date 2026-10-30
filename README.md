# 信贷知识库智能问答助手 · 模型评测框架

基于 DeepEval + 自研多轮流水线的信贷问答评测工程。参考 making_decison_test（pytest 框架）重构，
支持**单轮 / 多轮双流水线**、**技能文档驱动测试集生成**、**知识库注入作答**、**批量合并请求降本**、
**指纹缓存**、**可插拔指标**、**性能指标 + 雷达图报告**。

## 目录结构

```
├── config.py                        # 全部配置：路径 / 阈值 / 批量 / 缓存 / 权重
├── common/modelHttpAction.py        # Kimi HTTP 客户端（两项目共用，勿改）
├── utils/handle_log.py              # loguru 日志
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
│   └── metrics/                     # 指标包（可插拔）
│       ├── single_turn.py               # 单轮 4 标准指标（DeepEval，多阶段逐条核实）
│       ├── multi_turn_judge.py          # 多轮 G1/G2 门禁 + Q1~Q5 加权评分
│       ├── state_validator.py           # 会话状态机验证（YAML 驱动，strict/lenient/category）
│       ├── local_asserts.py             # 本地断言：数值一致性 + 意图规则分类（零成本）
│       └── custom_example.py            # ★ 自定义指标注释样例（GEval / BaseMetric 两条路线）
├── results/                         # 评测产物（结果 JSON / HTML 报告 / 作答缓存）
└── logs/                            # 执行日志（按日 txt，自动清理 30 天前的日志）
```

## 快速开始

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

参照 `eval/metrics/custom_example.py` 两条路线：

- **路线一（GEval）**：自定义评分标准，复用 DeepEval 裁判调用框架，如"拒答恰当性"
- **路线二（BaseMetric）**：完全掌控 `measure()`，可纯本地规则零成本，如"话术合规红线"

接入两步：① 注册进 `eval/metrics/single_turn.py` 的 `SINGLE_METRIC_REGISTRY`；
② 环境变量 `EVAL_SINGLE_METRICS=faithfulness,refusal_appropriateness` 或
`build_selected_metrics(judge, names)` 选用。批量降本用 `batch_runner.SingleTurnJudgeStrategy`。

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
  && python -m eval.synthesize_cases && python -m eval.metrics.state_validator \
  && python -m eval.metrics.local_asserts && python -m eval.metrics.multi_turn_judge \
  && python -m eval.metrics.custom_example && python -m eval.report
```
