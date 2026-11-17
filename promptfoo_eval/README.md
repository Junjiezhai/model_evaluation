# promptfoo 独立 benchmark 工作区

本目录是 **promptfoo 学习/实验专用目录：只跑公共评测集（GSM8K、C-Eval）**，与 DeepEval 信贷评测链路（`eval/`、`common/`、`config.py`、`results/`）**完全解耦**：

- 代码零耦合：本目录所有 Python 脚本不 import 项目内任何模块（`config` / `common` / `eval` 一概不引用），可用 grep 验证；
- 数据零共享：不读 `results/`、不读 `test_cases/`；数据集自包含在本目录 `benchmark_data/`；
- 凭据沿用根目录 `.env`（你的决策 ③）：promptfoo 启动时自动加载**当前工作目录**的 `.env`，
  因此所有命令**必须在仓库根目录执行**。

## 目录结构

```
promptfoo_eval/
├── benchmark/
│   ├── gsm8k.yaml / ceval.yaml      # 两个评测配置（prompts × provider × tests × 断言）
│   ├── prompts/                     # 题面模板（CoT / 5-shot / 0-shot）
│   ├── scripts/                     # 数据转换 + 判分断言（自包含）
│   ├── tests/                       # 生成的用例 CSV（gitignore）
│   └── outputs/                     # 评测结果 JSON（gitignore）
└── benchmark_data/                  # 原始数据集（gitignore，复现方式见其 README.md）
```

## 快速开始

```bash
# ① 生成用例（先小流量）
python promptfoo_eval/benchmark/scripts/build_gsm8k_tests.py --limit 300
python promptfoo_eval/benchmark/scripts/build_ceval_tests.py --limit-per-subject 50

# ② 评测（仓库根目录执行；先 --filter-first-n 20 验证链路，再去掉跑全量）
promptfoo eval -c promptfoo_eval/benchmark/gsm8k.yaml --filter-first-n 20
promptfoo eval -c promptfoo_eval/benchmark/ceval.yaml --filter-first-n 20

# ③ 查看（本地 Web UI：逐条看输出 / 判分理由 / pass rate）
promptfoo view
```

## 模型接入说明（与 DeepEval 底层调用保持一致）

两个 YAML 都使用 promptfoo 内置 `anthropic:` provider：

```yaml
providers:
  - id: anthropic:kimi-for-coding
    config:
      apiBaseUrl: https://api.kimi.com/coding      # = ModelHttpAction 预设 'kimi'
      apiKey: '{{env.KIMI_API_KEY}}'               # = 根目录 .env 的同一凭据
      temperature: 0
```

即与 `common/modelHttpAction.py` 走**同一端点、同一模型名、同一 Key**（Anthropic Messages 协议）。
差异仅在于 promptfoo 请求体不带 `cache_control` 与 `thinking` 字段——后者与 deepeval 评测场景
`thinking_type=None` 的口径一致；前者为 Kimi for Coding 网关的缓存标记，若小流量验证发现端点报错，
再考虑改用自定义 provider YAML（在请求模板里补该字段）。

## 判分口径（本地断言，零模型调用、零裁判成本）

| 数据集 | 断言脚本 | 口径 |
|---|---|---|
| GSM8K | `scripts/gsm8k_assert.py` | 抽取"答案：N"（取最后一次），兜底取末位数字；与 gold 数值相等即 PASS |
| C-Eval | `scripts/ceval_assert.py` | "答案：B" → 整行字母 → 首个字母，三级容错；与 gold 字母相等即 PASS |

**口径声明**：prompt 模板与官方榜单不一致，分数**不可与 C-Eval/GSM8K 榜单对比**；
本轨价值在于同一口径横向对比模型/提示词/参数。

## 成本控制

| 数据集 | 全量规模 | 体验建议 |
|---|---|---|
| GSM8K | 1319 题 × CoT 输出 | `--limit 300` 起步，链路 OK 再全量（约几十万 tokens） |
| C-Eval | 52 科目 12342 题 | 默认 3 科目；`--limit-per-subject 50` 起步；全量前先估算 token |

promptfoo 自带缓存：配置未变时重复跑不重复计费；正式出分前如担心缓存干扰，加 `--no-cache`。

## 依赖

- `pandas` + `pyarrow`：仅 `build_ceval_tests.py` 使用（读 parquet）；GSM8K 脚本纯标准库
- promptfoo CLI：全局已装 0.125.1（Node.js ≥ 18）
