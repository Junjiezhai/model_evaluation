# Langfuse 使用手册（LLM 可观测性学习轨）

> 本文档是 `../observability` 目录的权威文档。
> 目标： **不依赖真实 agent、不安装任何中间件**，通过构造 trace 与评测数据体验完整的 LLM 可观测平台。
> 全部代码已适配 Langfuse SDK v4 + Cloud（2026-11 版 API）。

---

## 一、是什么 & 为什么

本工作区在两条评测轨（DeepEval 信贷评测 / promptfoo 公共集）之上，增加了第三条轨：
**基于 Langfuse 的 LLM 可观测性体验轨**。它解决三个评测框架回答不了的问题：

| 问题 | 评测框架 | Langfuse |
|---|---|---|
| 单次调用结果对不对 | ✅ 出分 | ✅ score 关联 trace |
| **每次模型调用 prompt/response 原文、token、耗时** | ❌ 日志里散落 | ✅ generation 明细 |
| **多步链路怎么一步步走到这个结果的**（路由→决策→动作） | ❌ | ✅ span 树回放 |

设计特点：

- **双模式运行**：配置凭据 → 上报 Langfuse Cloud；未配置 → 本地控制台渲染 span 树。学习、调试、CI 零依赖。
- **三层数据源**：trace 模拟器（假 agent 拓扑）→ 真实模型调用埋点 → 评测结果导入，由浅入深。
- **观测模型字段对齐 `rc-credit-agent` 真实实现**（`node_path` / `decision_log` / `next_action_decision` / `fallback_reason` 等），学到的结构将来接真实 agent 时原样复用。

```
┌──────────────────────────────────────────────────────────┐
│   Layer 1  真实 LLM 调用埋点                                │
│    common/modelHttpAction.py::chat_messages                │
│    → 每次模型调用自动生成 generation（prompt/response/token）│
├──────────────────────────────────────────────────────────┤
│  Layer 2  trace 模拟器 observability/simulate_agent_traces │
│    → 5 种场景构造完整 agent span 树（含异步跨线程）           │
├──────────────────────────────────────────────────────────┤
│  Layer 3  评测结果导入 observability/import_eval_results   │
│    → case→trace、judge→score，打通「评测 ↔ 观测」闭环       │
└──────────────────────────────────────────────────────────┘
            ↓ 统一经 observability/langfuse_obs.py（双模式）
       Langfuse Cloud（jp 区域）/ 本地控制台渲染
```

---

## 二、环境与配置

### 2.1 依赖

```bash
pip install langfuse          # v4 SDK（已加入 requirements.txt 则跳过）
pip install python-dotenv     # 读取 .env（评测框架已用）
```

### 2.2 .env 配置

在项目根目录 `../.env` 追加（gitignored，不入库）：

```bash
# Langfuse 观测（体验轨）
LANGFUSE_ENABLED=true                          # false 时降级为本地渲染模式
LANGFUSE_PUBLIC_KEY=pk-lf-xxxx                 # Cloud → Project Settings → API Keys
LANGFUSE_SECRET_KEY=sk-lf-xxxx
LANGFUSE_HOST=                                 # 留空即可，SDK 按 key 自动解析区域
```

### 2.3 注册与拿 Key（一次性）

1. 打开 https://cloud.langfuse.com → 注册（GitHub / Google 账号均可）
2. 新建 Project（如 `rc-credit-eval`）
3. **Project Settings → API Keys** → 分别复制 public / secret key 填入 `../.env`

> **区域说明**：Langfuse Cloud 按项目分配区域（本工作区在 `jp.cloud.langfuse.com`）。
> `LANGFUSE_HOST` 留空时 SDK 自动解析，查询 API 时需显式用区域域名（见第六节）。

### 2.4 企业代理注意

公司代理（iOA）会拦截到 `*.langfuse.com` 的 TLS 连接（报 `SSLEOFError`）。
`langfuse_obs.py` 已在进程内把 `langfuse.com / .langfuse.com` 加入 `no_proxy` 白名单
（只影响本进程，不影响其他域名走代理），**无需手工设置环境变量**。

---

## 三、快速开始

```bash
# ① 模拟器灌入 5 种 agent 场景（约 10 秒）
python -m observability.simulate_agent_traces

# ② 导入最近一次评测结果（case 树 + judge 分数）
python -m observability.import_eval_results

# ③ （可选）真实 LLM 调用：generation 是真实 prompt/response，消耗 token
python -m observability.simulate_agent_traces --real --scenario normal_limit_increase
```

导入后直接打开 Cloud UI（见第四节）查看。不配置凭据时，以上命令在控制台输出等价的 span 树，流程完全一致。

---

## 四、UI 查看指南

Cloud 入口：https://cloud.langfuse.com（登录后自动跳转所属区域）。

### 4.1 Traces（span 树回放）

左侧 **Traces** → 按 `sessionId` 过滤：

| session | 内容 |
|---|---|
| `learning-session-001` | 模拟器 5 场景 |
| `eval-run-{run_id}` | 一次评测运行的全部 case（导入脚本生成） |

点进一条 trace 展开 span 树，重点观察：

- **generation 节点**：点开可见完整 prompt / response / token 用量 / model 参数——评测框架日志里没有的信息
- **场景对比学习**：
  - `turn.llm_fallback`：第一个 generation 为 ERROR（fast_primary 超时），第二个带 `fallback_from` / `fallback_reason` metadata → **模型降级链可视化**
  - `turn.max_loop_guard`：ERROR 级 `max_loop` 兜底 span → **循环保护触发可视**
  - `turn.async_gate`：`async_task.T-1001` 与在线 turn 挂在**同一 trace** 下 → **异步任务跨线程不断链**
  - `turn.guard_intercept`：`input_guard` 以 guardrail 类型拦截，链路短路 → **安全护栏可视**

### 4.2 Sessions（一次运行的聚合视图）

**Sessions** 页 → 点 `eval-run-{run_id}`：一次评测运行的所有 case 聚合在一起，
可整体看待"这批用例的模型表现"，与 `../results/index.html` 历史索引页互补（一个看分，一个看过程）。
该 session 由评测主入口**运行时自动产生**（见 6.4），不再需要先跑导入脚本。

### 4.3 Scores（评测 ↔ 观测闭环）

**Scores** 页按 name 过滤（断言体系已统一三层命名空间 `e2e.*` / `traj.*` / `comp.*`）：

| score name | 含义 | 来源 |
|---|---|---|
| `e2e.overall` | 多轮综合分（0-100） | judge |
| `e2e.gate.G1_compliance` / `e2e.gate.G2_complaint_control` | 底线门禁（CATEGORICAL: PASS/FAIL） | judge |
| `e2e.dim.Q1_context_memory` ~ `e2e.dim.Q5_consistency` | 五维均分 | judge |
| `e2e.Qx_xxx.turnN` / `e2e.Qx.turnN.reason` | 逐轮评分与理由 | judge |
| `traj.business_result` / `traj.degradation_chain` / `traj.loop_guard` / `traj.guardrail` / `traj.async_promise` | 模拟器场景结局（轨迹层） | 模拟器 |

点任意 score 可跳到对应 trace——**分数能下钻到产生它的完整调用链**，这是评测框架单独做不到的。

### 4.4 Dashboards（自建看板）

左侧 **Dashboards** → New：可按 model / session / metadata 字段聚合，
常用图表：token 成本按 model 分布、generation 延迟 P90、ERROR 率趋势。
体验阶段可跳过，接真实流量后再建。

---

## 五、调试与排障

### 5.1 三套组合拳

```bash
# ① 本地渲染模式：不上报，控制台看 span 树（结构排障最快）
#    把 .env 里 LANGFUSE_ENABLED 改为 false 后重跑任意命令

# ② SDK debug 日志：看每条上报请求的成功/失败原因
set LANGFUSE_DEBUG=true && python -m observability.simulate_agent_traces --scenario guard_intercept

# ③ API 抽查：脚本化验证落库（注意用 v2/v3，v1 已下线，见 5.2）
```

### 5.2 查询 API（v1 已于 2026-11-16 下线）

```python
import os, requests
from dotenv import load_dotenv

load_dotenv('../.env')
auth = (os.getenv('LANGFUSE_PUBLIC_KEY'), os.getenv('LANGFUSE_SECRET_KEY'))
host = 'https://jp.cloud.langfuse.com'  # 按项目区域填写

# 拉最近 1 小时的观测（traces 概念由 observations 按 traceId 分组得到）
# 注意：API 一律 UTC——startTime 是 UTC 时刻，换算北京时间 = +8 小时；
#       曾踩坑：按北京时间子串过滤 UTC 时间戳导致误判"数据没上报"
from datetime import datetime, timedelta, timezone

now = datetime.now(timezone.utc)
r = requests.get(host + '/api/public/v2/observations', auth=auth, params={
  'fields': 'core,basic,usage',
  'fromStartTime': (now - timedelta(hours=1)).strftime('%Y-%m-%dT%H:%M:%SZ'),
  'toStartTime': now.strftime('%Y-%m-%dT%H:%M:%SZ'),
  'limit': 100,
}, timeout=30)

# 拉 scores（v3）
r = requests.get(host + '/api/public/v3/scores', auth=auth, params={'limit': 50}, timeout=30)
```

### 5.3 常见错误速查表

| 现象 | 原因 | 处理 |
|---|---|---|
| 控制台出现 `Authentication error: ... Client will be disabled` | 未配置凭据 | 正常——自动降级本地渲染；要上报就补 `.env` |
| `ProxyError ... SSLEOFError` | 企业代理拦截 | 已内置 no_proxy 白名单；若升级 SDK 后复发，检查 `langfuse_obs._bypass_proxy_for_langfuse` 是否生效 |
| `401 Unauthorized`（SDK） | key 错/项目删了 | 重新生成 API key；确认 `LANGFUSE_ENABLED=true` |
| `400 ... Provide exactly one of: traceId, sessionId or datasetRunId` | score 归属冲突 | 已在 `langfuse_obs.score()` 内处理（有 trace 丢 session）；自行调 SDK 时注意 |
| `410 Gone`（查询 API） | 用了 v1 REST API | traces→`v2/observations`（按 traceId 分组），scores→`v3/scores`（见 5.2） |
| 数据延迟几秒不可见 |  ingestion 异步处理 | 正常，刷新即可；脚本末尾已 `flush()` |
| 查询"没数据"但 UI 有 | 时区换算错：API 时间戳是 UTC，北京时间 = UTC+8 | 用 `datetime.now(timezone.utc)` 构造窗口，别按北京时间子串过滤（见 5.2） |
| 脚本退出后尾部数据丢失 | 进程被杀未 flush | 交互/长驻进程在退出前调 `langfuse_obs.flush()` |

---

## 六、三层数据接入详解

### 6.1 Layer 2：trace 模拟器

```bash
python -m observability.simulate_agent_traces                  # 全部 5 场景
python -m observability.simulate_agent_traces --scenario llm_fallback
python -m observability.simulate_agent_traces --real           # think/分类用真实 LLM
```

| 场景 | 模拟的业务现实 | 学习点 |
|---|---|---|
| `normal_limit_increase` | 提额正常链路（主图 13 节点 + 子图 think-action ×2） | span 树基本结构、action 事件与幂等键 |
| `llm_fallback` | fast 主模型超时 → 备模型接管 | 降级链、ERROR 级 generation |
| `max_loop_guard` | 循环打满 `MAX_LIMIT_INCREASE_ACTION_LOOPS` 兜底 | 循环保护、status_message |
| `guard_intercept` | input_guard 命中拒答规则短路 | guardrail 类型、短链路 |
| `async_worker` | 异步 gate 命中，后台线程完成子图 | **ContextVar 跨线程传播**（真实接入最大坑） |

实现要点（都在源码注释里）：

- 节点名/metadata 与 `rc-credit-agent` 真实字段同名，接入真实 agent 时观测代码原样复用
- 异步场景用 `asyncio.to_thread`（自动拷贝 ContextVar 上下文）；**若用裸 `threading.Thread` 必须显式 `contextvars.copy_context()`，否则 span 断链**
- `--real` 模式下模拟器调 `common.modelHttpAction`（自带 Layer 1 埋点），span 树里会出现"业务编排 span + 客户端 generation"的分层结构

### 6.2 Layer 1：模型调用埋点

位置：`common/modelHttpAction.py::chat_messages`（对所有调用方透明生效，包括 DeepEval answer/judge、synthesizer、promptfoo 间接调用）。

每次调用上报一个 **generation**：

- `input/output`：完整 messages / 响应文本
- `usage_details`：input / output / total tokens（anthropic 与 openai 两种协议自动归一，见 `_normalize_usage`）
- `model_parameters`：temperature / max_tokens / thinking_type
- `metadata`：provider / api_type / desc（如 `【Kimi-judge】` 可区分裁判调用）
- 失败时 `level=ERROR` + `status_message`（含重试后最终错误）

**未配置凭据时零开销直通**（一次 `is_enabled()` 判断，不 import SDK）。

> 验证方式：跑一次评测，Langfuse Traces 里应出现大量 `llm.chat_messages` generation，
> 按 metadata.desc 可分出 answer / judge / synthesizer 三类调用，成本一目了然。

### 6.3 Layer 3：评测结果导入

```bash
python -m observability.import_eval_results            # 导入 results/evaluation_results.json
python -m observability.import_eval_results --logs     # 连同 logs/*.txt 一起（批量灌入演示）
```

映射规则：

| 评测数据 | Langfuse | 说明 |
|---|---|---|
| 每个 case | 一条 trace（chain） | `session_id = eval-run-{run_id}`，Traces/Sessions 双视角 |
| 每轮对话 | `round_N` span | input=问题，output=作答，metadata=意图/延迟/缓存命中 |
| `overall_score` / `dialogue_score` / `dimension_scores` | NUMERIC score | 挂 trace |
| `gate_checks` | CATEGORICAL score | PASS/FAIL + 判定理由放 comment |
| `per_turn_scores` | 逐轮 NUMERIC + reason (TEXT) | `Qx_xxx.turnN` |
| 日志行（`--logs`） | span（level 映射） | 演示批量灌入；真实场景应按 trace_id 关联 |

可重复执行（trace id 由 SDK 生成，重复导入产生新 trace；如需幂等可后续改固定 id）。

### 6.4 评测链路原生 trace 结构（run_evaluation 内建）

**不需要导入脚本**——`eval/run_evaluation.py` 执行时自动产生结构化观测（2026-11-21 起内建）：

```
session: eval-run-{run_id}                       ← eval_run() 注入，一次运行一个 session
├─ trace eval_case.MD_0001（chain）               ← 每个多轮 case 一条
│   ├─ round_1 span（input=问题, output=作答, metadata: cache_hit/latency/意图标注）
│   ├─ round_2 span ...
│   └─ round_3 span ...                          ← 缓存命中也留痕（cache_hit=True，无子 generation）
├─ trace judge.MD_0001（span）                    ← judge 独立 span（真实 generation 嵌在其下）
│   └─ generation llm.chat_messages（prompt/response/token 可下钻）
└─ trace eval_case.ST_0001（chain）               ← 单轮 case（作答 generation 嵌在其下）
```

- **score 回挂**：judge 分数在评测结束后按 `trace_id` 回写到对应 case trace
  （`e2e.overall` / `e2e.gate.*` CATEGORICAL / `e2e.dim.*` / `e2e.Qx.turnN` /
  单轮 `e2e.metric.*`），Scores 页点分数直达 case 轨迹
- **单轮批量模式**：合并请求无法拆到各 case，以 `eval_case.batch_group_N` 组级 chain 留痕
- **已知限制**：单轮 DeepEval `evaluate()` 在 worker 线程池里跑指标，线程不继承
  contextvars → judge generation 不在 case chain 下（仍在同 session）；分数已按 trace_id
  回挂，关联不丢。多轮链路全在主线程，结构完整
- 未配置凭据时全部降级为控制台渲染（本地看树），行为一致；`qa_system` 自测强制本地模式，不向 Cloud 灌数据

> 排查入口：UI → Sessions → `eval-run-{run_id}` 看本次运行全貌；点进 case trace 看逐轮
> 作答与 judge 判定过程。这就是"评测 run 在 Langfuse 里的样子"，替代了此前只能看散点
> `llm.chat_messages` 的局面。

---

## 七、批量轨迹合法性断言（assertion/trajectory）

解决"评分通过 ≠ 轨迹正确"：把轨迹合法性从人肉排查变成**声明式规则的批量检查**
（业界 trajectory evaluation 的确定性断言层——能写成规则的不交给 LLM judge）。

```bash
python -m assertion.trajectory                          # 本地模拟数据全场景（零副作用）
python -m assertion.trajectory --source langfuse        # 对 Cloud 数据检查，score 回写
```

- **规则文件** `assertion/trajectory/rules.yaml`：8 条规则覆盖结构完整性（R-001）、安全红线
  （R-002 护栏 BLOCK 后禁止动作）、循环预算（R-003）、决策质量（R-004）、幂等完整性（R-005）、
  异步闭环（R-006/007）、兜底可视（R-008）。规则可带 `when` 前置条件（不满足记 SKIP 不算违反），
  锚定**不变量**而非固定路径，避免误杀合法新路径。
- **双数据源**：`simulate`（capture() 捕获观测树，默认不上报）/ `langfuse`（v2 observations
  按 `parentObservationId` 重建 span 树，metadata 需显式加 `fields=...,metadata` 才返回）。
  两种源对同一批数据断言结果一致。
- **结果回写**：Cloud 源检查结果以 CATEGORICAL score `traj.assert`（PASS/FAIL）回写（断言体系三层命名空间：e2e.\*/traj.\*/comp.\*），
  UI → Scores 页按该名过滤，点 score 直达问题 trace。
- **CI 友好**：存在 critical 违规时退出码为 1。
- 模拟器含故意埋 bug 的场景 `guard_bypass_bug`（护栏 BLOCK 后仍执行 silent_apply），
  用于验证断言链路的抓捕能力——正常场景全过、bug 场景被 R-002 逮住才算系统有效。

## 八、往真实 agent 接入的迁移路径

观测模型已对齐 `rc-credit-agent`（`D:\工作\rc-credit-agent`），迁移时：

| 模拟器中的概念 | rc-credit-agent 真实接入点 |
|---|---|
| 根 trace（turn） | `agent/trace/reporter.py::report_turn_started/completed` 之间 |
| 节点 span（`_node()`） | `node_trace_context`（每个图节点已包 `_trace_node()`），把 `report_trace_event` 镜像为 span |
| generation（`_mock_generation`） | 无需手工——`agent/llm/client.py::_record_llm_trace` 处加 `input/output/usage` 即可（架构同 `modelHttpAction` 埋点） |
| action 事件（`_run_action`） | `LimitIncreaseActionLogService.record_action()` 双写，或从 `agent_action_log` 表 ETL |
| 异步场景 | `AsyncActionWorker`（`asyncio.to_thread`）已验证跨线程方案 |
| score | promptfoo/deepeval 结果经 `create_score` 回写（本文 Layer 3 即雏形） |

部署形态：体验用 Cloud；生产建议自托管（ClickHouse 后端可与现有观测链路同构运维），届时只改 `LANGFUSE_HOST`，代码零改动。

---

## 八、文件清单

```
observability/
├── langfuse_obs.py               # ★ 统一观测层：Langfuse 上报 / 本地渲染 / 轨迹捕获 三模式
│                                 #   observe() / score() / flush() / get_trace_url() / capture()
│                                 #   内置：.env 加载、代理白名单、score 归属约束处理
├── simulate_agent_traces.py      # trace 模拟器：6 场景（正常/降级/循环兜底/护栏/护栏绕过bug演示/异步）
├── import_eval_results.py        # 评测结果导入：case→trace、judge→score、日志批量灌入
└── （轨迹断言在 assertion/trajectory/：engine.py + rules.yaml + __main__.py，
    入口 python -m assertion.trajectory）
```
