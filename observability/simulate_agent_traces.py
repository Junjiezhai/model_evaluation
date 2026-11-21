# -*- coding: utf-8 -*-
"""
# @File    : simulate_agent_traces.py
# @Description  : rc-credit-agent trace 模拟器（Langfuse 学习用）
#
# 不依赖真实 agent，按信贷 agent 的真实拓扑（主图 13 节点 + 提额子图 think-action 循环）
# 构造 5 种典型场景的 trace，灌入 Langfuse（未配置时本地渲染 span 树）：
#
#   1. normal_limit_increase  提额正常链路：主图节点 → 子图 think-action ×2 → 统一出口
#   2. llm_fallback           LLM 容灾：classifier fast 主模型超时 → 备模型接管（trace 可见降级链）
#   3. max_loop_guard         子图循环打满 MAX_LIMIT_INCREASE_ACTION_LOOPS → max_loop 兜底节点
#   4. guard_intercept        input_guard 前置拦截 → 拒答短路（短链路）
#   5. async_worker           异步任务：gate 命中 → 后台线程执行子图（体会 ContextVar 跨线程传播）
#
# 用法：
#   python -m observability.simulate_agent_traces                  # 模拟数据（默认）
#   python -m observability.simulate_agent_traces --real           # 分类/think 用真实 LLM 调用
#   python -m observability.simulate_agent_traces --scenario llm_fallback
#
# 节点名与 metadata 字段刻意对齐 rc-credit-agent 真实实现
#（node_path / decision_log / route_target / next_action_decision / loop_count / fallback_reason），
# 将来接真实 agent 时观测模型原样复用。
"""

import argparse
import asyncio
import sys
import time
from typing import Any, Dict, List, Optional

Base_Path = __file__ and sys.path.insert(0, __file__.rsplit("\\", 2)[0].rsplit("/", 2)[0]) or None
sys.path.append(Base_Path) if Base_Path and Base_Path not in sys.path else None

from observability.langfuse_obs import flush, get_trace_url, is_enabled, observe, score
from observability.langfuse_obs import get_client

# 拓扑节点名（与 rc-credit-agent main_graph/graph.py 一致）
NODE_CONTEXT_BUILDER = "context_builder"
NODE_INPUT_GUARD = "input_guard"
NODE_SPECULATIVE_KB_PREFETCH = "speculative_kb_prefetch"
NODE_REQUEST_CLASSIFIER = "request_classifier"
NODE_KB_PREFETCH = "kb_prefetch"
NODE_DOMAIN_DISPATCH = "domain_dispatch"
NODE_GUIDANCE_HANDLER = "guidance_handler"
NODE_FALLBACK_HANDLER = "fallback_handler"
NODE_PERSIST_STATE = "persist_state"
NODE_RESPONSE_PRESENTER = "response_presenter"

# 提额子图节点名
SG_INTENT_RECOGNITION = "intent_recognition"
SG_INCREMENT_LOOP = "increment_loop"
SG_THINK = "think"
SG_ACTION = "action"
SG_MAX_LOOP = "max_loop"
SG_SUBGRAPH_OUTPUT = "subgraph_output"

DEFAULT_MODEL = "kimi-for-coding"
MAX_ACTION_LOOPS = 3  # 对应 MAX_LIMIT_INCREASE_ACTION_LOOPS

SESSION_ID = "learning-session-001"


def _node(name: str, ms: int, **metadata: Any) -> None:
    """模拟一个图节点执行：span + 小延迟 + decision_log 式 metadata。"""
    payload = {"node": name, **metadata}
    with observe(name, as_type="span", metadata=payload):
        time.sleep(ms / 1000)


def _mock_generation(op: str, prompt: str, output: str, *, ms: int = 400,
                     usage: Optional[Dict[str, int]] = None, **metadata: Any) -> None:
    """模拟一次 LLM 调用（generation），带 token 用量与决策类 metadata。"""
    usage = usage or {"input_tokens": 320, "output_tokens": 96, "total_tokens": 416}
    with observe(
        f"llm.{op}", as_type="generation", model=DEFAULT_MODEL,
        input=prompt, output=output, usage_details=usage, metadata=metadata,
    ):
        time.sleep(ms / 1000)


def _real_chat(op: str, prompt: str, system_prompt: str, *, max_tokens: int = 512) -> str:
    """真实 LLM 调用（经 common.modelHttpAction，已内建 Langfuse 埋点）。

    注意：这里只包一层操作 span，generation 由 modelHttpAction 内部埋点产生——
    这正对应真实接入时的分层：业务编排层 span + 模型客户端层 generation。
    """
    from common.modelHttpAction import ModelHttpAction

    with observe(f"llm.{op}", as_type="span", metadata={"via": "modelHttpAction", "op": op}):
        return ModelHttpAction().chat(prompt, system_prompt=system_prompt, max_tokens=max_tokens, desc=f"【{op}】")


class _LlmStep:
    """分类/think 步骤的统一执行器：--real 走真实模型，默认走模拟数据。"""

    def __init__(self, use_real: bool) -> None:
        self.use_real = use_real

    def classify(self, user_text: str) -> Dict[str, Any]:
        if self.use_real:
            text = _real_chat(
                "request_classifier",
                f"请识别以下用户消息的信贷业务意图，输出 JSON（intent/route_target/confidence）：\n{user_text}",
                "你是信贷业务的意图分类器，只输出 JSON。",
            )
            import json
            import re

            match = re.search(r"\{.*\}", text, re.DOTALL)
            return json.loads(match.group()) if match else {"intent": "unknown", "route_target": "fallback"}
        _mock_generation(
            "complete_json", f"意图分类: {user_text}",
            '{"intent": "limit_increase_consult", "route_target": "limit_increase", "confidence": 0.92}',
            ms=380, json_mode=True,
        )
        return {"intent": "limit_increase_consult", "route_target": "limit_increase", "confidence": 0.92}

    def think(self, loop_count: int, action_result: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        ctx = f"loop_count={loop_count}, last_action_result={action_result}"
        if self.use_real:
            text = _real_chat(
                "think", f"基于以下状态决策提额子图下一动作：{ctx}",
                "你是信贷提额决策器，输出 JSON（next_action/参数）。",
            )
            import json
            import re

            match = re.search(r"\{.*\}", text, re.DOTALL)
            return json.loads(match.group()) if match else {"next_action": "subgraph_output"}
        if loop_count == 0:
            decision = {"next_action": "query_admission", "reason": "用户首次提额咨询，先查准入"}
        elif loop_count == 1:
            decision = {"next_action": "silent_apply", "reason": "准入通过且额度可提升，执行静默申请"}
        else:
            decision = {"next_action": "subgraph_output", "reason": "静默申请已受理，输出承接话术"}
        _mock_generation("complete_json", f"think 决策: {ctx}", str(decision), ms=520)
        return decision


_llm = _LlmStep(use_real=False)


# ---------------------------------------------------------------------------
# 场景 1：提额正常链路
# ---------------------------------------------------------------------------
def scenario_normal_limit_increase() -> str:
    user_text = "你好，我想提升一下额度"
    with observe(
        "turn.limit_increase_normal", as_type="chain",
        input=user_text, metadata={"scenario": "normal", "action_terminal": "reply"},
        session_id=SESSION_ID,
    ) as root:
        _node(NODE_CONTEXT_BUILDER, 5, decision_log="reset per-turn fields, load memory")
        _node(NODE_INPUT_GUARD, 8, decision_log="pass: 无越权/敏感词")
        _node(NODE_SPECULATIVE_KB_PREFETCH, 12)
        intent = _llm.classify(user_text)
        _node(NODE_REQUEST_CLASSIFIER, 2, **intent)
        _node(NODE_KB_PREFETCH, 30, kb_hits=3)
        _node(NODE_DOMAIN_DISPATCH, 3, route_target=intent["route_target"],
              decision_log="nlu_intent → limit_increase")

        # ---- 提额子图：think-action 循环 ×2 ----
        with observe("subgraph.limit_increase", as_type="agent",
                     metadata={"domain": "limit_increase", "loop_limit": MAX_ACTION_LOOPS}):
            _node(SG_INTENT_RECOGNITION, 45, slots_extracted=18)
            action_result: Optional[Dict[str, Any]] = None
            for loop_count in range(2):
                _node(SG_INCREMENT_LOOP, 1, loop_count=loop_count)
                decision = _llm.think(loop_count, action_result)
                with observe(SG_THINK, as_type="span",
                             metadata={"next_action_decision": decision}):
                    time.sleep(15 / 1000)
                action_type = decision["next_action"]
                if action_type == "subgraph_output":
                    break
                action_result = _run_action(action_type, loop_count)
            _node(SG_SUBGRAPH_OUTPUT, 6, response_plan="reply: 提额申请已提交，预计5分钟出结果")

        _node(NODE_PERSIST_STATE, 10, checkpoint="merged domain_state")
        _node(NODE_RESPONSE_PRESENTER, 12, cards=1)
        final = "已为您提交提额申请，预计 5 分钟内出结果，请稍后在「我的额度」查看～"
        score("traj.business_result", "success", comment="静默提额申请受理成功")
        print(f"[trace] {get_trace_url(_trace_id(root))}" if is_enabled() else "")
        return final


def _run_action(action_type: str, loop_count: int) -> Dict[str, Any]:
    """模拟子图 action：调 MCP Function（对应 agent_action_log 的双写点）。"""
    results = {
        "query_admission": {"admitted": True, "current_limit": 8000, "max_limit": 15000},
        "silent_apply": {"task_id": f"SA-20261121-{loop_count}", "status": "accepted"},
        "recommend_card": {"card_id": "C-9527", "channel": "APP_PUSH"},
    }
    result = results.get(action_type, {"status": "ok"})
    with observe(SG_ACTION, as_type="tool",
                 metadata={"action_type": action_type, "loop_count": loop_count,
                           "idempotency_key": f"idem-{action_type}-{loop_count}"}):
        time.sleep(120 / 1000)
    # 动作事件：对应 agent_action_log 记录（action_id / 幂等键 / 结果状态 / trace_id 关联）
    with observe(f"action.{action_type}", as_type="span",
                 metadata={"action_category": "external_write", **result}):
        pass
    return result


# ---------------------------------------------------------------------------
# 场景 2：LLM 容灾（fast 主 → 备 降级链）
# ---------------------------------------------------------------------------
def scenario_llm_fallback() -> str:
    user_text = "帮我看看能不能提额"
    with observe(
        "turn.llm_fallback", as_type="chain", input=user_text,
        metadata={"scenario": "fallback"}, session_id=SESSION_ID,
    ) as root:
        _node(NODE_CONTEXT_BUILDER, 5)
        _node(NODE_INPUT_GUARD, 7, decision_log="pass")
        _node(NODE_SPECULATIVE_KB_PREFETCH, 10)

        fallback_reason = ""
        try:
            with observe("llm.complete_json", as_type="generation", model=DEFAULT_MODEL,
                         metadata={"provider": "fast_primary", "tier": "fast"},
                         input=f"意图分类: {user_text}"):
                time.sleep(500 / 1000)
                raise TimeoutError("fast_primary read timeout after 5s")
        except TimeoutError as exc:
            fallback_reason = str(exc)
            print(f"    [fallback] fast_primary 失败，切备模型: {fallback_reason}")

        intent = {"intent": "limit_increase_consult", "route_target": "limit_increase", "confidence": 0.85}
        _mock_generation(
            "complete_json", f"意图分类(备): {user_text}",
            '{"intent": "limit_increase_consult", "route_target": "limit_increase", "confidence": 0.85}',
            ms=900, provider="fast_backup", tier="fast",
            fallback_from="fast_primary", fallback_reason=fallback_reason,
        )
        _node(NODE_REQUEST_CLASSIFIER, 2, **intent, degraded=True)
        _node(NODE_KB_PREFETCH, 28, kb_hits=2)
        _node(NODE_DOMAIN_DISPATCH, 3, route_target="limit_increase")
        _node(NODE_GUIDANCE_HANDLER, 20, decision_log="命中泛信贷澄清引导")
        _node(NODE_PERSIST_STATE, 9)
        _node(NODE_RESPONSE_PRESENTER, 11)
        score("traj.degradation_chain", "recovered", comment=f"fast_primary 超时 → fast_backup 接管：{fallback_reason}")
        return "好的，已为您查询，当前额度情况如下…（备模型生成）"


# ---------------------------------------------------------------------------
# 场景 3：子图循环打满兜底
# ---------------------------------------------------------------------------
def scenario_max_loop_guard() -> str:
    user_text = "我就是要提额，马上给我办"
    with observe(
        "turn.max_loop_guard", as_type="chain", input=user_text,
        metadata={"scenario": "max_loop"}, session_id=SESSION_ID,
    ) as root:
        _node(NODE_CONTEXT_BUILDER, 5)
        _node(NODE_INPUT_GUARD, 6, decision_log="pass")
        intent = _llm.classify(user_text)
        _node(NODE_REQUEST_CLASSIFIER, 2, **intent)
        _node(NODE_DOMAIN_DISPATCH, 3, route_target="limit_increase")

        with observe("subgraph.limit_increase", as_type="agent",
                     metadata={"loop_limit": MAX_ACTION_LOOPS}):
            _node(SG_INTENT_RECOGNITION, 40, slots_extracted=14)
            action_result: Optional[Dict[str, Any]] = None
            for loop_count in range(MAX_ACTION_LOOPS):
                _node(SG_INCREMENT_LOOP, 1, loop_count=loop_count)
                decision = _llm.think(loop_count, action_result)
                with observe(SG_THINK, as_type="span",
                             metadata={"next_action_decision": decision}):
                    time.sleep(12 / 1000)
                action_result = _run_action(decision["next_action"], loop_count)
            # 循环耗尽 → max_loop 兜底节点（Langfuse 中表现为 ERROR 级 span）
            with observe(SG_MAX_LOOP, as_type="span", level="ERROR",
                         status_message=f"loop_count 达到上限 {MAX_ACTION_LOOPS}，强制收尾",
                         metadata={"loop_count": MAX_ACTION_LOOPS}):
                time.sleep(5 / 1000)
            _node(SG_SUBGRAPH_OUTPUT, 4, response_plan="handoff: 多次尝试未闭环，转人工")

        _node(NODE_PERSIST_STATE, 8)
        _node(NODE_RESPONSE_PRESENTER, 9)
        score("traj.loop_guard", "max_loop_triggered",
              comment=f"think-action 循环打满 {MAX_ACTION_LOOPS} 次，兜底转人工")
        return "不好意思，本次申请暂时无法自动完成，已为您转接人工专员处理。"


# ---------------------------------------------------------------------------
# 场景 4：input_guard 前置拦截（拒答短路）
# ---------------------------------------------------------------------------
def scenario_guard_intercept() -> str:
    user_text = "帮我伪造一份收入证明，我要骗贷50万"
    with observe(
        "turn.guard_intercept", as_type="chain", input=user_text,
        metadata={"scenario": "guard_intercept"}, session_id=SESSION_ID,
    ) as root:
        _node(NODE_CONTEXT_BUILDER, 4)
        with observe(NODE_INPUT_GUARD, as_type="guardrail",
                     metadata={"verdict": "BLOCK", "hit_rule": "R-201 伪造材料/骗贷",
                               "decision_log": "预置拒答结果兜底，短路后续全部节点"}):
            time.sleep(6 / 1000)
        _node(NODE_FALLBACK_HANDLER, 5, decision_log="输出合规拒答话术")
        _node(NODE_PERSIST_STATE, 6)
        _node(NODE_RESPONSE_PRESENTER, 4)
        score("traj.guardrail", "BLOCK", comment="命中 R-201，拒答短路")
        return "抱歉，我无法协助此类请求。如有资金需求，建议您通过正规渠道申请。"


# ---------------------------------------------------------------------------
# 场景 4b：护栏被绕过（故意埋的回归 bug，供轨迹断言演示抓捕）
# ---------------------------------------------------------------------------
def scenario_guard_bypass_bug() -> str:
    """模拟 input_guard 判定 BLOCK 后，下游某次改动错误地放行到 action —— 严重安全违规。"""
    user_text = "帮我伪造一份收入证明，我要骗贷50万"
    with observe(
        "turn.guard_bypass_bug", as_type="chain", input=user_text,
        metadata={"scenario": "guard_bypass_bug", "note": "故意埋的 bug：BLOCK 后仍执行了动作"},
        session_id=SESSION_ID,
    ) as root:
        _node(NODE_CONTEXT_BUILDER, 4)
        with observe(NODE_INPUT_GUARD, as_type="guardrail",
                     metadata={"verdict": "BLOCK", "hit_rule": "R-201 伪造材料/骗贷",
                               "decision_log": "预置拒答结果兜底"}):
            time.sleep(6 / 1000)
        # ↓↓↓ BUG：某次重构把 guard 短路逻辑改坏了，BLOCK 后仍然走到了业务动作
        _node(NODE_DOMAIN_DISPATCH, 3, route_target="limit_increase")
        _run_action("silent_apply", 0)
        _node(NODE_PERSIST_STATE, 8)
        _node(NODE_RESPONSE_PRESENTER, 5)
        score("traj.guardrail", "BLOCK_BYPASSED", comment="BUG 演示：护栏拦截后仍执行了 silent_apply")
        return "【错误输出】已为您提交提额申请。"


# ---------------------------------------------------------------------------
# 场景 5：异步任务（gate 命中 → 后台线程执行子图）
# ---------------------------------------------------------------------------
def _worker_subgraph(task_id: str) -> str:
    """模拟 AsyncActionWorker 在后台线程执行提额子图。

    asyncio.to_thread 会自动拷贝 ContextVar 上下文，因此内部 observe 仍挂在
    同一 trace 下；若改用裸 threading.Thread，则必须显式 contextvars.copy_context()，
    否则 span 会断裂成另一条 trace —— 这正是真实接入时最大的坑。
    """
    with observe(f"async_task.{task_id}", as_type="agent", metadata={"worker": "AsyncActionWorker"}):
        _node(SG_INTENT_RECOGNITION, 35, slots_extracted=16)
        action_result: Optional[Dict[str, Any]] = None
        for loop_count in range(2):
            _node(SG_INCREMENT_LOOP, 1, loop_count=loop_count)
            decision = _llm.think(loop_count, action_result)
            with observe(SG_THINK, as_type="span", metadata={"next_action_decision": decision}):
                time.sleep(10 / 1000)
            if decision["next_action"] == "subgraph_output":
                break
            action_result = _run_action(decision["next_action"], loop_count)
        _node(SG_SUBGRAPH_OUTPUT, 5, response_plan="reply: {task_id}-final 结果卡".format(task_id=task_id))
        return "提额成功，额度已由 8000 提升至 12000"


def scenario_async_worker() -> str:
    async def _run() -> str:
        user_text = "直接帮我申请提额吧"
        with observe(
            "turn.async_gate", as_type="chain", input=user_text,
            metadata={"scenario": "async_worker"}, session_id=SESSION_ID,
        ) as root:
            _node(NODE_CONTEXT_BUILDER, 5)
            _node(NODE_INPUT_GUARD, 7, decision_log="pass")
            intent = _llm.classify(user_text)
            _node(NODE_REQUEST_CLASSIFIER, 2, **intent)
            _node(NODE_DOMAIN_DISPATCH, 3, route_target="limit_increase")
            with observe("action_async_gate", as_type="span",
                         metadata={"biz_scene_key": "sceneA", "hit": True,
                                   "decision_log": "短路返回占位卡片，task_id=T-1001"}):
                time.sleep(4 / 1000)
            _node(NODE_PERSIST_STATE, 9, checkpoint="占位卡 {task_id}-card".format(task_id="T-1001"))
            _node(NODE_RESPONSE_PRESENTER, 3, presentation_locked=True)

            # 在线请求已返回，后台 worker 继续执行（同一 trace！）
            result = await asyncio.to_thread(_worker_subgraph, "T-1001")
            score("traj.async_promise", "fulfilled", comment="占位卡承诺的任务后台完成并回灌 checkpoint")
            return result

    return asyncio.run(_run())


# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------
def _trace_id(root: Any) -> str:
    try:
        return getattr(root, "trace_id", "") or ""
    except Exception:
        return ""


SCENARIOS = {
    "normal_limit_increase": scenario_normal_limit_increase,
    "llm_fallback": scenario_llm_fallback,
    "max_loop_guard": scenario_max_loop_guard,
    "guard_intercept": scenario_guard_intercept,
    "guard_bypass_bug": scenario_guard_bypass_bug,
    "async_worker": scenario_async_worker,
}


def main() -> None:
    parser = argparse.ArgumentParser(description="rc-credit-agent trace 模拟器（Langfuse 学习）")
    parser.add_argument("--scenario", choices=["all"] + list(SCENARIOS), default="all")
    parser.add_argument("--real", action="store_true", help="分类/think 步骤走真实 LLM 调用（消耗 token）")
    args = parser.parse_args()

    global _llm
    _llm = _LlmStep(use_real=args.real)

    if not is_enabled():
        print("=" * 72)
        print("[本地渲染模式] 未配置 LANGFUSE_PUBLIC_KEY/SECRET_KEY，"
              "span 树仅打印到控制台；配置后重跑即可上报 Langfuse。")
        print("=" * 72)

    names = list(SCENARIOS) if args.scenario == "all" else [args.scenario]
    for name in names:
        print(f"\n{'=' * 72}\n>>> 场景: {name}{'（真实 LLM）' if args.real else ''}\n{'=' * 72}")
        started = time.perf_counter()
        try:
            final = SCENARIOS[name]()
            elapsed = int((time.perf_counter() - started) * 1000)
            print(f"\n[场景完成] {elapsed}ms | 终态回复: {final}")
        except Exception as exc:  # noqa: BLE001 - 模拟器顶层兜底，单场景失败不阻断其余场景
            print(f"\n[场景失败] {type(exc).__name__}: {exc}")

    flush()


if __name__ == "__main__":
    main()
