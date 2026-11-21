# -*- coding: utf-8 -*-
"""断言包入口：按标准评测体系三层组织（端到端 → 轨迹 → 组件）。

分层判据是「断言对象」而非「实现方式」：
  e2e/        端到端层——最终回答/综合表现对不对（单轮指标、多轮 judge、数值一致性）
  trajectory/ 轨迹层——过程/路径合不合法（状态机、逐轮意图、YAML 规则引擎）
  component/  组件层——单次调用/检索/工具等部件好不好（骨架，随接入加深）

公共抽象在 core.py：AssertionResult / ASSERTION_REGISTRY / 报告渲染 / Langfuse sink。
层间依赖只允许向下（e2e 可消费轨迹层产出，反之不行）。

  - build_single_turn_metrics(judge): DeepEval 标准指标（可用 EVAL_SINGLE_METRICS 过滤）
  - build_multi_turn_judge(judge_llm): 多轮 G1/G2 + Q1~Q5 评估器
  - build_state_validator(): 状态机验证器（YAML 驱动）
"""


def build_single_turn_metrics(judge):
    """单轮：DeepEval 标准指标（默认 4 项；EVAL_SINGLE_METRICS 环境变量可按名过滤）"""
    from assertion.e2e.single_turn import build_selected_metrics
    return build_selected_metrics(judge)


def build_multi_turn_judge(judge_llm):
    """多轮：G1/G2 门禁 + Q1~Q5 加权评估器"""
    from assertion.e2e.multi_turn_judge import MultiTurnLLMJudge
    return MultiTurnLLMJudge(judge_llm)


def build_state_validator():
    """状态机验证器（strict/lenient/category，config.STATE_VALIDATION_MODE）"""
    from assertion.trajectory.state_machine import SessionStateValidator
    return SessionStateValidator()
