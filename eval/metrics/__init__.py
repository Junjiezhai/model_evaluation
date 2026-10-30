# -*- coding: utf-8 -*-
"""
指标包入口：按模式构建评测组件

  - build_single_turn_metrics(judge): DeepEval 标准指标（可用 EVAL_SINGLE_METRICS 过滤）
  - build_multi_turn_judge(judge_llm): 多轮 G1/G2 + Q1~Q5 评估器
  - build_state_validator(): 状态机验证器（YAML 驱动）
"""

import config


def build_single_turn_metrics(judge):
    """单轮：DeepEval 标准指标（默认 4 项；EVAL_SINGLE_METRICS 环境变量可按名过滤）"""
    from eval.metrics.single_turn import build_selected_metrics
    return build_selected_metrics(judge)


def build_multi_turn_judge(judge_llm):
    """多轮：G1/G2 门禁 + Q1~Q5 加权评估器"""
    from eval.metrics.multi_turn_judge import MultiTurnLLMJudge
    return MultiTurnLLMJudge(judge_llm)


def build_state_validator():
    """状态机验证器（strict/lenient/category，config.STATE_VALIDATION_MODE）"""
    from eval.metrics.state_validator import SessionStateValidator
    return SessionStateValidator()
