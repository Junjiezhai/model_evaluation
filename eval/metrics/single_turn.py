# -*- coding: utf-8 -*-
"""
单轮评测指标构建

默认 4 个 DeepEval 标准指标（多阶段逐条核实，保留 DeepEval 内置算法以抑制 halo effect）：
  - FaithfulnessMetric      忠实度（抽声明→逐条核实）
  - AnswerRelevancyMetric   答案相关性（生成扰动问题比对）
  - ContextualRecallMetric  上下文召回（逐句归因）
  - GEval(correctness)      答案正确性（对照标准答案）

新增自定义单轮指标：
  1. 在 eval/metrics/custom_example.py 参照样例实现（GEval 或 BaseMetric 路线）
  2. 在本文件 build_metrics() 中按同样模式追加，或注册进 SINGLE_METRIC_REGISTRY
"""

import os
from typing import List

from deepeval.metrics import FaithfulnessMetric, AnswerRelevancyMetric, ContextualRecallMetric, GEval
from deepeval.test_case import SingleTurnParams

import config


def build_default_metrics(judge) -> List:
    """构建默认 4 个单轮指标（阈值取自 config.METRIC_THRESHOLDS）"""
    th = config.METRIC_THRESHOLDS
    return [
        FaithfulnessMetric(threshold=th['faithfulness'], model=judge, include_reason=True),
        AnswerRelevancyMetric(threshold=th['answer_relevancy'], model=judge, include_reason=True),
        ContextualRecallMetric(threshold=th['contextual_recall'], model=judge, include_reason=True),
        GEval(
            name='correctness',
            criteria=(
                '对照标准答案，评估候选答案的事实正确性：关键信息（条件、数字、流程、结论）'
                '是否与标准答案一致；允许换一种说法，但不得增加标准答案中没有的事实，'
                '也不得遗漏标准答案中的关键约束。'
            ),
            evaluation_params=[
                SingleTurnParams.INPUT,
                SingleTurnParams.ACTUAL_OUTPUT,
                SingleTurnParams.EXPECTED_OUTPUT,
            ],
            threshold=th['correctness'],
            model=judge,
        ),
    ]


# ---- 指标注册表：name -> 构造函数（judge) -> metric ----
# 自定义指标实现后在此注册，即可被 build_selected_metrics() 按名字挑选。
SINGLE_METRIC_REGISTRY = {
    'faithfulness': lambda judge: FaithfulnessMetric(
        threshold=config.METRIC_THRESHOLDS['faithfulness'], model=judge, include_reason=True),
    'answer_relevancy': lambda judge: AnswerRelevancyMetric(
        threshold=config.METRIC_THRESHOLDS['answer_relevancy'], model=judge, include_reason=True),
    'contextual_recall': lambda judge: ContextualRecallMetric(
        threshold=config.METRIC_THRESHOLDS['contextual_recall'], model=judge, include_reason=True),
    'correctness': lambda judge: GEval(
        name='correctness',
        criteria=(
            '对照标准答案，评估候选答案的事实正确性：关键信息（条件、数字、流程、结论）'
            '是否与标准答案一致；允许换一种说法，但不得增加标准答案中没有的事实，'
            '也不得遗漏标准答案中的关键约束。'
        ),
        evaluation_params=[SingleTurnParams.INPUT, SingleTurnParams.ACTUAL_OUTPUT,
                           SingleTurnParams.EXPECTED_OUTPUT],
        threshold=config.METRIC_THRESHOLDS['correctness'],
        model=judge),
}


def build_selected_metrics(judge, names: str = None) -> List:
    """按名字挑选指标；names 为空时取环境变量 EVAL_SINGLE_METRICS（逗号分隔），
    未配置则返回默认全量 4 指标。"""
    if names is None:
        names = os.getenv('EVAL_SINGLE_METRICS', '')
    if not names.strip():
        return build_default_metrics(judge)
    metrics = []
    for name in [n.strip() for n in names.split(',') if n.strip()]:
        if name not in SINGLE_METRIC_REGISTRY:
            raise KeyError(f'未知指标 "{name}"，可选: {list(SINGLE_METRIC_REGISTRY)}')
        metrics.append(SINGLE_METRIC_REGISTRY[name](judge))
    return metrics
