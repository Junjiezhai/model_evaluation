# -*- coding: utf-8 -*-
"""端到端层（e2e）：断言对象是最终回答/综合表现。

- single_turn.py    DeepEval 标准指标（忠实度/相关性/召回/正确性 + 阈值）
- multi_turn_judge  多轮 G1/G2 门禁（一票否决）+ Q1~Q5 加权综合分
- deterministic.py  确定性结果断言：数值一致性（本地零成本）
- custom_example.py 自定义指标两条路线样例（GEval / BaseMetric）
"""
