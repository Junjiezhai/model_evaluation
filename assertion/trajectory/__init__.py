# -*- coding: utf-8 -*-
"""轨迹层（trajectory）：断言对象是过程/路径。

- state_machine.py  会话状态机转移合规（YAML 矩阵驱动，strict/lenient/category）
- intent.py         逐轮意图断言 + 关键词规则分类器（7 类）
- engine.py         通用 YAML 规则引擎（对观测树做结构断言）
- rules.yaml        默认规则集（8 条：结构完整性/安全红线/循环预算/幂等/异步闭环）
- __main__          CLI：python -m assertion.trajectory（对模拟器/Cloud 数据批量断言）
"""
