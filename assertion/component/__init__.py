# -*- coding: utf-8 -*-
"""组件层（component）：断言对象是单次调用/检索/工具等部件的输出质量。

当前为骨架：信贷评测链路以端到端问答为主，组件级断言随真实 agent 接入加深后填充
（如 rc-credit-agent 的 LLM 调用级 generation 质量、检索命中率、工具参数 schema 校验）。
"""

from assertion.core import build_assertion_result, register, CRITICAL  # noqa: F401

# 导入即注册：组件层内置断言进 ASSERTION_REGISTRY
from assertion.component import generation, retrieval  # noqa: E402,F401
