# -*- coding: utf-8 -*-
"""
DeepEval 自定义模型适配层：将 Kimi 接入 DeepEval 框架。

DeepEval 的指标（FaithfulnessMetric / AnswerRelevancyMetric / GEval 等）内部会调用
`model.generate(prompt)`，这里通过继承 DeepEvalBaseLLM 把调用桥接到
common.modelHttpAction.ModelHttpAction（统一 Kimi HTTP 客户端）。

同一个底层模型（kimi-for-coding）通过不同的 system_prompt / temperature
分饰多角：
  - 出题人（synthesizer）：生成测试问题变体
  - 问答助手（answer，被测系统）：信贷客服角色提示词
  - 评测裁判（judge，LLM-as-a-Judge）：公正评审提示词，temperature=0
"""

import asyncio
from typing import Optional

from deepeval.models import DeepEvalBaseLLM

from common.modelHttpAction import ModelHttpAction
import config


class KimiDeepEvalLLM(DeepEvalBaseLLM):
    """把 Kimi HTTP 客户端包装成 DeepEval 认可的 LLM 接口。"""

    def __init__(
        self,
        role: str = 'judge',
        model_name: str = 'kimi-for-coding',
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: int = 4096,
    ):
        """
        Args:
            role: 'answer'（问答助手）/ 'judge'（评测裁判）/ 其他自定义角色，仅用于标识
            model_name: 模型名（DeepEval 展示用）
            system_prompt: 角色提示词；None 时按 role 取 config 中的默认
            temperature: 采样温度；None 时按 role 取默认
            max_tokens: 最大生成 token 数
        """
        self.role = role
        self._model_name = f'{model_name}({role})'
        if system_prompt is None:
            system_prompt = (
                config.ANSWER_SYSTEM_PROMPT if role == 'answer' else config.JUDGE_SYSTEM_PROMPT
            )
        if temperature is None:
            temperature = 0.3 if role == 'answer' else config.JUDGE_TEMPERATURE
        self.system_prompt = system_prompt
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._client = ModelHttpAction()
        super().__init__(model_name=self._model_name)

    def load_model(self):
        """DeepEvalBaseLLM 要求实现：返回底层模型句柄。"""
        return self._client

    def generate(self, prompt: str, schema=None) -> str:
        """同步生成。DeepEval 指标评测时调用此方法。"""
        return self._client.chat(
            prompt=prompt,
            system_prompt=self.system_prompt,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            thinking_type=None,  # 评测/出题场景不使用思考模式，降低耗时
            desc=f'【Kimi-{self.role}】',
        )

    async def a_generate(self, prompt: str, schema=None) -> str:
        """异步生成。DeepEval 并发评测时调用；转线程执行实现真并发。"""
        return await asyncio.to_thread(self.generate, prompt, schema)

    def get_model_name(self) -> str:
        return self._model_name


def build_answer_llm() -> KimiDeepEvalLLM:
    """构建「智能问答助手」角色的模型实例（被测系统）。"""
    return KimiDeepEvalLLM(role='answer')


def build_judge_llm() -> KimiDeepEvalLLM:
    """构建「评测裁判」角色的模型实例（LLM-as-a-Judge）。"""
    return KimiDeepEvalLLM(role='judge')


def build_synthesizer_llm(system_prompt: str = None, temperature: float = 0.7) -> KimiDeepEvalLLM:
    """构建「出题人」角色的模型实例（测试用例合成）。

    Args:
        system_prompt: 生成指令；默认由调用方传入 skills/*.md 全文
        temperature: 合成需要多样性，默认 0.7
    """
    return KimiDeepEvalLLM(
        role='synthesizer',
        system_prompt=system_prompt or '你是一位资深的信贷业务测试用例设计专家。',
        temperature=temperature,
    )
