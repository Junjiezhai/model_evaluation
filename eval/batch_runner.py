# -*- coding: utf-8 -*-
"""
批量合并请求运行框架

"批量 = 合并请求降本"：把多条用例合并为一次 LLM 请求（JSON 结构化输出），
解析拆分后回填各条结果；任一组调用/解析失败时整组回退为逐条处理，保证鲁棒性。

内置策略：
  - SingleTurnAnswerStrategy  单轮作答合并（N 个问题 → 一次请求返回 N 个答案）
  - MultiTurnJudgeStrategy    多轮 Judge 合并（多个 case 的评分对象一次返回）
  - SingleTurnJudgeStrategy   单轮 Judge 合并（示例：供自定义/业务指标使用；
                              默认单轮流水线的 4 个标准指标由 DeepEval 内部并发管理，
                              不参与合并，以保留其多阶段逐条核实算法）

使用方式：
    from eval.batch_runner import BatchRunner, SingleTurnAnswerStrategy

    runner = BatchRunner(answer_llm, SingleTurnAnswerStrategy(group_size=8))
    records = runner.run(items)   # items: List[dict]，返回各条的作答结果
"""

import json
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from utils.handle_log import log


class BatchParseError(Exception):
    """批量响应解析失败（触发整组回退）"""


class BatchStrategy(ABC):
    """批量策略基类：组 prompt 构建 + 组响应解析 + 单条回退"""

    def __init__(self, group_size: int = 8):
        self.group_size = group_size

    @abstractmethod
    def build_group_prompt(self, items: List[dict]) -> str:
        """把组内条目合并为一个 prompt（须要求模型输出与组大小一致的 JSON 数组/对象）"""
        raise NotImplementedError

    @abstractmethod
    def parse_group_response(self, data: Any, items: List[dict]) -> List[Any]:
        """把结构化响应拆分回与 items 一一对应的结果列表；不完整时抛 BatchParseError"""
        raise NotImplementedError

    @abstractmethod
    def fallback_one(self, item: dict, runner: 'BatchRunner') -> Any:
        """整组回退时逐条处理单个条目"""
        raise NotImplementedError

    def group_key(self, item: dict) -> Optional[str]:
        """分组键：相同 key 的条目进同一组（默认不分组）"""
        return None


class BatchRunner:
    """批量执行器：分组 → 合并请求（json_schema 结构化输出）→ 解析拆分 → 失败回退"""

    def __init__(self, llm, strategy: BatchStrategy):
        """
        Args:
            llm: 具备 generate(prompt)->str 或底层 client 的对象。
                 优先使用 llm.client（ModelHttpAction）；否则要求 llm 有 generate 方法。
            strategy: 批量策略实例
        """
        self.llm = llm
        self.strategy = strategy
        # KimiDeepEvalLLM 持有 _client；自定义桩对象可直接是 client
        self.client = getattr(llm, '_client', llm)
        self.stats = {'groups': 0, 'group_ok': 0, 'group_fallback': 0, 'items': 0}

    def run(self, items: List[dict]) -> List[Any]:
        """执行批量处理，返回与 items 等长的结果列表（回退条目由 fallback_one 产出）"""
        if not items:
            return []
        self.stats['items'] = len(items)

        groups = self._group_items(items)
        results: List[Any] = [None] * len(items)

        for group in groups:
            self.stats['groups'] += 1
            group_items = [items[i] for i in group]
            try:
                merged = self.strategy.build_group_prompt(group_items)
                data = self.client.chat_with_json_output(
                    merged,
                    json_schema=self.strategy_output_schema(),
                    system_prompt=self.strategy_system_prompt(),
                    max_tokens=8192,
                    desc=f'【批量调用】组大小={len(group)}',
                )
                parsed = self.strategy.parse_group_response(data, group_items)
                if len(parsed) != len(group):
                    raise BatchParseError(
                        f'解析结果数量 {len(parsed)} 与组大小 {len(group)} 不一致')
                for idx, value in zip(group, parsed):
                    results[idx] = value
                self.stats['group_ok'] += 1
            except Exception as e:  # noqa: BLE001 - 任一组失败整组回退，不中断整体
                self.stats['group_fallback'] += 1
                log.warning(f'【批量调用】第 {self.stats["groups"]} 组失败（{e}），整组回退逐条处理（{len(group)} 条）')
                for idx in group:
                    try:
                        results[idx] = self.strategy.fallback_one(items[idx], self)
                    except Exception as fe:  # noqa: BLE001
                        log.error(f'【批量调用】逐条回退也失败（第 {idx} 条）：{fe}')
                        results[idx] = None

        log.info(
            f'【批量调用】完成：组 {self.stats["group_ok"]}/{self.stats["groups"]} 成功，'
            f'回退 {self.stats["group_fallback"]} 组，共 {self.stats["items"]} 条'
        )
        return results

    # ---- 供策略覆写的钩子 ----

    def strategy_output_schema(self) -> Optional[dict]:
        """请求级 JSON Schema；None 时由客户端默认行为处理"""
        return getattr(self.strategy, 'output_schema', None)

    def strategy_system_prompt(self) -> Optional[str]:
        return getattr(self.strategy, 'system_prompt', None)

    # ---- 内部 ----

    def _group_items(self, items: List[dict]) -> List[List[int]]:
        """按 group_key 分桶 + 按 group_size 切分，返回索引组列表"""
        buckets: Dict[Any, List[int]] = {}
        for idx, item in enumerate(items):
            key = self.strategy.group_key(item)
            buckets.setdefault(key, []).append(idx)

        groups: List[List[int]] = []
        for bucket in buckets.values():
            for i in range(0, len(bucket), self.strategy.group_size):
                groups.append(bucket[i:i + self.strategy.group_size])
        return groups


# ============================================================
# 内置策略一：单轮作答合并
# ============================================================

class SingleTurnAnswerStrategy(BatchStrategy):
    """
    单轮作答合并策略：一次请求回答 N 个问题。

    要求被测模型返回 JSON 对象 {"answers": [{"index": 1, "answer": "..."}, ...]}，
    index 从 1 开始，与组内顺序一致。
    """

    output_schema = {
        'type': 'object',
        'properties': {
            'answers': {
                'type': 'array',
                'items': {
                    'type': 'object',
                    'properties': {
                        'index': {'type': 'integer'},
                        'answer': {'type': 'string'},
                    },
                    'required': ['index', 'answer'],
                },
            },
        },
        'required': ['answers'],
    }

    def __init__(self, group_size: int = 8, questions_key: str = 'input'):
        super().__init__(group_size)
        self.questions_key = questions_key

    def build_group_prompt(self, items: List[dict]) -> str:
        lines = [
            '你是一名信贷客服助手。请依次回答下面 {} 个用户问题。'.format(len(items)),
            '回答要求：严格依据提供的知识库内容，不得编造数字；每个回答独立完整。',
            '请严格按 JSON 格式输出：{"answers": [{"index": 序号, "answer": 回答}, ...]}，序号从 1 开始。',
            '',
            '【问题列表】',
        ]
        for i, item in enumerate(items, start=1):
            lines.append(f'{i}. {item[self.questions_key]}')
        return '\n'.join(lines)

    def parse_group_response(self, data: Any, items: List[dict]) -> List[str]:
        answers = data.get('answers') if isinstance(data, dict) else None
        if not isinstance(answers, list):
            raise BatchParseError(f'响应缺少 answers 数组: {str(data)[:200]}')
        by_index = {}
        for entry in answers:
            if isinstance(entry, dict) and 'index' in entry:
                by_index[int(entry['index'])] = str(entry.get('answer', '')).strip()
        ordered = []
        for i in range(1, len(items) + 1):
            if i not in by_index or not by_index[i]:
                raise BatchParseError(f'响应缺少第 {i} 条答案（共返回 {len(answers)} 条）')
            ordered.append(by_index[i])
        return ordered

    def fallback_one(self, item: dict, runner: 'BatchRunner') -> Any:
        # 回退：走正常单轮作答链路（由 qa_system.answer_single_turn 在批量失败后重试时处理）
        raise BatchParseError('单轮作答回退由调用方逐条链路处理')


# ============================================================
# 内置策略二：多轮 Judge 合并
# ============================================================

class MultiTurnJudgeStrategy(BatchStrategy):
    """
    多轮 Judge 合并策略：一次请求返回多个 case 的 G/Q 评分对象。

    要求返回 {"results": [{"case_index": 1, "score_object": {...}}, ...]}，
    score_object 的结构由具体 judge（multi_turn_judge.MultiTurnLLMJudge）定义。
    """

    output_schema = {
        'type': 'object',
        'properties': {
            'results': {
                'type': 'array',
                'items': {
                    'type': 'object',
                    'properties': {
                        'case_index': {'type': 'integer'},
                        'score_object': {'type': 'object'},
                    },
                    'required': ['case_index', 'score_object'],
                },
            },
        },
        'required': ['results'],
    }

    def __init__(self, judge_prompt_builder, group_size: int = 2):
        """
        Args:
            judge_prompt_builder: Callable[[List[dict]], str]
                由 MultiTurnLLMJudge 注入；把多个 case 的评估材料合并为一个评估 prompt
        """
        super().__init__(group_size)
        self.judge_prompt_builder = judge_prompt_builder

    def build_group_prompt(self, items: List[dict]) -> str:
        return self.judge_prompt_builder(items)

    def parse_group_response(self, data: Any, items: List[dict]) -> List[dict]:
        results = data.get('results') if isinstance(data, dict) else None
        if not isinstance(results, list):
            raise BatchParseError(f'响应缺少 results 数组: {str(data)[:200]}')
        by_index = {}
        for entry in results:
            if isinstance(entry, dict) and 'case_index' in entry:
                by_index[int(entry['case_index'])] = entry.get('score_object')
        ordered = []
        for i in range(1, len(items) + 1):
            if i not in by_index or not isinstance(by_index[i], dict):
                raise BatchParseError(f'响应缺少第 {i} 个 case 的评分对象')
            ordered.append(by_index[i])
        return ordered

    def fallback_one(self, item: dict, runner: 'BatchRunner') -> Any:
        # 回退：由 MultiTurnLLMJudge.evaluate() 单条链路处理
        raise BatchParseError('多轮 judge 回退由调用方逐条链路处理')


# ============================================================
# 内置策略三：单轮 Judge 合并（自定义/业务指标示例）
# ============================================================

class SingleTurnJudgeStrategy(BatchStrategy):
    """
    单轮 Judge 合并策略（示例）。

    默认单轮流水线的 4 个标准指标（Faithfulness 等）由 DeepEval 内部并发管理，
    保留其"多阶段逐条核实"算法，不参与合并；本策略供**自定义业务指标**批量降本使用，
    典型场景：逐条检查清单评分（拒答恰当性、话术合规性等）。
    使用示例见 eval/metrics/custom_example.py。
    """

    output_schema = {
        'type': 'object',
        'properties': {
            'judgements': {
                'type': 'array',
                'items': {
                    'type': 'object',
                    'properties': {
                        'index': {'type': 'integer'},
                        'pass': {'type': 'boolean'},
                        'reason': {'type': 'string'},
                    },
                    'required': ['index', 'pass', 'reason'],
                },
            },
        },
        'required': ['judgements'],
    }

    def __init__(self, rubric: str, group_size: int = 10,
                 question_key: str = 'input', answer_key: str = 'actual_output'):
        """
        Args:
            rubric: 评判标准描述（注入 prompt，作为统一评分依据）
        """
        super().__init__(group_size)
        self.rubric = rubric
        self.question_key = question_key
        self.answer_key = answer_key

    def build_group_prompt(self, items: List[dict]) -> str:
        lines = [
            '你是信贷问答质量评审专家。请按以下标准逐条评判 {} 个候选答案：'.format(len(items)),
            f'【评判标准】{self.rubric}',
            '请严格按 JSON 输出：{"judgements": [{"index": 序号, "pass": true/false, "reason": "理由"}, ...]}',
            '',
            '【待评判条目】',
        ]
        for i, item in enumerate(items, start=1):
            lines.append(f'--- 第 {i} 条 ---')
            lines.append(f'问题：{item.get(self.question_key, "")}')
            lines.append(f'候选答案：{item.get(self.answer_key, "")}')
        return '\n'.join(lines)

    def parse_group_response(self, data: Any, items: List[dict]) -> List[dict]:
        judgements = data.get('judgements') if isinstance(data, dict) else None
        if not isinstance(judgements, list):
            raise BatchParseError(f'响应缺少 judgements 数组: {str(data)[:200]}')
        by_index = {}
        for entry in judgements:
            if isinstance(entry, dict) and 'index' in entry:
                by_index[int(entry['index'])] = {
                    'pass': bool(entry.get('pass')),
                    'reason': str(entry.get('reason', '')),
                }
        ordered = []
        for i in range(1, len(items) + 1):
            if i not in by_index:
                raise BatchParseError(f'响应缺少第 {i} 条评判结果')
            ordered.append(by_index[i])
        return ordered

    def fallback_one(self, item: dict, runner: 'BatchRunner') -> Any:
        raise BatchParseError('单轮 judge 回退由调用方逐条链路处理')


# ============================================================
# 自测（stub client，不调用真实 LLM）
# ============================================================

if __name__ == '__main__':
    print('=' * 60)
    print('  batch_runner 自测（stub client）')
    print('=' * 60)

    class _StubClient:
        """模拟 ModelHttpAction：按 prompt 内容返回预设 JSON"""

        def __init__(self, mode='ok'):
            self.mode = mode
            self.calls = 0

        def chat_with_json_output(self, prompt, **kwargs):
            self.calls += 1
            if self.mode == 'ok':
                # 从 prompt 中数出条目数，返回同数量的答案
                n = prompt.count('--- 第') if '--- 第' in prompt else max(
                    1, len([l for l in prompt.splitlines() if l[:2] in
                            {f'{i}.' for i in range(1, 100)}]))
                if '【问题列表】' in prompt:
                    n = len([l for l in prompt.splitlines()
                             if l and l[0].isdigit() and '. ' in l[:4]])
                    return {'answers': [{'index': i, 'answer': f'答案{i}'} for i in range(1, n + 1)]}
                if '【待评判条目】' in prompt:
                    n = prompt.count('--- 第')
                    return {'judgements': [{'index': i, 'pass': True, 'reason': 'ok'}
                                           for i in range(1, n + 1)]}
                raise ValueError('未知 prompt 类型')
            if self.mode == 'partial':
                # 只返回一半结果，触发解析失败
                return {'answers': [{'index': 1, 'answer': '答案1'}]}
            raise RuntimeError('模拟网络错误')

    # ---- 1. 单轮作答：正常拆分 ----
    client = _StubClient('ok')
    strategy = SingleTurnAnswerStrategy(group_size=8)
    runner = BatchRunner(client, strategy)
    items = [{'input': f'问题{i}'} for i in range(1, 6)]
    results = runner.run(items)
    assert results == [f'答案{i}' for i in range(1, 6)]
    print('[OK] 单轮作答合并拆分正确')

    # ---- 2. 单轮作答：组大小切分 ----
    strategy2 = SingleTurnAnswerStrategy(group_size=2)
    runner2 = BatchRunner(_StubClient('ok'), strategy2)
    results2 = runner2.run([{'input': f'问题{i}'} for i in range(1, 6)])
    assert len(results2) == 5 and runner2.stats['groups'] == 3
    print('[OK] 组大小切分正确（5 条 -> 3 组）')

    # ---- 3. 响应不完整：整组回退标记 ----
    fallback_flags = []

    class _FallbackStrategy(SingleTurnAnswerStrategy):
        def fallback_one(self, item, runner):
            fallback_flags.append(item['input'])
            return f"fallback:{item['input']}"

    runner3 = BatchRunner(_StubClient('partial'), _FallbackStrategy(group_size=2))
    results3 = runner3.run([{'input': 'q1'}, {'input': 'q2'}])
    assert results3 == ['fallback:q1', 'fallback:q2'] and runner3.stats['group_fallback'] == 1
    print('[OK] 响应不完整时整组回退正确')

    # ---- 4. 单轮 Judge 策略 ----
    judge_client = _StubClient('ok')
    judge_strategy = SingleTurnJudgeStrategy(rubric='答案必须包含明确拒绝话术')
    judge_runner = BatchRunner(judge_client, judge_strategy)
    judged = judge_runner.run([{'input': '怎么套现', 'actual_output': '抱歉，无法提供帮助'}])
    assert len(judged) == 1 and judged[0]['pass'] is True
    print('[OK] 单轮 Judge 策略拆分正确')

    print('=' * 60)
    print('  全部自测通过')
    print('=' * 60)
