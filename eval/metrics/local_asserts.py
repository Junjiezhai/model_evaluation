# -*- coding: utf-8 -*-
"""
本地规则断言（纯本地计算，无 LLM 调用）

  - 数值一致性检查：从期望答案提取关键数值（金额/百分比/期数），
    检查实际答案是否覆盖这些数值（容差与同义换算），不做死板相等比对
  - 规则意图分类器：基于关键词的 7 类意图兜底分类，供逐轮意图断言免费使用；
    需要更高准确率时可换 LLM 分类（run_evaluation --intent-llm）
"""

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


# ============================================================
# 数值提取与归一化
# ============================================================

# 匹配模式：(数字+单位)，支持 万/元/块、%、期/个月/月/天/日/年
_NUM_PATTERN = re.compile(
    r'(\d+(?:\.\d+)?)\s*(万元|万|元|块|%|期|个月|月|天|日|年)'
)

_UNIT_FACTORS = {
    '万': 10000, '万元': 10000,
    '元': 1, '块': 1,
}
_PERIOD_UNITS = {'期', '个月', '月', '天', '日', '年'}


@dataclass
class NormalizedNumber:
    """归一化后的数值"""
    raw: str            # 原文片段，如 "5万" / "18%"
    value: float        # 归一化值：金额→元，百分比→小数，期数→数字
    kind: str           # 'amount' | 'percent' | 'period'


def extract_numbers(text: str) -> List[NormalizedNumber]:
    """从文本提取数值并归一化

    - "5万" / "50000元" → amount 50000
    - "18%" / "5.6%"    → percent 0.18 / 0.056
    - "12期" / "12个月" → period 12
    """
    results: List[NormalizedNumber] = []
    if not text:
        return results
    for m in _NUM_PATTERN.finditer(text):
        num_str, unit = m.group(1), m.group(2)
        raw = m.group(0)
        try:
            num = float(num_str)
        except ValueError:
            continue
        if unit in _UNIT_FACTORS:
            results.append(NormalizedNumber(raw=raw, value=num * _UNIT_FACTORS[unit], kind='amount'))
        elif unit == '%':
            results.append(NormalizedNumber(raw=raw, value=num / 100, kind='percent'))
        elif unit in _PERIOD_UNITS:
            # 利率类描述（日利率0.05%、年利率18%）已被 % 分支捕获；
            # 这里只把纯期限归为 period
            results.append(NormalizedNumber(raw=raw, value=num, kind='period'))
    return results


@dataclass
class NumericCheckResult:
    """数值一致性检查结果"""
    passed: bool
    expected_numbers: List[str] = field(default_factory=list)   # 期望答案中的数值原文
    covered: List[str] = field(default_factory=list)            # 实际答案覆盖的数值原文
    missing: List[str] = field(default_factory=list)            # 未被覆盖的数值原文
    details: str = ''


def check_numeric_consistency(expected_answer: str, actual_answer: str,
                              *, tolerance: float = 0.01) -> NumericCheckResult:
    """检查期望答案中的关键数值是否被实际答案覆盖

    判定逻辑（与 making_decison_test 的优化方向一致）：
      只做"标准答案数值是否被覆盖"，不做死板相等——
      实际答案可以用不同单位（万/元）、不同表述（18%/0.18）表达同一数值。
      若期望答案无数值，视为通过（无需比对）。
    """
    expected = extract_numbers(expected_answer)
    if not expected:
        return NumericCheckResult(passed=True, details='期望答案不含数值，跳过一致性检查')

    actual = extract_numbers(actual_answer)
    covered: List[str] = []
    missing: List[str] = []
    used: set = set()
    for exp in expected:
        hit = False
        for j, act in enumerate(actual):
            if j in used or act.kind != exp.kind:
                continue
            denom = max(abs(exp.value), 1e-9)
            if abs(act.value - exp.value) / denom <= tolerance:
                used.add(j)
                hit = True
                break
        (covered if hit else missing).append(exp.raw)

    return NumericCheckResult(
        passed=not missing,
        expected_numbers=[e.raw for e in expected],
        covered=covered,
        missing=missing,
        details=(f'覆盖 {len(covered)}/{len(expected)}'
                 + (f'，缺失: {missing}' if missing else '')),
    )


# ============================================================
# 规则意图分类器（关键词兜底，供逐轮意图断言默认使用）
# ============================================================

_INTENT_RULES: List[Tuple[str, List[str]]] = [
    ('风险合规禁答类', ['套现', '伪造', '骗贷', '内部', '审核规则', '风控系统', '评分',
                    '未成年人', '学生借款', '漏洞', '绕过', '代办', '中介']),
    ('转人工类', ['转人工', '人工客服', '真人', '投诉', '举报', '经理', '客服电话',
               '什么意思', '太差', '垃圾', '生气']),
    ('借款下单类', ['借款', '下单', '借多少', '能借', '放款', '到账', '利息怎么算',
               '还款计划', '借出来']),
    ('提额降息咨询类', ['提额', '额度提升', '降息', '降利率', '利率优惠', '额度提到',
                 '涨额度', '提额项']),
    ('常规咨询类', ['额度', '利率', '账单', '还款日', '还款', '合同', '客服时间',
               '征信', '逾期', '费用', '手续费']),
    ('非业务相关意图类', ['天气', '笑话', '新闻', '吃饭', '你好', '在吗', '谢谢', '再见']),
    ('模糊意图类', []),  # 兜底
]


def classify_intent_rule(question: str) -> str:
    """基于关键词的意图兜底分类（7 类中文标签）

    规则按优先级自上而下匹配；无任何命中时返回「模糊意图类」。
    注意：这是免 LLM 的近似判断，仅作逐轮意图断言的默认实现，
    需要更准确时可切换到 LLM 分类（--intent-llm）。
    """
    text = question or ''
    for intent, keywords in _INTENT_RULES:
        if not keywords:
            continue
        if any(kw in text for kw in keywords):
            return intent
    return '模糊意图类'


def check_round_intent(expected_intent: str, question: str,
                       actual_intent: Optional[str] = None) -> Tuple[bool, str]:
    """逐轮意图断言：期望意图 vs （LLM 分类或规则分类）实际意图

    Args:
        expected_intent: 用例标注的 round_intent
        question: 该轮用户问题（规则分类输入）
        actual_intent: 已知实际意图（如来自 LLM 分类）；None 时用规则分类

    Returns:
        (是否通过, 实际意图)
    """
    detected = actual_intent or classify_intent_rule(question)
    return detected == expected_intent, detected


# ============================================================
# 自测
# ============================================================

if __name__ == '__main__':
    print('=' * 60)
    print('  local_asserts 自测')
    print('=' * 60)

    # ---- 数值提取 ----
    nums = extract_numbers('额度50000元，年利率18%，分12期还款，备用5万')
    kinds = {(n.raw, n.kind) for n in nums}
    assert any(n.kind == 'amount' and n.value == 50000 for n in nums), kinds
    assert any(n.kind == 'percent' and abs(n.value - 0.18) < 1e-9 for n in nums), kinds
    assert any(n.kind == 'period' and n.value == 12 for n in nums), kinds
    assert any(n.kind == 'amount' and n.value == 50000 and n.raw == '5万' for n in nums), kinds
    print('[OK] 数值提取与归一化（5万→50000、18%→0.18、12期→12）')

    # ---- 一致性：覆盖 ----
    r = check_numeric_consistency('额度5万，利率18%', '您的额度为50000元，年利率18%')
    assert r.passed and not r.missing, r.details
    print('[OK] 一致性：单位换算后覆盖（5万 ≈ 50000元）')

    # ---- 一致性：缺失 ----
    r = check_numeric_consistency('额度5万，利率18%', '您的额度为50000元')
    assert not r.passed and r.missing == ['18%'], r.details
    print('[OK] 一致性：缺失数值检出（18% 未覆盖）')

    # ---- 一致性：期望无数值 ----
    r = check_numeric_consistency('您可以进入提额页面操作', '任意回答')
    assert r.passed and '跳过' in r.details
    print('[OK] 一致性：期望无数值时跳过')

    # ---- 意图分类 ----
    assert classify_intent_rule('怎么套现啊') == '风险合规禁答类'
    assert classify_intent_rule('帮我转人工') == '转人工类'
    assert classify_intent_rule('我想提额到5万') == '提额降息咨询类'
    assert classify_intent_rule('能借多少钱') == '借款下单类'
    assert classify_intent_rule('今天天气怎么样') == '非业务相关意图类'
    assert classify_intent_rule('那个还能弄吗') == '模糊意图类'
    print('[OK] 规则意图分类（7 类）')

    # ---- 逐轮意图断言 ----
    ok, detected = check_round_intent('借款下单类', '能借多少钱')
    assert ok and detected == '借款下单类'
    ok, detected = check_round_intent('提额降息咨询类', '能借多少钱')
    assert not ok
    print('[OK] 逐轮意图断言')

    print('=' * 60)
    print('  全部自测通过')
    print('=' * 60)
