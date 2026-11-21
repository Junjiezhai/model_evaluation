# -*- coding: utf-8 -*-
"""
端到端层 · 确定性结果断言：数值一致性（纯本地计算，无 LLM 调用）

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
# 自测
# ============================================================

if __name__ == '__main__':
    print('=' * 60)
    print('  e2e/deterministic 自测')
    print('=' * 60)
    nums = extract_numbers('额度50000元，年利率18%，分12期还款，备用5万')
    kinds = {(n.raw, n.kind) for n in nums}
    assert any(n.kind == 'amount' and n.value == 50000 for n in nums), kinds
    assert any(n.kind == 'percent' and abs(n.value - 0.18) < 1e-9 for n in nums), kinds
    assert any(n.kind == 'period' and n.value == 12 for n in nums), kinds
    assert any(n.kind == 'amount' and n.value == 50000 and n.raw == '5万' for n in nums), kinds
    print('[OK] 数值提取与归一化（5万→50000、18%→0.18、12期→12）')
    r = check_numeric_consistency('额度5万，利率18%', '您的额度为50000元，年利率18%')
    assert r.passed and not r.missing, r.details
    print('[OK] 一致性：单位换算后覆盖（5万 ≈ 50000元）')
    r = check_numeric_consistency('额度5万，利率18%', '您的额度为50000元')
    assert not r.passed and r.missing == ['18%'], r.details
    print('[OK] 一致性：缺失数值检出（18% 未覆盖）')
    r = check_numeric_consistency('您可以进入提额页面操作', '任意回答')
    assert r.passed and '跳过' in r.details
    print('[OK] 一致性：期望无数值时跳过')
    print('=' * 60)
    print('  全部自测通过')
    print('=' * 60)
