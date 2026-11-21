# -*- coding: utf-8 -*-
"""
轨迹层 · 逐轮意图断言（纯本地计算，无 LLM 调用）

  - 规则意图分类器：基于关键词的 7 类意图兜底分类，供逐轮意图断言免费使用；
    需要更高准确率时可换 LLM 分类（run_evaluation --intent-llm）
  - 判定对象是「过程标注」（每轮意图是否与用例标注一致），故归入轨迹层
"""

import re
from typing import List, Optional, Tuple

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
    print('  trajectory/intent 自测')
    print('=' * 60)
    assert classify_intent_rule('怎么套现啊') == '风险合规禁答类'
    assert classify_intent_rule('帮我转人工') == '转人工类'
    assert classify_intent_rule('我想提额到5万') == '提额降息咨询类'
    assert classify_intent_rule('能借多少钱') == '借款下单类'
    assert classify_intent_rule('今天天气怎么样') == '非业务相关意图类'
    assert classify_intent_rule('那个还能弄吗') == '模糊意图类'
    print('[OK] 规则意图分类（7 类）')
    ok, detected = check_round_intent('借款下单类', '能借多少钱')
    assert ok and detected == '借款下单类'
    ok, detected = check_round_intent('提额降息咨询类', '能借多少钱')
    assert not ok
    print('[OK] 逐轮意图断言')
    print('=' * 60)
    print('  全部自测通过')
    print('=' * 60)
