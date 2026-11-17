# -*- coding: utf-8 -*-
"""
GSM8K 判分断言（promptfoo python assertion，本地执行、零模型调用）。

约定：导出 get_assert(output, context)，返回 {'pass', 'score', 'reason'}。
判分口径：
  1. 优先匹配"答案：N"格式（取最后一次出现，容忍模型中途复述）；
  2. 匹配不到则兜底取输出中最后一个数字（容错模型不按格式输出）；
  3. 与 vars.gold 做数值相等比较（容忍千分位逗号、浮点尾差）。
"""

import re

ANSWER_RE = re.compile(r'答案\s*[:：]\s*\(?\s*(-?[\d,]+(?:\.\d+)?)\s*\)?')
NUMBER_RE = re.compile(r'-?[\d,]+(?:\.\d+)?')


def _to_float(s: str) -> float:
    return float(str(s).replace(',', '').rstrip('.'))


def get_assert(output, context):
    vars_ = (context or {}).get('vars') or {}
    gold_raw = str(vars_.get('gold', '')).strip()
    text = (output or '').strip()
    if not text:
        return {'pass': False, 'score': 0.0, 'reason': '模型输出为空'}

    try:
        gold = _to_float(gold_raw)
    except ValueError:
        return {'pass': False, 'score': 0.0, 'reason': f'gold 无法解析: {gold_raw!r}'}

    matches = ANSWER_RE.findall(text)
    if matches:
        pred_raw, via = matches[-1], '答案行抽取'
    else:
        nums = NUMBER_RE.findall(text)
        if not nums:
            return {'pass': False, 'score': 0.0, 'reason': '输出中找不到任何数字'}
        pred_raw, via = nums[-1], '末位数字兜底'

    try:
        pred = _to_float(pred_raw)
    except ValueError:
        return {'pass': False, 'score': 0.0, 'reason': f'预测值无法解析: {pred_raw!r}'}

    passed = abs(pred - gold) < 1e-9
    return {
        'pass': passed,
        'score': 1.0 if passed else 0.0,
        'reason': f'{via}：预测={pred:g}，gold={gold:g}',
    }
