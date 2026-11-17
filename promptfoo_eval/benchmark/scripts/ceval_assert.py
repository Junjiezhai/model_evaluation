# -*- coding: utf-8 -*-
"""
C-Eval 判分断言（promptfoo python assertion，本地执行、零模型调用）。

约定：导出 get_assert(output, context)，返回 {'pass', 'score', 'reason'}。
抽取优先级（模型被要求"只输出字母"，但按需容错多种格式）：
  1. "答案：B" / "答案是B" / "答案(B)" 等带"答案"前缀的格式（取最后一次出现）；
  2. 整个输出就是一个字母（如 "B" 或 "(B)"）；
  3. 输出中第一个独立出现的 A-D 字母（兜底，reason 中标注）。
"""

import re

ANSWER_RE = re.compile(r'答案\s*(?:是|为)?\s*[:：]?\s*\(?([ABCD])\)?', re.IGNORECASE)
BARE_RE = re.compile(r'^\(?([ABCD])\)?$')
LETTER_RE = re.compile(r'\b([ABCD])\b')


def get_assert(output, context):
    vars_ = (context or {}).get('vars') or {}
    gold = str(vars_.get('gold', '')).strip().upper()
    text = (output or '').strip()
    if not text:
        return {'pass': False, 'score': 0.0, 'reason': '模型输出为空'}
    if gold not in ('A', 'B', 'C', 'D'):
        return {'pass': False, 'score': 0.0, 'reason': f'gold 异常: {gold!r}'}

    matches = ANSWER_RE.findall(text)
    if matches:
        pred, via = matches[-1].upper(), '答案前缀抽取'
    elif BARE_RE.match(text):
        pred, via = BARE_RE.match(text).group(1).upper(), '整行字母'
    elif LETTER_RE.search(text):
        pred, via = LETTER_RE.search(text).group(1).upper(), '首个字母兜底'
    else:
        return {'pass': False, 'score': 0.0, 'reason': '输出中找不到 A-D 选项字母'}

    passed = pred == gold
    return {
        'pass': passed,
        'score': 1.0 if passed else 0.0,
        'reason': f'{via}：预测={pred}，gold={gold}',
    }
