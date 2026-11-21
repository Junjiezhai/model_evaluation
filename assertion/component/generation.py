# -*- coding: utf-8 -*-
"""组件层 · generation 断言：单次模型调用输出质量。

当前实现：结构化输出合法性（JSON 可解析性）统计——批量作答/判分链路的组件级健康指标。
预留：schema 合规校验（json_schema 关键字校验）、拒答组件判定、幻觉声明抽检等。
"""

import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from assertion.core import CRITICAL, build_assertion_result, register


@dataclass
class JsonValidityResult:
    """结构化输出合法性统计"""
    total: int
    valid: int
    repaired: int          # 正则提取后修复成功
    failed: int
    failures: List[str] = field(default_factory=list)

    @property
    def validity_rate(self) -> float:
        return (self.valid + self.repaired) / self.total if self.total else 1.0


def _try_parse_json_object(text: str) -> tuple:
    """返回 (obj, repaired)。与 common/modelHttpAction.chat_with_json_output 同策略：
    先直接解析，失败用正则提取 {...} 兜底。"""
    text = (text or "").strip()
    if not text:
        return None, False
    try:
        obj = json.loads(text)
        return (obj if isinstance(obj, (dict, list)) else None), False
    except ValueError:
        pass
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return None, False
    try:
        return json.loads(m.group()), True
    except ValueError:
        return None, False


@register("structured_output_validity")
def check_structured_output_validity(outputs: List[str], *,
                                       min_rate: float = 1.0,
                                       source: str = "") :
    """批量断言：结构化输出合法率 ≥ min_rate（默认全量合法才通过）。

    用法（注册中心模式）：
        from assertion.core import ASSERTION_REGISTRY
        result = ASSERTION_REGISTRY['structured_output_validity'](outputs)
    """
    result = JsonValidityResult(total=len(outputs), valid=0, repaired=0, failed=0)
    for text in outputs:
        obj, repaired = _try_parse_json_object(text)
        if obj is None:
            result.failed += 1
            result.failures.append((text or "")[:80])
        elif repaired:
            result.repaired += 1
        else:
            result.valid += 1
    return build_assertion_result(
        "comp", "structured_output_validity", result.validity_rate >= min_rate,
        detail=(f"合法率 {result.validity_rate:.0%}（valid={result.valid} "
                f"repaired={result.repaired} failed={result.failed}）"),
        severity=CRITICAL if result.failed else "warning",
        source=source,
        validity=result.validity_rate, failed_samples=result.failures[:5],
    )


if __name__ == "__main__":
    ok = ['{"a": 1}', '{"answers": []}']
    repaired = ['前言 {"a": 1} 后记']
    bad = ['不是json', '']
    r = check_structured_output_validity(ok + repaired, min_rate=1.0)
    assert r.passed and r.extra["validity"] == 1.0, r.detail
    r = check_structured_output_validity(ok + bad, min_rate=1.0)
    assert not r.passed and r.extra["validity"] == 0.5, r.detail
    r = check_structured_output_validity(ok + bad, min_rate=0.5)
    assert r.passed, r.detail
    print("[OK] component/generation 自测通过")
