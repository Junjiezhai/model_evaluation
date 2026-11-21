# -*- coding: utf-8 -*-
"""组件层 · retrieval 断言：知识检索部件质量（骨架）。

预留：bigram 检索命中率、注入上下文与答案的归因一致性等。
接入点：eval/knowledge_base.py 的检索结果可按本层接口包装为组件断言。
"""

from typing import List

from assertion.core import build_assertion_result, register


@register("retrieval_hit_rate")
def check_retrieval_hit_rate(hit_flags: List[bool], *, min_rate: float = 0.8,
                             source: str = ""):
    """批量断言：检索命中率 ≥ min_rate。

    Args:
        hit_flags: 每个查询是否命中可用知识（调用方按业务口径判定，如 top_k 结果非空）
        min_rate:  最低命中率要求
    """
    total = len(hit_flags)
    hits = sum(1 for h in hit_flags if h)
    rate = hits / total if total else 1.0
    return build_assertion_result(
        "comp", "retrieval_hit_rate", rate >= min_rate,
        detail=f"命中率 {rate:.0%}（{hits}/{total}）",
        severity="warning", source=source,
        hit_rate=rate,
    )


if __name__ == "__main__":
    assert check_retrieval_hit_rate([True] * 9 + [False], min_rate=0.8).passed
    assert not check_retrieval_hit_rate([True] * 5 + [False] * 5, min_rate=0.8).passed
    print("[OK] component/retrieval 自测通过")
