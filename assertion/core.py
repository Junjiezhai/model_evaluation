# -*- coding: utf-8 -*-
"""
断言内核：统一结果模型 / 注册表 / 报告渲染 / Langfuse 出口。

三层体系（e2e / trajectory / component）共享本内核：
  - AssertionResult    所有断言的统一结果结构（层内保留各自的专用结果类型，出口处统一）
  - ASSERTION_REGISTRY 规则类断言的注册中心（新断言 = 注册一个实现，不改报告代码）
  - sink.score()       统一 Langfuse 出口，score 名自动带层前缀（e2e.*/traj.*/comp.*）

层间依赖纪律：只允许向下（e2e 可消费轨迹层产出，反之不行）。
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

# 层前缀（Langfuse score 命名空间，与目录分层同构）
LAYER_E2E = "e2e"
LAYER_TRAJ = "traj"
LAYER_COMP = "comp"

CRITICAL = "critical"
WARNING = "warning"


@dataclass
class AssertionResult:
    """断言统一结果模型：任何层的断言都可归一化为本结构进同一张报告 / 同一 score 命名空间。

    layer    所属层（e2e / traj / comp）
    name     断言名（不含层前缀，出口自动加）
    passed   是否通过
    detail   未通过原因 / 补充说明（进 comment）
    severity critical = 违反即红线（CI 退出码语义）；warning = 值得审查
    source   被断言对象标识（case_id / trace_id / scenario，便于聚合定位）
    """

    layer: str
    name: str
    passed: bool
    detail: str = ""
    severity: str = WARNING
    source: str = ""
    extra: Dict[str, Any] = field(default_factory=dict)

    @property
    def namespaced_name(self) -> str:
        return f"{self.layer}.{self.name}"


# ---------------------------------------------------------------------------
# 注册表：规则类断言（本地/结构断言）的注册中心
# ---------------------------------------------------------------------------
# name -> callable(*args, **kwargs) -> AssertionResult
ASSERTION_REGISTRY: Dict[str, Callable[..., AssertionResult]] = {}


def register(name: str) -> Callable:
    """装饰器：注册规则类断言实现。

    用法：
        @register('numeric_consistency')
        def check_numeric(...) -> AssertionResult: ...
    """
    def wrap(fn: Callable) -> Callable:
        ASSERTION_REGISTRY[name] = fn
        return fn
    return wrap


def build_assertion_result(layer: str, name: str, passed: bool, *,
                           detail: str = "", severity: str = WARNING,
                           source: str = "", **extra: Any) -> AssertionResult:
    """便捷构造函数（各层专用结果类型在出口处转换用）。"""
    return AssertionResult(layer=layer, name=name, passed=passed, detail=detail,
                           severity=severity, source=source, extra=extra)


# ---------------------------------------------------------------------------
# 报告渲染
# ---------------------------------------------------------------------------
def render_report(results: List[AssertionResult], title: str = "断言报告") -> str:
    """把跨层断言结果渲染为统一文本报告（严重违规置顶）。"""
    lines = ["=" * 74, title, "=" * 74]
    fails = [r for r in results if not r.passed]
    oks = [r for r in results if r.passed]
    for r in fails:
        flag = "🔴" if r.severity == CRITICAL else "🟡"
        lines.append(f"{flag} [{r.namespaced_name}] {r.source}  {r.detail}")
    lines.append("-" * 74)
    lines.append(f"总计 {len(results)} 项 | 通过 {len(oks)} | 未通过 {len(fails)}"
                 f"（critical {sum(1 for r in fails if r.severity == CRITICAL)}）")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Langfuse 出口：score 名统一加层前缀
# ---------------------------------------------------------------------------
class sink:
    """统一观测出口（Langfuse score），自动按层加命名空间前缀。"""

    @staticmethod
    def score(result: AssertionResult, *, trace_id: Optional[str] = None,
              session_id: Optional[str] = None, comment: Optional[str] = None,
              data_type: Optional[str] = None) -> None:
        from observability.langfuse_obs import score as lf_score

        value: Any = "PASS" if result.passed else "FAIL"
        lf_score(result.namespaced_name, value, trace_id=trace_id, session_id=session_id,
                 comment=comment or result.detail,
                 data_type=data_type or "CATEGORICAL", metadata={
                     "severity": result.severity,
                     "source": result.source,
                 })

    @staticmethod
    def flush() -> None:
        from observability.langfuse_obs import flush as lf_flush
        lf_flush()
