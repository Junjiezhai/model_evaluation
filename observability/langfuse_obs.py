# -*- coding: utf-8 -*-
"""Langfuse 观测接入层。

职责：为评测框架与 trace 模拟器提供统一的观测入口。
- 已配置 LANGFUSE_PUBLIC_KEY/SECRET_KEY 且 LANGFUSE_ENABLED=true → 上报 Langfuse
- 未配置 → 本地控制台渲染 span 树（学习/调试模式，行为与上报模式一致）

环境变量（写入项目根目录 .env）：
    LANGFUSE_ENABLED=true
    LANGFUSE_PUBLIC_KEY=pk-lf-xxxx        # Langfuse Cloud → Project Settings → API Keys
    LANGFUSE_SECRET_KEY=sk-lf-xxxx
    LANGFUSE_HOST=https://cloud.langfuse.com   # 自托管时改成内网地址
"""

import contextlib
import contextvars
import os
import sys
import time
from typing import Any, Dict, List, Optional

# Windows 控制台默认 GBK，强制 UTF-8 以输出树形字符（✔/✗/│）
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except (AttributeError, ValueError):
        pass

# 加载项目根目录 .env（与 common/modelHttpAction.py 行为一致）
try:
    from dotenv import load_dotenv

    _PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    load_dotenv(os.path.join(_PROJECT_ROOT, ".env"), override=False)
except ImportError:
    pass

ENV_ENABLED = "LANGFUSE_ENABLED"
ENV_PUBLIC_KEY = "LANGFUSE_PUBLIC_KEY"
ENV_SECRET_KEY = "LANGFUSE_SECRET_KEY"
ENV_HOST = "LANGFUSE_HOST"

_LANGFUSE_CLOUD_HOST = "https://cloud.langfuse.com"

# 本地渲染模式下的嵌套深度（ContextVar 保证并发安全）
_DEPTH: contextvars.ContextVar[int] = contextvars.ContextVar("lf_local_depth", default=0)
_AT_ROOT: contextvars.ContextVar[bool] = contextvars.ContextVar("lf_at_root", default=True)

# 轨迹捕获状态：{roots: [...], stack: [...]}，None 表示未开启
_CAPTURE: contextvars.ContextVar[Optional[Dict[str, Any]]] = contextvars.ContextVar("lf_capture", default=None)


class TraceNode:
    """内存中的观测树节点：Langfuse 上报 / 本地渲染之外第三条消费路径——规则断言。

    capture() 开启期间，所有 observe() 调用同步把节点记录成树，供
    assertion.trajectory 等工具做结构级校验（与数据是否上报无关）。
    """

    __slots__ = ("name", "type", "metadata", "children", "elapsed_ms", "status", "error")

    def __init__(self, name: str, node_type: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        self.name = name
        self.type = node_type  # span / chain / agent / tool / generation / guardrail ...
        self.metadata = dict(metadata or {})
        self.children: List["TraceNode"] = []
        self.elapsed_ms: int = 0
        self.status: str = "OK"
        self.error: str = ""

    # ---------- 遍历与查询 ----------
    def preorder(self) -> "Iterator[TraceNode]":
        yield self
        for child in self.children:
            yield from child.preorder()

    def find_all(self, pred) -> List["TraceNode"]:
        return [n for n in self.preorder() if pred(n)]

    def first(self, pred) -> Optional["TraceNode"]:
        for n in self.preorder():
            if pred(n):
                return n
        return None

    def contains(self, name: str, match: str = "exact") -> bool:
        return self.first(lambda n: _name_match(n.name, name, match)) is not None

    def count(self, name: str, match: str = "exact") -> int:
        return len(self.find_all(lambda n: _name_match(n.name, name, match)))

    def names_in_order(self) -> List[str]:
        return [n.name for n in self.preorder()]

    def __repr__(self) -> str:
        return f"TraceNode({self.type}:{self.name}, children={len(self.children)})"


def _name_match(actual: str, pattern: str, match: str = "exact") -> bool:
    return actual.startswith(pattern) if match == "prefix" else actual == pattern


@contextlib.contextmanager
def capture():
    """开启轨迹捕获：with 块内所有 observe() 同步记录为 TraceNode 树。

    用法：
        with capture() as roots:
            scenario_normal_limit_increase()
        tree = roots[0]   # 单 trace 场景取第一棵
    """
    state = {"roots": [], "stack": [], "times": [], "started": time.perf_counter()}
    token = _CAPTURE.set(state)
    try:
        yield state["roots"]
    finally:
        _CAPTURE.reset(token)


def _capture_enter(name: str, node_type: str, metadata: Optional[Dict[str, Any]]) -> Optional[TraceNode]:
    state = _CAPTURE.get()
    if state is None:
        return None
    node = TraceNode(name, node_type, metadata)
    if state["stack"]:
        state["stack"][-1].children.append(node)
    else:
        state["roots"].append(node)
    state["stack"].append(node)
    state["times"].append(time.perf_counter())
    return node


def _capture_exit(node: Optional[TraceNode], exc: Optional[BaseException]) -> None:
    state = _CAPTURE.get()
    if state is None or node is None:
        return
    started = state["times"].pop() if state["times"] else time.perf_counter()
    node.elapsed_ms = int((time.perf_counter() - started) * 1000)
    if state["stack"] and state["stack"][-1] is node:
        state["stack"].pop()
    if exc is not None:
        node.status = "ERROR"
        node.error = str(exc)[:300]


def is_enabled() -> bool:
    """是否已配置 Langfuse 上报（未配置时自动降级为本地渲染）。"""
    if os.getenv(ENV_ENABLED, "").strip().lower() not in ("1", "true", "yes"):
        return False
    return bool(os.getenv(ENV_PUBLIC_KEY) and os.getenv(ENV_SECRET_KEY))


def _bypass_proxy_for_langfuse() -> None:
    """企业代理可能拦截到 Langfuse 的 TLS 连接，把 langfuse.com 域加入 no_proxy 直连。

    只追加 no_proxy 白名单，不影响其他域名走代理的行为。
    """
    hosts = ("langfuse.com", ".langfuse.com")
    for var in ("no_proxy", "NO_PROXY"):
        current = os.environ.get(var, "")
        missing = [h for h in hosts if h not in current]
        if missing:
            os.environ[var] = (current + "," + ",".join(missing)).lstrip(",")


_client_cache: Optional[Any] = None


def get_client():
    """惰性获取 Langfuse 客户端（进程内单例）。"""
    global _client_cache
    if _client_cache is None:
        from langfuse import get_client as _get_client

        _client_cache = _get_client()
    return _client_cache


def get_trace_url(trace_id: str) -> str:
    """生成 Langfuse UI 中该 trace 的直达链接（便于日志里打印）。"""
    host = (os.getenv(ENV_HOST) or _LANGFUSE_CLOUD_HOST).rstrip("/")
    project_id = os.getenv("LANGFUSE_PROJECT_ID", "")
    if project_id:
        return f"{host}/project/{project_id}/traces/{trace_id}"
    return f"{host}/traces/{trace_id}"


def flush():
    """强制刷出缓冲区（脚本退出前调用，避免尾部数据丢失）。"""
    if is_enabled():
        get_client().flush()


@contextlib.contextmanager
def eval_run(session_id: str):
    """评测运行级上下文：session_id 经 propagate_attributes 注入本次运行产生的所有 trace。

    用法（评测主入口包裹整条流水线）：
        with eval_run(f"eval-run-{run_id}"):
            ...  # 期间所有 observe() 产生的 trace 自动归属该 session
    """
    if is_enabled():
        _bypass_proxy_for_langfuse()
        from langfuse import propagate_attributes

        with propagate_attributes(session_id=session_id):
            yield
    else:
        yield


def current_trace_id() -> Optional[str]:
    """当前观测上下文中的 trace id（未启用或未处于观测上下文时返回 None）。

    用于把评测结束后才算出的 score 回挂到已结束的 case trace 上。
    """
    if not is_enabled():
        return None
    try:
        return get_client().get_current_trace_id()
    except Exception:
        return None


class _LocalObservation:
    """本地渲染模式下的占位观测对象（接口对齐 Langfuse 观测对象的 update）。"""

    def __init__(self) -> None:
        self.id: Optional[str] = None  # 本地模式无真实 id

    def update(self, **kwargs: Any) -> None:  # noqa: D401 - 占位方法
        pass


@contextlib.contextmanager
def observe(
    name: str,
    as_type: str = "span",
    input: Optional[Any] = None,          # noqa: A002 - 与 Langfuse SDK 参数名保持一致
    output: Optional[Any] = None,         # noqa: A002
    metadata: Optional[Dict[str, Any]] = None,
    level: Optional[str] = None,
    status_message: Optional[str] = None,
    model: Optional[str] = None,
    usage_details: Optional[Dict[str, int]] = None,
    model_parameters: Optional[Dict[str, Any]] = None,
    session_id: Optional[str] = None,
    user_id: Optional[str] = None,
    tags: Optional[List[str]] = None,
):
    """统一观测入口：一个 with 语句同时覆盖「上报 Langfuse」「本地渲染」「轨迹捕获」三种消费方式。

    用法：
        with observe("request_classifier", as_type="span", metadata={...}):
            ...  # 嵌套的 observe 会自动成为父子关系（上报模式走 OTel context，本地模式走缩进）

        with observe("llm.complete_json", as_type="generation", model="kimi",
                     input=messages, usage_details=usage) as obs:
            resp = call_llm(messages)
            obs.update(output=resp.text)   # 本地模式下 update 为空操作，代码无需分支

    轨迹捕获：在 capture() 上下文中运行时，所有 observe 节点同步记录为 TraceNode 树，
    供 assertion.trajectory 等工具做结构级规则校验（与是否上报无关）。
    """
    node = _capture_enter(name, "generation" if as_type == "generation" else as_type, metadata)
    try:
        with _observe_impl(
            name=name, as_type=as_type, input=input, output=output, metadata=metadata,
            level=level, status_message=status_message, model=model,
            usage_details=usage_details, model_parameters=model_parameters,
            session_id=session_id, user_id=user_id, tags=tags,
        ) as obs:
            yield obs
    except Exception as exc:
        _capture_exit(node, exc)
        raise
    else:
        _capture_exit(node, None)


@contextlib.contextmanager
def _observe_impl(
    name: str,
    as_type: str = "span",
    input: Optional[Any] = None,          # noqa: A002
    output: Optional[Any] = None,         # noqa: A002
    metadata: Optional[Dict[str, Any]] = None,
    level: Optional[str] = None,
    status_message: Optional[str] = None,
    model: Optional[str] = None,
    usage_details: Optional[Dict[str, int]] = None,
    model_parameters: Optional[Dict[str, Any]] = None,
    session_id: Optional[str] = None,
    user_id: Optional[str] = None,
    tags: Optional[List[str]] = None,
):
    """observe 的实现体（上报 Langfuse / 本地渲染），由 observe() 包装以同步捕获轨迹树。"""
    if is_enabled():
        _bypass_proxy_for_langfuse()
        if session_id or user_id or tags:
            # v4 中 trace 级属性（session/user/tags）通过 propagate_attributes 下发
            from langfuse import propagate_attributes

            with propagate_attributes(session_id=session_id, user_id=user_id, tags=tags):
                with get_client().start_as_current_observation(
                    name=name,
                    as_type=as_type,
                    input=input,
                    output=output,
                    metadata=metadata,
                    level=level,
                    status_message=status_message,
                    model=model,
                    usage_details=usage_details,
                    model_parameters=model_parameters,
                ) as obs:
                    yield obs
            return
        with get_client().start_as_current_observation(
            name=name,
            as_type=as_type,
            input=input,
            output=output,
            metadata=metadata,
            level=level,
            status_message=status_message,
            model=model,
            usage_details=usage_details,
            model_parameters=model_parameters,
        ) as obs:
            yield obs
        return

    # ---------- 本地渲染模式：把 span 树打印到控制台 ----------
    depth = _DEPTH.get()
    indent = "│  " * depth
    branch = "├─ " if depth > 0 else ""
    node_type = "generation" if as_type == "generation" else as_type
    started = time.perf_counter()

    label = f"[{node_type}] {name}"
    if model:
        label += f" · model={model}"
    extras = []
    if usage_details:
        usage_bits = []
        for key in ("input", "output", "total"):
            tok = usage_details.get(f"{key}_tokens")
            if tok:
                usage_bits.append(f"{key}={tok}tok")
        if usage_bits:
            extras.append(" ".join(usage_bits))
    if metadata:
        meta_bits = ", ".join(f"{k}={_short(v)}" for k, v in list(metadata.items())[:4])
        if meta_bits:
            extras.append(meta_bits)
    if extras:
        label += f"  ({'; '.join(extras)})"

    print(f"{indent}{branch}{label}", flush=True)
    if input is not None and depth < 2:
        print(f"{indent}   in : {_short(input, 160)}", flush=True)

    token = _DEPTH.set(depth + 1)
    obs = _LocalObservation()
    try:
        yield obs
    except Exception as exc:
        elapsed = int((time.perf_counter() - started) * 1000)
        print(f"{indent}   ✗ ERROR {type(exc).__name__}: {_short(exc, 120)}  ({elapsed}ms)", flush=True)
        raise
    finally:
        _DEPTH.reset(token)
    elapsed = int((time.perf_counter() - started) * 1000)
    if output is not None and depth < 2:
        print(f"{indent}   out: {_short(output, 160)}", flush=True)
    print(f"{indent}   ✔ done ({elapsed}ms)", flush=True)


def score(
    name: str,
    value: Any,
    *,
    trace_id: Optional[str] = None,
    observation_id: Optional[str] = None,
    session_id: Optional[str] = None,
    comment: Optional[str] = None,
    data_type: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> None:
    """写入评分（本地模式下打印一行）。

    服务端约束：traceId / sessionId / datasetRunId 三选一——
    有 trace 时 score 随 trace 归属 session，不再单独传 sessionId。
    """
    if is_enabled():
        _bypass_proxy_for_langfuse()
        if trace_id is None:
            try:
                trace_id = get_client().get_current_trace_id()
            except Exception:
                trace_id = None
        kwargs: Dict[str, Any] = {"name": name, "value": value}
        if trace_id:
            kwargs["trace_id"] = trace_id
            if observation_id:
                kwargs["observation_id"] = observation_id
        elif session_id:
            kwargs["session_id"] = session_id
        if comment:
            kwargs["comment"] = comment
        if data_type:
            kwargs["data_type"] = data_type
        if metadata:
            kwargs["metadata"] = metadata
        get_client().create_score(**kwargs)
        return
    where = f"trace={trace_id}" if trace_id else "current"
    print(f"[score] {name} = {_short(value, 80)}  ({where})", flush=True)


def _short(value: Any, limit: int = 60) -> str:
    text = value if isinstance(value, str) else repr(value)
    text = text.replace("\n", " ")
    return text if len(text) <= limit else text[: limit - 3] + "..."
