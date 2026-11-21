# -*- coding: utf-8 -*-
"""轨迹层规则引擎：对观测树（TraceNode）执行声明式 YAML 规则断言。

数据源（函数级解耦，引擎本身不关心数据从哪来）：
  - load_simulated_trees()   本地跑模拟器场景 + capture() 捕获（零副作用）
  - load_langfuse_trees()    Cloud v2 observations 按 parentObservationId 重建 span 树

结果出口：core.sink（Langfuse score 名 traj.assert，CATEGORICAL PASS/FAIL）。
CLI 入口见 __main__.py（python -m assertion.trajectory）。
"""

import os
import sys
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import yaml  # noqa: E402

from observability.langfuse_obs import TraceNode, _bypass_proxy_for_langfuse, capture  # noqa: E402

PASS, FAIL, SKIP = "PASS", "FAIL", "SKIP"

DEFAULT_RULES_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "rules.yaml")


# ---------------------------------------------------------------------------
# 规则求值
# ---------------------------------------------------------------------------
def _match(actual: str, pattern: str, mode: str) -> bool:
    return actual.startswith(pattern) if mode == "prefix" else actual == pattern


def _check_when(tree: TraceNode, when: Dict[str, Any]) -> bool:
    """前置条件：树中须存在满足条件的节点（node_name 精确 / node_prefix 前缀 + metadata 全等）。"""
    name_pat = when.get("node_name")
    prefix_pat = when.get("node_prefix")
    meta = when.get("metadata") or {}

    def ok(node: TraceNode) -> bool:
        if name_pat and node.name != name_pat:
            return False
        if prefix_pat and not node.name.startswith(prefix_pat):
            return False
        return all(node.metadata.get(k) == v for k, v in meta.items())

    return tree.first(ok) is not None


def _eval_assert(tree: TraceNode, assertion: Dict[str, Any]) -> Tuple[bool, str]:
    """返回 (通过?, 违规细节)。恰好一种断言类型。"""
    if "must_contain" in assertion:
        spec = assertion["must_contain"]
        names, match_mode = spec["names"], spec.get("match", "exact")
        missing = [pat for pat in names if not tree.contains(pat, match_mode)]
        if missing:
            return False, f"缺少节点: {missing}（匹配模式 {match_mode}）"
        return True, ""

    if "must_contain_ordered" in assertion:
        spec = assertion["must_contain_ordered"]
        names, match_mode = spec["names"], spec.get("match", "exact")
        seq = tree.names_in_order()
        idx = 0
        for name in seq:
            if idx < len(names) and _match(name, names[idx], match_mode):
                idx += 1
        if idx == len(names):
            return True, ""
        return False, f"缺少/乱序：{names[idx]}（匹配模式 {match_mode}）"

    if "must_not_contain" in assertion:
        spec = assertion["must_not_contain"]
        names, match_mode = spec["names"], spec.get("match", "exact")
        hits = [n.name for n in tree.preorder()
                if any(_match(n.name, pat, match_mode) for pat in names)]
        if hits:
            return False, f"出现被禁节点: {sorted(set(hits))}"
        return True, ""

    if "max_occurrences" in assertion:
        spec = assertion["max_occurrences"]
        cnt = tree.count(spec["name"], spec.get("match", "exact"))
        if cnt > spec["max"]:
            return False, f"{spec['name']} 出现 {cnt} 次 > 上限 {spec['max']}"
        return True, ""

    if "min_type_count" in assertion:
        spec = assertion["min_type_count"]
        cnt = len(tree.find_all(lambda n: n.type == spec["type"]))
        if cnt < spec["min"]:
            return False, f"{spec['type']} 仅 {cnt} 个 < 要求 {spec['min']}"
        return True, ""

    if "require_metadata" in assertion:
        spec = assertion["require_metadata"]
        key = spec["key"]
        expected = spec.get("equals", "__absent__")

        def selected(n: TraceNode) -> bool:
            if spec.get("node_type") and n.type != spec["node_type"]:
                return False
            if spec.get("node_name") and n.name != spec["node_name"]:
                return False
            if spec.get("node_prefix") and not n.name.startswith(spec["node_prefix"]):
                return False
            return True

        candidates = tree.find_all(selected)
        missing = [n.name for n in candidates if key not in n.metadata]
        wrong = [n.name for n in candidates
                 if key in n.metadata and expected != "__absent__" and n.metadata[key] != expected]
        if missing or wrong:
            parts = []
            if missing:
                parts.append(f"{len(missing)} 个节点缺键 {key}: {sorted(set(missing))[:5]}")
            if wrong:
                parts.append(f"{len(wrong)} 个节点值不符: {sorted(set(wrong))[:5]}")
            return False, "; ".join(parts)
        return True, ""

    return False, f"未知断言类型: {list(assertion)}"


def eval_rule(tree: TraceNode, scenario: str, rule: Dict[str, Any]) -> Tuple[str, str]:
    """单规则 × 单 trace → (PASS|FAIL|SKIP, 细节)。"""
    applies = rule.get("applies_to") or []
    if applies and scenario not in applies:
        return SKIP, "场景不在适用列表"
    when = rule.get("when")
    if when and not _check_when(tree, when):
        return SKIP, "前置条件不满足"
    passed, detail = _eval_assert(tree, rule["assert"])
    return (PASS, "") if passed else (FAIL, detail)


# ---------------------------------------------------------------------------
# 数据源一：本地模拟器（capture 捕获，默认零副作用）
# ---------------------------------------------------------------------------
def load_simulated_trees(scenarios: List[str], upload: bool) -> List[Dict[str, Any]]:
    if not upload:
        os.environ["LANGFUSE_ENABLED"] = "false"

    from observability import simulate_agent_traces as sim

    results = []
    for name in scenarios:
        with capture() as roots:
            sim.SCENARIOS[name]()
        for tree in roots:
            results.append({"scenario": name, "tree": tree, "trace_id": None})
    return results


# ---------------------------------------------------------------------------
# 数据源二：Langfuse Cloud（v2 observations → TraceNode 树）
# ---------------------------------------------------------------------------
def _parse_metadata(raw: Any) -> Dict[str, Any]:
    """v2 API 的 metadata 是 JSON 字符串（且混有 OTel baggage 噪音键），解析并清洗。"""
    if isinstance(raw, dict):
        data = raw
    elif isinstance(raw, str) and raw:
        import ast
        import json

        try:
            data = json.loads(raw)
        except (ValueError, TypeError):
            try:
                data = ast.literal_eval(raw)
            except (ValueError, SyntaxError):
                return {}
    else:
        return {}
    if not isinstance(data, dict):
        return {}
    return {k: v for k, v in data.items() if "scope.attributes" not in k}


def load_langfuse_trees(session_id: str, include_eval: bool, limit: int) -> List[Dict[str, Any]]:
    from datetime import datetime, timedelta, timezone

    import requests

    _bypass_proxy_for_langfuse()
    host = (os.getenv("LANGFUSE_HOST") or "https://cloud.langfuse.com").rstrip("/")
    auth = (os.getenv("LANGFUSE_PUBLIC_KEY", ""), os.getenv("LANGFUSE_SECRET_KEY", ""))
    if not all(auth):
        raise SystemExit("未配置 LANGFUSE_PUBLIC_KEY/SECRET_KEY，无法从 Cloud 拉取数据")

    # 区域解析：凭据属于区域域而 host 留空时，先探测真实区域
    try:
        probe = requests.get(f"{host}/api/public/v3/scores?limit=1", auth=auth, timeout=15)
        if probe.status_code in (401, 403):
            alt = "https://jp.cloud.langfuse.com"
            if requests.get(f"{alt}/api/public/v3/scores?limit=1", auth=auth, timeout=15).ok:
                host = alt
    except requests.RequestException:
        pass

    now = datetime.now(timezone.utc)
    params = {
        "fields": "core,basic,usage,metadata",  # metadata 单独申请才返回
        "fromStartTime": (now - timedelta(days=7)).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "toStartTime": now.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "limit": limit,
    }
    r = requests.get(f"{host}/api/public/v2/observations", params=params, auth=auth, timeout=30)
    r.raise_for_status()
    rows = r.json().get("data", [])
    if len(rows) >= limit:
        print(f"[警告] 观测数达到 limit={limit}，可能被截断，建议缩小时间范围", flush=True)

    # 按 traceId 分组 → parentObservationId 重建树
    by_trace: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        if session_id and row.get("sessionId") != session_id:
            continue
        by_trace.setdefault(row["traceId"], []).append(row)

    results = []
    for trace_id, obs in by_trace.items():
        obs.sort(key=lambda o: o.get("startTime") or "")
        nodes = {}
        roots: List[TraceNode] = []
        for o in obs:
            node = TraceNode(o.get("name", "?"), (o.get("type") or "span").lower(),
                             _parse_metadata(o.get("metadata")))
            node.elapsed_ms = int((o.get("latency") or 0) * 1000)
            node.status = "ERROR" if o.get("level") == "ERROR" else "OK"
            node.error = o.get("statusMessage") or ""
            nodes[o["id"]] = node
        for o in obs:
            parent_id = o.get("parentObservationId")
            if parent_id and parent_id in nodes:
                nodes[parent_id].children.append(nodes[o["id"]])
            else:
                roots.append(nodes[o["id"]])
        if not roots:
            continue
        root_name = roots[0].name
        if root_name.startswith("turn."):
            scenario = root_name[len("turn."):]
        elif root_name.startswith("eval_case."):
            if not include_eval:
                continue
            scenario = root_name[len("eval_case."):]
        else:
            scenario = root_name
        results.append({"scenario": scenario, "tree": roots[0], "trace_id": trace_id})

    # 同场景多次上报只保留最新一次
    deduped: Dict[str, Dict[str, Any]] = {}
    for item in results:
        deduped[item["scenario"]] = item
    return list(deduped.values())


# ---------------------------------------------------------------------------
# 报告与 score 回写（统一走 assertion.core.sink，命名空间 traj.*）
# ---------------------------------------------------------------------------
def run_assertions(trees: List[Dict[str, Any]], rules: List[Dict[str, Any]],
                   writeback: bool) -> Dict[str, int]:
    from assertion.core import CRITICAL, build_assertion_result, sink

    stats = {PASS: 0, FAIL: 0, SKIP: 0}
    critical_fails: List[Tuple[str, str, str]] = []  # (scenario, rule_id, detail)

    print("=" * 78)
    print("轨迹断言报告（traj.*）")
    print("=" * 78)

    # 逐规则求值（含 SKIP 统计）
    rule_results: Dict[str, List[Tuple[Dict[str, Any], str, str]]] = {}
    for rule in rules:
        checked = [(t, *eval_rule(t["tree"], t["scenario"], rule)) for t in trees]
        rule_results[rule["id"]] = checked
        for _, st, _ in checked:
            stats[st] = stats.get(st, 0) + 1

    # 违规详情（全过/全跳的规则不展开）
    for rule in rules:
        rid, sev = rule["id"], rule.get("severity", "warning")
        fails = [(t, detail) for t, st, detail in rule_results[rid] if st == FAIL]
        if not fails:
            continue
        print(f"\n[{rid}|{sev}] {rule['description']}")
        for t, detail in fails:
            print(f"  ✗ {t['scenario']:24s} {detail}")
            if sev == CRITICAL:
                critical_fails.append((t["scenario"], rid, detail))
            if writeback and t["trace_id"]:
                sink.score(build_assertion_result(
                    "traj", "assert", False, detail=f"{rid}: {detail}",
                    severity=sev, source=t["scenario"]), trace_id=t["trace_id"])

    # PASS 的 score 回写（同样落 traj.assert，与 FAIL 成对便于聚合统计通过率）
    if writeback:
        for rule in rules:
            for t, st, _ in rule_results[rule["id"]]:
                if st == PASS and t["trace_id"]:
                    sink.score(build_assertion_result(
                        "traj", "assert", True, detail=rule["id"],
                        severity=rule.get("severity", "warning"),
                        source=t["scenario"]), trace_id=t["trace_id"])

    # 汇总矩阵
    print("\n" + "-" * 78)
    print("scenario".ljust(26) + "".join(r["id"].ljust(8) for r in rules))
    for t in trees:
        cells = []
        for rule in rules:
            st, _ = eval_rule(t["tree"], t["scenario"], rule)
            cells.append({"PASS": "✓", "FAIL": "✗", "SKIP": "·"}[st].ljust(8))
        print(t["scenario"][:25].ljust(26) + "".join(cells))
    print("-" * 78)
    print(f"汇总: PASS={stats[PASS]}  FAIL={stats[FAIL]}  SKIP={stats[SKIP]}"
          f"  |  critical 违规: {len(critical_fails)}")
    for sc, rid, detail in critical_fails:
        print(f"  🔴 {sc} ← {rid}: {detail[:60]}")
    return {"fails": stats[FAIL], "critical_fails": len(critical_fails)}


def load_rules(path: Optional[str] = None) -> List[Dict[str, Any]]:
    with open(path or DEFAULT_RULES_PATH, encoding="utf-8") as f:
        return yaml.safe_load(f)["rules"]
