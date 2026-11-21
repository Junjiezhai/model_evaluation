# -*- coding: utf-8 -*-
"""轨迹断言 CLI：python -m assertion.trajectory

用法：
  python -m assertion.trajectory                          # 本地模拟数据全场景（零副作用）
  python -m assertion.trajectory --scenario guard_bypass_bug
  python -m assertion.trajectory --source langfuse        # Cloud 数据检查 + score 回写（traj.assert）

退出码：存在 critical 违规时返回 1（可挂 CI / 评测流水线）。
"""

import argparse
import sys

from assertion.trajectory.engine import (
    load_langfuse_trees, load_rules, load_simulated_trees, run_assertions,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="批量轨迹合法性断言（轨迹层）")
    parser.add_argument("--source", choices=["simulate", "langfuse"], default="simulate")
    parser.add_argument("--scenario", help="限定模拟器场景（逗号分隔），默认全部")
    parser.add_argument("--rules", default=None, help="规则 YAML 路径（默认 assertion/trajectory/rules.yaml）")
    parser.add_argument("--session", default="learning-session-001", help="langfuse 源：session 过滤")
    parser.add_argument("--include-eval", action="store_true", help="langfuse 源：包含 eval_case trace")
    parser.add_argument("--upload", action="store_true",
                        help="simulate 源：允许上报 Cloud（默认仅本地检查，零副作用）")
    parser.add_argument("--limit", type=int, default=1000, help="langfuse 源：观测拉取上限")
    args = parser.parse_args()

    rules = load_rules(args.rules)

    if args.source == "simulate":
        from observability.simulate_agent_traces import SCENARIOS
        names = [s.strip() for s in args.scenario.split(",")] if args.scenario else list(SCENARIOS)
        names = [n for n in names if n in SCENARIOS] or list(SCENARIOS)
        trees = load_simulated_trees(names, upload=args.upload)
        writeback = args.upload
    else:
        trees = load_langfuse_trees(args.session, args.include_eval, args.limit)
        writeback = True  # Cloud 源：结果回写 score，UI 可按 traj.assert 过滤

    if not trees:
        print("没有可检查的 trace")
        sys.exit(0)

    print(f"规则 {len(rules)} 条 × trace {len(trees)} 条（source={args.source}）\n")
    result = run_assertions(trees, rules, writeback)

    if writeback:
        from assertion.core import sink
        sink.flush()
        print("\n[score 回写] traj.assert 已写回 Cloud，UI → Scores 页按该名称过滤")

    sys.exit(1 if result["critical_fails"] > 0 else 0)


if __name__ == "__main__":
    main()
