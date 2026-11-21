# -*- coding: utf-8 -*-
"""
# @File    : import_eval_results.py
# @Description  : 把评测框架的既有结果导入 Langfuse，练习「trace ↔ 评测 score」关联
#
# 数据来源：
#   1. results/evaluation_results.json —— 每个评测 case 一条 trace：
#        - 每轮对话一个 round span（input=用户问题, output=模型作答, metadata=意图/延迟/缓存命中）
#        - judge 结果写入 score（overall / 各维度 / 门禁项），session=run_id 聚合一次评测运行
#   2. logs/*.txt（--logs 可选）—— loguru 日志按行导入为 span，练习批量灌入
#
# 用法：
#   python -m observability.import_eval_results              # 导入最新评测结果
#   python -m observability.import_eval_results --logs       # 连同历史日志一起导入
#
# 未配置 LANGFUSE 凭据时降级为本地渲染（span 树 + score 打印），流程完全一致。
"""

import argparse
import json
import os
import re
import sys
import time
from typing import Any, Dict, List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from observability.langfuse_obs import flush, is_enabled, observe, score  # noqa: E402

RESULTS_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "results", "evaluation_results.json",
)
LOGS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs")

_LEVEL_MAP = {"DEBUG": "DEBUG", "INFO": "DEFAULT", "WARNING": "WARNING", "ERROR": "ERROR"}


def _import_case(case: Dict[str, Any], run_id: str, session_id: str) -> None:
    """单个评测 case → 一条 trace（round span 树 + judge scores）。"""
    case_id = case.get("case_id", "unknown")
    rounds: List[Dict[str, Any]] = case.get("rounds") or []
    judge: Dict[str, Any] = case.get("judge") or {}

    with observe(
        f"eval_case.{case_id}", as_type="chain",
        input=case.get("case_desc", ""),
        metadata={
            "case_type": case.get("case_type"),
            "run_id": run_id,
            "rounds": len(rounds),
            "source": "evaluation_results.json",
        },
        session_id=session_id,
    ) as root:
        # ---- 逐轮对话：round span ----
        for rd in rounds:
            round_no = rd.get("round_no", 0)
            numeric = rd.get("numeric_check") or {}
            with observe(
                f"round_{round_no}", as_type="chain",
                input=rd.get("user_question", ""),
                output=rd.get("actual_answer", ""),
                metadata={
                    "expected_intent": rd.get("round_intent"),
                    "actual_intent": rd.get("actual_intent"),
                    "intent_pass": rd.get("intent_pass"),
                    "expect_refuse": rd.get("expect_refuse"),
                    "state_intent": rd.get("state_intent"),
                    "context_assert_type": rd.get("context_assert_type"),
                    "latency_s": rd.get("latency"),
                    "cache_hit": rd.get("cache_hit"),
                    "numeric_check_passed": numeric.get("passed"),
                },
            ):
                time.sleep(0.001)  # 仅本地模式可感知耗时，上报模式无影响

        # ---- judge 评分：trace 级 score（session=run_id 可在 UI 按次运行聚合）----
        trace_id = _trace_id_of(root)
        common = {"trace_id": trace_id, "session_id": session_id}

        if "overall_score" in judge:
            score("e2e.overall", judge["overall_score"], comment=judge.get("overall_comment", ""), **common)
        if "dialogue_score" in judge:
            dialogue = judge["dialogue_score"]
            if isinstance(dialogue, dict):
                # 该字段实际是逐维度评分明细（如 Q4_coherence）
                for dim, value in dialogue.items():
                    score(f"e2e.dialogue.{dim}", value, **common)
            else:
                score("e2e.dialogue", dialogue, **common)
        if "gate_passed" in judge:
            score("e2e.gate_passed", bool(judge["gate_passed"]), data_type="BOOLEAN", **common)

        for dim, value in (judge.get("dimension_scores") or {}).items():
            score(f"e2e.dim.{dim}", value, **common)

        for gate in judge.get("gate_checks") or []:
            score(
                f"e2e.gate.{gate.get('dimension', 'unknown')}",
                gate.get("verdict", "UNKNOWN"),
                data_type="CATEGORICAL",
                comment=gate.get("reason", ""),
                **common,
            )

        # 逐轮 Q 分（多轮上下文记忆/意图响应/污染等多维评分）
        for turn_score in judge.get("per_turn_scores") or []:
            turn = turn_score.get("turn", 0)
            for key, value in turn_score.items():
                if key == "turn":
                    continue
                if key.endswith("_reason"):
                    score(f"e2e.{key[:-7]}.turn{turn}.reason", str(value), data_type="TEXT", **common)
                elif isinstance(value, (int, float)):
                    score(f"e2e.{key}.turn{turn}", value, **common)


def _import_log_file(path: str, limit: int = 100) -> None:
    """loguru 日志按行导入为 span（练习批量灌入；真实场景应按 trace_id 关联）。"""
    filename = os.path.basename(path)
    pattern = re.compile(r"^(?P<ts>\S+ \S+) \| (?P<level>\w+)\s*\| (?P<loc>[^|]+) \| (?P<msg>.*)$")
    ingested = 0

    with observe(f"log.{filename}", as_type="chain", metadata={"source": path, "note": "批量灌入演示，未关联业务 trace"}):
        with open(path, encoding="utf-8") as f:
            for line in f:
                if ingested >= limit:
                    break
                m = pattern.match(line.rstrip("\n"))
                if not m:
                    continue
                level = _LEVEL_MAP.get(m.group("level"), "DEFAULT")
                with observe(
                    f'log.{m.group("loc").strip().split(":")[0]}', as_type="span",
                    input=m.group("msg")[:500], level=level,
                    metadata={"ts": m.group("ts"), "locator": m.group("loc").strip()},
                ):
                    ingested += 1
                    time.sleep(0.001)
        score("comp.log_lines_ingested", ingested, comment=f"{filename} 导入行数（上限 {limit}）")


def _trace_id_of(root: Any) -> Any:
    try:
        return getattr(root, "trace_id", None)
    except Exception:
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description="评测结果导入 Langfuse")
    parser.add_argument("--results", default=RESULTS_PATH, help="evaluation_results.json 路径")
    parser.add_argument("--logs", action="store_true", help="同时导入 logs/ 下的历史日志")
    parser.add_argument("--log-limit", type=int, default=100, help="每个日志文件最多导入行数")
    args = parser.parse_args()

    if not is_enabled():
        print("=" * 72)
        print("[本地渲染模式] 未配置 LANGFUSE_PUBLIC_KEY/SECRET_KEY，"
              "仅打印 span 树；配置后重跑即可上报 Langfuse。")
        print("=" * 72)

    with open(args.results, encoding="utf-8") as f:
        data = json.load(f)

    run_id = data.get("run_id", "unknown_run")
    answer_model = data.get("answer_model") or data.get("model", "unknown_model")
    session_id = f"eval-run-{run_id}"
    print(f"[导入] run_id={run_id} | answer_model={answer_model} | mode={data.get('mode')}")

    sections = []
    for section in ("single_turn", "multi_turn"):
        part = data.get(section)
        if isinstance(part, dict) and part.get("cases"):
            sections.append((section, part["cases"]))

    if not sections:
        print("[导入] 未找到任何 case（single_turn/multi_turn 均为空），退出")
        return

    for section, cases in sections:
        print(f"\n== 分区 {section}: {len(cases)} 个 case ==")
        for case in cases:
            _import_case(case, run_id, session_id)

    if args.logs and os.path.isdir(LOGS_DIR):
        for name in sorted(os.listdir(LOGS_DIR)):
            if name.endswith(".txt"):
                print(f"\n== 日志 {name} ==")
                _import_log_file(os.path.join(LOGS_DIR, name), limit=args.log_limit)

    flush()
    print("\n[导入完成] 提示：Langfuse UI 的 Sessions 页按 session 聚合可看到本次评测运行全貌。")


if __name__ == "__main__":
    main()
