# -*- coding: utf-8 -*-
"""
评测主入口：CLI 路由单轮 / 多轮 / 全量流水线，统一落盘 evaluation_results.json。

单轮流水线（DeepEval 标准指标）：
  读 CSV 用例 → 知识库检索注入 → 被测模型作答（可批量/缓存）→ 4 指标 LLM 评分
  → 落盘 + HTML 报告（含性能区块）

多轮流水线（自研，不经 deepeval.evaluate）：
  读 CSV 用例 → 逐轮对话执行（历史累积 + 逐轮计时）→ 本地断言（数值一致性 + 意图分类）
  → 会话状态机验证 → LLM Judge（G1/G2 门禁 + Q1~Q5 加权，可批量）→ 落盘 + 雷达图报告

用法：
  python -m eval.run_evaluation                          # 单轮，非批量，读已有测试集
  python -m eval.run_evaluation --mode multi --limit 5   # 多轮前 5 个 case
  python -m eval.run_evaluation --mode all               # 单轮 + 多轮全量
  python -m eval.run_evaluation --batch                  # 作答合并请求降本
  python -m eval.run_evaluation --force-answer           # 绕过作答缓存
  python -m eval.run_evaluation --generate-testset       # 先按技能文档重新生成测试集
  python -m eval.run_evaluation --intent-llm             # 多轮意图分类走 LLM（默认本地规则）
"""

import argparse
import json
import math
import os
import re
import shutil
import sys
import time
from dataclasses import asdict
from datetime import datetime, timedelta
from typing import Dict, List

os.environ.setdefault('DEEPEVAL_TELEMETRY_OPT_OUT', 'YES')
# deepeval 默认把 .deepeval 缓存目录建在"当前工作目录"下（cwd 不同会散落多处），
# 这里钉死到项目根目录，与运行时的 cwd 无关
Base_Path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault('DEEPEVAL_CACHE_FOLDER', os.path.join(Base_Path, '.deepeval'))
# deepeval 默认每个指标任务 180s 超时预算，Kimi 端并发高时较慢，放宽到 900s
os.environ.setdefault('DEEPEVAL_PER_TASK_TIMEOUT_SECONDS_OVERRIDE', '900')
sys.path.append(Base_Path)

from deepeval import evaluate

import config
from eval import qa_system
from eval.batch_runner import BatchRunner, MultiTurnJudgeStrategy
from eval.case_loader import VALID_INTENT_LABELS, load_multi_turn, load_single_turn
from eval.kimi_llm import build_answer_llm, build_judge_llm
from eval.knowledge_base import KnowledgeBase
from eval.metrics import build_multi_turn_judge, build_single_turn_metrics, build_state_validator
from eval.metrics.local_asserts import check_numeric_consistency, check_round_intent
from eval.metrics.multi_turn_judge import aggregate_results
from eval.metrics.state_validator import DriftSeverity
from eval.report import generate_multi_turn_report, generate_report
from eval.history_index import generate_index
from utils.handle_log import log


# ============================================================
# 工具
# ============================================================

# 历史快照文件夹名格式：YYYYMMDD_HHMMSS（用于按日期清理）
_RUN_ID_PATTERN = re.compile(r'^(\d{8})_\d{6}$')


def _new_run_id() -> str:
    """生成本次运行的唯一标识：时间戳；同秒冲突时追加 _2/_3..."""
    run_id = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_dir = os.path.join(config.RESULTS_HISTORY_DIR, run_id)
    n = 2
    while os.path.exists(run_dir):
        run_id = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{n}"
        run_dir = os.path.join(config.RESULTS_HISTORY_DIR, run_id)
        n += 1
    return run_id


def _cleanup_history() -> int:
    """清理超过保留期的历史评测快照文件夹，返回清理数量。
    只清理文件夹名可解析为日期的目录，其他内容不动（与 logs 清理同约定）。"""
    history_dir = config.RESULTS_HISTORY_DIR
    if not os.path.isdir(history_dir):
        return 0
    cutoff = datetime.now() - timedelta(days=config.RESULTS_HISTORY_RETENTION_DAYS)
    removed = 0
    for name in os.listdir(history_dir):
        m = _RUN_ID_PATTERN.match(name)
        if not m:
            continue
        try:
            run_date = datetime.strptime(m.group(1), '%Y%m%d')
        except ValueError:
            continue
        if run_date < cutoff:
            shutil.rmtree(os.path.join(history_dir, name))
            removed += 1
    if removed:
        log.info(f'[评测] 已自动清理 {removed} 个 {config.RESULTS_HISTORY_RETENTION_DAYS} 天前的历史快照')
    return removed


def _persist_results(data: Dict, run_dir: str) -> None:
    """统一落盘：写历史快照 + 更新 results/ 根下的 latest 副本（唯一写入口）"""
    os.makedirs(run_dir, exist_ok=True)
    snapshot = os.path.join(run_dir, 'evaluation_results.json')
    with open(snapshot, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    with open(config.RAW_RESULTS_PATH, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    log.info(f'[评测] 原始结果已保存 -> {snapshot}（latest 副本 -> {config.RAW_RESULTS_PATH}）')


def _write_run_meta(run_dir: str, run_id: str, started_at: datetime,
                    duration: float, args, data: Dict) -> None:
    """写运行配置快照 run_meta.json（阈值/权重/模式等，便于追溯当次评测口径）"""
    single = data.get('single_turn', {})
    multi = data.get('multi_turn', {})
    cases_s = single.get('cases', [])
    cases_m = multi.get('cases', [])
    passed_s = sum(1 for c in cases_s if all(m['success'] for m in c['metrics'])) if cases_s else 0
    meta = {
        'run_id': run_id,
        'mode': data.get('mode', ''),
        'model': data.get('model', ''),
        'answer_model': data.get('answer_model', ''),
        'started_at': started_at.strftime('%Y-%m-%d %H:%M:%S'),
        'finished_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'duration_sec': round(duration, 1),
        'cli_args': vars(args) if hasattr(args, '__dict__') else {},
        'config_snapshot': {
            'METRIC_THRESHOLDS': config.METRIC_THRESHOLDS,
            'MULTI_TURN_THRESHOLD': config.MULTI_TURN_THRESHOLD,
            'MULTI_TURN_DIMENSION_WEIGHTS': config.MULTI_TURN_DIMENSION_WEIGHTS,
            'STATE_VALIDATION_MODE': config.STATE_VALIDATION_MODE,
            'INTENT_DRIFT_RATE_THRESHOLD': config.INTENT_DRIFT_RATE_THRESHOLD,
            'PROMPT_VERSION': config.PROMPT_VERSION,
            'CACHE_ENABLED': config.CACHE_ENABLED,
            'BATCH_ENABLED_DEFAULT': config.BATCH_ENABLED_DEFAULT,
            'BATCH_GROUP_SIZE': config.BATCH_GROUP_SIZE,
            'BATCH_GROUP_SIZE_MULTI_JUDGE': config.BATCH_GROUP_SIZE_MULTI_JUDGE,
            'RETRIEVAL_TOP_K': config.RETRIEVAL_TOP_K,
        },
        'counts': {
            'single_turn_cases': len(cases_s),
            'multi_turn_cases': len(cases_m),
        },
        'summary': {
            'single_pass_rate': f'{passed_s / len(cases_s) * 100:.1f}%' if cases_s else None,
            'single_pass_rate_value': round(passed_s / len(cases_s) * 100, 1) if cases_s else None,
            'single_perf': single.get('perf'),
            'multi_aggregate': multi.get('aggregate'),
            'multi_perf': multi.get('perf'),
        },
    }
    with open(os.path.join(run_dir, 'run_meta.json'), 'w', encoding='utf-8') as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)


def _percentile(sorted_vals: List[float], p: float) -> float:
    """最近秩分位数（纯 Python，p 取 0~1）"""
    if not sorted_vals:
        return 0.0
    k = max(1, math.ceil(p * len(sorted_vals)))
    return sorted_vals[min(k, len(sorted_vals)) - 1]


def _perf_stats(latencies: List[float]) -> Dict:
    """耗时统计：平均 / P90 / P95 / 最大 / 最小"""
    vals = sorted(latencies)
    if not vals:
        return {'count': 0, 'avg': 0, 'p90': 0, 'p95': 0, 'max': 0, 'min': 0}
    return {
        'count': len(vals),
        'avg': round(sum(vals) / len(vals), 3),
        'p90': round(_percentile(vals, 0.9), 3),
        'p95': round(_percentile(vals, 0.95), 3),
        'max': round(vals[-1], 3),
        'min': round(vals[0], 3),
    }


def _unwrap(item):
    """兼容 deepeval 不同版本的返回形态：EvaluationResult / 元组 / TestResult。"""
    if hasattr(item, 'metrics_data'):
        return item
    if isinstance(item, (tuple, list)):
        for sub in item:
            if hasattr(sub, 'metrics_data'):
                return sub
    return item


# ============================================================
# 单轮流水线
# ============================================================

def run_single_turn(args, kb, data: Dict):
    """单轮评测：读 CSV → 作答（可批量/缓存）→ DeepEval 4 指标 → 序列化"""
    cases = load_single_turn(limit=args.limit)
    log.info(f'[单轮] 测试用例数: {len(cases)}')

    answer_llm = build_answer_llm()
    records = qa_system.answer_single_turn(
        cases, kb, answer_llm, force=args.force_answer,
        batch=args.batch, cache=qa_system.AnswerCache() if config.CACHE_ENABLED else None)
    answered = [r for r in records if r.actual_output]
    cache_hits = sum(1 for r in records if r.cache_hit)
    log.info(f'[单轮] 成功作答 {len(answered)}/{len(cases)} 条（缓存命中 {cache_hits} 条，'
             f'批量={"开" if args.batch else "关"}）')

    # DeepEval 指标评分（默认 4 项，EVAL_SINGLE_METRICS 可过滤）
    judge = build_judge_llm()
    metrics = build_single_turn_metrics(judge)
    log.info(f'[单轮] 指标: {[getattr(m, "name", type(m).__name__) for m in metrics]}')
    eval_results = evaluate(qa_system.build_llm_test_cases(answered), metrics)
    # 兼容 deepeval 不同版本返回形态：对象.test_results / 列表 / 元组
    results = getattr(eval_results, 'test_results', None) or eval_results

    # 序列化：records 与 evaluate 返回一一对应（顺序保持）
    case_dicts = []
    for rec, raw in zip(answered, results):
        r = _unwrap(raw)
        case_dicts.append({
            'case_id': rec.case_id,
            'input': rec.input,
            'expected_output': rec.expected_output,
            'actual_output': rec.actual_output,
            'retrieval_context': rec.retrieval_context,
            'category': rec.category,
            'priority': rec.priority,
            'test_intent': rec.test_intent,
            'test_dimension': rec.test_dimension,
            'expect_refuse': rec.expect_refuse,
            'source': 'csv',
            'latency': round(rec.latency, 3),
            'latency_is_amortized': rec.latency_is_amortized,
            'cache_hit': rec.cache_hit,
            'metrics': [{
                'name': m.name, 'score': m.score, 'threshold': m.threshold,
                'success': bool(m.success), 'reason': m.reason, 'error': m.error,
            } for m in (r.metrics_data or [])],
        })

    perf = _perf_stats([r.latency for r in answered])
    perf['amortized_count'] = sum(1 for r in answered if r.latency_is_amortized)
    data['single_turn'] = {'cases': case_dicts, 'perf': perf}
    return judge.get_model_name(), answer_llm.get_model_name()


# ============================================================
# 多轮流水线
# ============================================================

def _classify_intents_llm(questions: List[str], judge_llm) -> List[str]:
    """LLM 意图分类（--intent-llm 时启用）：一次调用批量分类，失败回退规则分类

    返回与 questions 等长的意图标签列表（7 类中文标签）。
    """
    from eval.metrics.local_asserts import classify_intent_rule

    if not questions:
        return []
    labels = ' / '.join(VALID_INTENT_LABELS)
    lines = '\n'.join(f'{i + 1}. {q}' for i, q in enumerate(questions))
    prompt = (
        '你是意图分类器。请把下面的用户问题逐条分类为以下意图之一：\n'
        f'{labels}\n\n{lines}\n\n'
        '严格输出 JSON：{"labels": ["意图1", "意图2", ...]}，数量与问题数一致。'
    )
    try:
        schema = {'type': 'object',
                  'properties': {'labels': {'type': 'array', 'items': {'type': 'string'}}},
                  'required': ['labels']}
        client = getattr(judge_llm, '_client', None)
        resp = client.chat_with_json_output(prompt, json_schema=schema, temperature=0)
        got = resp.get('labels', [])
        if len(got) == len(questions):
            return [g if g in VALID_INTENT_LABELS else classify_intent_rule(q)
                    for g, q in zip(got, questions)]
    except Exception as e:  # noqa: BLE001 - 分类失败降级规则兜底
        log.info(f'[多轮] LLM 意图分类失败，回退规则分类: {e}')
    return [classify_intent_rule(q) for q in questions]


def _serialize_state_result(result) -> Dict:
    """MultiTurnResult → 可落盘 dict（枚举转字符串）"""
    if result is None:
        return {}
    return {
        'total_turns': result.total_turns,
        'valid_turns': result.valid_turns,
        'invalid_turns': result.invalid_turns,
        'drift_rate': result.drift_rate,
        'severity_distribution': result.severity_distribution,
        'transitions': [{
            'turn': t.turn,
            'current_state': t.current_state,
            'detected_intent': t.detected_intent,
            'expected_next_state': t.expected_next_state,
            'is_valid': t.is_valid,
            'drift_severity': t.drift_severity.value
            if isinstance(t.drift_severity, DriftSeverity) else str(t.drift_severity),
            'matched_rule': t.matched_rule,
            'detail': t.detail,
        } for t in result.transitions],
        'summary': result.summary,
    }


def _serialize_judge_result(result) -> Dict:
    """MultiTurnEvalResult → 可落盘 dict（剔除冗余 report_text）"""
    d = asdict(result)
    d.pop('report_text', None)
    return d


def _intent_confusion(rounds: List[Dict]) -> Dict:
    """逐轮意图混淆矩阵：期望 round_intent × 实际 actual_intent"""
    labels = list(VALID_INTENT_LABELS)
    idx = {l: i for i, l in enumerate(labels)}
    matrix = [[0] * len(labels) for _ in labels]
    for r in rounds:
        e, a = r.get('round_intent'), r.get('actual_intent')
        if e in idx and a in idx:
            matrix[idx[e]][idx[a]] += 1
    return {'labels': labels, 'matrix': matrix}


def run_multi_turn(args, kb, data: Dict):
    """多轮评测：对话执行 → 本地断言 → 状态机 → LLM Judge → 序列化"""
    cases = load_multi_turn(limit=args.limit)
    log.info(f'[多轮] 测试用例数: {len(cases)}')

    answer_llm = build_answer_llm()
    cache = qa_system.AnswerCache() if config.CACHE_ENABLED else None
    judge_llm = build_judge_llm()

    # 1. 逐 case 执行对话（历史累积 + 逐轮计时）
    trajectories = []
    for case in cases:
        traj = qa_system.run_multi_turn_dialog(case, kb, answer_llm,
                                               force=args.force_answer, cache=cache)
        trajectories.append(traj)
        log.info(f'[多轮] {case.case_id} 对话执行完成（{traj.round_count} 轮）')

    # 2. 逐轮本地断言：数值一致性 + 意图分类（规则 or LLM）
    all_rounds: List[Dict] = []
    for traj in trajectories:
        for r in traj.rounds:
            numeric = check_numeric_consistency(r.expected_answer, r.actual_answer)
            all_rounds.append({
                'case_id': traj.case_id,
                'round_no': r.round_no,
                'user_question': r.user_question,
                'expected_answer': r.expected_answer,
                'actual_answer': r.actual_answer,
                'round_intent': r.round_intent,
                'state_intent': r.state_intent,
                'context_assert_type': r.context_assert_type,
                'expect_refuse': r.expect_refuse,
                'numeric_check': asdict(numeric),
                'latency': round(r.latency, 3),
                'cache_hit': r.cache_hit,
            })

    if args.intent_llm:
        detected = _classify_intents_llm([r['user_question'] for r in all_rounds], judge_llm)
        for r, intent in zip(all_rounds, detected):
            r['actual_intent'] = intent
    for r in all_rounds:
        actual = r.get('actual_intent') or check_round_intent(
            r['round_intent'], r['user_question'])[1]
        r['actual_intent'] = actual
        r['intent_pass'] = actual == r['round_intent']

    # 3. 会话状态机验证（仅对标注了 state_intent 的轮次；状态链式推进）
    from collections import Counter as _Counter

    from eval.metrics.state_validator import MultiTurnResult
    validator = build_state_validator()
    state_results: Dict[str, Dict] = {}
    for traj in trajectories:
        annotated = [(r.round_no, r.state_intent) for r in traj.rounds if r.state_intent]
        if not annotated:
            state_results[traj.case_id] = {}
            continue
        transitions = []
        current, prev_state = 'INIT', None
        for turn, intent in annotated:
            res = validator.validate_single_turn(
                current_state=current, detected_intent=intent,
                turn=turn, previous_state=prev_state)
            transitions.append(res)
            prev_state = res.expected_next_state or current
            current = prev_state
        valid_n = sum(1 for t in transitions if t.is_valid)
        invalid_n = len(transitions) - valid_n
        result = MultiTurnResult(
            total_turns=len(transitions), valid_turns=valid_n, invalid_turns=invalid_n,
            drift_rate=round(invalid_n / len(transitions), 4) if transitions else 0.0,
            severity_distribution=dict(
                _Counter(t.drift_severity.value for t in transitions if not t.is_valid)),
            transitions=transitions,
            summary=(f'共{len(transitions)}轮，合法{valid_n}，非法{invalid_n}，'
                     f'漂移率{invalid_n / len(transitions):.2%}' if transitions else '空轨迹'))
        state_results[traj.case_id] = _serialize_state_result(result)

    # 4. LLM Judge（G1/G2 门禁 + Q1~Q5 加权；可批量合并请求）
    judge = build_multi_turn_judge(judge_llm)
    items = [{'dialogue': t.to_dialogue(), 'case_id': t.case_id, 'case_desc': t.case_desc}
             for t in trajectories]
    if args.batch:
        strategy = MultiTurnJudgeStrategy(
            judge_prompt_builder=judge.build_batch_prompt,
            group_size=config.BATCH_GROUP_SIZE_MULTI_JUDGE)
        batch_runner = BatchRunner(judge_llm, strategy)
        judge_results = judge.evaluate_batch(items, batch_runner)
    else:
        judge_results = [judge.evaluate(it['dialogue'], case_id=it['case_id'],
                                        case_desc=it['case_desc']) for it in items]

    # 5. 序列化：rounds 按 case 分组 + judge + state
    case_dicts = []
    for traj, jr in zip(trajectories, judge_results):
        rounds = [r for r in all_rounds if r['case_id'] == traj.case_id]
        case_dicts.append({
            'case_id': traj.case_id,
            'case_type': traj.case_type,
            'case_desc': traj.case_desc,
            'rounds': rounds,
            'judge': _serialize_judge_result(jr),
            'state': state_results.get(traj.case_id, {}),
        })

    # 6. 聚合：judge 维度均分 + 门禁通过率 + 漂移率 + 意图混淆矩阵
    aggregate = aggregate_results(judge_results)
    drift_vals = [c['state'].get('drift_rate') for c in case_dicts if c['state']]
    aggregate['avg_drift_rate'] = round(sum(drift_vals) / len(drift_vals), 4) if drift_vals else 0
    aggregate['intent_confusion'] = _intent_confusion(all_rounds)

    perf = _perf_stats([r['latency'] for r in all_rounds])
    perf['amortized_count'] = 0  # 多轮对话流不可合并，无均摊耗时
    data['multi_turn'] = {'cases': case_dicts, 'aggregate': aggregate, 'perf': perf}
    return judge_llm.get_model_name(), answer_llm.get_model_name()


# ============================================================
# 入口
# ============================================================

def main(argv=None):
    parser = argparse.ArgumentParser(description='信贷问答助手评测（单轮/多轮）')
    parser.add_argument('--mode', choices=['single', 'multi', 'all'],
                        default='single', help='评测模式（默认 single）')
    parser.add_argument('--generate-testset', action='store_true',
                        help='先按 skills/*.md 技能文档重新生成测试集（默认直接读已有 CSV）')
    parser.add_argument('--batch', action='store_true',
                        help='批量模式：作答/judge 合并请求降本（默认非批量）')
    parser.add_argument('--force-answer', action='store_true',
                        help='忽略作答缓存，重新调用问答模型')
    parser.add_argument('--limit', type=int, default=None, help='只评测前 N 条用例')
    parser.add_argument('--intent-llm', action='store_true',
                        help='多轮意图分类走 LLM（默认本地关键词规则）')
    args = parser.parse_args(argv)
    run_id = _new_run_id()
    run_dir = os.path.join(config.RESULTS_HISTORY_DIR, run_id)
    started_at = datetime.now()
    start_ts = time.time()
    log.info(f'[评测] 本次运行标识 run_id={run_id}')

    # 0. 测试集（默认读已有 CSV；--generate-testset 才走技能文档生成）
    if args.generate_testset:
        from eval.synthesize_cases import generate_testset
        modes = ['single', 'multi'] if args.mode == 'all' else [args.mode]
        for m in modes:
            generate_testset(m, force=True)

    # 1. 知识库注入（被测模型作答时检索）
    kb = KnowledgeBase.from_markdown()
    log.info(f'[评测] 知识库载入 {len(kb.pairs)} 条问答对')

    data: Dict = {
        'mode': args.mode,
        'run_id': run_id,
        'model': '', 'answer_model': '',
        'generated_at': started_at.strftime('%Y-%m-%d %H:%M:%S'),
    }

    # 2. 执行流水线
    judge_name = answer_name = ''
    if args.mode in ('single', 'all'):
        judge_name, answer_name = run_single_turn(args, kb, data)
    if args.mode in ('multi', 'all'):
        j, a = run_multi_turn(args, kb, data)
        judge_name = judge_name or j
        answer_name = answer_name or a
    data['model'] = judge_name
    data['answer_model'] = answer_name

    # 3. 统一落盘：历史快照 + latest 副本 + 运行配置快照
    duration = time.time() - start_ts
    _persist_results(data, run_dir)
    _write_run_meta(run_dir, run_id, started_at, duration, args, data)

    # 4. 报告（latest 副本）+ 复制进历史快照
    report_files = []
    if args.mode in ('single', 'all'):
        generate_report()
        report_files.append(config.HTML_REPORT_PATH)
    if args.mode in ('multi', 'all'):
        generate_multi_turn_report()
        report_files.append(config.MULTI_TURN_HTML_REPORT_PATH)
    for rp in report_files:
        shutil.copy2(rp, os.path.join(run_dir, os.path.basename(rp)))

    # 5. 自动清理过期历史快照，并重建历史索引页
    _cleanup_history()
    generate_index()
    log.info(f'[评测] 本次运行完成，历史快照目录 -> {run_dir}')


if __name__ == '__main__':
    main()
