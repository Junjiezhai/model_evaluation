# -*- coding: utf-8 -*-
"""
评测结果可视化：自包含静态 HTML 报告（无外部 CDN 依赖）

两个模板：
  - generate_report()         单轮报告：总览 / 指标详情 / 分类聚合 / 性能区块 / 失败置顶 / 明细
  - generate_multi_turn_report() 多轮报告：总览 / Q1~Q5 雷达图(内联 SVG) / G1-G2 门禁 /
                              维度得分 / 状态转移合规 / 本地断言 / 意图混淆矩阵 / 性能 / 逐 case 明细

输入：results/evaluation_results.json（统一 schema，run_evaluation.py 产出）
"""

import html
import json
import math
import os
from collections import defaultdict
from typing import Dict, List

import config
from utils.handle_log import log

METRIC_LABELS = {
    'Faithfulness': '忠实度',
    'AnswerRelevancy': '答案相关性',
    'ContextualRecall': '上下文召回',
    'correctness': '答案正确性（GEval）',
}

Q_DIMENSIONS = [
    ('Q1_context_memory', 'Q1.上下文记忆'),
    ('Q2_intent_responsiveness', 'Q2.意图响应度'),
    ('Q3_contamination', 'Q3.上下文污染度'),
    ('Q4_coherence', 'Q4.对话连贯性'),
    ('Q5_consistency', 'Q5.信息一致性'),
]


# ============================================================
# 通用工具
# ============================================================

def _esc(text) -> str:
    return html.escape(str(text if text is not None else '')).replace('\n', '<br>')


def _metric_label(name: str) -> str:
    return METRIC_LABELS.get(name, name)


def _bar(pct: float, color: str) -> str:
    pct = max(0.0, min(100.0, pct))
    return (f'<div class="bar-track"><div class="bar-fill" '
            f'style="width:{pct:.1f}%;background:{color}"></div></div>')


def _score_color(score: float, threshold: float) -> str:
    if score is None:
        return '#9ca3af'
    return '#16a34a' if score >= threshold else '#dc2626'


def _percentile(sorted_vals: List[float], p: float) -> float:
    """最近秩分位数（纯 Python，p 取 0~1）"""
    if not sorted_vals:
        return 0.0
    k = max(1, math.ceil(p * len(sorted_vals)))
    return sorted_vals[min(k, len(sorted_vals)) - 1]


def _perf_block(perf: Dict, title: str = '作答性能') -> str:
    """性能区块：平均 / P90 / P95 / 最大 / 最小耗时"""
    if not perf or not perf.get('count'):
        return '<p class="muted">无性能数据</p>'
    rows = ''.join([
        f'<tr><td>{_esc(k)}</td><td><b>{v:.3f}s</b></td></tr>'
        for k, v in [('平均耗时', perf.get('avg', 0)), ('P90 耗时', perf.get('p90', 0)),
                     ('P95 耗时', perf.get('p95', 0)), ('最大耗时', perf.get('max', 0)),
                     ('最小耗时', perf.get('min', 0))]
    ])
    amortized = perf.get('amortized_count', 0)
    note = (f'<p class="muted">共 {perf["count"]} 次作答'
            + (f'，其中 {amortized} 次为批量模式均摊耗时' if amortized else '') + '</p>')
    return f'<h2>{title}</h2><table style="width:auto"><tr><th>指标</th><th>耗时</th></tr>{rows}</table>{note}'


# ============================================================
# 单轮报告
# ============================================================

def _distribution_bars(scores: List[float], threshold: float) -> str:
    if not scores:
        return '<span class="muted">无数据</span>'
    bins = [0] * 10
    for s in scores:
        bins[min(int(s * 10), 9)] += 1
    max_cnt = max(bins) or 1
    labels = [f'{i / 10:.1f}' for i in range(10)]
    cells = []
    for i, cnt in enumerate(bins):
        h = cnt / max_cnt * 100
        color = '#16a34a' if i / 10 >= threshold else '#dc2626'
        cells.append(
            f'<div class="hist-cell" title="{labels[i]}~{labels[i]}+: {cnt} 条">'
            f'<div class="hist-bar" style="height:{h:.0f}%;background:{color if cnt else "#e5e7eb"}"></div>'
            f'<div class="hist-count">{cnt or ""}</div>'
            f'<div class="hist-label">{labels[i]}</div></div>'
        )
    return f'<div class="hist">{"".join(cells)}</div>'


def _metric_chips(case: Dict) -> str:
    chips = []
    for m in case['metrics']:
        color = _score_color(m['score'], m['threshold'])
        chips.append(
            f'<div class="chip" style="border-color:{color}">'
            f'<span class="chip-name">{_esc(_metric_label(m["name"]))}</span>'
            f'<span class="chip-score" style="color:{color}">{m["score"]:.2f}</span>'
            f'<span class="chip-th">/{m["threshold"]:.2f}</span></div>'
        )
    latency = case.get('latency')
    if latency is not None:
        mark = '*' if case.get('latency_is_amortized') else ''
        chips.append(f'<div class="chip chip-latency">⏱ {latency:.3f}s{mark}</div>')
    return ''.join(chips)


def _reason_block(case: Dict) -> str:
    items = []
    for m in case['metrics']:
        items.append(
            f'<div class="reason-item"><b>{_esc(_metric_label(m["name"]))}'
            f'（{m["score"]:.2f} / 阈值 {m["threshold"]:.2f}）：</b>'
            f'{_esc(m.get("reason") or "（无理由）")}</div>'
        )
    return f'<details><summary>Judge 评分理由</summary>{"".join(items)}</details>'


def _case_row(case: Dict) -> str:
    contexts = '<hr>'.join(_esc(c) for c in case.get('retrieval_context', []))
    return f'''
    <tr>
      <td class="col-input"><b>{_esc(case.get('case_id', ''))}</b><br>{_esc(case['input'])}
        <div class="tags"><span class="tag">{_esc(case.get('category', ''))}</span>
        <span class="tag">{_esc(case.get('test_intent', ''))}</span>
        <span class="tag">{_esc(case.get('priority', ''))}</span></div>
      </td>
      <td class="col-expected">{_esc(case['expected_output'])}</td>
      <td class="col-actual">{_esc(case['actual_output'])}</td>
      <td class="col-ctx">{contexts}</td>
      <td class="col-metrics">{_metric_chips(case)}{_reason_block(case)}</td>
    </tr>'''


_BASE_CSS = '''
  * { box-sizing: border-box; }
  body { font-family: "Microsoft YaHei", "PingFang SC", sans-serif; margin: 0; background: #f3f4f6; color: #1f2937; }
  .container { max-width: 1500px; margin: 0 auto; padding: 24px; }
  h1 { font-size: 22px; } h2 { font-size: 17px; margin-top: 32px; border-left: 4px solid #2563eb; padding-left: 8px; }
  .muted { color: #6b7280; font-size: 12px; }
  .cards { display: flex; gap: 16px; flex-wrap: wrap; margin: 16px 0; }
  .card { background: #fff; border-radius: 10px; padding: 18px 32px; box-shadow: 0 1px 3px rgba(0,0,0,.08); text-align: center; }
  .card-num { font-size: 28px; font-weight: 700; } .card-label { color: #6b7280; font-size: 13px; margin-top: 4px; }
  .metrics-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 16px; }
  .metric-box { background: #fff; border-radius: 10px; padding: 16px; box-shadow: 0 1px 3px rgba(0,0,0,.08); }
  .metric-head { display: flex; align-items: baseline; gap: 10px; margin-bottom: 8px; }
  .metric-name { font-weight: 600; } .metric-avg { font-size: 22px; font-weight: 700; margin-left: auto; }
  .bar-track { background: #e5e7eb; border-radius: 6px; height: 10px; overflow: hidden; margin: 6px 0; }
  .bar-fill { height: 100%; border-radius: 6px; }
  .hist { display: flex; align-items: flex-end; gap: 3px; height: 90px; margin-top: 12px; }
  .hist-cell { flex: 1; display: flex; flex-direction: column; align-items: center; height: 100%; justify-content: flex-end; }
  .hist-bar { width: 100%; border-radius: 3px 3px 0 0; min-height: 2px; }
  .hist-count { font-size: 10px; color: #6b7280; height: 14px; }
  .hist-label { font-size: 10px; color: #9ca3af; }
  table { width: 100%; border-collapse: collapse; background: #fff; font-size: 13px; box-shadow: 0 1px 3px rgba(0,0,0,.08); }
  th, td { border: 1px solid #e5e7eb; padding: 8px 10px; vertical-align: top; text-align: left; }
  th { background: #f9fafb; position: sticky; top: 0; }
  .col-input { width: 16%; } .col-expected { width: 18%; } .col-actual { width: 20%; } .col-ctx { width: 22%; } .col-metrics { width: 24%; }
  .chip { display: inline-flex; gap: 4px; border: 1px solid; border-radius: 20px; padding: 2px 10px; margin: 2px; font-size: 12px; align-items: baseline; }
  .chip-score { font-weight: 700; } .chip-th { color: #9ca3af; font-size: 10px; }
  .chip-latency { border-color: #d1d5db; color: #4b5563; }
  .tags { margin-top: 6px; }
  .tag { display: inline-block; background: #eef2ff; color: #4338ca; border-radius: 4px; padding: 1px 6px; font-size: 11px; margin-right: 4px; }
  details { margin-top: 6px; font-size: 12px; color: #4b5563; }
  details summary { cursor: pointer; color: #2563eb; }
  .reason-item { margin: 6px 0; padding: 6px 8px; background: #f9fafb; border-radius: 6px; }
'''


def render_report(data: Dict) -> str:
    """单轮报告渲染（输入 data['single_turn']）"""
    cases: List[Dict] = data.get('single_turn', {}).get('cases', data.get('cases', []))
    total = len(cases)

    metric_stats: Dict[str, Dict] = {}
    for case in cases:
        for m in case['metrics']:
            st = metric_stats.setdefault(m['name'], {'scores': [], 'pass': 0, 'threshold': m['threshold']})
            st['scores'].append(m['score'])
            st['pass'] += int(m['success'])
    for name, st in metric_stats.items():
        st['avg'] = sum(st['scores']) / len(st['scores']) if st['scores'] else 0
        st['pass_rate'] = st['pass'] / len(st['scores']) if st['scores'] else 0

    overall_pass = sum(1 for c in cases if all(m['success'] for m in c['metrics']))
    failed_cases = [c for c in cases if not all(m['success'] for m in c['metrics'])]

    summary_cards = f'''
      <div class="card"><div class="card-num">{total}</div><div class="card-label">测试用例</div></div>
      <div class="card"><div class="card-num">{overall_pass}</div><div class="card-label">全部通过</div></div>
      <div class="card"><div class="card-num" style="color:{'#16a34a' if overall_pass == total else '#dc2626'}">{overall_pass / total * 100 if total else 0:.0f}%</div><div class="card-label">整体通过率</div></div>'''

    metric_sections = []
    for name, st in metric_stats.items():
        color = _score_color(st['avg'], st['threshold'])
        metric_sections.append(f'''
      <div class="metric-box">
        <div class="metric-head">
          <span class="metric-name">{_esc(_metric_label(name))}</span>
          <span class="metric-avg" style="color:{color}">{st['avg']:.2f}</span>
          <span class="metric-th muted">阈值 {st['threshold']:.2f}</span>
        </div>
        {_bar(st['avg'] * 100, color)}
        <div class="muted">通过率 {st['pass_rate'] * 100:.0f}%（{st['pass']}/{len(st['scores'])}）</div>
        {_distribution_bars(st['scores'], st['threshold'])}
      </div>''')

    cat_stats = defaultdict(list)
    for c in cases:
        avg = sum(m['score'] for m in c['metrics']) / len(c['metrics'])
        cat_stats[c.get('category', '未分类')].append(avg)
    cat_rows = ''.join(
        f'<tr><td>{_esc(cat)}</td><td>{len(v)}</td><td>{sum(v) / len(v):.2f}</td></tr>'
        for cat, v in sorted(cat_stats.items(), key=lambda x: -sum(x[1]) / len(x[1]))
    )

    failed_rows = ''.join(_case_row(c) for c in failed_cases)
    all_rows = ''.join(_case_row(c) for c in cases)
    perf = data.get('single_turn', {}).get('perf')

    return f'''<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<title>信贷问答助手评测报告（单轮）</title>
<style>{_BASE_CSS}</style>
</head>
<body>
<div class="container">
  <h1>信贷知识库智能问答助手 · 单轮评测报告</h1>
  <div class="muted">评测模型：{_esc(data.get('model', ''))} ｜ 被测问答模型：{_esc(data.get('answer_model', ''))} ｜ 生成时间：{_esc(data.get('generated_at', ''))}</div>

  <h2>总体概览</h2>
  <div class="cards">{summary_cards}</div>

  <h2>指标详情（Judge 打分）</h2>
  <div class="metrics-grid">{''.join(metric_sections)}</div>

  <h2>按知识分类聚合（各用例指标均分）</h2>
  <table><tr><th>分类</th><th>用例数</th><th>平均得分</th></tr>{cat_rows}</table>

  {_perf_block(perf) if perf else ''}

  <h2>失败用例（任一指标未达阈值）：{len(failed_cases)} 条</h2>
  {f'<table><tr><th>问题</th><th>标准答案</th><th>模型答案</th><th>检索上下文</th><th>指标得分</th></tr>{failed_rows}</table>' if failed_cases else '<p class="muted">无失败用例 🎉</p>'}

  <h2>全部用例明细</h2>
  <table><tr><th>问题</th><th>标准答案</th><th>模型答案</th><th>检索上下文</th><th>指标得分</th></tr>{all_rows}</table>
  <p class="muted">⏱ 为单次作答耗时；带 * 为批量模式均摊耗时</p>
</div>
</body>
</html>'''


# ============================================================
# 多轮报告（含雷达图）
# ============================================================

def _radar_svg(dim_scores: List[float], labels: List[str], size: int = 360) -> str:
    """Q1~Q5 五维雷达图（纯内联 SVG，无外部依赖）

    Args:
        dim_scores: 5 个 0~100 的维度均分（顺序与 labels 一致）
    """
    n = len(labels)
    cx = cy = size / 2
    radius = size / 2 - 60
    colors = {'grid': '#e5e7eb', 'axis': '#d1d5db', 'poly': '#2563eb',
              'poly_fill': 'rgba(37, 99, 235, 0.18)', 'label': '#374151', 'value': '#1d4ed8'}

    def point(i: int, ratio: float):
        angle = -math.pi / 2 + 2 * math.pi * i / n
        return cx + radius * ratio * math.cos(angle), cy + radius * ratio * math.sin(angle)

    parts = [f'<svg width="{size}" height="{size}" viewBox="0 0 {size} {size}" role="img" aria-label="多轮质量维度雷达图">']
    # 网格环（20/40/60/80/100%）
    for ratio in (0.2, 0.4, 0.6, 0.8, 1.0):
        pts = ' '.join(f'{point(i, ratio)[0]:.1f},{point(i, ratio)[1]:.1f}' for i in range(n))
        parts.append(f'<polygon points="{pts}" fill="none" stroke="{colors["grid"]}" stroke-width="1"/>')
    # 轴 + 标签 + 刻度值
    for i, label in enumerate(labels):
        x, y = point(i, 1.0)
        parts.append(f'<line x1="{cx}" y1="{cy}" x2="{x:.1f}" y2="{y:.1f}" stroke="{colors["axis"]}"/>')
        lx, ly = point(i, 1.18)
        anchor = 'middle'
        if lx > cx + 10:
            anchor = 'start'
        elif lx < cx - 10:
            anchor = 'end'
        score = dim_scores[i] if i < len(dim_scores) else 0
        parts.append(
            f'<text x="{lx:.1f}" y="{ly:.1f}" text-anchor="{anchor}" font-size="12" '
            f'fill="{colors["label"]}">{_esc(label)}</text>')
        vx, vy = point(i, 1.18)
        parts.append(
            f'<text x="{vx:.1f}" y="{vy + 15:.1f}" text-anchor="{anchor}" font-size="11" '
            f'font-weight="bold" fill="{colors["value"]}">{score:.1f}</text>')
    # 数据多边形
    def _data_pt(i: int):
        ratio = max(0.0, min(1.0, dim_scores[i] / 100))
        return point(i, ratio)

    data_pts = ' '.join(f'{_data_pt(i)[0]:.1f},{_data_pt(i)[1]:.1f}'
                        for i in range(min(n, len(dim_scores))))
    parts.append(f'<polygon points="{data_pts}" fill="{colors["poly_fill"]}" stroke="{colors["poly"]}" stroke-width="2"/>')
    for i in range(min(n, len(dim_scores))):
        x, y = point(i, max(0.0, min(1.0, dim_scores[i] / 100)))
        parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3.5" fill="{colors["poly"]}"/>')
    parts.append('</svg>')
    return ''.join(parts)


def _heat_color(ratio: float) -> str:
    """混淆矩阵热力着色（红→黄→绿）"""
    ratio = max(0.0, min(1.0, ratio))
    r = int(220 + (74 - 220) * ratio)
    g = int(38 + (222 - 38) * ratio)
    b = int(38 + (128 - 38) * ratio)
    return f'#{r:02x}{g:02x}{b:02x}'


def _round_rows_html(rounds: List[Dict]) -> str:
    rows = []
    for r in rounds:
        intent_ok = r.get('intent_pass')
        intent_cell = (f'<span style="color:{"#16a34a" if intent_ok else "#dc2626"}">'
                       f'{_esc(r.get("actual_intent", ""))}{"✓" if intent_ok else "✗"}</span>')
        num = r.get('numeric_check') or {}
        num_cell = ('<span style="color:#16a34a">一致</span>' if num.get('passed')
                    else f'<span style="color:#dc2626">缺失: {_esc(", ".join(num.get("missing", [])))}</span>')
        rows.append(
            f'<tr><td>{r["round_no"]}</td><td>{_esc(r["user_question"])}</td>'
            f'<td>{_esc(r["expected_answer"])}</td><td>{_esc(r["actual_answer"])}</td>'
            f'<td>{_esc(r.get("round_intent", ""))}<br>{intent_cell}</td>'
            f'<td>{num_cell}</td><td>{r.get("latency", 0):.3f}s</td></tr>')
    return ''.join(rows)


def render_multi_turn_report(data: Dict) -> str:
    """多轮报告渲染（输入 data['multi_turn']）"""
    mt = data.get('multi_turn', {})
    cases: List[Dict] = mt.get('cases', [])
    agg: Dict = mt.get('aggregate', {})
    perf: Dict = mt.get('perf', {})
    total = len(cases)
    gate_pass = agg.get('gate_pass_count', 0)
    avg_score = agg.get('avg_overall_score', 0)
    dim_avgs: Dict[str, float] = agg.get('dimension_avgs', {})

    # 总览卡片
    score_color = '#16a34a' if avg_score >= config.MULTI_TURN_THRESHOLD else '#dc2626'
    summary_cards = f'''
      <div class="card"><div class="card-num">{total}</div><div class="card-label">多轮用例</div></div>
      <div class="card"><div class="card-num">{gate_pass}/{total}</div><div class="card-label">底线通过（G1+G2）</div></div>
      <div class="card"><div class="card-num" style="color:{score_color}">{avg_score:.1f}</div><div class="card-label">平均综合分（及格线 {config.MULTI_TURN_THRESHOLD}）</div></div>
      <div class="card"><div class="card-num">{agg.get("avg_drift_rate", 0) * 100:.1f}%</div><div class="card-label">平均意图漂移率</div></div>'''

    # 雷达图 + 维度表
    dim_scores = [dim_avgs.get(k, 0) for k, _ in Q_DIMENSIONS]
    labels = [label for _, label in Q_DIMENSIONS]
    radar = _radar_svg(dim_scores, labels)
    dim_rows = ''.join(
        f'<tr><td>{_esc(label)}</td>'
        f'<td><b style="color:{_score_color(dim_avgs.get(k, 0), 60)}">{dim_avgs.get(k, 0):.1f}</b></td>'
        f'<td>{config.MULTI_TURN_DIMENSION_WEIGHTS.get(k, 0) * 100:.0f}%</td></tr>'
        for k, label in Q_DIMENSIONS)

    # G1/G2 门禁表
    gate_rows = []
    for c in cases:
        for g in (c.get('judge', {}).get('gate_checks') or []):
            verdict = g.get('verdict', '')
            name = 'G1.业务合规' if g.get('dimension') == 'G1_compliance' else 'G2.客诉控制'
            extra = ', '.join(g.get('failed_items') or g.get('detected_signals') or [])
            gate_rows.append(
                f'<tr><td>{_esc(c["case_id"])}</td><td>{name}</td>'
                f'<td style="color:{"#16a34a" if verdict == "PASS" else "#dc2626"};font-weight:bold">{verdict}</td>'
                f'<td>{_esc(extra)}</td><td>{_esc(g.get("reason", ""))}</td></tr>')

    # 状态转移合规
    state_rows = []
    for c in cases:
        st = c.get('state', {})
        for t in st.get('transitions', []):
            ok = t.get('is_valid')
            state_rows.append(
                f'<tr><td>{_esc(c["case_id"])}</td><td>第{t.get("turn")}轮</td>'
                f'<td>{_esc(t.get("current_state", ""))} + [{_esc(t.get("detected_intent", ""))}]</td>'
                f'<td>{_esc(t.get("expected_next_state") or "—")}</td>'
                f'<td style="color:{"#16a34a" if ok else "#dc2626"}">{"✓ 合法" if ok else "✗ " + _esc(t.get("drift_severity", ""))}</td></tr>')

    # 意图混淆矩阵
    conf = agg.get('intent_confusion') or {}
    conf_html = ''
    if conf.get('labels'):
        labels_c = conf['labels']
        matrix = conf['matrix']
        max_cnt = max((max(row) for row in matrix), default=1) or 1
        head = '<tr><th>期望 \\ 实际</th>' + ''.join(f'<th>{_esc(l)}</th>' for l in labels_c) + '</tr>'
        body = ''
        for i, row in enumerate(matrix):
            cells = ''.join(
                f'<td style="background:{_heat_color(cnt / max_cnt) if cnt else "#fff"};'
                f'text-align:center">{cnt or ""}</td>' for cnt in row)
            body += f'<tr><th>{_esc(labels_c[i])}</th>{cells}</tr>'
        conf_html = f'<h2>逐轮意图混淆矩阵</h2><table style="width:auto">{head}{body}</table>'

    # 质量分布
    qd = agg.get('quality_distribution', {})
    dist_text = ' ｜ '.join(
        f'{name}: {qd.get(key, 0)}' for key, name in
        [('excellent', '优秀(≥85)'), ('good', '良好(70-85)'), ('fair', '一般(50-70)'), ('poor', '较差(<50)')])

    # 失败 case 置顶（门禁失败或综合分低于及格线）
    failed_cases = [c for c in cases
                    if not c.get('judge', {}).get('gate_passed', True)
                    or c.get('judge', {}).get('overall_score', 100) < config.MULTI_TURN_THRESHOLD]
    case_blocks = []
    for c in failed_cases:
        judge = c.get('judge', {})
        case_blocks.append(f'''
    <details open class="case-detail">
      <summary><b>{_esc(c["case_id"])}</b> ｜ {_esc(c.get("case_desc", ""))} ｜
        综合分 <b style="color:{_score_color(judge.get("overall_score", 0), config.MULTI_TURN_THRESHOLD)}">{judge.get("overall_score", 0):.1f}</b>
        ｜ 漂移率 {(c.get("state", {}).get("drift_rate") or 0) * 100:.0f}%</summary>
      <table><tr><th>轮次</th><th>用户问题</th><th>期望应答</th><th>模型应答</th><th>意图(期望/实际)</th><th>数值一致性</th><th>耗时</th></tr>
      {_round_rows_html(c.get("rounds", []))}</table>
      <div class="reason-item"><b>亮点：</b>{_esc('；'.join(judge.get("highlights", [])) or '—')}</div>
      <div class="reason-item"><b>问题：</b>{_esc('；'.join(judge.get("issues", [])) or '—')}</div>
      <div class="reason-item"><b>总评：</b>{_esc(judge.get("overall_comment", ""))}</div>
    </details>''')

    return f'''<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<title>信贷问答助手多轮评测报告</title>
<style>{_BASE_CSS}
  .radar-wrap {{ background:#fff; border-radius:10px; padding:16px; box-shadow:0 1px 3px rgba(0,0,0,.08); display:inline-block; }}
  .case-detail {{ background:#fff; border-radius:10px; padding:12px; margin:12px 0; box-shadow:0 1px 3px rgba(0,0,0,.08); }}
  .case-detail summary {{ font-size:14px; }}
</style>
</head>
<body>
<div class="container">
  <h1>信贷知识库智能问答助手 · 多轮对话评测报告</h1>
  <div class="muted">评测模型：{_esc(data.get('model', ''))} ｜ 被测问答模型：{_esc(data.get('answer_model', ''))} ｜ 生成时间：{_esc(data.get('generated_at', ''))}
    ｜ 状态验证模式：{_esc(config.STATE_VALIDATION_MODE)}</div>

  <h2>总体概览</h2>
  <div class="cards">{summary_cards}</div>
  <p class="muted">质量分布：{_esc(dist_text)}</p>

  <h2>Q1~Q5 质量维度雷达图（均分）</h2>
  <div class="radar-wrap">{radar}</div>
  <table style="width:auto; margin-top:12px"><tr><th>维度</th><th>均分</th><th>权重</th></tr>{dim_rows}</table>

  <h2>G1/G2 底线门禁结果</h2>
  {f'<table><tr><th>用例</th><th>门禁</th><th>判定</th><th>违规项/信号</th><th>说明</th></tr>{"".join(gate_rows)}</table>' if gate_rows else '<p class="muted">无数据</p>'}

  <h2>会话状态转移合规（逐轮）</h2>
  {f'<table><tr><th>用例</th><th>轮次</th><th>状态+意图</th><th>期望下一状态</th><th>判定</th></tr>{"".join(state_rows)}</table>' if state_rows else '<p class="muted">无用例标注 state_intent，已跳过状态机验证</p>'}

  {conf_html}

  {_perf_block(perf, '作答性能（逐轮）')}

  <h2>失败用例（门禁未通过或综合分低于 {config.MULTI_TURN_THRESHOLD}）：{len(failed_cases)} 条</h2>
  {''.join(case_blocks) if case_blocks else '<p class="muted">无失败用例 🎉</p>'}
</div>
</body>
</html>'''


# ============================================================
# 入口
# ============================================================

def generate_report(results_path: str = None, output_path: str = None) -> str:
    """单轮报告入口"""
    results_path = results_path or config.RAW_RESULTS_PATH
    output_path = output_path or config.HTML_REPORT_PATH
    with open(results_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    html_text = render_report(data)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_text)
    log.info(f'[报告] 单轮 HTML 报告已生成 -> {output_path}')
    return output_path


def generate_multi_turn_report(results_path: str = None, output_path: str = None) -> str:
    """多轮报告入口"""
    results_path = results_path or config.RAW_RESULTS_PATH
    output_path = output_path or config.MULTI_TURN_HTML_REPORT_PATH
    with open(results_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    html_text = render_multi_turn_report(data)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_text)
    log.info(f'[报告] 多轮 HTML 报告已生成 -> {output_path}')
    return output_path


# ============================================================
# 自测（构造数据渲染，不依赖真实结果文件）
# ============================================================

if __name__ == '__main__':
    import tempfile
    print('=' * 60)
    print('  report 自测（构造数据渲染）')
    print('=' * 60)

    mock_data = {
        'mode': 'all',
        'model': 'kimi-for-coding(judge)',
        'answer_model': 'kimi-for-coding(answer)',
        'generated_at': '2026-10-28 20:00:00',
        'single_turn': {
            'cases': [
                {'case_id': 'ST_0001', 'input': '怎么提额', 'expected_output': '标准答案',
                 'actual_output': '模型回答', 'retrieval_context': ['ctx'], 'category': '提额降息',
                 'test_intent': '提额降息咨询类', 'priority': 'P1', 'latency': 1.234,
                 'latency_is_amortized': False,
                 'metrics': [{'name': 'Faithfulness', 'score': 0.9, 'threshold': 0.7,
                              'success': True, 'reason': '忠实', 'error': None}]},
            ],
            'perf': {'count': 1, 'avg': 1.234, 'p90': 1.234, 'p95': 1.234,
                     'max': 1.234, 'min': 1.234, 'amortized_count': 0},
        },
        'multi_turn': {
            'cases': [
                {'case_id': 'MD_0001', 'case_desc': '[合规底线]测试 状态链路:INIT→JK_CQZX',
                 'case_type': 'risk_compliance_insert',
                 'rounds': [
                     {'round_no': 1, 'user_question': '怎么借款', 'expected_answer': 'ea',
                      'actual_answer': 'aa', 'round_intent': '借款下单类', 'actual_intent': '借款下单类',
                      'intent_pass': True, 'numeric_check': {'passed': True, 'missing': []},
                      'latency': 1.1},
                     {'round_no': 2, 'user_question': '怎么套现', 'expected_answer': 'eb',
                      'actual_answer': 'ab', 'round_intent': '风险合规禁答类', 'actual_intent': '风险合规禁答类',
                      'intent_pass': True, 'numeric_check': {'passed': False, 'missing': ['18%']},
                      'latency': 1.2},
                 ],
                 'judge': {'gate_passed': True, 'overall_score': 82.5,
                           'gate_checks': [{'dimension': 'G1_compliance', 'verdict': 'PASS',
                                            'failed_items': [], 'detected_signals': [],
                                            'reason': '未触碰红线'}],
                           'highlights': [], 'issues': ['数值缺失'], 'overall_comment': 'ok'},
                 'state': {'drift_rate': 0.0, 'transitions': [
                     {'turn': 1, 'current_state': 'INIT', 'detected_intent': '借款操作咨询',
                      'expected_next_state': 'JK_CQZX', 'is_valid': True, 'drift_severity': 'none'}]}},
            ],
            'aggregate': {
                'gate_pass_count': 1, 'gate_pass_rate': 1.0, 'avg_overall_score': 82.5,
                'dimension_avgs': {'Q1_context_memory': 87.5, 'Q2_intent_responsiveness': 100,
                                   'Q3_contamination': 100, 'Q4_coherence': 75, 'Q5_consistency': 100},
                'quality_distribution': {'excellent': 0, 'good': 1, 'fair': 0, 'poor': 0},
                'avg_drift_rate': 0.0,
                'intent_confusion': {'labels': ['借款下单类', '风险合规禁答类'],
                                     'matrix': [[2, 0], [0, 1]]},
            },
            'perf': {'count': 2, 'avg': 1.15, 'p90': 1.2, 'p95': 1.2, 'max': 1.2,
                     'min': 1.1, 'amortized_count': 0},
        },
    }

    single_html = render_report(mock_data)
    assert '作答性能' in single_html and '1.234s' in single_html
    assert '失败用例' in single_html and 'ST_0001' in single_html
    print('[OK] 单轮报告渲染（含性能区块）')

    multi_html = render_multi_turn_report(mock_data)
    assert '<svg' in multi_html and '雷达图' in multi_html
    assert 'G1/G2 底线门禁' in multi_html
    assert '会话状态转移合规' in multi_html and 'JK_CQZX' in multi_html
    assert '意图混淆矩阵' in multi_html
    assert 'Q1.上下文记忆' in multi_html and '87.5' in multi_html
    assert '失败用例' in multi_html and 'MD_0001' in multi_html
    print('[OK] 多轮报告渲染（雷达图/门禁/状态/混淆矩阵/性能）')

    with tempfile.TemporaryDirectory() as tmp:
        rp = os.path.join(tmp, 'r.json')
        with open(rp, 'w', encoding='utf-8') as f:
            json.dump(mock_data, f, ensure_ascii=False)
        out1 = generate_report(rp, os.path.join(tmp, 'single.html'))
        out2 = generate_multi_turn_report(rp, os.path.join(tmp, 'multi.html'))
        assert os.path.exists(out1) and os.path.exists(out2)
    print('[OK] 报告落盘入口')

    print('=' * 60)
    print('  全部自测通过')
    print('=' * 60)
