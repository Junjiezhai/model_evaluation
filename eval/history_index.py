# -*- coding: utf-8 -*-
"""
历史评测记录索引页（results/index.html）—— 类 Allure 首页

功能：
  - 趋势图：单轮通过率 / 多轮综合均分 / 平均作答耗时（内联 SVG 折线图，无 CDN）
  - 历史运行表：每次运行一行（时间/模式/用例数/通过率/均分/耗时），
    点击跳转该次快照的单轮报告、多轮报告与原始 JSON
  - 总览卡片：运行次数 / 最近运行 / 最近通过率 / 最近均分

数据来源：results/history/{run_id}/run_meta.json（run_evaluation 每次运行落盘）
生成时机：每次评测运行结束后自动重建；也可独立运行 python -m eval.history_index
"""

import json
import math
import os
import re
from typing import Dict, List, Optional

import config
from eval.report import _BASE_CSS, _esc, _score_color
from utils.handle_log import log

_RUN_ID_PATTERN = re.compile(r'^\d{8}_\d{6}(_\d+)?$')
_PCT_PATTERN = re.compile(r'([\d.]+)')


# ============================================================
# 数据加载
# ============================================================

def load_run_metas(history_dir: str = None) -> List[Dict]:
    """扫描历史快照目录，读取全部 run_meta.json（按 run_id 升序 = 时间升序）

    每个 meta 额外注入 '_run_dir'（相对 results/ 的路径）与
    '_reports'（该次快照实际包含的报告文件清单）。
    """
    history_dir = history_dir or config.RESULTS_HISTORY_DIR
    metas: List[Dict] = []
    if not os.path.isdir(history_dir):
        return metas
    for name in sorted(os.listdir(history_dir)):
        if not _RUN_ID_PATTERN.match(name):
            continue
        run_dir = os.path.join(history_dir, name)
        meta_path = os.path.join(run_dir, 'run_meta.json')
        if not os.path.exists(meta_path):
            continue
        try:
            with open(meta_path, 'r', encoding='utf-8') as f:
                meta = json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            log.info(f'[索引] 跳过损坏的 run_meta: {meta_path} ({e})')
            continue
        meta['_run_dir'] = f'history/{name}'
        meta['_reports'] = {
            'single': os.path.exists(os.path.join(run_dir, 'evaluation_report.html')),
            'multi': os.path.exists(os.path.join(run_dir, 'multi_turn_report.html')),
            'json': os.path.exists(os.path.join(run_dir, 'evaluation_results.json')),
        }
        metas.append(meta)
    return metas


def _parse_pct(value) -> Optional[float]:
    """'83.3%' / 83.3 / None -> float 或 None"""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    m = _PCT_PATTERN.search(str(value))
    return float(m.group(1)) if m else None


# ============================================================
# 趋势图（内联 SVG 折线图）
# ============================================================

def _trend_svg(points, title: str, color: str, y_fmt: str = '{:.0f}',
               width: int = 760, height: int = 240) -> str:
    """折线趋势图

    Args:
        points: [(x_label, Optional[float]), ...] 按时间升序；None 表示该次无数据（断点）
    """
    vals = [v for _, v in points if v is not None]
    if len(vals) < 2:
        return f'<p class="muted">{_esc(title)}：数据不足（至少 2 次有效运行）</p>'

    pad_l, pad_r, pad_t, pad_b = 52, 20, 30, 42
    cw, ch = width - pad_l - pad_r, height - pad_t - pad_b
    vmin, vmax = min(vals), max(vals)
    if vmax - vmin < 1e-9:
        vmin, vmax = vmin - 1, vmax + 1
    span = vmax - vmin
    vmin -= span * 0.1
    vmax += span * 0.1
    n = len(points)

    def xy(i: int, v: float):
        x = pad_l + (cw * i / (n - 1) if n > 1 else cw / 2)
        y = pad_t + ch * (1 - (v - vmin) / (vmax - vmin))
        return x, y

    parts = [f'<svg width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img">']
    parts.append(f'<text x="{pad_l}" y="18" font-size="13" font-weight="bold" fill="#374151">'
                 f'{_esc(title)}</text>')
    # 横向网格 + Y 轴刻度
    for g in range(5):
        gv = vmin + (vmax - vmin) * g / 4
        _, gy = xy(0, gv)
        parts.append(f'<line x1="{pad_l}" y1="{gy:.1f}" x2="{width - pad_r}" y2="{gy:.1f}" stroke="#e5e7eb"/>')
        parts.append(f'<text x="{pad_l - 6}" y="{gy + 4:.1f}" text-anchor="end" font-size="10" '
                     f'fill="#9ca3af">{y_fmt.format(gv)}</text>')
    # 折线（None 处断点）+ 数据点
    segments = [(i, *xy(i, v), v) for i, (_, v) in enumerate(points) if v is not None]
    d_cmds = []
    prev_i = None
    for i, x, y, _ in segments:
        d_cmds.append(('M' if prev_i is None or i - prev_i > 1 else 'L') + f'{x:.1f},{y:.1f}')
        prev_i = i
    parts.append(f'<path d="{" ".join(d_cmds)}" fill="none" stroke="{color}" stroke-width="2"/>')
    for i, x, y, v in segments:
        label = f'{_esc(points[i][0])}: {y_fmt.format(v)}'
        parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3.5" fill="{color}">'
                     f'<title>{label}</title></circle>')
        parts.append(f'<text x="{x:.1f}" y="{y - 8:.1f}" text-anchor="middle" font-size="10" '
                     f'fill="{color}">{y_fmt.format(v)}</text>')
    # X 轴标签（过多时隔点显示）
    step = max(1, math.ceil(n / 8))
    for i in range(0, n, step):
        x, _ = xy(i, vmin)
        parts.append(f'<text x="{x:.1f}" y="{height - pad_b + 16:.1f}" text-anchor="middle" '
                     f'font-size="10" fill="#9ca3af">{_esc(points[i][0])}</text>')
    parts.append('</svg>')
    return ''.join(parts)


# ============================================================
# 页面渲染
# ============================================================

def _trend_points(metas: List[Dict], key) -> List:
    """[(短标签, 值)]；短标签 = MM-DD HH:MM"""
    pts = []
    for m in metas:
        label = m.get('started_at', m.get('run_id', ''))[5:16]  # 'MM-DD HH:MM'
        pts.append((label or m.get('run_id', ''), key(m)))
    return pts


def render_index(metas: List[Dict]) -> str:
    """渲染历史索引页（输入按时间升序）"""
    if not metas:
        return f'''<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8"><title>评测历史索引</title>
<style>{_BASE_CSS}</style></head><body><div class="container">
<h1>信贷问答助手 · 评测历史索引</h1>
<p class="muted">暂无历史评测记录。运行 <code>python -m eval.run_evaluation</code> 后自动生成本页。</p>
</div></body></html>'''

    latest = metas[-1]
    last_single = _parse_pct(latest.get('summary', {}).get('single_pass_rate_value')
                             or latest.get('summary', {}).get('single_pass_rate'))
    last_multi_agg = latest.get('summary', {}).get('multi_aggregate') or {}
    last_multi = last_multi_agg.get('avg_overall_score')

    summary_cards = f'''
      <div class="card"><div class="card-num">{len(metas)}</div><div class="card-label">历史运行次数</div></div>
      <div class="card"><div class="card-num" style="font-size:16px">{_esc(latest.get('started_at', ''))}</div><div class="card-label">最近运行</div></div>
      <div class="card"><div class="card-num" style="color:{_score_color(last_single, 70) if last_single is not None else '#9ca3af'}">{f'{last_single:.1f}%' if last_single is not None else '—'}</div><div class="card-label">最近单轮通过率</div></div>
      <div class="card"><div class="card-num" style="color:{_score_color(last_multi, config.MULTI_TURN_THRESHOLD) if last_multi is not None else '#9ca3af'}">{f'{last_multi:.1f}' if last_multi is not None else '—'}</div><div class="card-label">最近多轮综合均分</div></div>'''

    # 三张趋势图（时间升序）
    def _avg_latency(m: Dict) -> Optional[float]:
        s = (m.get('summary', {}).get('single_perf') or {}).get('avg')
        mt = (m.get('summary', {}).get('multi_perf') or {}).get('avg')
        v = s if s is not None else mt
        return round(float(v), 2) if v is not None else None

    chart_boxes = ''.join(f'<div class="chart-box">{c}</div>' for c in [
        _trend_svg(_trend_points(metas, lambda m: _parse_pct(
            m.get('summary', {}).get('single_pass_rate_value')
            or m.get('summary', {}).get('single_pass_rate'))),
            '单轮通过率趋势（%）', '#2563eb', '{:.0f}%'),
        _trend_svg(_trend_points(metas, lambda m: (m.get('summary', {}).get('multi_aggregate') or {}).get('avg_overall_score')),
                   '多轮综合均分趋势', '#16a34a', '{:.1f}'),
        _trend_svg(_trend_points(metas, _avg_latency),
                   '平均作答耗时趋势（秒）', '#d97706', '{:.1f}s'),
    ])

    # 历史运行表（新的在前）
    rows = []
    for m in reversed(metas):
        run_dir = m.get('_run_dir', '')
        rid = m.get('run_id', '')
        counts = m.get('counts', {})
        summary = m.get('summary', {})
        single_pct = _parse_pct(summary.get('single_pass_rate_value') or summary.get('single_pass_rate'))
        multi_score = (summary.get('multi_aggregate') or {}).get('avg_overall_score')
        duration = m.get('duration_sec')
        reports = m.get('_reports', {})
        links = []
        if reports.get('single'):
            links.append(f'<a href="{run_dir}/evaluation_report.html">单轮报告</a>')
        if reports.get('multi'):
            links.append(f'<a href="{run_dir}/multi_turn_report.html">多轮报告</a>')
        if reports.get('json'):
            links.append(f'<a href="{run_dir}/evaluation_results.json">原始 JSON</a>')
        single_cell = (f'<span style="color:{_score_color(single_pct, 70)}">{single_pct:.1f}%</span>'
                       if single_pct is not None else '<span class="muted">—</span>')
        multi_cell = (f'<span style="color:{_score_color(multi_score, config.MULTI_TURN_THRESHOLD)}">{multi_score:.1f}</span>'
                      if multi_score is not None else '<span class="muted">—</span>')
        dur_cell = f'{duration:.0f}s' if duration is not None else '<span class="muted">—</span>'
        rows.append(
            f'<tr><td><b>{_esc(rid)}</b></td>'
            f'<td>{_esc(m.get("started_at", ""))}</td>'
            f'<td>{_esc(m.get("mode", ""))}</td>'
            f'<td>{_esc(m.get("answer_model", ""))}</td>'
            f'<td>{counts.get("single_turn_cases", 0)} / {counts.get("multi_turn_cases", 0)}</td>'
            f'<td>{single_cell}</td><td>{multi_cell}</td><td>{dur_cell}</td>'
            f'<td>{" ｜ ".join(links)}</td></tr>')

    from datetime import datetime as _dt
    return f'''<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<title>评测历史索引</title>
<style>{_BASE_CSS}
  a {{ color:#2563eb; text-decoration:none; }}
  a:hover {{ text-decoration:underline; }}
  .charts {{ display:flex; gap:16px; flex-wrap:wrap; }}
  .chart-box {{ background:#fff; border-radius:10px; padding:12px; box-shadow:0 1px 3px rgba(0,0,0,.08); overflow-x:auto; }}
</style>
</head>
<body>
<div class="container">
  <h1>信贷问答助手 · 评测历史索引</h1>
  <div class="muted">生成时间：{_dt.now():%Y-%m-%d %H:%M:%S} ｜ 快照保留 {config.RESULTS_HISTORY_RETENTION_DAYS} 天（到期自动清理）</div>

  <h2>总览</h2>
  <div class="cards">{summary_cards}</div>

  <h2>质量趋势</h2>
  <div class="charts">{chart_boxes}</div>

  <h2>历史运行明细（点击跳转）</h2>
  <table><tr><th>运行标识</th><th>开始时间</th><th>模式</th><th>被测模型</th>
  <th>单轮/多轮用例</th><th>单轮通过率</th><th>多轮均分</th><th>耗时</th><th>产物</th></tr>
  {''.join(rows)}</table>
  <p class="muted">通过率为 全部指标通过 的用例占比；多轮均分为 G1/G2 门禁 + Q1~Q5 加权综合分。</p>
</div>
</body>
</html>'''


# ============================================================
# 入口
# ============================================================

def generate_index(history_dir: str = None, output_path: str = None) -> str:
    """重建历史索引页（评测运行结束自动调用；也可独立运行）"""
    metas = load_run_metas(history_dir)
    output_path = output_path or os.path.join(config.RESULTS_DIR, 'index.html')
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    html_text = render_index(metas)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_text)
    log.info(f'[索引] 历史索引页已生成 -> {output_path}（{len(metas)} 次运行）')
    return output_path


# ============================================================
# 自测（构造历史快照渲染，不依赖真实运行）
# ============================================================

if __name__ == '__main__':
    import tempfile

    print('=' * 60)
    print('  history_index 自测')
    print('=' * 60)

    def _fake_meta(rid, started, mode, sp, ms, dur, reports=('single', 'multi', 'json')):
        return {
            'run_id': rid, 'mode': mode, 'model': 'judge-x', 'answer_model': 'answer-x',
            'started_at': started, 'duration_sec': dur,
            'cli_args': {'mode': mode}, 'config_snapshot': {},
            'counts': {'single_turn_cases': 10, 'multi_turn_cases': 2},
            'summary': {
                'single_pass_rate': f'{sp}%' if sp is not None else None,
                'single_pass_rate_value': sp,
                'single_perf': {'avg': 3.5} if 'single' in reports else None,
                'multi_aggregate': {'avg_overall_score': ms} if ms is not None else None,
                'multi_perf': {'avg': 7.2} if 'multi' in reports else None,
            },
            '_run_dir': f'history/{rid}',
            '_reports': {'single': 'single' in reports, 'multi': 'multi' in reports,
                         'json': 'json' in reports},
        }

    metas = [
        _fake_meta('20261027_100000', '2026-10-27 10:00:00', 'single', 80.0, None, 60.0, ('single', 'json')),
        _fake_meta('20261028_100000', '2026-10-28 10:00:00', 'all', 90.0, 85.5, 120.0),
        _fake_meta('20261029_100000', '2026-10-29 10:00:00', 'multi', None, 92.0, 90.0, ('multi', 'json')),
    ]

    # 1) 趋势图：断点与正常渲染
    svg_full = _trend_svg([('a', 1.0), ('b', 2.0), ('c', None), ('d', 3.0)], '测试', '#2563eb')
    assert '<svg' in svg_full and svg_full.count('<circle') == 3
    assert 'M' in svg_full.split('d="')[1]  # None 处断点重新 M
    svg_sparse = _trend_svg([('a', 1.0)], '测试', '#2563eb')
    assert '数据不足' in svg_sparse
    print('[OK] 趋势图：正常渲染 + None 断点 + 数据不足降级')

    # 2) 页面渲染
    html = render_index(metas)
    assert '历史运行次数' in html and '质量趋势' in html and '历史运行明细' in html
    assert '单轮通过率趋势' in html and '多轮综合均分趋势' in html and '平均作答耗时趋势' in html
    assert 'history/20261029_100000/multi_turn_report.html' in html   # 多轮快照链接
    assert 'history/20261027_100000/evaluation_report.html' in html   # 单轮快照链接
    assert 'history/20261028_100000/evaluation_results.json' in html  # JSON 链接
    assert html.index('20261029_100000') < html.index('20261028_100000')  # 新的在前
    assert '90.0%' in html and '92.0' in html
    print('[OK] 索引页渲染：卡片/三趋势图/表格/跳转链接/新在前')

    # 3) 空历史
    empty_html = render_index([])
    assert '暂无历史评测记录' in empty_html
    print('[OK] 空历史降级页')

    # 4) 落盘入口（构造临时 history 目录）
    with tempfile.TemporaryDirectory() as tmp:
        hist = os.path.join(tmp, 'history')
        for m in metas:
            os.makedirs(os.path.join(hist, m['run_id']), exist_ok=True)
        # 其中两个写 meta 文件，一个缺失 meta（应被跳过）
        for m in metas[:2]:
            body = {k: v for k, v in m.items() if not k.startswith('_')}
            with open(os.path.join(hist, m['run_id'], 'run_meta.json'), 'w', encoding='utf-8') as f:
                json.dump(body, f, ensure_ascii=False)
        out = generate_index(hist, os.path.join(tmp, 'index.html'))
        assert os.path.exists(out)
        loaded = load_run_metas(hist)
        assert len(loaded) == 2 and loaded[0]['_run_dir'].endswith('20261027_100000')
        assert loaded[0]['_reports']['json'] is False  # 未写 json 文件 → 链接不出现在表中
    print('[OK] load_run_metas 跳过损坏/缺失 meta，_reports 探测正确')

    print('=' * 60)
    print('  全部自测通过')
    print('=' * 60)
