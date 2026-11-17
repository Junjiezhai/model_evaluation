# -*- coding: utf-8 -*-
"""
# @File    : debug_benchmark_by_promptfoo.py
# @Author  : jeremyzhai
# @Description : promptfoo benchmark 评测调试入口（IDE 直接运行）

使用方式：
直接修改下方 main() 里 DebugConfig(...) 的覆盖项后运行本文件（PyCharm 右键 Run 即可）：

    python promptfoo_eval/debug_benchmark_by_promptfoo.py

与命令行的关系：
  - 正式用法见 promptfoo_eval/README.md（build 脚本 + promptfoo eval -c ...）；
  - 本文件是 IDE 调试入口：一键完成「生成用例 CSV → promptfoo eval → 本地汇总分数
    → 启动 promptfoo view 查看器（打印可点击链接、可自动开浏览器）」，
    参数集中在 DebugConfig 一个类里，无需敲命令行。
  - 查看器前台常驻：PyCharm 运行窗口会保持"运行中"，看完结果点 IDE 停止按钮即可，
    不会产生后台孤儿进程（serve_view=False 可跳过这一步）。

解耦声明：本脚本不 import 项目内任何模块（config / common / eval / utils 一概不用，
         日志用 print），仅以子进程方式调用 benchmark/scripts 下的转换脚本与 promptfoo CLI。
注意：promptfoo eval 必须在仓库根目录执行（outputPath 以 cwd 为基准、.env 由 promptfoo
      从 cwd 加载），本脚本已自动处理 cwd，IDE 内从任意工作目录启动均可。
"""

import json
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

# promptfoo_eval/ 与仓库根目录（本文件位于 promptfoo_eval/ 下）
_BASE = Path(__file__).resolve().parent
_REPO_ROOT = _BASE.parent
_BENCHMARK = _BASE / 'benchmark'

# PyCharm/重定向场景下 stdout 是块缓冲：改为行缓冲，保证链接/汇总及时可见
try:
    sys.stdout.reconfigure(line_buffering=True)
except (AttributeError, ValueError):
    pass

# 数据集 → (转换脚本, 评测配置, 结果 JSON)
_DATASETS = {
    'gsm8k': ('build_gsm8k_tests.py', 'gsm8k.yaml', 'gsm8k_output.json'),
    'ceval': ('build_ceval_tests.py', 'ceval.yaml', 'ceval_output.json'),
}


# ============================================================
# 调试配置（统一配置类：改这里，不用敲命令行）
# ============================================================

@dataclass
class DebugConfig:
    """promptfoo benchmark 调试配置 —— 所有可调参数集中在此

    参数对应关系：
    - dataset/limit/subjects/limit_per_subject → benchmark/scripts/build_*.py 命令行参数
    - filter_first_n / no_cache                → promptfoo eval 命令行参数
    """

    # 告诉 pytest 不要收集这个类（避免 PytestCollectionWarning）
    __test__ = False

    # ---- 数据集选择 ----
    # 'gsm8k' : GSM8K 数学推理（0-shot CoT）
    # 'ceval' : C-Eval 中文知识选择题（5-shot，默认 3 个上手科目）
    dataset: str = 'gsm8k'

    # ---- 建库配置 ----
    rebuild: bool = True                       # 评测前重新生成用例 CSV（改数据/参数后必开）
    limit: Optional[int] = 300                 # GSM8K：只取前 N 题（None = 全量 1319）
    subjects: Optional[tuple] = None           # C-Eval：科目元组，None = 默认 3 科目
    limit_per_subject: Optional[int] = 50      # C-Eval：每科目只取前 N 题（None = 该科目全量）

    # ---- 评测配置 ----
    filter_first_n: Optional[int] = 20         # 只评前 N 条（调试建议 20；None = 跑全量 CSV）
    no_cache: bool = False                     # True: 绕过 promptfoo 缓存重新调 API

    # ---- 结果查看 ----
    serve_view: bool = True                    # 评测完成后启动 promptfoo view 并打印可点击链接
    view_port: int = 15500                     # 查看器端口（被占用时先关掉旧查看器或改端口）
    open_browser: bool = True                  # 查看器就绪后自动打开浏览器

    def __str__(self) -> str:
        parts = [f'数据集: {self.dataset}', f'重建CSV: {"是" if self.rebuild else "否"}']
        if self.dataset == 'gsm8k':
            parts.append(f'题数: {self.limit if self.limit is not None else "全部1319"}')
        else:
            parts.append(f'科目: {self.subjects or "默认3科目"}')
            parts.append(f'每科目: {self.limit_per_subject or "全量"}')
        parts.append(f'评测条数: {self.filter_first_n or "全量"}')
        parts.append(f'绕缓存: {"是" if self.no_cache else "否"}')
        return ' ｜ '.join(parts)


# ============================================================
# 执行逻辑
# ============================================================

def _run(cmd: list, step: str, ok_codes=(0,)) -> None:
    """在仓库根目录执行子进程命令，返回码不在 ok_codes 内即中断。

    promptfoo eval 的退出码语义（CI 门禁设计）：0 = 全部 PASS；
    100 = 评测正常完成但有断言 FAIL（对我们是正常结果，不是错误）。
    """
    print(f'\n>>> [{step}] {" ".join(str(c) for c in cmd)}')
    completed = subprocess.run(cmd, cwd=_REPO_ROOT)
    if completed.returncode not in ok_codes:
        raise RuntimeError(f'[{step}] 执行失败（退出码 {completed.returncode}），评测中断')


def _promptfoo_cmd() -> list:
    """解析 promptfoo CLI 可执行文件（Windows 下处理 npm 的 .cmd shim）"""
    exe = shutil.which('promptfoo')
    if not exe:
        raise RuntimeError('未找到 promptfoo CLI，请先: npm install -g promptfoo')
    if exe.lower().endswith(('.cmd', '.bat')):
        return ['cmd', '/c', exe]
    return [exe]


def _print_summary(output_json: Path, filter_first_n: Optional[int]) -> None:
    """从 output.json 本地汇总 pass rate 与失败样本（零模型调用）"""
    if not output_json.exists():
        print(f'\n[汇总] 未找到结果文件 {output_json}（eval 可能未产出）')
        return
    with open(output_json, 'r', encoding='utf-8') as f:
        results = json.load(f).get('results', {}).get('results', [])

    total, passed, failures = len(results), 0, []
    for r in results:
        comp = ((r.get('gradingResult') or {}).get('componentResults') or [{}])[0]
        if comp.get('pass'):
            passed += 1
        else:
            resp = r.get('response')
            out = resp.get('output') if isinstance(resp, dict) else resp
            failures.append({
                'id': (r.get('vars') or {}).get('id', '?'),
                'reason': comp.get('reason', ''),
                'output': str(out or '')[:150].replace('\n', ' '),
            })

    print('\n' + '=' * 70)
    print(f'[汇总] {output_json.name}: {passed}/{total} PASS'
          + (f'（{passed / total:.1%}）' if total else ''))
    if failures:
        print(f'[汇总] 失败 {len(failures)} 条（promptfoo 因此返回退出码 100，'
              f'属 CI 门禁语义，评测结果有效），前 {min(5, len(failures))} 条如下:')
        for f_ in failures[:5]:
            print(f'  ✗ {f_["id"]} ｜ {f_["reason"]} ｜ 输出: {f_["output"]}...')
    else:
        print('[汇总] 无失败样本')
    if filter_first_n is not None:
        print('[汇总] 提示: 本次为抽样评测，全量请把 filter_first_n 设为 None 后重跑')
    print('[汇总] 逐条可视化: promptfoo view')
    print('=' * 70)


def _serve_view(port: int, open_browser: bool) -> None:
    """启动 promptfoo view（前台常驻）并打印可点击链接。

    前台常驻是刻意的：PyCharm 运行窗口保持"运行中"状态托管查看器进程，
    看完结果直接在 IDE 点停止按钮（或 Ctrl+C）即可，不会留下孤儿进程。
    浏览器行为交给 promptfoo 自己：-y 自动打开 / -n 不打开（避免交互式询问
    "Open URL in browser?" 在子进程里挂起）。
    """
    url = f'http://localhost:{port}'
    print('\n' + '=' * 70)
    print(f'[查看] 结果查看器: {url}')
    print('[查看] 页面顶部下拉框选中最新一轮 eval，点表格行看单题 prompt/输出/判分明细')
    print('[查看] 关闭方式: PyCharm 停止按钮 / Ctrl+C（端口被占用说明旧查看器还在，直接访问上面的链接即可）')
    print('=' * 70)

    cmd = _promptfoo_cmd() + ['view', '--port', str(port), '-y' if open_browser else '-n']
    proc = subprocess.Popen(cmd, cwd=_REPO_ROOT)
    try:
        proc.wait()  # 前台常驻，直到用户停止
    except KeyboardInterrupt:
        proc.terminate()
        print('\n[查看] 查看器已关闭')


def run(cfg: DebugConfig) -> None:
    if cfg.dataset not in _DATASETS:
        raise ValueError(f"未知数据集 {cfg.dataset!r}，可选: {list(_DATASETS.keys())}")
    build_script, yaml_name, output_name = _DATASETS[cfg.dataset]

    print('=' * 70)
    print('promptfoo benchmark 调试执行器')
    print(f'配置: {cfg}')
    print('=' * 70)

    # ① 生成用例 CSV
    if cfg.rebuild:
        cmd = [sys.executable, str(_BENCHMARK / 'scripts' / build_script)]
        if cfg.dataset == 'gsm8k':
            if cfg.limit is not None:
                cmd += ['--limit', str(cfg.limit)]
        else:
            if cfg.subjects:
                cmd += ['--subjects', *cfg.subjects]
            if cfg.limit_per_subject is not None:
                cmd += ['--limit-per-subject', str(cfg.limit_per_subject)]
        _run(cmd, '建库')

    # ② promptfoo 评测（cwd=仓库根目录：.env 加载 + outputPath 基准）
    cmd = _promptfoo_cmd() + ['eval', '-c', str(_BENCHMARK / yaml_name)]
    if cfg.filter_first_n is not None:
        cmd += ['--filter-first-n', str(cfg.filter_first_n)]
    if cfg.no_cache:
        cmd.append('--no-cache')
    _run(cmd, '评测', ok_codes=(0, 100))  # 100 = 有断言 FAIL（CI 门禁语义），结果仍有效

    # ③ 本地汇总
    _print_summary(_REPO_ROOT / 'promptfoo_eval' / 'benchmark' / 'outputs' / output_name,
                   cfg.filter_first_n)

    # ④ 启动结果查看器（前台常驻，打印可点击链接；serve_view=False 可跳过）
    if cfg.serve_view:
        _serve_view(cfg.view_port, cfg.open_browser)


# ============================================================
# 主入口
# ============================================================

def main():
    """promptfoo benchmark 调试执行器主入口：改 DebugConfig 后运行本文件"""
    # ===================================================================
    # 【IDE 调试参数：直接修改 DebugConfig(...) 中的覆盖项即可】
    # 示例：
    #   DebugConfig(dataset='gsm8k', filter_first_n=None, limit=300)      # GSM8K 全量
    #   DebugConfig(dataset='ceval', subjects=('middle_school_physics',)) # C-Eval 单科目
    #   DebugConfig(dataset='ceval', filter_first_n=None, no_cache=True)  # C-Eval 全量且绕缓存
    # ===================================================================
    cfg = DebugConfig(
        dataset='gsm8k',
        filter_first_n=None,
        limit=50,
    )
    # ===================================================================
    run(cfg)


if __name__ == '__main__':
    main()
