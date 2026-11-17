# -*- coding: utf-8 -*-
"""
C-Eval 测试集转换：parquet → promptfoo 用例 CSV。

与 DeepEval 链路完全解耦：只读 benchmark_data/、只写 benchmark/tests/，
唯一第三方依赖是 pandas + pyarrow（仅本脚本使用，与 DeepEval 链路无关）。

特点：
  - 每科目的 dev 集（5 道带答案示范题）拼成 fewshot 列，供 5-shot prompt 引用；
  - 默认只导出 3 个上手科目，可用 --subjects 指定任意科目组合；
  - --limit-per-subject 控制每科目题数，体验期建议先小流量。

用法：
  python promptfoo_eval/benchmark/scripts/build_ceval_tests.py                          # 默认 3 科目全量
  python promptfoo_eval/benchmark/scripts/build_ceval_tests.py --limit-per-subject 50   # 每科目 50 题
  python promptfoo_eval/benchmark/scripts/build_ceval_tests.py \
      --subjects middle_school_physics high_school_mathematics professional_tour_guide
"""

import argparse
import csv
import sys
from pathlib import Path

import pandas as pd

# promptfoo_eval/benchmark/scripts/ → promptfoo_eval/
_BASE = Path(__file__).resolve().parents[2]
_BENCHMARK_DATA = _BASE / 'benchmark_data' / 'ceval'
_SUBJECTS_FILE = _BASE / 'benchmark_data' / 'subjects.txt'
DEFAULT_OUTPUT = Path(__file__).resolve().parents[1] / 'tests' / 'ceval_test.csv'

COLUMNS = ['id', 'subject', 'question', 'A', 'B', 'C', 'D', 'gold', 'fewshot']

# 上手推荐：理科 + 数学 + 文科各一，验证判分抽取在不同题型下的鲁棒性
DEFAULT_SUBJECTS = [
    'middle_school_physics',
    'high_school_mathematics',
    'professional_tour_guide',
]

FEWSHOT_TEMPLATE = '题目：{question}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\n答案：{answer}'


def load_subjects() -> list:
    with open(_SUBJECTS_FILE, 'r', encoding='utf-8') as f:
        return [s.strip() for s in f if s.strip()]


def build_fewshot(subject: str) -> str:
    """把该科目 dev 集的 5 道示范题拼成示例文本块。"""
    dev = pd.read_parquet(_BENCHMARK_DATA / subject / 'dev-00000-of-00001.parquet')
    blocks = [
        FEWSHOT_TEMPLATE.format(
            question=r['question'], A=r['A'], B=r['B'], C=r['C'], D=r['D'], answer=r['answer']
        )
        for _, r in dev.iterrows()
    ]
    return '\n\n'.join(blocks)


def main() -> int:
    parser = argparse.ArgumentParser(description='C-Eval → promptfoo 用例 CSV')
    parser.add_argument('--subjects', nargs='+', default=DEFAULT_SUBJECTS, help='科目名（见 benchmark_data/subjects.txt）')
    parser.add_argument('--limit-per-subject', type=int, default=None, help='每科目只取前 N 题')
    parser.add_argument('--output', default=str(DEFAULT_OUTPUT), help='输出 CSV 路径')
    args = parser.parse_args()

    valid = set(load_subjects())
    unknown = [s for s in args.subjects if s not in valid]
    if unknown:
        print(f'[错误] 未知科目: {unknown}，可选范围见 promptfoo_eval/benchmark_data/subjects.txt')
        return 1

    rows = []
    for subject in args.subjects:
        test = pd.read_parquet(_BENCHMARK_DATA / subject / 'test-00000-of-00001.parquet')
        if args.limit_per_subject:
            test = test.head(args.limit_per_subject)
        fewshot = build_fewshot(subject)
        for _, r in test.iterrows():
            rows.append({
                'id': f'{subject}-{r["id"]}',
                'subject': subject,
                'question': r['question'],
                'A': r['A'], 'B': r['B'], 'C': r['C'], 'D': r['D'],
                'gold': r['answer'],
                'fewshot': fewshot,
            })

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    print(f'[OK] 已导出 {len(args.subjects)} 个科目共 {len(rows)} 题到 {out}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
