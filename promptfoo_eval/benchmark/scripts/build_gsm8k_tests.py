# -*- coding: utf-8 -*-
"""
GSM8K 测试集转换：JSONL → promptfoo 用例 CSV。

与 DeepEval 链路完全解耦：只读 benchmark_data/、只写 benchmark/tests/，
不 import 项目内任何模块（纯标准库）。

用法：
  python promptfoo_eval/benchmark/scripts/build_gsm8k_tests.py                 # 全量 1319 题
  python promptfoo_eval/benchmark/scripts/build_gsm8k_tests.py --limit 300     # 只导出前 300 题（体验推荐）
"""

import argparse
import csv
import json
import sys
from pathlib import Path

# promptfoo_eval/benchmark/scripts/ → promptfoo_eval/
_BASE = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = _BASE / 'benchmark_data' / 'gsm8k' / 'test.jsonl'
DEFAULT_OUTPUT = Path(__file__).resolve().parents[1] / 'tests' / 'gsm8k_test.csv'

COLUMNS = ['id', 'question', 'gold']


def extract_gold(answer: str) -> str:
    """GSM8K 标准答案在 answer 字段末尾，形如 '…\n#### 42'。"""
    if '####' not in answer:
        return ''
    return answer.split('####')[-1].strip()


def main() -> int:
    parser = argparse.ArgumentParser(description='GSM8K → promptfoo 用例 CSV')
    parser.add_argument('--input', default=str(DEFAULT_INPUT), help='GSM8K test.jsonl 路径')
    parser.add_argument('--output', default=str(DEFAULT_OUTPUT), help='输出 CSV 路径')
    parser.add_argument('--limit', type=int, default=None, help='只导出前 N 题（体验期建议 300）')
    args = parser.parse_args()

    src = Path(args.input)
    if not src.exists():
        print(f'[错误] 找不到 {src}，请先按 promptfoo_eval/benchmark_data/README.md 下载数据集')
        return 1

    rows, skipped = [], 0
    with open(src, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            gold = extract_gold(item.get('answer', ''))
            if not gold:
                skipped += 1
                continue
            rows.append({'id': i, 'question': item['question'], 'gold': gold})
            if args.limit and len(rows) >= args.limit:
                break

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    print(f'[OK] 已导出 {len(rows)} 题到 {out}' + (f'（跳过无 gold {skipped} 题）' if skipped else ''))
    return 0


if __name__ == '__main__':
    sys.exit(main())
