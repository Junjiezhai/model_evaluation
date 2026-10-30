# -*- coding: utf-8 -*-
"""
# @File    : handle_csv.py
# @Author  : jeremyzhai
# @Description : 项目内 CSV 读写统一入口（read_csv_data / parse_csv_text /
#               write_csv_data / csv_to_string），其他模块禁止直接 import csv。
#
# 统一约定：
#   - 文件读写一律 utf-8-sig（兼容 Excel BOM）
#   - 行 dict 附带两个下划线元字段：_line_no（文件读取时从 2 起，1 是表头；
#     文本解析时为 0）、_header（该文件/文本块的表头列表）
#   - 读取时清洗：键与值去首尾空白，None 归一为 ''
"""

import csv
import io
import os
from typing import Dict, List, Sequence

from utils.handle_log import log


def _clean_row(row: Dict, line_no: int, header: List[str]) -> Dict:
    """清洗单行：键/值去空白，None 归一 ''，附带 _line_no / _header 元字段"""
    cleaned = {k.strip() if k else k: (v.strip() if isinstance(v, str) else v or '')
               for k, v in row.items()}
    cleaned['_line_no'] = line_no
    cleaned['_header'] = header
    return cleaned


# csv文件读取
def read_csv_data(path: str) -> List[Dict]:
    """
    读取 CSV 文件，返回清洗后的行 dict 列表
    :param path: 文件路径
    :return: 行 dict 列表；每行附带 _line_no（从 2 起）与 _header
    :raises FileNotFoundError: 文件不存在
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f'CSV 文件不存在: {path}')
    rows: List[Dict] = []
    with open(path, 'r', encoding='utf-8-sig', newline='') as f:
        reader = csv.DictReader(f)
        header = [h.strip() if h else '' for h in (reader.fieldnames or [])]
        for i, row in enumerate(reader, start=2):  # 行号从 2 开始（1 是表头）
            rows.append(_clean_row(row, i, header))
    return rows


def parse_csv_text(text: str) -> List[Dict]:
    """
    从字符串解析 CSV（用于解析模型响应中的 CSV 文本块）
    :param text: 含表头的 CSV 文本
    :return: 行 dict 列表；_line_no=0；内容全空的行被跳过
    """
    reader = csv.DictReader(io.StringIO(text.strip()))
    header = [h.strip() if h else '' for h in (reader.fieldnames or [])]
    rows: List[Dict] = []
    for row in reader:
        cleaned = _clean_row(row, 0, header)
        if any(v for k, v in cleaned.items() if not k.startswith('_')):
            rows.append(cleaned)
    return rows


# csv文件写入
def write_csv_data(path: str, header: Sequence[str], rows: Sequence[Sequence]) -> str:
    """
    将表头 + 行数据写入 CSV 文件（目录不存在时自动创建）
    :param path: 文件路径
    :param header: 表头列名序列
    :param rows: 行数据序列（与表头等长）
    :return: 写入路径
    """
    dir_name = os.path.dirname(path)
    if dir_name:
        os.makedirs(dir_name, exist_ok=True)
    with open(path, 'w', encoding='utf-8-sig', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(list(header))
        writer.writerows(rows)
    log.info(f"CSV 文件写入成功: {path}（{len(rows)} 行）")
    return path


def csv_to_string(header: Sequence[str], rows: Sequence[Sequence]) -> str:
    """
    表头 + 行数据序列化为 CSV 文本（内存中构造测试数据用，配合临时文件）
    :return: CSV 文本（含表头行）
    """
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(list(header))
    writer.writerows(rows)
    return buf.getvalue()


if __name__ == '__main__':
    import tempfile
    print('=' * 60)
    print('  handle_csv 自测')
    print('=' * 60)
    header = ('case_id', 'question', 'answer')
    rows = [('ST_0001', ' 怎么提额 ', '您可以在提额页面选择提额项。')]

    text = csv_to_string(header, rows)
    parsed = parse_csv_text(text)
    assert len(parsed) == 1 and parsed[0]['question'] == '怎么提额'  # 值已去空白
    assert parsed[0]['_line_no'] == 0 and parsed[0]['_header'] == list(header)
    print('[OK] csv_to_string -> parse_csv_text 往返（含空白清洗）')

    with tempfile.TemporaryDirectory() as tmp:
        p = os.path.join(tmp, 'sub', 't.csv')  # 目录不存在自动创建
        write_csv_data(p, header, rows)
        got = read_csv_data(p)
        assert len(got) == 1 and got[0]['case_id'] == 'ST_0001'
        assert got[0]['_line_no'] == 2 and got[0]['answer'].startswith('您可以')
        # BOM 兼容：Excel 可直开
        assert open(p, 'rb').read(3) == b'\xef\xbb\xbf'
        try:
            read_csv_data(os.path.join(tmp, 'missing.csv'))
            raise AssertionError('应当抛 FileNotFoundError')
        except FileNotFoundError:
            pass
    print('[OK] write_csv_data -> read_csv_data 往返（utf-8-sig BOM、_line_no=2、缺文件抛错）')

    print('=' * 60)
    print('  全部自测通过')
    print('=' * 60)
