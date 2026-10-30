# -*- coding: utf-8 -*-
"""
测试集 CSV 读取/校验/归一化模块

职责：
  - 读取 test_cases/single_turn_cases.csv（10 列，一行一个用例）
  - 读取 test_cases/multi_turn_cases.csv（13 列，一行一轮，按 case_id 聚合）
  - 统一校验字段格式，错误聚合为 CaseFormatError（带行号）一次性抛出，不静默跳过

内部字段命名沿用下游（qa_system / run_evaluation）已有约定：
  SingleTurnCase.input / .expected_output 对应 CSV 的 question / expected_answer 列。
"""

import os
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import config
from utils.handle_csv import csv_to_string, read_csv_data


class CaseFormatError(Exception):
    """测试集格式错误（聚合多个错误，带行号信息）"""

    def __init__(self, errors: List[str]):
        self.errors = errors
        message = '测试集格式校验失败，共 {} 处错误：\n  - {}'.format(
            len(errors), '\n  - '.join(errors[:50]))
        if len(errors) > 50:
            message += '\n  ... 以及另外 {} 处错误'.format(len(errors) - 50)
        super().__init__(message)


# ============================================================
# 数据结构
# ============================================================

@dataclass
class SingleTurnCase:
    """单轮测试用例（一行即一个 case）"""
    case_id: str
    case_type: str
    test_intent: str
    test_dimension: str
    input: str                # = CSV question 列
    expected_output: str      # = CSV expected_answer 列
    category: str
    priority: str
    expect_refuse: bool
    remark: str
    line_no: int = 0          # 在 CSV 中的行号（含表头，用于报错）


@dataclass
class MultiTurnRound:
    """多轮对话中的一轮（一行）"""
    round_no: int
    user_question: str
    expected_answer: str
    intent: str
    round_intent: str
    state_intent: str         # 状态机意图名，空串表示该轮跳过状态验证
    context_assert_type: str
    previous_intent: str
    expect_refuse: bool
    remark: str
    line_no: int = 0


@dataclass
class MultiTurnCase:
    """多轮测试用例（同一 case_id 的所有轮聚合）"""
    case_id: str
    case_type: str
    case_desc: str
    rounds: List[MultiTurnRound] = field(default_factory=list)

    @property
    def round_count(self) -> int:
        return len(self.rounds)


# ============================================================
# 常量定义
# ============================================================

SINGLE_TURN_COLUMNS: Tuple[str, ...] = (
    'case_id', 'case_type', 'test_intent', 'test_dimension', 'question',
    'expected_answer', 'category', 'priority', 'expect_refuse', 'remark',
)

MULTI_TURN_COLUMNS: Tuple[str, ...] = (
    'case_id', 'case_type', 'case_desc', 'round_no', 'user_question',
    'expected_answer', 'intent', 'round_intent', 'state_intent',
    'context_assert_type', 'previous_intent', 'expect_refuse', 'remark',
)

# round_intent 合法取值（多轮，7 个类后缀标签，见 base_knowledge/intent_taxonomy.md）
VALID_INTENT_LABELS = frozenset({
    '提额降息咨询类', '常规咨询类', '借款下单类', '转人工类',
    '风险合规禁答类', '模糊意图类', '非业务相关意图类',
})

# test_intent 合法取值（单轮，8 个短名标签，见 skills/gen_single_turn_cases.md）
SINGLE_TURN_INTENT_LABELS = frozenset({
    '提额降息', '常规咨询', '复合意图', '借款下单',
    '转人工', '风险合规禁答', '模糊意图', '非业务相关意图',
})

VALID_TEST_DIMENSIONS = frozenset({'标准表达', '同义变换', '边界模糊', '对抗干扰', '负向样本'})
VALID_PRIORITIES = frozenset({'P0', 'P1', 'P2', 'P3'})
VALID_CONTEXT_ASSERT_TYPES = frozenset({
    'normal', 'single_intent_followup', 'cross_intent_switch',
    'risk_compliance_insert', 'return_to_business', 'complaint_escalation',
})

_CASE_ID_PATTERNS = {
    'single': re.compile(r'^ST_\d{4}$'),
    'multi': re.compile(r'^MD_\d{4}$'),
}


# ============================================================
# 内部工具
# ============================================================

def _read_csv(path: str) -> List[Dict[str, str]]:
    """读取 CSV（utf-8-sig 兼容 Excel BOM），返回行 dict 列表（附 _line_no）"""
    try:
        return read_csv_data(path)
    except FileNotFoundError:
        raise FileNotFoundError(
            f'测试集文件不存在: {path}\n'
            f'请先生成测试集：python -m eval.run_evaluation --generate-testset --mode <single|multi>'
        )


def _check_columns(header: List[str], expected_columns: Tuple[str, ...], errors: List[str]) -> bool:
    """校验表头列名与顺序；返回是否通过"""
    missing = [c for c in expected_columns if c not in header]
    if missing:
        errors.append(f'表头缺少列: {missing}；实际表头: {header}')
        return False
    # 允许表尾有多余列（向前兼容），但前 N 列顺序必须一致
    if header[:len(expected_columns)] != list(expected_columns):
        errors.append(f'表头列顺序不符，期望前 {len(expected_columns)} 列为: {list(expected_columns)}；实际: {header}')
        return False
    return True


def _yes_no(value: str) -> Optional[bool]:
    """Y/N 解析；空返回 None"""
    v = (value or '').strip().upper()
    if v in ('Y', '是'):
        return True
    if v in ('N', '否'):
        return False
    return None


# ============================================================
# 单轮用例加载
# ============================================================

def _validate_single_row(row: Dict[str, str], errors: List[str]) -> Optional[SingleTurnCase]:
    line_no = row['_line_no']
    prefix = f'第{line_no}行'

    case_id = row.get('case_id', '')
    if not case_id:
        errors.append(f'{prefix}: case_id 为空')
        return None
    if not _CASE_ID_PATTERNS['single'].match(case_id):
        errors.append(f'{prefix}: case_id "{case_id}" 格式不符，期望 ST_XXXX')

    question = row.get('question', '')
    if not question:
        errors.append(f'{prefix}({case_id}): question 为空')

    expected_answer = row.get('expected_answer', '')
    if not expected_answer:
        errors.append(f'{prefix}({case_id}): expected_answer 为空')

    test_intent = row.get('test_intent', '')
    if test_intent and test_intent not in SINGLE_TURN_INTENT_LABELS:
        errors.append(f'{prefix}({case_id}): test_intent "{test_intent}" 不在合法标签集合内')

    test_dimension = row.get('test_dimension', '')
    if test_dimension and test_dimension not in VALID_TEST_DIMENSIONS:
        errors.append(f'{prefix}({case_id}): test_dimension "{test_dimension}" 不在合法集合内')

    priority = row.get('priority', '')
    if priority and priority not in VALID_PRIORITIES:
        errors.append(f'{prefix}({case_id}): priority "{priority}" 不在合法集合内')

    expect_refuse_raw = row.get('expect_refuse', '')
    expect_refuse = _yes_no(expect_refuse_raw)
    if expect_refuse is None and expect_refuse_raw:
        errors.append(f'{prefix}({case_id}): expect_refuse "{expect_refuse_raw}" 应为 Y/N')

    return SingleTurnCase(
        case_id=case_id,
        case_type=row.get('case_type', 'single_turn') or 'single_turn',
        test_intent=test_intent,
        test_dimension=test_dimension,
        input=question,
        expected_output=expected_answer,
        category=row.get('category', ''),
        priority=priority,
        expect_refuse=bool(expect_refuse),
        remark=row.get('remark', ''),
        line_no=line_no,
    )


def _validate_single(rows: List[Dict[str, str]]) -> List[SingleTurnCase]:
    errors: List[str] = []
    if not rows:
        raise CaseFormatError(['CSV 无任何数据行'])

    _check_columns(rows[0]['_header'], SINGLE_TURN_COLUMNS, errors)

    cases: List[SingleTurnCase] = []
    seen_ids: Dict[str, int] = {}
    for row in rows:
        case = _validate_single_row(row, errors)
        if case is None:
            continue
        if case.case_id in seen_ids:
            errors.append(
                f'第{case.line_no}行: case_id "{case.case_id}" 重复'
                f'（首次出现在第{seen_ids[case.case_id]}行）')
        else:
            seen_ids[case.case_id] = case.line_no
        cases.append(case)

    if errors:
        raise CaseFormatError(errors)
    return cases


def load_single_turn(path: str = None, limit: int = None) -> List[SingleTurnCase]:
    """加载单轮测试集 CSV，返回 SingleTurnCase 列表

    Args:
        path: CSV 路径，默认 config.SINGLE_TURN_CASES_PATH
        limit: 只取前 N 个用例（None 表示全部）
    """
    rows = _read_csv(path or config.SINGLE_TURN_CASES_PATH)
    cases = _validate_single(rows)
    return cases[:limit] if limit else cases


# ============================================================
# 多轮用例加载
# ============================================================

def _validate_multi_row(row: Dict[str, str], errors: List[str]) -> Optional[MultiTurnRound]:
    line_no = row['_line_no']
    prefix = f'第{line_no}行'

    case_id = row.get('case_id', '')
    if not case_id:
        errors.append(f'{prefix}: case_id 为空')
        return None
    if not _CASE_ID_PATTERNS['multi'].match(case_id):
        errors.append(f'{prefix}: case_id "{case_id}" 格式不符，期望 MD_XXXX')

    round_no_raw = row.get('round_no', '')
    try:
        round_no = int(round_no_raw)
    except (TypeError, ValueError):
        errors.append(f'{prefix}({case_id}): round_no "{round_no_raw}" 不是整数')
        return None
    if round_no < 1:
        errors.append(f'{prefix}({case_id}): round_no 必须 >= 1')

    user_question = row.get('user_question', '')
    if not user_question:
        errors.append(f'{prefix}({case_id} 第{round_no}轮): user_question 为空')

    expected_answer = row.get('expected_answer', '')
    if not expected_answer:
        errors.append(f'{prefix}({case_id} 第{round_no}轮): expected_answer 为空')

    round_intent = row.get('round_intent', '')
    if not round_intent:
        errors.append(f'{prefix}({case_id} 第{round_no}轮): round_intent 为空')
    elif round_intent not in VALID_INTENT_LABELS:
        errors.append(f'{prefix}({case_id} 第{round_no}轮): round_intent "{round_intent}" 不在合法标签集合内')

    context_assert_type = row.get('context_assert_type', '')
    if context_assert_type and context_assert_type not in VALID_CONTEXT_ASSERT_TYPES:
        errors.append(
            f'{prefix}({case_id} 第{round_no}轮): context_assert_type "{context_assert_type}" 不在合法集合内')

    expect_refuse_raw = row.get('expect_refuse', '')
    expect_refuse = _yes_no(expect_refuse_raw)
    if expect_refuse is None:
        errors.append(f'{prefix}({case_id} 第{round_no}轮): expect_refuse "{expect_refuse_raw}" 应为 Y/N')

    return MultiTurnRound(
        round_no=round_no,
        user_question=user_question,
        expected_answer=expected_answer,
        intent=row.get('intent', ''),
        round_intent=round_intent,
        state_intent=row.get('state_intent', ''),
        context_assert_type=context_assert_type or 'normal',
        previous_intent=row.get('previous_intent', ''),
        expect_refuse=bool(expect_refuse),
        remark=row.get('remark', ''),
        line_no=line_no,
    )


def _validate_and_group_multi(rows: List[Dict[str, str]]) -> List[MultiTurnCase]:
    errors: List[str] = []
    if not rows:
        raise CaseFormatError(['CSV 无任何数据行'])

    _check_columns(rows[0]['_header'], MULTI_TURN_COLUMNS, errors)

    grouped: Dict[str, List[MultiTurnRound]] = {}
    meta: Dict[str, Dict[str, str]] = {}  # case_id -> {case_type, case_desc, first_line}
    for row in rows:
        case_id = row.get('case_id', '')
        round_obj = _validate_multi_row(row, errors)
        if round_obj is None:
            continue
        grouped.setdefault(case_id, []).append(round_obj)
        if case_id not in meta:
            meta[case_id] = {
                'case_type': row.get('case_type', ''),
                'case_desc': row.get('case_desc', ''),
                'first_line': row.get('_line_no'),
            }
        else:
            # 同 case 的 case_type / case_desc 必须一致
            if row.get('case_type', '') != meta[case_id]['case_type']:
                errors.append(
                    f'第{row["_line_no"]}行({case_id}): case_type "{row.get("case_type")}"'
                    f' 与同 case 第{meta[case_id]["first_line"]}行不一致')
            if row.get('case_desc', '') != meta[case_id]['case_desc']:
                errors.append(
                    f'第{row["_line_no"]}行({case_id}): case_desc 与同 case 第{meta[case_id]["first_line"]}行不一致')

    # round_no 连续性校验 + 组装
    cases: List[MultiTurnCase] = []
    for case_id, rounds in grouped.items():
        rounds.sort(key=lambda r: r.round_no)
        expected_seq = list(range(1, len(rounds) + 1))
        actual_seq = [r.round_no for r in rounds]
        if actual_seq != expected_seq:
            errors.append(
                f'用例 {case_id}: round_no 序列 {actual_seq} 不连续，'
                f'期望 1..{len(rounds)}（检查是否跳号或重复）')
        if len(rounds) < 3:
            errors.append(f'用例 {case_id}: 共 {len(rounds)} 轮，少于 3 轮（多轮用例至少 3 轮）')
        cases.append(MultiTurnCase(
            case_id=case_id,
            case_type=meta[case_id]['case_type'],
            case_desc=meta[case_id]['case_desc'],
            rounds=rounds,
        ))

    if errors:
        raise CaseFormatError(errors)

    # 按 case_id 排序，保证执行顺序稳定
    cases.sort(key=lambda c: c.case_id)
    return cases


def load_multi_turn(path: str = None, limit: int = None) -> List[MultiTurnCase]:
    """加载多轮测试集 CSV，返回 MultiTurnCase 列表（limit 作用于 case 数）"""
    rows = _read_csv(path or config.MULTI_TURN_CASES_PATH)
    cases = _validate_and_group_multi(rows)
    return cases[:limit] if limit else cases


# ============================================================
# 自测
# ============================================================

if __name__ == '__main__':
    import tempfile

    print('=' * 60)
    print('  case_loader 自测')
    print('=' * 60)

    # ---- 1. 合法单轮 CSV ----
    single_csv = '\n'.join([
        ','.join(SINGLE_TURN_COLUMNS),
        'ST_0001,single_turn,提额降息,标准表达,个税认证怎么操作,您可以在提额页面选择"个人所得税认证"。,提额降息,P1,N,备注',
        'ST_0002,single_turn,转人工,同义变换,帮我转人工,好的正在为您转接人工客服。,转人工,P0,N,',
    ])
    with tempfile.NamedTemporaryFile('w', encoding='utf-8-sig', suffix='.csv', delete=False) as f:
        f.write(single_csv)
        tmp_single = f.name
    cases = load_single_turn(tmp_single)
    assert len(cases) == 2 and cases[0].case_id == 'ST_0001' and cases[0].input.startswith('个税认证')
    print(f'[OK] 合法单轮 CSV -> {len(cases)} 条')

    # ---- 2. 非法单轮 CSV（重复 id + 非法枚举 + 空问题） ----
    bad_single_csv = '\n'.join([
        ','.join(SINGLE_TURN_COLUMNS),
        'ST_0001,single_turn,提额降息,标准表达,问题1,答案1,提额降息,P1,N,',
        'ST_0001,single_turn,不存在类,标准表达,,答案2,提额降息,P9,N,',
    ])
    with tempfile.NamedTemporaryFile('w', encoding='utf-8-sig', suffix='.csv', delete=False) as f:
        f.write(bad_single_csv)
        tmp_bad_single = f.name
    try:
        load_single_turn(tmp_bad_single)
        raise AssertionError('应当抛出 CaseFormatError')
    except CaseFormatError as e:
        assert '重复' in str(e) and 'P9' in str(e) and 'question 为空' in str(e)
        print(f'[OK] 非法单轮 CSV 正确聚合报错（{len(e.errors)} 处）')

    # ---- 3. 合法多轮 CSV（每行一轮，含引号转义） ----
    multi_rows = []
    for round_no, (q, a, ri, si, cat, prev, refuse, remark) in enumerate([
        ('怎么借款', '您可以进入借款页面选择金额和期限。', '借款下单类', '借款操作咨询', 'normal', '', 'N', '状态:INIT→JK_CQZX'),
        ('需要什么条件', '需要年满18周岁且有稳定收入。', '借款下单类', '借款条件咨询', 'single_intent_followup', '借款下单类', 'N', '状态:JK_CQZX→JK_TJZX'),
        ('怎么套现', '抱歉，套现属于违规操作。', '风险合规禁答类', '违规操作', 'risk_compliance_insert', '借款下单类', 'Y', '状态:JK_CQZX→FX_WGCZ'),
    ], start=1):
        multi_rows.append(['MD_0001', 'risk_compliance_insert', '[合规底线]测试 状态链路:INIT→JK_CQZX',
                           round_no, q, a, '借款下单类', ri, si, cat, prev, refuse, remark])
    with tempfile.NamedTemporaryFile('w', encoding='utf-8-sig', suffix='.csv', delete=False) as f:
        f.write(csv_to_string(MULTI_TURN_COLUMNS, multi_rows))
        tmp_multi = f.name
    multi_cases = load_multi_turn(tmp_multi)
    assert len(multi_cases) == 1 and multi_cases[0].round_count == 3
    assert multi_cases[0].rounds[2].expect_refuse is True
    assert multi_cases[0].rounds[1].state_intent == '借款条件咨询'
    print(f'[OK] 合法多轮 CSV -> {len(multi_cases)} 个 case，{multi_cases[0].round_count} 轮')

    # ---- 4. 非法多轮 CSV（round_no 跳号） ----
    bad_multi_csv = csv_to_string(MULTI_TURN_COLUMNS, [
        ['MD_0001', 'single_intent_followup', 'desc', 1, 'q1', 'a1', 'i', '借款下单类', '', 'normal', '', 'N', ''],
        ['MD_0001', 'single_intent_followup', 'desc', 3, 'q3', 'a3', 'i', '借款下单类', '', 'normal', '借款下单类', 'N', ''],
    ])
    with tempfile.NamedTemporaryFile('w', encoding='utf-8-sig', suffix='.csv', delete=False) as f:
        f.write(bad_multi_csv)
        tmp_bad_multi = f.name
    try:
        load_multi_turn(tmp_bad_multi)
        raise AssertionError('应当抛出 CaseFormatError')
    except CaseFormatError as e:
        assert '不连续' in str(e)
        print(f'[OK] round_no 跳号正确报错')

    for p in (tmp_single, tmp_bad_single, tmp_multi, tmp_bad_multi):
        os.unlink(p)

    print('=' * 60)
    print('  全部自测通过')
    print('=' * 60)
