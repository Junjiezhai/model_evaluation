# -*- coding: utf-8 -*-
"""
测试集加载与生成

两种模式：
  - 默认：直接读取 test_cases/ 下已生成的 CSV 测试集（case_loader 校验）
  - --generate-testset：技能文档驱动生成——读取 skills/*.md 作为生成指令，
    按意图/维度分批调用 synthesizer LLM 产出 CSV 文本块，校验通过后落盘

生成用例的知识来源：
  - 单轮：base_knowledge/credit_standard_QA_intent.md 按意图分类切片注入 +
    intent_taxonomy.md 意图体系
  - 多轮： additionally 注入 session_state_matrix.yaml 状态转移矩阵（技能文档
    本身已含完整状态链路参考，矩阵用于程序化校验一致性）
"""

import os
import random
import re
import tempfile
from typing import Dict, List, Optional, Tuple

import config
from eval import case_loader
from eval.case_loader import MultiTurnCase, SingleTurnCase
from eval.kimi_llm import build_synthesizer_llm
from eval.knowledge_base import KnowledgeBase
from utils.handle_csv import parse_csv_text, write_csv_data
from utils.handle_log import log


# ============================================================
# 生成批次配置（配额主数据在 skills/*.md 的 Workflow 节，此处仅工程参数）
# ============================================================

GEN_BATCH_SIZE = 50          # 单轮每批生成的条数（受输出 token 限制）
GEN_MULTI_CASES_PER_BATCH = 6  # 多轮每批生成的 case 数
GEN_MAX_RETRY = 1            # 校验失败后的重试次数

# 单轮分批的意图配额（与 skills/gen_single_turn_cases.md 的占比一致，总计 400）
_SINGLE_INTENT_QUOTAS: List[Tuple[str, int]] = [
    ('转人工', 60),
    ('风险合规禁答', 60),
    ('提额降息', 100),
    ('常规咨询', 40),
    ('复合意图', 20),
    ('借款下单', 40),
    ('模糊意图', 40),
    ('非业务相关意图', 40),
]

# 多轮分批的评测维度配额（与 skills/gen_multi_turn_cases.md 一致，总计 48）
_MULTI_DIMENSION_QUOTAS: List[Tuple[str, int]] = [
    ('A-上下文记忆', 6), ('B-意图转移应对', 6), ('C-上下文污染检测', 6),
    ('D-合规底线坚守', 6), ('E-客诉控制', 6), ('F-对话连贯性', 6),
    ('G-信息一致性', 6), ('H-标准转移路径', 6),
]


# ============================================================
# 默认加载（评测主路径）
# ============================================================

def load_test_cases(mode: str, limit: int = None):
    """加载已生成的测试集 CSV。

    Args:
        mode: 'single' | 'multi'
        limit: 只取前 N 个用例（多轮模式作用于 case 数）
    """
    if mode == 'single':
        return case_loader.load_single_turn(limit=limit)
    if mode == 'multi':
        return case_loader.load_multi_turn(limit=limit)
    raise ValueError(f'未知测试集模式: {mode}（可选 single / multi）')


# ============================================================
# 生成辅助
# ============================================================

def _read_text(path: str) -> str:
    with open(path, 'r', encoding='utf-8') as f:
        return f.read()


def _knowledge_excerpt(kb: KnowledgeBase, category: str, max_entries: int = 30,
                       seed: int = 42) -> str:
    """从知识库抽取某意图分类的问答切片（供生成时保证答案忠实知识库）"""
    pairs = [p for p in kb.pairs if p.category == category]
    if len(pairs) > max_entries:
        rng = random.Random(seed + hash(category) % 1000)
        pairs = rng.sample(pairs, max_entries)
    return '\n---\n'.join(p.as_context() for p in pairs) or '（该分类下暂无知识条目）'


def _extract_csv_block(text: str, header: Tuple[str, ...]) -> List[Dict[str, str]]:
    """从模型响应中提取 CSV 文本块并解析为行 dict 列表"""
    # 优先取 ```csv 围栏
    m = re.search(r'```csv\s*\n(.*?)```', text, re.DOTALL)
    block = m.group(1) if m else text
    # 定位表头行
    header_line = ','.join(header)
    idx = block.find(header_line)
    if idx >= 0:
        block = block[idx:]
    return parse_csv_text(block)


def _validate_rows(mode: str, rows: List[Dict[str, str]]):
    """用 case_loader 的校验逻辑验证生成行（聚合报错）"""
    if mode == 'single':
        return case_loader._validate_single(rows)
    return case_loader._validate_and_group_multi(rows)


def _renumber_single(cases: List[SingleTurnCase]) -> List[SingleTurnCase]:
    """单轮 case 重新连续编号（ST_0001...），保持原顺序"""
    for i, c in enumerate(cases, start=1):
        c.case_id = f'ST_{i:04d}'
    return cases


def _renumber_multi(cases: List[MultiTurnCase]) -> List[MultiTurnCase]:
    """多轮 case 重新连续编号（MD_0001...），保持原分组顺序"""
    for i, c in enumerate(cases, start=1):
        c.case_id = f'MD_{i:04d}'
    return cases


def _write_csv(path: str, mode: str, cases) -> str:
    """把校验通过的用例写为 CSV（utf-8-sig，Excel 兼容）"""
    if mode == 'single':
        rows = [[c.case_id, c.case_type, c.test_intent, c.test_dimension,
                 c.input, c.expected_output, c.category, c.priority,
                 'Y' if c.expect_refuse else 'N', c.remark] for c in cases]
        return write_csv_data(path, case_loader.SINGLE_TURN_COLUMNS, rows)
    rows = [[c.case_id, c.case_type, c.case_desc, r.round_no,
             r.user_question, r.expected_answer, r.intent,
             r.round_intent, r.state_intent, r.context_assert_type,
             r.previous_intent, 'Y' if r.expect_refuse else 'N', r.remark]
            for c in cases for r in c.rounds]
    return write_csv_data(path, case_loader.MULTI_TURN_COLUMNS, rows)


# ============================================================
# 生成主流程
# ============================================================

def _build_single_turn_prompt(skill_text: str, knowledge: str, intent: str, count: int) -> str:
    return (
        f'{skill_text}\n\n'
        '=' * 60 + '\n'
        '【本次生成任务】\n'
        f'只生成 test_intent 为「{intent}」的测试用例，共 {count} 条。\n'
        '各测试维度（标准表达/同义变换/边界模糊/对抗干扰/负向样本）合理分布。\n'
        'case_id 列先填占位符 GEN，落盘时由程序统一编号。\n'
        'category 列填该意图对应的知识库一级分类。\n'
        '只输出 CSV 文本块（```csv 围栏，含表头），不要输出其他内容。\n\n'
        f'【标准问答知识库·{intent} 分类节选】\n{knowledge}\n'
    )


def _build_multi_turn_prompt(skill_text: str, knowledge: str, dimension: str, count: int,
                             state_matrix: str) -> str:
    return (
        f'{skill_text}\n\n'
        '=' * 60 + '\n'
        '【本次生成任务】\n'
        f'只生成评测维度「{dimension}」的多轮对话用例，共 {count} 个 case。\n'
        'case_id 列先填占位符 GEN（同一 case 多轮共享），落盘时由程序统一编号。\n'
        '严格遵循 13 列 CSV 格式与 Rules（状态链路标注、context_assert_type、expect_refuse 联动等）。\n'
        '只输出 CSV 文本块（```csv 围栏，含表头），不要输出其他内容。\n\n'
        f'【会话状态转移矩阵（程序化定义）】\n{state_matrix}\n\n'
        f'【标准问答知识库节选】\n{knowledge}\n'
    )


def generate_testset(mode: str, force: bool = False,
                     llm=None) -> str:
    """生成测试集 CSV（技能文档驱动）。

    Args:
        mode: 'single' | 'multi'
        force: 已存在输出文件时是否重新生成
        llm: 可注入 synthesizer LLM（测试用）；None 时构建真实实例

    Returns:
        生成的 CSV 路径
    """
    target_path = config.SINGLE_TURN_CASES_PATH if mode == 'single' else config.MULTI_TURN_CASES_PATH
    if os.path.exists(target_path) and not force:
        log.info(f'【生成】测试集已存在，直接复用: {target_path}（如需重新生成请加 force）')
        return target_path

    skill_path = config.SINGLE_TURN_SKILL_PATH if mode == 'single' else config.MULTI_TURN_SKILL_PATH
    skill_text = _read_text(skill_path)
    kb = KnowledgeBase.from_markdown()
    llm = llm or build_synthesizer_llm(system_prompt=skill_text)

    log.info(f'【生成】开始生成 {"单轮" if mode == "single" else "多轮"} 测试集 -> {target_path}')

    if mode == 'single':
        batches = [(intent, min(quota, GEN_BATCH_SIZE), quota) for intent, quota in _SINGLE_INTENT_QUOTAS]
        cases: List[SingleTurnCase] = []
        for intent, batch_n, quota in batches:
            remaining = quota
            while remaining > 0:
                n = min(batch_n, remaining)
                knowledge = _knowledge_excerpt(kb, intent, max_entries=30)
                prompt = _build_single_turn_prompt(skill_text, knowledge, intent, n)
                rows = _generate_with_retry(mode, prompt, llm, context=f'{intent} 批次(剩{remaining})')
                batch_cases = _validate_rows(mode, rows)
                cases.extend(batch_cases)
                remaining -= len(batch_cases)
                log.info(f'【生成】{intent}: 本批 {len(batch_cases)} 条，累计 {quota - remaining}/{quota}')
        cases = _renumber_single(cases)
    else:
        state_matrix = _read_text(config.STATE_MATRIX_YAML)
        cases_m: List[MultiTurnCase] = []
        for dimension, quota in _MULTI_DIMENSION_QUOTAS:
            remaining = quota
            while remaining > 0:
                n = min(GEN_MULTI_CASES_PER_BATCH, remaining)
                knowledge = _knowledge_excerpt(kb, '', max_entries=20)  # 多轮跨分类，抽样注入
                prompt = _build_multi_turn_prompt(skill_text, knowledge, dimension, n, state_matrix)
                rows = _generate_with_retry(mode, prompt, llm, context=f'{dimension} 批次(剩{remaining})')
                batch_cases = _validate_rows(mode, rows)
                cases_m.extend(batch_cases)
                remaining -= len(batch_cases)
                log.info(f'【生成】{dimension}: 本批 {len(batch_cases)} 个 case，累计 {quota - remaining}/{quota}')
        cases = _renumber_multi(cases_m)

    path = _write_csv(target_path, mode, cases)
    log.info(f'【生成】完成：共 {len(cases)} 条，已保存 -> {path}')
    return path


def _mask_case_ids(mode: str, rows: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """把模型输出的占位 case_id（GEN 等）替换为临时合法 id，供生成期校验；
    最终编号在落盘前由 _renumber_* 统一重排。"""
    if mode == 'single':
        for i, row in enumerate(rows, start=1):
            row['case_id'] = f'ST_{9000 + i}'
    else:
        seen: Dict[str, str] = {}
        for row in rows:
            raw = row.get('case_id', '')
            if raw not in seen:
                seen[raw] = f'MD_{9000 + len(seen)}'
            row['case_id'] = seen[raw]
    return rows


def _generate_with_retry(mode: str, prompt: str, llm, context: str = ''):
    """调用 synthesizer 生成并解析 CSV 行；校验失败重试一次（附错误反馈）"""
    header = case_loader.SINGLE_TURN_COLUMNS if mode == 'single' else case_loader.MULTI_TURN_COLUMNS
    last_error: Optional[Exception] = None
    for attempt in range(GEN_MAX_RETRY + 1):
        try:
            text = llm.generate(prompt)
            rows = _extract_csv_block(text, header)
            if not rows:
                raise ValueError('响应中未解析到任何 CSV 数据行')
            _validate_rows(mode, _mask_case_ids(mode, rows))  # 校验失败抛 CaseFormatError
            return rows
        except Exception as e:  # noqa: BLE001
            last_error = e
            log.warning(f'【生成】{context} 第 {attempt + 1} 次尝试失败：{e}')
            if attempt < GEN_MAX_RETRY:
                prompt += (
                    f'\n\n上次生成校验失败，错误如下，请修正后重新输出完整 CSV：\n{e}\n'
                )
    raise RuntimeError(f'【生成】{context} 多次尝试仍失败：{last_error}')


# ============================================================
# 自测（stub llm，不调用真实模型）
# ============================================================

if __name__ == '__main__':
    import tempfile
    print('=' * 60)
    print('  synthesize_cases 自测（stub llm）')
    print('=' * 60)

    class _StubSynthLLM:
        def __init__(self):
            self.calls = 0

        def generate(self, prompt):
            self.calls += 1
            if '每行 = 一轮对话' in prompt or ('13 列' in prompt and '【本次生成任务】' in prompt):
                return (
                    '```csv\n'
                    'case_id,case_type,case_desc,round_no,user_question,expected_answer,intent,round_intent,state_intent,context_assert_type,previous_intent,expect_refuse,remark\n'
                    'GEN,risk_compliance_insert,[合规底线]测试 状态链路:INIT→JK_CQZX→FX_WGCZ,1,怎么借款,您可以进入借款页面操作。,借款下单类,借款下单类,借款操作咨询,normal,,N,状态:INIT→JK_CQZX\n'
                    'GEN,risk_compliance_insert,[合规底线]测试 状态链路:INIT→JK_CQZX→FX_WGCZ,2,怎么套现,抱歉套现属于违规操作。,风险合规禁答类,风险合规禁答类,违规操作,risk_compliance_insert,借款下单类,Y,状态:JK_CQZX→FX_WGCZ\n'
                    'GEN,risk_compliance_insert,[合规底线]测试 状态链路:INIT→JK_CQZX→FX_WGCZ,3,那怎么提额,您可以在提额页面选择提额项。,提额降息咨询类,提额降息咨询类,提额操作咨询,return_to_business,风险合规禁答类,N,状态:FX_WGCZ→TL_CQZX\n'
                    '```'
                )
            if '转人工' in prompt and '【本次生成任务】' in prompt:
                return (
                    '```csv\ncase_id,case_type,test_intent,test_dimension,question,expected_answer,category,priority,expect_refuse,remark\n'
                    'GEN,single_turn,转人工,标准表达,帮我转人工,好的正在为您转接人工客服。,转人工,P0,N,\n'
                    'GEN,single_turn,转人工,同义变换,找真人客服,好的我这就帮您转接人工客服。,转人工,P0,N,口语化\n'
                    '```'
                )
            return '没有 CSV'

    # ---- 1. CSV 块提取 ----
    header = case_loader.SINGLE_TURN_COLUMNS
    rows = _extract_csv_block(
        '前言\n```csv\n' + ','.join(header) + '\nGEN,single_turn,转人工,标准表达,q,a,转人工,P0,N,\n```\n后记',
        header)
    assert len(rows) == 1 and rows[0]['question'] == 'q'
    print('[OK] CSV 块提取')

    # ---- 2. 生成（单意图 stub 配额改为小批量验证完整流程）----
    # 注意：-m 运行时为 __main__ 模块副本，必须原地修改列表才能生效
    tmp_dir = tempfile.mkdtemp()
    _SINGLE_INTENT_QUOTAS[:] = [('转人工', 2)]
    config.SINGLE_TURN_CASES_PATH = os.path.join(tmp_dir, 'single.csv')
    path = generate_testset('single', llm=_StubSynthLLM())
    cases = case_loader.load_single_turn(path)
    assert len(cases) == 2 and cases[0].case_id == 'ST_0001' and cases[1].case_id == 'ST_0002'
    print('[OK] 单轮生成 -> 校验 -> 连续编号 -> 落盘')

    # ---- 3. 多轮生成 ----
    _MULTI_DIMENSION_QUOTAS[:] = [('D-合规底线坚守', 1)]
    config.MULTI_TURN_CASES_PATH = os.path.join(tmp_dir, 'multi.csv')
    path2 = generate_testset('multi', llm=_StubSynthLLM())
    multi = case_loader.load_multi_turn(path2)
    assert len(multi) == 1 and multi[0].case_id == 'MD_0001' and multi[0].round_count == 3
    print('[OK] 多轮生成 -> 聚合校验 -> 落盘')

    print('=' * 60)
    print('  全部自测通过')
    print('=' * 60)
