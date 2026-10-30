# -*- coding: utf-8 -*-
"""
问答链路：单轮作答 / 多轮对话执行 / 作答缓存 / 耗时采集

要点：
  - 知识库注入：检索 top_k 条知识以【知识库内容】注入 prompt（沿用原模板）
  - 缓存：AnswerCache 以"问题 + 模型 + 提示词版本 + 检索参数"指纹为 key；
         默认开启（业界主流做法，与 RAGAS / promptfoo 一致），--force-answer 绕过
  - 批量模式：单轮作答支持合并请求降本（--batch），组失败整组回退逐条；
             多轮对话流因历史依赖无法合并，始终逐轮调用
  - 耗时：调用方以 time.perf_counter() 包裹采集（modelHttpAction 为两项目共用，不改）；
          批量模式下整组耗时均摊到各条并打 latency_is_amortized 标记
"""

import hashlib
import json
import os
import time
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional

from deepeval.test_case import LLMTestCase

import config
from eval.case_loader import MultiTurnCase, SingleTurnCase
from utils.handle_log import log


# ============================================================
# Prompt 模板（知识库注入）
# ============================================================

ANSWER_PROMPT_TEMPLATE = (
    '【知识库内容】\n{context}\n\n'
    '【对话历史】\n{history}\n\n'
    '【用户问题】\n{question}\n\n'
    '请根据知识库内容回答用户问题。'
)


def _format_context(pairs) -> str:
    """把检索到的知识条目格式化为上下文文本"""
    if not pairs:
        return '（未检索到相关知识）'
    return '\n---\n'.join(p.as_context() for p in pairs)


# ============================================================
# 作答缓存（指纹 key）
# ============================================================

def _fingerprint(text: str) -> str:
    return hashlib.sha1(text.encode('utf-8')).hexdigest()[:16]


def answer_cache_key(question: str, *, model_name: str = None,
                     prompt_version: str = None, top_k: int = None,
                     temperature: float = None) -> str:
    """单轮作答缓存 key：问题 + 模型 + 提示词版本 + 检索参数 的指纹

    任一要素变化（换模型、改 prompt、调 top_k）都会使旧缓存自动失效。
    """
    model_name = model_name or 'kimi-for-coding'
    prompt_version = prompt_version or config.PROMPT_VERSION
    top_k = config.RETRIEVAL_TOP_K if top_k is None else top_k
    temperature = 0.3 if temperature is None else temperature
    raw = f'{question}|{model_name}|{prompt_version}|{top_k}|{temperature}'
    return _fingerprint(raw)


def multi_turn_cache_key(case_id: str, round_no: int, history_fingerprint: str, *,
                         model_name: str = None, prompt_version: str = None,
                         top_k: int = None) -> str:
    """多轮单轮次缓存 key：额外包含历史指纹，历史变化缓存自动失效"""
    model_name = model_name or 'kimi-for-coding'
    prompt_version = prompt_version or config.PROMPT_VERSION
    top_k = config.RETRIEVAL_TOP_K if top_k is None else top_k
    raw = f'{case_id}|{round_no}|{history_fingerprint}|{model_name}|{prompt_version}|{top_k}'
    return _fingerprint(raw)


class AnswerCache:
    """作答缓存：内存 dict + 增量落盘（每 put 一条即全量重写，防中断丢失）"""

    def __init__(self, path: str = None):
        self.path = path or config.QA_CACHE_PATH
        self._data: Dict[str, dict] = {}
        self._load()

    def _load(self):
        if os.path.exists(self.path):
            try:
                with open(self.path, 'r', encoding='utf-8') as f:
                    raw = json.load(f)
                # 兼容两种结构：{key: record} 或旧格式 [{input, actual_output, ...}]
                if isinstance(raw, dict):
                    self._data = {k: v for k, v in raw.items() if isinstance(v, dict)}
                elif isinstance(raw, list):
                    self._data = {}
                    log.warning(f'作答缓存为旧列表结构，将通过 migrate_legacy 迁移: {self.path}')
            except (json.JSONDecodeError, OSError) as e:
                log.warning(f'作答缓存读取失败（将重建）：{e}')
                self._data = {}

    def get(self, key: str) -> Optional[dict]:
        return self._data.get(key)

    def put(self, key: str, record: dict):
        self._data[key] = record
        self._save()

    def _save(self):
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        tmp = self.path + '.tmp'
        with open(tmp, 'w', encoding='utf-8') as f:
            json.dump(self._data, f, ensure_ascii=False, indent=2)
        os.replace(tmp, self.path)

    def migrate_legacy(self, legacy_records: List[dict], *, model_name: str = None) -> int:
        """把旧格式（key=问题原文的列表结构）缓存导入为指纹 key，返回迁移条数"""
        migrated = 0
        for rec in legacy_records:
            question = rec.get('input', '')
            if not question:
                continue
            key = answer_cache_key(question, model_name=model_name)
            if key not in self._data:
                self._data[key] = rec
                migrated += 1
        if migrated:
            self._save()
        return migrated

    def __len__(self):
        return len(self._data)


# ============================================================
# 数据结构：作答记录 / 多轮轨迹
# ============================================================

@dataclass
class AnswerRecord:
    """单轮作答记录"""
    case_id: str
    input: str
    actual_output: str
    expected_output: str
    retrieval_context: List[str]
    latency: float                       # 秒
    latency_is_amortized: bool = False   # True = 批量模式下整组耗时均摊
    cache_hit: bool = False
    category: str = ''
    priority: str = ''
    test_intent: str = ''
    test_dimension: str = ''
    expect_refuse: bool = False
    remark: str = ''


@dataclass
class RoundRecord:
    """多轮对话中的一轮执行结果"""
    round_no: int
    user_question: str
    expected_answer: str
    actual_answer: str
    retrieval_context: List[str]
    latency: float
    # 用例元数据（透传自 MultiTurnRound，供评估与报告）
    round_intent: str = ''
    actual_intent: str = ''      # 由 local_asserts 意图分类后回填
    state_intent: str = ''
    context_assert_type: str = ''
    previous_intent: str = ''
    expect_refuse: bool = False
    remark: str = ''
    cache_hit: bool = False


@dataclass
class DialogTrajectory:
    """多轮对话完整执行轨迹"""
    case_id: str
    case_type: str
    case_desc: str
    rounds: List[RoundRecord] = field(default_factory=list)

    @property
    def round_count(self) -> int:
        return len(self.rounds)

    def to_dialogue(self) -> List[dict]:
        """转换为 multi_turn_judge 期望的 dialogue 格式"""
        return [
            {'turn': r.round_no, 'user_question': r.user_question, 'model_response': r.actual_answer}
            for r in self.rounds
        ]


# ============================================================
# 单轮作答
# ============================================================

def answer_one(case: SingleTurnCase, kb, answer_llm, cache: AnswerCache = None) -> AnswerRecord:
    """单条作答（含检索、计时、缓存写入）。kb 为 KnowledgeBase，answer_llm 为 KimiDeepEvalLLM"""
    pairs = kb.retrieve(case.input, config.RETRIEVAL_TOP_K)
    context = _format_context(pairs)
    prompt = ANSWER_PROMPT_TEMPLATE.format(context=context, history='（无）', question=case.input)

    t0 = time.perf_counter()
    text = answer_llm.generate(prompt)
    latency = round(time.perf_counter() - t0, 3)

    record = AnswerRecord(
        case_id=case.case_id,
        input=case.input,
        actual_output=text.strip(),
        expected_output=case.expected_output,
        retrieval_context=[p.as_context() for p in pairs],
        latency=latency,
        category=case.category,
        priority=case.priority,
        test_intent=case.test_intent,
        test_dimension=case.test_dimension,
        expect_refuse=case.expect_refuse,
        remark=case.remark,
    )
    if cache is not None:
        cache.put(answer_cache_key(case.input), asdict(record))
    return record


def _cached_to_record(case: SingleTurnCase, cached: dict) -> AnswerRecord:
    """把缓存 dict 转回答录（缺失字段用用例数据补齐）"""
    return AnswerRecord(
        case_id=case.case_id,
        input=cached.get('input', case.input),
        actual_output=cached.get('actual_output', ''),
        expected_output=case.expected_output,
        retrieval_context=list(cached.get('retrieval_context') or []),
        latency=float(cached.get('latency', 0.0)),
        latency_is_amortized=bool(cached.get('latency_is_amortized', False)),
        cache_hit=True,
        category=case.category,
        priority=case.priority,
        test_intent=case.test_intent,
        test_dimension=case.test_dimension,
        expect_refuse=case.expect_refuse,
        remark=case.remark,
    )


def answer_single_turn(cases: List[SingleTurnCase], kb, answer_llm, *,
                       force: bool = False, batch: bool = False,
                       batch_size: int = None, cache: AnswerCache = None) -> List[AnswerRecord]:
    """单轮作答主入口。

    Args:
        cases: 用例列表
        kb: KnowledgeBase
        answer_llm: KimiDeepEvalLLM（role='answer'）
        force: 忽略缓存重新作答
        batch: 批量模式（合并请求降本）；组失败整组回退逐条
        batch_size: 合并组大小，默认 config.BATCH_GROUP_SIZE
        cache: AnswerCache 实例；None 时按 config.CACHE_ENABLED 内部创建
    """
    if cache is None and config.CACHE_ENABLED:
        cache = AnswerCache()
        # 旧列表结构缓存自动迁移
        if len(cache) == 0 and os.path.exists(cache.path):
            try:
                with open(cache.path, 'r', encoding='utf-8') as f:
                    legacy = json.load(f)
                if isinstance(legacy, list):
                    n = cache.migrate_legacy(legacy)
                    log.info(f'【缓存】已迁移旧格式作答缓存 {n} 条')
            except (json.JSONDecodeError, OSError):
                pass

    use_cache = (cache is not None) and (not force)
    results: List[Optional[AnswerRecord]] = [None] * len(cases)
    todo: List[int] = []

    # 第一遍：缓存命中直接复用
    for idx, case in enumerate(cases):
        if use_cache:
            cached = cache.get(answer_cache_key(case.input))
            if cached is not None and cached.get('actual_output'):
                results[idx] = _cached_to_record(case, cached)
                continue
        todo.append(idx)

    if not todo:
        log.info(f'【作答】全部 {len(cases)} 条命中缓存')
        return results  # type: ignore

    if batch:
        _answer_in_batches(cases, todo, results, kb, answer_llm, cache, batch_size)
    else:
        for idx in todo:
            record = answer_one(cases[idx], kb, answer_llm, cache)
            log.info(f'【作答】{cases[idx].case_id} 完成，耗时 {record.latency}s')
            results[idx] = record

    return results  # type: ignore


def _answer_in_batches(cases, todo, results, kb, answer_llm, cache, batch_size):
    """批量作答：分组合并请求，失败整组回退逐条"""
    group_size = batch_size or config.BATCH_GROUP_SIZE
    client = answer_llm._client  # KimiDeepEvalLLM 持有 ModelHttpAction

    for start in range(0, len(todo), group_size):
        group_idx = todo[start:start + group_size]
        group_cases = [cases[i] for i in group_idx]

        # 组内逐条检索（检索无 LLM 成本），上下文随问题一起注入合并 prompt
        contexts = []
        for case in group_cases:
            pairs = kb.retrieve(case.input, config.RETRIEVAL_TOP_K)
            contexts.append(_format_context(pairs))

        lines = [
            '你是一名信贷客服助手。请依次回答下面 {} 个用户问题。'.format(len(group_cases)),
            '每个问题都附有相关知识库内容，请严格依据知识库回答，不得编造数字；'
            '每个回答独立完整、语气礼貌口语化。',
            '请严格按 JSON 格式输出：{"answers": [{"index": 序号, "answer": 回答}, ...]}，序号从 1 开始。',
            '',
            '【问题列表】',
        ]
        for i, (case, ctx) in enumerate(zip(group_cases, contexts), start=1):
            lines.append(f'--- 第 {i} 个问题 ---')
            lines.append(f'【知识库内容】\n{ctx}')
            lines.append(f'【用户问题】\n{case.input}')

        t0 = time.perf_counter()
        try:
            data = client.chat_with_json_output(
                '\n'.join(lines),
                json_schema={
                    'type': 'object',
                    'properties': {
                        'answers': {
                            'type': 'array',
                            'items': {
                                'type': 'object',
                                'properties': {
                                    'index': {'type': 'integer'},
                                    'answer': {'type': 'string'},
                                },
                                'required': ['index', 'answer'],
                            },
                        },
                    },
                    'required': ['answers'],
                },
                system_prompt=config.ANSWER_SYSTEM_PROMPT,
                max_tokens=8192,
                desc=f'【批量作答】组大小={len(group_cases)}',
            )
            group_latency = round(time.perf_counter() - t0, 3)
            answers = {int(a['index']): str(a.get('answer', '')).strip()
                       for a in data.get('answers', []) if isinstance(a, dict) and 'index' in a}
            if len(answers) != len(group_cases) or any(not answers.get(i + 1) for i in range(len(group_cases))):
                raise ValueError(f'批量响应不完整：期望 {len(group_cases)} 条，实得 {len(answers)} 条')
            per_latency = round(group_latency / len(group_cases), 3)
            for i, idx in enumerate(group_idx):
                case = cases[idx]
                record = AnswerRecord(
                    case_id=case.case_id,
                    input=case.input,
                    actual_output=answers[i + 1],
                    expected_output=case.expected_output,
                    retrieval_context=[p.as_context() for p in kb.retrieve(case.input, config.RETRIEVAL_TOP_K)],
                    latency=per_latency,
                    latency_is_amortized=True,
                    category=case.category,
                    priority=case.priority,
                    test_intent=case.test_intent,
                    test_dimension=case.test_dimension,
                    expect_refuse=case.expect_refuse,
                    remark=case.remark,
                )
                if cache is not None:
                    cache.put(answer_cache_key(case.input), asdict(record))
                results[idx] = record
            log.info(f'【批量作答】组 {start // group_size + 1} 完成，整组耗时 {group_latency}s')
        except Exception as e:  # noqa: BLE001 - 整组回退逐条
            log.warning(f'【批量作答】组失败（{e}），回退逐条作答（{len(group_cases)} 条）')
            for idx in group_idx:
                results[idx] = answer_one(cases[idx], kb, answer_llm, cache)


# ============================================================
# 多轮对话执行
# ============================================================

def _history_fingerprint(history: List[dict]) -> str:
    raw = json.dumps(history, ensure_ascii=False, sort_keys=True)
    return _fingerprint(raw)


def run_multi_turn_dialog(case: MultiTurnCase, kb, answer_llm, *,
                          force: bool = False, cache: AnswerCache = None) -> DialogTrajectory:
    """执行多轮对话：逐轮检索→注入上下文与历史→调用被测模型，逐轮计时。

    多轮对话流因历史依赖无法合并请求，始终逐轮调用。
    """
    if cache is None and config.CACHE_ENABLED:
        cache = AnswerCache()
    use_cache = (cache is not None) and (not force)

    trajectory = DialogTrajectory(case_id=case.case_id, case_type=case.case_type, case_desc=case.case_desc)
    history: List[dict] = []  # [{'role','content'}...]，system 消息由 chat() 注入

    for rnd in case.rounds:
        pairs = kb.retrieve(rnd.user_question, config.RETRIEVAL_TOP_K)
        context = _format_context(pairs)
        history_text = '\n'.join(
            f'{"用户" if m["role"] == "user" else "助手"}：{m["content"]}' for m in history
        ) or '（无）'
        prompt = ANSWER_PROMPT_TEMPLATE.format(
            context=context, history=history_text, question=rnd.user_question)

        key = multi_turn_cache_key(
            case.case_id, rnd.round_no, _history_fingerprint(history))
        cached = cache.get(key) if use_cache else None

        if cached is not None and cached.get('actual_answer'):
            actual = cached['actual_answer']
            latency = float(cached.get('latency', 0.0))
            cache_hit = True
            log.info(f'【多轮作答】{case.case_id} 第{rnd.round_no}轮命中缓存')
        else:
            # 复用 answer_llm 的 generate（内部走 client.chat，system_prompt=ANSWER_SYSTEM_PROMPT）
            t0 = time.perf_counter()
            actual = answer_llm.generate(prompt).strip()
            latency = round(time.perf_counter() - t0, 3)
            cache_hit = False
            if cache is not None:
                cache.put(key, {'actual_answer': actual, 'latency': latency})

        trajectory.rounds.append(RoundRecord(
            round_no=rnd.round_no,
            user_question=rnd.user_question,
            expected_answer=rnd.expected_answer,
            actual_answer=actual,
            retrieval_context=[p.as_context() for p in pairs],
            latency=latency,
            round_intent=rnd.round_intent,
            state_intent=rnd.state_intent,
            context_assert_type=rnd.context_assert_type,
            previous_intent=rnd.previous_intent,
            expect_refuse=rnd.expect_refuse,
            remark=rnd.remark,
            cache_hit=cache_hit,
        ))
        # 维护对话历史（供下一轮注入）
        history.append({'role': 'user', 'content': rnd.user_question})
        history.append({'role': 'assistant', 'content': actual})
        log.info(f'【多轮作答】{case.case_id} 第{rnd.round_no}轮完成，耗时 {latency}s')

    return trajectory


# ============================================================
# 组装 DeepEval 用例
# ============================================================

def build_llm_test_cases(records: List[AnswerRecord]) -> List[LLMTestCase]:
    """把单轮作答记录组装为 DeepEval LLMTestCase 列表"""
    test_cases = []
    for r in records:
        test_cases.append(LLMTestCase(
            input=r.input,
            actual_output=r.actual_output,
            expected_output=r.expected_output,
            retrieval_context=r.retrieval_context,
            metadata={
                'case_id': r.case_id,
                'category': r.category,
                'priority': r.priority,
                'test_intent': r.test_intent,
                'test_dimension': r.test_dimension,
                'source': 'single_turn_csv',
                'latency': r.latency,
                'latency_is_amortized': r.latency_is_amortized,
                'cache_hit': r.cache_hit,
            },
        ))
    return test_cases


# ============================================================
# 自测（stub answer_llm / kb，不调用真实 LLM）
# ============================================================

if __name__ == '__main__':
    import tempfile
    print('=' * 60)
    print('  qa_system 自测（stub）')
    print('=' * 60)

    class _StubPair:
        def __init__(self, q):
            self.question = q
        def as_context(self):
            return f'【测试·咨询】\n问：{self.question}\n答：标准答案'

    class _StubKB:
        def retrieve(self, query, top_k):
            return [_StubPair(query)]

    class _StubLLM:
        """模拟 KimiDeepEvalLLM"""
        def __init__(self):
            self._client = None
            self.calls = 0
        def generate(self, prompt):
            self.calls += 1
            return f'针对"{prompt.split("【用户问题】")[-1].strip()[:12]}"的模拟回答'

    tmp_dir = tempfile.mkdtemp()
    cache_path = os.path.join(tmp_dir, 'cache.json')

    cases = [
        SingleTurnCase(case_id='ST_0001', case_type='single_turn', test_intent='提额降息咨询类',
                       test_dimension='标准表达', input='怎么提额', expected_output='答案1',
                       category='提额降息', priority='P1', expect_refuse=False, remark=''),
        SingleTurnCase(case_id='ST_0002', case_type='single_turn', test_intent='转人工类',
                       test_dimension='标准表达', input='帮我转人工', expected_output='答案2',
                       category='转人工', priority='P0', expect_refuse=False, remark=''),
    ]

    # ---- 1. 非批量作答 + 缓存 ----
    llm = _StubLLM()
    cache = AnswerCache(cache_path)
    records = answer_single_turn(cases, _StubKB(), llm, cache=cache)
    assert len(records) == 2 and all(r.latency >= 0 for r in records)
    assert llm.calls == 2 and len(cache) == 2
    print('[OK] 非批量作答 + 缓存写入')

    # ---- 2. 缓存命中（不再调用模型） ----
    llm2 = _StubLLM()
    records2 = answer_single_turn(cases, _StubKB(), llm2, cache=AnswerCache(cache_path))
    assert llm2.calls == 0 and all(r.cache_hit for r in records2)
    print('[OK] 缓存命中，0 次模型调用')

    # ---- 3. force 绕过缓存 ----
    llm3 = _StubLLM()
    records3 = answer_single_turn(cases, _StubKB(), llm3, force=True, cache=AnswerCache(cache_path))
    assert llm3.calls == 2 and not any(r.cache_hit for r in records3)
    print('[OK] --force-answer 绕过缓存')

    # ---- 4. 缓存 key 指纹敏感性 ----
    k1 = answer_cache_key('问题', prompt_version='v2')
    k2 = answer_cache_key('问题', prompt_version='v3')
    assert k1 != k2
    print('[OK] 缓存 key 随 prompt_version 变化')

    # ---- 5. 多轮对话流（stub） ----
    from eval.case_loader import MultiTurnCase, MultiTurnRound
    mt_case = MultiTurnCase(
        case_id='MD_0001', case_type='single_intent_followup', case_desc='[上下文记忆]测试',
        rounds=[
            MultiTurnRound(round_no=1, user_question='我想提额', expected_answer='a1',
                           intent='提额降息咨询类', round_intent='提额降息咨询类', state_intent='提额业务咨询',
                           context_assert_type='normal', previous_intent='', expect_refuse=False, remark=''),
            MultiTurnRound(round_no=2, user_question='需要什么条件', expected_answer='a2',
                           intent='提额降息咨询类', round_intent='提额降息咨询类', state_intent='提额准入条件咨询',
                           context_assert_type='single_intent_followup', previous_intent='提额降息咨询类',
                           expect_refuse=False, remark=''),
            MultiTurnRound(round_no=3, user_question='怎么套现', expected_answer='a3',
                           intent='风险合规禁答类', round_intent='风险合规禁答类', state_intent='违规操作',
                           context_assert_type='risk_compliance_insert', previous_intent='提额降息咨询类',
                           expect_refuse=True, remark=''),
        ])
    llm4 = _StubLLM()
    cache2 = AnswerCache(os.path.join(tmp_dir, 'cache2.json'))
    traj = run_multi_turn_dialog(mt_case, _StubKB(), llm4, cache=cache2)
    assert traj.round_count == 3 and llm4.calls == 3
    assert traj.rounds[1].actual_answer and traj.rounds[2].expect_refuse is True
    print('[OK] 多轮对话流 3 轮执行 + 逐轮计时')

    # ---- 6. 多轮缓存 key 随历史变化 ----
    ka = multi_turn_cache_key('MD_0001', 2, _history_fingerprint([{'role': 'user', 'content': 'x'}]))
    kb_ = multi_turn_cache_key('MD_0001', 2, _history_fingerprint([{'role': 'user', 'content': 'y'}]))
    assert ka != kb_
    print('[OK] 多轮缓存 key 随历史指纹变化')

    print('=' * 60)
    print('  全部自测通过')
    print('=' * 60)
