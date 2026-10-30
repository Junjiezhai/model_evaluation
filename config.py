# -*- coding: utf-8 -*-
"""
评测全局配置

集中管理路径、模型角色、缓存、批量模式与多轮评测阈值。
新增配置项时请在对应分组内添加，并同步 README 的"配置说明"章节。
"""

import os

Base_Path = os.path.dirname(os.path.abspath(__file__))

# ---------------- 路径配置 ----------------
KNOWLEDGE_BASE_PATH = os.path.join(Base_Path, 'base_knowledge', 'credit_standard_QA_intent.md')
SINGLE_TURN_CASES_PATH = os.path.join(Base_Path, 'test_cases', 'single_turn_cases.csv')
MULTI_TURN_CASES_PATH = os.path.join(Base_Path, 'test_cases', 'multi_turn_cases.csv')
SKILLS_DIR = os.path.join(Base_Path, 'skills')
RESULTS_DIR = os.path.join(Base_Path, 'results')
RAW_RESULTS_PATH = os.path.join(RESULTS_DIR, 'evaluation_results.json')
HTML_REPORT_PATH = os.path.join(RESULTS_DIR, 'evaluation_report.html')
MULTI_TURN_HTML_REPORT_PATH = os.path.join(RESULTS_DIR, 'multi_turn_report.html')
QA_CACHE_PATH = os.path.join(RESULTS_DIR, 'qa_answers.json')
# 历史评测快照：每次运行一个 {YYYYMMDD_HHMMSS}/ 文件夹，到期自动清理
RESULTS_HISTORY_DIR = os.path.join(RESULTS_DIR, 'history')
RESULTS_HISTORY_RETENTION_DAYS = 30

# ---------------- base_knowledge 素材 ----------------
INTENT_TAXONOMY_PATH = os.path.join(Base_Path, 'base_knowledge', 'intent_taxonomy.md')
SESSION_STATE_DOC_PATH = os.path.join(Base_Path, 'base_knowledge', 'session_state_transition.md')
STATE_MATRIX_YAML = os.path.join(Base_Path, 'base_knowledge', 'session_state_matrix.yaml')
MULTI_TURN_DIMENSIONS_YAML = os.path.join(Base_Path, 'base_knowledge', 'multi_turn_eval_dimensions.yaml')

# 生成测试集时各模式需要注入的技能文档与知识素材
SINGLE_TURN_SKILL_PATH = os.path.join(SKILLS_DIR, 'gen_single_turn_cases.md')
MULTI_TURN_SKILL_PATH = os.path.join(SKILLS_DIR, 'gen_multi_turn_cases.md')

# ---------------- 检索配置 ----------------
RETRIEVAL_TOP_K = 3  # 每条问题检索的上下文条数

# ---------------- 模型角色配置 ----------------
# 角色一：智能问答助手（被测系统）
ANSWER_SYSTEM_PROMPT = (
    "你是某信贷平台的智能客服助手。请严格依据下方提供的知识库内容回答用户问题。\n"
    "要求：\n"
    "1. 只使用知识库中提供的信息，不得编造利率、额度、时效等具体数字；\n"
    "2. 语气礼貌、简洁、口语化，适合客服场景；\n"
    "3. 如果知识库未覆盖该问题，请明确告知用户“该问题暂无法确认，建议联系人工客服”，不要强行回答。"
)

# 角色二：LLM 裁判（评测模型）
JUDGE_SYSTEM_PROMPT = (
    "你是一位严谨、公正的信贷客服问答质量评审专家。"
    "你的任务是根据给定的评分标准，对候选答案进行客观打分。"
    "评分时必须严格对照参考材料，不受答案表达流畅度的干扰，只关注事实准确性与任务达成度。"
    "请始终按照要求的格式输出评分与理由。"
)
JUDGE_TEMPERATURE = 0.0

# ---------------- 作答/缓存配置 ----------------
# 提示词版本号：ANSWER_PROMPT_TEMPLATE 变更时手动 +1，使旧作答缓存自动失效
PROMPT_VERSION = 'v2'
CACHE_ENABLED = True  # 作答缓存默认开启；命令行 --force-answer 可绕过

# ---------------- 评测阈值 ----------------
METRIC_THRESHOLDS = {
    'faithfulness': 0.7,
    'answer_relevancy': 0.7,
    'contextual_recall': 0.7,
    'correctness': 0.7,
}

# ---------------- 批量模式配置 ----------------
# 批量 = 合并请求降本：把多条用例合并为一次 LLM 请求，解析拆分后回填；
# 任一组解析/调用失败时整组回退为逐条处理。默认关闭（非批量）。
BATCH_ENABLED_DEFAULT = False
BATCH_GROUP_SIZE = 8            # 单轮作答合并组大小
BATCH_GROUP_SIZE_JUDGE = 10     # 单轮批量 Judge 每组最大数量
BATCH_GROUP_SIZE_MULTI_JUDGE = 2  # 多轮批量 Judge 每组最大数量（prompt 较长，取小值）

# ---------------- 多轮评测配置 ----------------
MULTI_TURN_THRESHOLD = 70  # 多轮综合分及格线（0-100），门禁失败直接判 0
STATE_VALIDATION_MODE = 'strict'  # 状态机验证模式：strict / lenient / category
INTENT_DRIFT_RATE_THRESHOLD = 0.2  # 意图漂移率告警阈值
MULTI_TURN_DIMENSION_WEIGHTS = {
    'Q1_context_memory': 0.25,
    'Q2_intent_responsiveness': 0.25,
    'Q3_contamination': 0.20,
    'Q4_coherence': 0.15,
    'Q5_consistency': 0.15,
}
