# -*- coding: utf-8 -*-
"""
# @File    : debug_evaluation_by_deepeval.py
# @Author  : jeremyzhai
# @Description : DeepEval 评测调试入口文件（对应 making_decison_test 的 debug_model_by_pytest.py）

使用方式：
直接修改下方 DebugConfig 的参数（或 main() 里 DebugConfig(...) 调用中的覆盖项）后运行本文件：

    python debug_evaluation_by_deepeval.py

与 eval/run_evaluation.py 的关系：
  - run_evaluation.py 是 CLI 正式入口（argparse 参数）；
  - 本文件是 IDE 调试入口：参数集中在 DebugConfig 一个类里，支持额外覆盖
    config.py 中的阈值/权重/批量组大小等运行时配置，无需改命令行。
"""

import os
import sys
from dataclasses import dataclass
from typing import Optional

# 脚本所在目录加入 path（保证从任意 cwd 启动都能 import eval 包）
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _SCRIPT_DIR)

import config  # noqa: E402  (path 设置后才能 import)
from eval import run_evaluation  # noqa: E402
from utils.handle_log import log  # noqa: E402


# ============================================================
# 调试配置（统一配置类：改这里，不用敲命令行）
# ============================================================

@dataclass
class DebugConfig:
    """DeepEval 评测调试配置 —— 所有可调参数集中在此

    参数分两类：
    - CLI 同名参数   : 与 `python -m eval.run_evaluation` 的命令行参数一一对应
    - config 覆盖参数 : 运行时覆盖 config.py 的模块属性（None = 沿用 config.py 默认值）
    """

    # 告诉 pytest 不要收集这个类（避免 PytestCollectionWarning）
    __test__ = False

    # ---- 场景选择（对应 CLI --mode） ----
    # 'single' : 单轮评测（DeepEval 4 标准指标）
    # 'multi'  : 多轮评测（G1/G2 门禁 + Q1~Q5 加权 + 状态机 + 本地断言）
    # 'all'    : 单轮 + 多轮
    mode: str = 'multi'

    # ---- 用例配置（对应 CLI --limit / --generate-testset） ----
    # None 表示全部；调试时建议先跑 1~5 条
    limit: Optional[int] = 5
    generate_testset: bool = False  # True: 先按 skills/*.md 重新生成测试集再评测

    # ---- 作答配置（对应 CLI --force-answer / --batch / --intent-llm） ----
    force_answer: bool = False      # True: 绕过作答缓存，重新调用问答模型
    batch: bool = False             # True: 作答/judge 合并请求降本（默认非批量）
    intent_llm: bool = False        # True: 多轮意图分类走 LLM（默认本地关键词规则，零成本）

    # ---- 单轮指标选择（None = 默认全量 4 指标）----
    # 逗号分隔的指标名，对应 eval/metrics/single_turn.py 的 SINGLE_METRIC_REGISTRY，如：
    # 'faithfulness,correctness'
    single_metrics: Optional[str] = None

    # ---- 多轮评估覆盖（None = 沿用 config.py 默认值）----
    multi_turn_threshold: Optional[int] = None       # 多轮综合分及格线（默认 70）
    state_validation_mode: Optional[str] = None      # 状态机验证：'strict'/'lenient'/'category'

    # ---- 批量参数覆盖（None = 沿用 config.py 默认值）----
    batch_group_size: Optional[int] = None              # 单轮作答合并组大小（默认 8）
    batch_group_size_multi_judge: Optional[int] = None  # 多轮 judge 合并组大小（默认 2）

    # ---- 缓存指纹覆盖（None = 沿用 config.py 默认值）----
    # 修改 prompt 模板后 bump 此版本号，让旧缓存自动失效
    prompt_version: Optional[str] = None

    def apply(self):
        """应用配置：覆盖 config 模块属性 + 设置指标选择环境变量"""
        if self.single_metrics:
            os.environ['EVAL_SINGLE_METRICS'] = self.single_metrics
        if self.multi_turn_threshold is not None:
            config.MULTI_TURN_THRESHOLD = self.multi_turn_threshold
        if self.state_validation_mode is not None:
            config.STATE_VALIDATION_MODE = self.state_validation_mode
        if self.batch_group_size is not None:
            config.BATCH_GROUP_SIZE = self.batch_group_size
        if self.batch_group_size_multi_judge is not None:
            config.BATCH_GROUP_SIZE_MULTI_JUDGE = self.batch_group_size_multi_judge
        if self.prompt_version is not None:
            config.PROMPT_VERSION = self.prompt_version

    def to_argv(self) -> list:
        """转成 run_evaluation.main(argv) 的 CLI 参数列表"""
        argv = ['--mode', self.mode]
        if self.limit is not None:
            argv += ['--limit', str(self.limit)]
        if self.generate_testset:
            argv.append('--generate-testset')
        if self.force_answer:
            argv.append('--force-answer')
        if self.batch:
            argv.append('--batch')
        if self.intent_llm:
            argv.append('--intent-llm')
        return argv

    def __str__(self) -> str:
        mode_label = {'single': '单轮', 'multi': '多轮', 'all': '单轮+多轮'}.get(self.mode, self.mode)
        parts = [
            f'模式: {mode_label}',
            f'用例限制: {self.limit if self.limit is not None else "全部"}',
            f'生成测试集: {"是" if self.generate_testset else "否（读已有 CSV）"}',
            f'批量降本: {"开" if self.batch else "关"}',
            f'绕缓存: {"是" if self.force_answer else "否"}',
            f'单轮指标: {self.single_metrics or "默认 4 指标"}',
        ]
        if self.mode in ('multi', 'all'):
            parts.append(f'意图分类: {"LLM" if self.intent_llm else "本地规则"}')
            parts.append(f'多轮及格线: {self.multi_turn_threshold or config.MULTI_TURN_THRESHOLD}')
            parts.append(f'状态机模式: {self.state_validation_mode or config.STATE_VALIDATION_MODE}')
        return ' ｜ '.join(parts)


# ============================================================
# 主入口
# ============================================================

def main():
    """DeepEval 评测调试执行器主入口：改 DebugConfig 后运行本文件"""
    # ===================================================================
    # 【IDE 调试参数：直接修改 DebugConfig(...) 中的覆盖项即可】
    # 示例：
    #   DebugConfig(mode='multi', limit=2, intent_llm=True)
    #   DebugConfig(mode='single', limit=10, batch=True, single_metrics='faithfulness,correctness')
    # ===================================================================
    cfg = DebugConfig(
        # mode='multi',
        # limit=2,
        # batch=True,
        # force_answer=True,
        # state_validation_mode='lenient',
    )
    # ===================================================================

    cfg.apply()
    log.info('=' * 80)
    log.info('DeepEval 评测调试执行器')
    log.info(f'配置: {cfg}')
    log.info('=' * 80)

    run_evaluation.main(cfg.to_argv())


if __name__ == '__main__':
    main()
