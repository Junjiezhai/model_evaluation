# -*- coding: utf-8 -*-
"""
会话状态转移合规性验证器（YAML 驱动）

移植自 making_decison_test 的 session_state_validator.py，数据源从类常量改为
base_knowledge/session_state_matrix.yaml（由 session_state_transition.md 派生）。

功能：
  - 单轮/多轮状态转移合规性验证（三层规则：P0 全局 → 主流程 → 兜底漂移）
  - 意图漂移率（Intent Drift Rate）量化
  - 漂移严重程度分级（none/mild/moderate/severe/critical）
  - 验证模式：strict / lenient / category
"""

import os
from collections import Counter
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

import config
from utils.handle_yaml import read_yaml_data


# ============================================================
# 枚举定义
# ============================================================

class ValidationMode(Enum):
    """验证模式"""
    STRICT = "strict"       # 严格模式：转移必须精确匹配
    LENIENT = "lenient"     # 宽松模式：允许同级业务域内跳跃、回退 INIT 视为合法
    CATEGORY = "category"   # 分类模式：只校验业务大类（TL/JX/JK/ZX 等）


class DriftSeverity(Enum):
    """漂移严重程度"""
    NONE = "none"
    MILD = "mild"             # 轻微：同级业务域内跳转
    MODERATE = "moderate"     # 中等：跨业务域但仍在信贷范围内
    SEVERE = "severe"         # 严重：跨到不相关域 / 业务域到风险人工
    CRITICAL = "critical"     # 致命：终态之后继续转移


# ============================================================
# 数据类
# ============================================================

@dataclass
class TransitionResult:
    """单次转移验证结果"""
    turn: int
    current_state: str
    detected_intent: str
    expected_next_state: Optional[str]
    is_valid: bool
    drift_severity: DriftSeverity
    matched_rule: Optional[str]
    detail: str = ''


@dataclass
class MultiTurnResult:
    """多轮验证结果"""
    total_turns: int
    valid_turns: int
    invalid_turns: int
    drift_rate: float
    severity_distribution: Dict[str, int]
    transitions: List[TransitionResult]
    summary: str = ''


@dataclass
class DriftReport:
    """意图漂移分析报告"""
    total_turns: int
    drift_rate: float
    hard_drift_rate: float
    soft_drift_rate: float
    category_drift_count: int
    severity_counts: Dict[str, int]
    top_drift_patterns: List[tuple]
    drift_transitions: List[TransitionResult]
    report_text: str = ''


# ============================================================
# 状态转移矩阵（YAML 加载）
# ============================================================

@dataclass
class StateTransitionMatrix:
    """由 session_state_matrix.yaml 构造的状态转移矩阵"""
    global_transitions: Dict[str, str] = field(default_factory=dict)
    global_intent_keywords: Dict[str, List[str]] = field(default_factory=dict)
    main_transitions: Dict[str, Dict[str, str]] = field(default_factory=dict)
    state_category_map: Dict[str, str] = field(default_factory=dict)
    terminal_states: set = field(default_factory=set)
    business_states: set = field(default_factory=set)


def load_state_matrix(yaml_path: str = None) -> StateTransitionMatrix:
    """从 YAML 加载状态转移矩阵（缺文件时抛 FileNotFoundError，不静默降级）"""
    path = yaml_path or config.STATE_MATRIX_YAML
    if not os.path.exists(path):
        raise FileNotFoundError(f'状态转移矩阵文件不存在: {path}')
    raw = read_yaml_data(path) or {}
    return StateTransitionMatrix(
        global_transitions=raw.get('global_transitions', {}),
        global_intent_keywords=raw.get('global_intent_keywords', {}),
        main_transitions=raw.get('main_transitions', {}),
        state_category_map=raw.get('state_category_map', {}),
        terminal_states=set(raw.get('terminal_states', [])),
        business_states=set(raw.get('business_states', [])),
    )


# ============================================================
# 会话状态验证器
# ============================================================

class SessionStateValidator:
    """信贷客服会话状态转移合规性验证器"""

    def __init__(self, matrix: StateTransitionMatrix = None,
                 mode: ValidationMode = None):
        """
        Args:
            matrix: 状态转移矩阵；None 时从 config.STATE_MATRIX_YAML 加载
            mode: 验证模式；None 时取 config.STATE_VALIDATION_MODE
        """
        self.matrix = matrix or load_state_matrix()
        mode_value = (mode or ValidationMode(config.STATE_VALIDATION_MODE)).value \
            if isinstance(mode or ValidationMode(config.STATE_VALIDATION_MODE), ValidationMode) \
            else (mode or config.STATE_VALIDATION_MODE)
        self.mode = ValidationMode(mode_value)

    # ---------------- 公共 API ----------------

    def validate_single_turn(
        self,
        current_state: str,
        detected_intent: str,
        *,
        turn: int = 1,
        previous_state: Optional[str] = None,
    ) -> TransitionResult:
        """验证单轮状态转移是否合规"""
        # Step 1: P0 全局规则
        global_match = self._match_global_intent(detected_intent)
        if global_match:
            next_state = self.matrix.global_transitions[global_match]
            return TransitionResult(
                turn=turn, current_state=current_state, detected_intent=detected_intent,
                expected_next_state=next_state, is_valid=True,
                drift_severity=DriftSeverity.NONE,
                matched_rule=f'全局P0规则: {global_match} → {next_state}',
                detail=f'触发全局转移规则，从 {current_state} 转移到 {next_state}',
            )

        # Step 2: 主流程精确匹配
        state_rules = self.matrix.main_transitions.get(current_state, {})
        if detected_intent in state_rules:
            next_state = state_rules[detected_intent]
            return TransitionResult(
                turn=turn, current_state=current_state, detected_intent=detected_intent,
                expected_next_state=next_state, is_valid=True,
                drift_severity=DriftSeverity.NONE,
                matched_rule=f'{current_state} + {detected_intent} → {next_state}',
                detail='精确匹配主流程转移规则',
            )

        # Step 3: 模糊匹配（双向包含 + 去"咨询/查询"后缀）
        fuzzy_match = self._fuzzy_match_intent(detected_intent, state_rules)
        if fuzzy_match:
            next_state = state_rules[fuzzy_match]
            return TransitionResult(
                turn=turn, current_state=current_state, detected_intent=detected_intent,
                expected_next_state=next_state, is_valid=True,
                drift_severity=DriftSeverity.MILD,
                matched_rule=f'{current_state} + {fuzzy_match}（模糊匹配）→ {next_state}',
                detail=f'意图 "{detected_intent}" 模糊匹配到规则 "{fuzzy_match}"',
            )

        # Step 4/5: 宽松 / 分类模式
        if self.mode == ValidationMode.LENIENT:
            return self._lenient_validate(current_state, detected_intent, turn)
        if self.mode == ValidationMode.CATEGORY:
            return self._category_validate(current_state, detected_intent, turn)

        # Step 6: 严格模式无匹配 → 漂移
        return self._build_drift_result(current_state, detected_intent, turn)

    def validate_multi_turn(
        self,
        trajectory: List[Dict[str, Any]],
        *,
        expected_trajectory: Optional[List[str]] = None,
    ) -> MultiTurnResult:
        """验证多轮会话的完整状态轨迹

        Args:
            trajectory: 每轮 {'turn', 'current_state', 'detected_intent', ...}
            expected_trajectory: 期望的状态序列（可选）
        """
        if not trajectory:
            return MultiTurnResult(0, 0, 0, 0.0, {}, [], '空轨迹，无数据可验证')

        transitions: List[TransitionResult] = []
        previous_state: Optional[str] = None
        for i, turn_data in enumerate(trajectory):
            turn_num = turn_data.get('turn', i + 1)
            current_state = turn_data.get('current_state', 'INIT')
            detected_intent = turn_data.get('detected_intent', '')
            result = self.validate_single_turn(
                current_state=current_state, detected_intent=detected_intent,
                turn=turn_num, previous_state=previous_state)
            transitions.append(result)
            previous_state = result.expected_next_state or current_state

        valid_count = sum(1 for t in transitions if t.is_valid)
        invalid_count = len(transitions) - valid_count
        drift_rate = invalid_count / len(transitions) if transitions else 0.0
        severity_dist = dict(Counter(t.drift_severity.value for t in transitions if not t.is_valid))

        summary_parts = [
            f"多轮验证完成：共 {len(transitions)} 轮",
            f"合法转移 {valid_count} 轮，非法转移 {invalid_count} 轮",
            f"意图漂移率: {drift_rate:.2%}",
        ]
        if severity_dist:
            summary_parts.append(f"严重程度分布: {severity_dist}")

        return MultiTurnResult(
            total_turns=len(transitions), valid_turns=valid_count,
            invalid_turns=invalid_count, drift_rate=round(drift_rate, 4),
            severity_distribution=severity_dist, transitions=transitions,
            summary='；'.join(summary_parts),
        )

    def analyze_drift(self, trajectory: List[Dict[str, Any]]) -> DriftReport:
        """意图漂移深度分析（硬/软漂移率、跨类别漂移、高频模式）"""
        multi_result = self.validate_multi_turn(trajectory)
        drift_transitions = [t for t in multi_result.transitions if not t.is_valid]
        total = multi_result.total_turns

        hard_sev = {DriftSeverity.MODERATE, DriftSeverity.SEVERE, DriftSeverity.CRITICAL}
        hard_count = sum(1 for t in drift_transitions if t.drift_severity in hard_sev)
        soft_count = sum(1 for t in drift_transitions if t.drift_severity == DriftSeverity.MILD)

        category_drift_count = 0
        for t in drift_transitions:
            current_cat = self.matrix.state_category_map.get(t.current_state, '??')
            if t.expected_next_state:
                next_cat = self.matrix.state_category_map.get(t.expected_next_state, '??')
                if current_cat != next_cat and next_cat not in ('INIT', 'END', '??'):
                    category_drift_count += 1

        drift_patterns = Counter()
        for t in drift_transitions:
            current_cat = self.matrix.state_category_map.get(t.current_state, '??')
            drift_patterns[(current_cat, t.detected_intent)] += 1
        top_patterns = drift_patterns.most_common(10)
        severity_counts = dict(Counter(t.drift_severity.value for t in drift_transitions))

        lines = [
            "=" * 60,
            "           意图漂移分析报告 (Intent Drift Report)",
            "=" * 60,
            f"  总轮数:         {total}",
            f"  合法转移:       {multi_result.valid_turns}",
            f"  非法转移(漂移): {multi_result.invalid_turns}",
            f"  总漂移率:       {multi_result.drift_rate:.2%}",
            f"  硬漂移率:       {hard_count / total:.2%}" if total else "  硬漂移率:       N/A",
            f"  软漂移率:       {soft_count / total:.2%}" if total else "  软漂移率:       N/A",
            f"  跨类别漂移次数: {category_drift_count}",
            "-" * 60,
            "  严重程度分布:",
        ]
        for severity in ['mild', 'moderate', 'severe', 'critical']:
            count = severity_counts.get(severity, 0)
            lines.append(f"    {severity:12s}: {count:3d} {'█' * min(count, 40)}")
        if top_patterns:
            lines.append('-' * 60)
            lines.append(f"  高频漂移模式 (Top-{min(5, len(top_patterns))})：")
            for (from_cat, intent), count in top_patterns[:5]:
                lines.append(f"    {from_cat} → [{intent}]: {count} 次")

        return DriftReport(
            total_turns=total, drift_rate=multi_result.drift_rate,
            hard_drift_rate=round(hard_count / total, 4) if total else 0.0,
            soft_drift_rate=round(soft_count / total, 4) if total else 0.0,
            category_drift_count=category_drift_count,
            severity_counts=severity_counts,
            top_drift_patterns=[(p[0][0], p[0][1], p[1]) for p in top_patterns],
            drift_transitions=drift_transitions,
            report_text='\n'.join(lines),
        )

    # ---------------- 内部方法 ----------------

    def _match_global_intent(self, detected_intent: str) -> Optional[str]:
        """P0 全局意图匹配：精确 → 关键词"""
        if detected_intent in self.matrix.global_transitions:
            return detected_intent
        intent_lower = detected_intent.lower()
        for intent_name, keywords in self.matrix.global_intent_keywords.items():
            for kw in keywords:
                if kw in intent_lower or kw in detected_intent:
                    return intent_name
        return None

    @staticmethod
    def _fuzzy_match_intent(detected_intent: str,
                            state_rules: Dict[str, str]) -> Optional[str]:
        """模糊匹配意图到转移规则（双向包含 + 去后缀核心词）"""
        for rule_intent in state_rules:
            if rule_intent in detected_intent or detected_intent in rule_intent:
                return rule_intent
            rule_core = rule_intent.replace('咨询', '').replace('查询', '').strip()
            intent_core = detected_intent.replace('咨询', '').replace('查询', '').strip()
            if rule_core and intent_core and \
                    (rule_core in intent_core or intent_core in rule_core):
                return rule_intent
        return None

    def _lenient_validate(self, current_state: str, detected_intent: str,
                          turn: int) -> TransitionResult:
        """宽松模式：允许回退 INIT / 同级业务域内跳跃"""
        current_cat = self.matrix.state_category_map.get(current_state, '??')
        init_rules = self.matrix.main_transitions.get('INIT', {})

        if detected_intent in init_rules:
            return TransitionResult(
                turn=turn, current_state=current_state, detected_intent=detected_intent,
                expected_next_state=init_rules[detected_intent], is_valid=True,
                drift_severity=DriftSeverity.MILD,
                matched_rule=f'宽松模式: 回退到 INIT → {init_rules[detected_intent]}',
                detail=f'允许从 {current_state} 回退到 INIT 重新发起 {detected_intent}',
            )
        for state, rules in self.matrix.main_transitions.items():
            state_cat = self.matrix.state_category_map.get(state, '??')
            if state_cat == current_cat and detected_intent in rules:
                return TransitionResult(
                    turn=turn, current_state=current_state, detected_intent=detected_intent,
                    expected_next_state=rules[detected_intent], is_valid=True,
                    drift_severity=DriftSeverity.MILD,
                    matched_rule=f'宽松模式: 同级域 {state} + {detected_intent} → {rules[detected_intent]}',
                    detail=f'允许在 {current_cat} 业务域内跳跃转移',
                )
        if current_state.startswith('ZX_') and detected_intent in init_rules:
            return TransitionResult(
                turn=turn, current_state=current_state, detected_intent=detected_intent,
                expected_next_state=init_rules[detected_intent], is_valid=True,
                drift_severity=DriftSeverity.MILD,
                matched_rule=f'宽松模式: 查询回业务 → {init_rules[detected_intent]}',
                detail='允许从查询态回到业务操作态',
            )
        return self._build_drift_result(current_state, detected_intent, turn)

    def _category_validate(self, current_state: str, detected_intent: str,
                           turn: int) -> TransitionResult:
        """分类模式：只校验业务大类"""
        current_cat = self.matrix.state_category_map.get(current_state, '??')
        init_rules = self.matrix.main_transitions.get('INIT', {})
        target_state = init_rules.get(detected_intent)
        if not target_state:
            fuzzy = self._fuzzy_match_intent(detected_intent, init_rules)
            target_state = init_rules.get(fuzzy) if fuzzy else None
        if target_state:
            target_cat = self.matrix.state_category_map.get(target_state, '??')
            if current_cat == target_cat or target_cat in ('INIT',):
                return TransitionResult(
                    turn=turn, current_state=current_state, detected_intent=detected_intent,
                    expected_next_state=target_state, is_valid=True,
                    drift_severity=DriftSeverity.MILD,
                    matched_rule=f'分类模式: {current_cat} → {target_cat}',
                    detail=f'业务大类匹配（{current_cat} → {target_cat}）',
                )
        return self._build_drift_result(current_state, detected_intent, turn)

    def _build_drift_result(self, current_state: str, detected_intent: str,
                            turn: int) -> TransitionResult:
        """构建漂移结果"""
        current_cat = self.matrix.state_category_map.get(current_state, '??')
        severity = self._classify_drift_severity(current_state, detected_intent)
        return TransitionResult(
            turn=turn, current_state=current_state, detected_intent=detected_intent,
            expected_next_state=None, is_valid=False, drift_severity=severity,
            matched_rule=None,
            detail=f'状态 {current_state}（{current_cat}类）下未定义意图 "{detected_intent}" 的合法转移',
        )

    def _classify_drift_severity(self, current_state: str,
                                 detected_intent: str) -> DriftSeverity:
        """漂移严重程度分级"""
        current_cat = self.matrix.state_category_map.get(current_state, '??')
        init_rules = self.matrix.main_transitions.get('INIT', {})
        target_state = init_rules.get(detected_intent)
        if not target_state:
            fuzzy = self._fuzzy_match_intent(detected_intent, init_rules)
            target_state = init_rules.get(fuzzy) if fuzzy else None
        target_cat = self.matrix.state_category_map.get(target_state, '??') if target_state else '??'

        if current_state in self.matrix.terminal_states:
            return DriftSeverity.CRITICAL
        if current_cat == target_cat and current_cat not in ('??', 'INIT', 'END'):
            return DriftSeverity.MILD
        business_cats = {'TL', 'JX', 'JK', 'ZX', 'FH'}
        if current_cat in business_cats and target_cat in business_cats:
            return DriftSeverity.MODERATE
        if current_cat in business_cats and target_cat in ('FY', 'MH'):
            return DriftSeverity.SEVERE
        if current_cat in business_cats and target_cat in ('RG', 'FX'):
            return DriftSeverity.SEVERE
        return DriftSeverity.MODERATE


def build_trajectory_from_intents(intents: List[str],
                                  *, start_state: str = 'INIT',
                                  validator: SessionStateValidator = None) -> List[Dict[str, Any]]:
    """从意图序列自动构建轨迹（按状态转移矩阵推导每轮 current_state）"""
    validator = validator or SessionStateValidator()
    trajectory = []
    current = start_state
    for i, intent in enumerate(intents):
        result = validator.validate_single_turn(current, intent, turn=i + 1)
        trajectory.append({'turn': i + 1, 'current_state': current, 'detected_intent': intent})
        current = result.expected_next_state or current
    return trajectory


# ============================================================
# 自测（照搬参考实现的 8 组用例）
# ============================================================

if __name__ == '__main__':
    print('=' * 60)
    print('  state_validator 自测')
    print('=' * 60)

    validator = SessionStateValidator()
    assert validator.matrix.main_transitions, 'YAML 矩阵加载失败'

    # 1. 正常单轮
    r = validator.validate_single_turn('INIT', '提额业务咨询')
    assert r.is_valid and r.expected_next_state == 'TL_YWZX'
    print('[OK] 1. 正常单轮转移')

    # 2. 非法转移
    r = validator.validate_single_turn('TL_ZRTJ', '借款操作咨询')
    assert not r.is_valid
    print('[OK] 2. 非法转移判定')

    # 3. P0 全局规则
    r = validator.validate_single_turn('TL_CQZX', '用户主动要求转人工')
    assert r.is_valid and r.expected_next_state == 'RG_BZSQ'
    print('[OK] 3. P0 全局规则（任意状态转人工）')

    # 4. 正常轨迹漂移率 0
    traj = build_trajectory_from_intents([
        '提额业务咨询', '提额准入条件咨询', '提额操作咨询',
        '提额材料咨询', '提额进度咨询', '提额结果咨询',
    ], validator=validator)
    result = validator.validate_multi_turn(traj)
    assert result.drift_rate == 0.0, result.summary
    print('[OK] 4. 正常轨迹漂移率 0%')

    # 5. 含漂移轨迹
    drift_traj = [
        {'turn': 1, 'current_state': 'INIT', 'detected_intent': '提额业务咨询'},
        {'turn': 2, 'current_state': 'TL_YWZX', 'detected_intent': '借款操作咨询'},
        {'turn': 3, 'current_state': 'JK_CQZX', 'detected_intent': '非业务相关'},
        {'turn': 4, 'current_state': 'INIT', 'detected_intent': '降息业务咨询'},
    ]
    result = validator.validate_multi_turn(drift_traj)
    # 第2轮（提额域→借款）为漂移；第3轮"非业务相关"命中 P0 全局规则合法；第4轮回 INIT 合法
    assert result.invalid_turns == 1, result.summary
    print(f'[OK] 5. 漂移轨迹判定（漂移率 {result.drift_rate:.0%}）')

    # 6. P0 全局规则优先级（任意业务态情绪升级）
    for state in ['TL_CQZX', 'JX_JG', 'JK_CQZX', 'ZX_EDCX', 'INIT']:
        r = validator.validate_single_turn(state, '用户情绪升级')
        assert r.is_valid and r.expected_next_state == 'RG_QX_SJ'
    print('[OK] 6. P0 全局规则优先级')

    # 7. 终态不可转移（CRITICAL）
    r = validator.validate_single_turn('FX_WGCZ', '提额业务咨询')
    assert not r.is_valid and r.drift_severity == DriftSeverity.CRITICAL
    print('[OK] 7. 终态继续转移 = CRITICAL')

    # 8. 提额成功 → 引导借款闭环
    r = validator.validate_single_turn('TL_JG', '借款操作咨询')
    assert r.is_valid and r.expected_next_state == 'JK_CQZX'
    print('[OK] 8. 提额成功→借款闭环')

    print('=' * 60)
    print('  全部自测通过')
    print('=' * 60)
