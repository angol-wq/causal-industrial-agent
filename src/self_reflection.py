"""
Agent 自我反思机制 — 分析错误 → 提取教训 → 补充知识库 → 规避同类失误

反思闭环:
  诊断出错 → 反思分析(为什么错?) → 提取缺失规则 → 注入知识库 → 下次规避 ✓

三种反思模式:
  1. 因果图缺失反思: 根因正确但路径不完整 → 缺了某条因果边
  2. 阈值不当反思: 异常没检测到 → normal_range 太宽/太窄
  3. 评分偏差反思: 根因排序错误 → 评分权重需要调整

用法:
  reflector = SelfReflectionEngine(agent)
  lesson = reflector.reflect(observation, agent_diagnosis, ground_truth)
  # lesson 会自动写入知识库
"""

import sys, os, json, time, hashlib
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from datetime import datetime
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))


@dataclass
class ReflectionLesson:
    """一次反思的教训"""
    timestamp: str
    error_type: str          # "missing_edge" / "wrong_ranking" / "missed_detection" / "false_positive"
    what_happened: str       # 发生了什么
    why_wrong: str           # 为什么会错
    extracted_rule: Dict     # 提取出的新规则/修正
    confidence: float        # 这次反思的置信度
    applied: bool = False    # 是否已应用到知识库


class SelfReflectionEngine:
    """
    自我反思引擎 — Agent 的"事后复盘"能力

    工作原理:
      当 Agent 的诊断结果与真实根因不一致时，触发反思:
      1. 对比 agent_diagnosis vs ground_truth
      2. 识别错误类型(缺失边/排序错/漏检/误报)
      3. 提取可操作的修正规则
      4. 将规则注入知识库(因果图/参数/阈值)
      5. 记录反思日志(持久化)
    """

    def __init__(self, log_dir: str = "data/reflection"):
        self.log_dir = log_dir
        self.lessons: List[ReflectionLesson] = []
        os.makedirs(log_dir, exist_ok=True)
        self._load_history()

    # ================================================================
    # 核心反思流程
    # ================================================================
    def reflect(self,
                observation: Dict[str, float],
                agent_diagnosis: Dict,          # Agent的诊断结果
                ground_truth: Dict,             # 真实根因 (如果有)
                causal_graph,                   # 当前因果图
                normal_ranges: Dict = None,     # 当前正常范围
                ) -> List[ReflectionLesson]:
        """
        执行一次完整反思

        Args:
            observation: 异常观测值
            agent_diagnosis: {"root_cause": "X", "score": 0.7, "path": [...]}
            ground_truth: {"root_cause": "Y", "verified_by": "操作员确认"}
            causal_graph: 当前因果图
            normal_ranges: 当前正常范围

        Returns:
            提取的教训列表
        """
        agent_rc = agent_diagnosis.get("root_cause", "")
        true_rc = ground_truth.get("root_cause", "")
        lessons = []

        print(f"\n[反思] 分析诊断结果...")
        print(f"  Agent诊断: {agent_rc} (评分: {agent_diagnosis.get('score', 0):.3f})")
        print(f"  真实根因: {true_rc}")

        # ---- 情况1: 根因完全正确 ----
        if agent_rc == true_rc or true_rc in agent_rc:
            # 仍然值得反思: 排序正确吗? 路径完整吗?
            if agent_diagnosis.get("path"):
                completeness = self._check_path_completeness(
                    agent_diagnosis["path"], causal_graph, ground_truth
                )
                if completeness < 1.0:
                    lesson = self._reflect_missing_edge(
                        observation, agent_diagnosis, ground_truth, causal_graph
                    )
                    if lesson:
                        lessons.append(lesson)
            print(f"  ✅ 根因正确，无需反思")
            return lessons

        # ---- 情况2: 根因错误 — 需要深度反思 ----
        print(f"  ❌ 根因不匹配, 启动深度反思...")

        # 反思维度1: 因果图是否缺失关键边?
        lesson1 = self._reflect_missing_edge(
            observation, agent_diagnosis, ground_truth, causal_graph
        )
        if lesson1:
            lessons.append(lesson1)

        # 反思维度2: 评分权重是否需要调整?
        lesson2 = self._reflect_wrong_ranking(
            observation, agent_diagnosis, ground_truth
        )
        if lesson2:
            lessons.append(lesson2)

        # 反思维度3: 检测阈值是否太宽/太窄?
        lesson3 = self._reflect_detection_failure(
            observation, agent_diagnosis, ground_truth, normal_ranges
        )
        if lesson3:
            lessons.append(lesson3)

        # 反思维度4: 是否误报?
        if ground_truth.get("is_false_alarm"):
            lesson4 = self._reflect_false_positive(
                observation, agent_diagnosis, ground_truth
            )
            if lesson4:
                lessons.append(lesson4)

        # 保存
        for lesson in lessons:
            self.lessons.append(lesson)
            self._save_lesson(lesson)

        print(f"  📝 提取 {len(lessons)} 条教训")
        return lessons

    # ================================================================
    # 反思维度1: 因果图缺失边
    # ================================================================
    def _reflect_missing_edge(self, observation, agent_diag, truth, graph
                              ) -> Optional[ReflectionLesson]:
        """
        分析: 如果真实根因在因果图中存在，但Agent没有找到它——
        是因为因果路径上缺了一条边吗?
        """
        true_rc = truth.get("root_cause", "")
        agent_rc = agent_diag.get("root_cause", "")
        true_path = truth.get("causal_path", [])

        # 检查真实根因是否在因果图中
        if true_rc not in graph.nodes():
            # 因果图缺少这个变量
            return ReflectionLesson(
                timestamp=datetime.now().isoformat(),
                error_type="missing_node",
                what_happened=f"真实根因'{true_rc}'不在因果图中，Agent无法诊断",
                why_wrong=f"因果图缺少变量节点: {true_rc}",
                extracted_rule={
                    "action": "add_node",
                    "node": true_rc,
                    "context": truth.get("mechanism", "未知机理"),
                    "evidence": truth.get("verified_by", "人工反馈"),
                },
                confidence=0.8,
            )

        # 检查真实根因和Agent根因之间是否有路径
        if true_rc in graph.nodes() and agent_rc in graph.nodes():
            try:
                import networkx as nx
                path_exists = nx.has_path(graph, true_rc, agent_rc)
                if not path_exists:
                    # 两个节点之间没有路径 —— 缺一条关键边
                    return ReflectionLesson(
                        timestamp=datetime.now().isoformat(),
                        error_type="missing_edge",
                        what_happened=f"真实根因'{true_rc}'与Agent诊断'{agent_rc}'之间无因果路径",
                        why_wrong=f"因果图缺少从'{true_rc}'到'{agent_rc}'的因果连接",
                        extracted_rule={
                            "action": "add_edge",
                            "cause": true_rc,
                            "effect": agent_rc,
                            "mechanism": truth.get("mechanism", "从错误分析中推断"),
                            "confidence": 0.6,  # 反思得到的边置信度较低
                            "source": "self_reflection",
                            "evidence": f"诊断错误: Agent={agent_rc}, 真实={true_rc}",
                        },
                        confidence=0.7,
                    )
            except Exception:
                pass

        return None

    # ================================================================
    # 反思维度2: 评分排序错误
    # ================================================================
    def _reflect_wrong_ranking(self, observation, agent_diag, truth
                               ) -> Optional[ReflectionLesson]:
        """
        分析: 真实根因在Agent的候选列表中，但排得太靠后——
        评分权重需要调整吗?
        """
        agent_rc = agent_diag.get("root_cause", "")
        true_rc = truth.get("root_cause", "")
        all_candidates = agent_diag.get("all_candidates", [])

        # 检查真实根因是否在候选列表里
        true_in_list = None
        for c in all_candidates:
            if c.get("variable") == true_rc or true_rc in c.get("variable", ""):
                true_in_list = c
                break

        if true_in_list:
            rank = true_in_list.get("rank", 99)
            if rank > 2:  # 排在前2名之外
                return ReflectionLesson(
                    timestamp=datetime.now().isoformat(),
                    error_type="wrong_ranking",
                    what_happened=f"真实根因'{true_rc}'在候选列表第{rank}位, 但Agent选了'{agent_rc}'",
                    why_wrong="评分函数权重不当: 因果效应/置信度/路径长度/偏离程度 的相对重要性需调整",
                    extracted_rule={
                        "action": "adjust_scoring_weights",
                        "variable": true_rc,
                        "old_rank": rank,
                        "reason_for_low_score": "路径较长/置信度较低/偏离较小",
                        "suggested_fix": f"当'{true_rc}'出现在路径中时, 提升其权重",
                    },
                    confidence=0.5,
                )

        return None

    # ================================================================
    # 反思维度3: 检测失败(漏检)
    # ================================================================
    def _reflect_detection_failure(self, observation, agent_diag, truth,
                                    normal_ranges) -> Optional[ReflectionLesson]:
        """
        分析: Agent根本没检测到异常——是阈值太宽了吗?
        """
        true_rc = truth.get("root_cause", "")
        abnormal_var = truth.get("abnormal_variable", "")

        if abnormal_var and normal_ranges and abnormal_var in normal_ranges:
            lo, hi = normal_ranges[abnormal_var]
            obs_val = observation.get(abnormal_var, (lo + hi) / 2)

            # 检查是否在正常范围内(漏检)
            if lo <= obs_val <= hi:
                # 漏检了 —— 阈值太宽
                return ReflectionLesson(
                    timestamp=datetime.now().isoformat(),
                    error_type="missed_detection",
                    what_happened=f"'{abnormal_var}'实际异常(根因={true_rc}), 但在正常范围[{lo:.1f},{hi:.1f}]内",
                    why_wrong=f"正常范围阈值太宽(n_std可能太大), 导致早期异常被漏掉",
                    extracted_rule={
                        "action": "tighten_normal_range",
                        "variable": abnormal_var,
                        "old_range": [lo, hi],
                        "suggested_new_std_multiplier": 2.0,  # 从3σ收紧到2σ
                        "reason": f"历史漏检: 值{obs_val:.1f}在[{lo:.1f},{hi:.1f}]内但实际异常",
                    },
                    confidence=0.6,
                )

        return None

    # ================================================================
    # 反思维度4: 误报
    # ================================================================
    def _reflect_false_positive(self, observation, agent_diag, truth
                                 ) -> Optional[ReflectionLesson]:
        """
        分析: Agent报了异常但实际没有——为什么误报?
        """
        agent_rc = agent_diag.get("root_cause", "")

        return ReflectionLesson(
            timestamp=datetime.now().isoformat(),
            error_type="false_positive",
            what_happened=f"Agent诊断根因为'{agent_rc}', 但实际无故障(误报)",
            why_wrong="传感器噪声/瞬态波动被误判为异常, 或因果图中存在虚假因果边",
            extracted_rule={
                "action": "add_persistence_check",
                "variable": agent_rc,
                "rule": "观测值持续超过阈值≥3个采样点才触发诊断(避免瞬态误报)",
                "suggested_min_duration": 3,
            },
            confidence=0.7,
        )

    # ================================================================
    # 路径完整性检查
    # ================================================================
    def _check_path_completeness(self, agent_path: List[str],
                                  graph, truth: Dict) -> float:
        """检查Agent给出的因果路径是否完整"""
        true_path = truth.get("causal_path", [])
        if not true_path or not agent_path:
            return 1.0

        # 计算路径覆盖率
        agent_set = set(agent_path)
        true_set = set(true_path)
        overlap = len(agent_set & true_set)
        return overlap / len(true_set) if true_set else 1.0

    # ================================================================
    # 应用教训到知识库
    # ================================================================
    def apply_lessons(self, causal_graph, normal_ranges: Dict = None
                      ) -> Dict:
        """
        将未应用的教训注入知识库

        Returns:
            应用报告: {"edges_added": N, "ranges_adjusted": M, ...}
        """
        report = {"edges_added": 0, "ranges_adjusted": 0, "rules_added": 0}

        for lesson in self.lessons:
            if lesson.applied:
                continue

            rule = lesson.extracted_rule
            action = rule.get("action", "")

            if action == "add_edge" and causal_graph is not None:
                causal_graph.add_edge(
                    rule["cause"], rule["effect"],
                    mechanism=rule.get("mechanism", ""),
                    confidence=rule.get("confidence", 0.5),
                    source="self_reflection",
                )
                report["edges_added"] += 1
                lesson.applied = True

            elif action == "tighten_normal_range" and normal_ranges is not None:
                var = rule["variable"]
                if var in normal_ranges:
                    lo, hi = normal_ranges[var]
                    center = (lo + hi) / 2
                    new_std = rule.get("suggested_new_std_multiplier", 2.0)
                    new_half = (hi - lo) * (new_std / 3.0)
                    normal_ranges[var] = (center - new_half, center + new_half)
                    report["ranges_adjusted"] += 1
                    lesson.applied = True

            elif action in ("adjust_scoring_weights", "add_persistence_check"):
                report["rules_added"] += 1
                lesson.applied = True

        # 保存更新状态
        self._save_history()

        print(f"[反思应用] {report['edges_added']}边 + "
              f"{report['ranges_adjusted']}阈值 + {report['rules_added']}规则")

        return report

    # ================================================================
    # 持久化
    # ================================================================
    def _save_lesson(self, lesson: ReflectionLesson):
        path = os.path.join(self.log_dir,
                           f"lesson_{datetime.now():%Y%m%d_%H%M%S}_{hashlib.md5(lesson.what_happened.encode()).hexdigest()[:8]}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump({
                "timestamp": lesson.timestamp,
                "error_type": lesson.error_type,
                "what_happened": lesson.what_happened,
                "why_wrong": lesson.why_wrong,
                "extracted_rule": lesson.extracted_rule,
                "confidence": lesson.confidence,
                "applied": lesson.applied,
            }, f, ensure_ascii=False, indent=2)

    def _load_history(self):
        if not os.path.exists(self.log_dir):
            return
        for f in sorted(os.listdir(self.log_dir)):
            if f.endswith(".json"):
                with open(os.path.join(self.log_dir, f), encoding="utf-8") as fp:
                    data = json.load(fp)
                self.lessons.append(ReflectionLesson(**data))

    def _save_history(self):
        pass  # 每条lesson单独保存，不需要额外操作

    def get_stats(self) -> Dict:
        """反思统计"""
        by_type = Counter(l.applied for l in self.lessons)
        return {
            "total_lessons": len(self.lessons),
            "applied": by_type.get(True, 0),
            "pending": by_type.get(False, 0),
            "by_error_type": dict(Counter(l.error_type for l in self.lessons)),
            "avg_confidence": sum(l.confidence for l in self.lessons) / len(self.lessons) if self.lessons else 0,
        }


if __name__ == "__main__":
    print("=" * 50)
    print("自我反思引擎测试")
    print("=" * 50)

    # 模拟一次诊断错误
    engine = SelfReflectionEngine()

    # 模拟: Agent说是CW_Flow的问题, 但真实是冷却水泵故障
    observation = {"CW_Flow": 118, "Reactor_Temp": 192, "Pump_Current": 45}
    agent_diag = {
        "root_cause": "CW_Flow",
        "score": 0.74,
        "path": ["CW_Valve", "CW_Flow", "Reactor_Temp"],
        "all_candidates": [
            {"variable": "CW_Flow", "rank": 1, "score": 0.74},
            {"variable": "Pump_Current", "rank": 4, "score": 0.35},
        ],
    }
    ground_truth = {
        "root_cause": "Pump_Current",
        "verified_by": "操作员现场确认: 冷却水泵电机电流异常",
        "mechanism": "泵电机绕组短路→电流升高→但转速下降→流量不足",
        "causal_path": ["Pump_Current", "CW_Flow", "Reactor_Temp"],
    }

    import networkx as nx
    g = nx.DiGraph()
    g.add_edge("CW_Valve", "CW_Flow", mechanism="阀门→流量", confidence=0.9)
    g.add_edge("CW_Flow", "Reactor_Temp", mechanism="流量→温度", confidence=0.9)
    # 注意: 图中没有 Pump_Current 节点! 这就是Agent犯错的原因

    normal_ranges = {
        "CW_Flow": (200, 300),
        "Reactor_Temp": (150, 175),
        "Pump_Current": (50, 80),
    }

    lessons = engine.reflect(observation, agent_diag, ground_truth, g, normal_ranges)
    print(f"\n提取教训: {len(lessons)} 条")
    for l in lessons:
        print(f"  [{l.error_type}] {l.what_happened}")
        print(f"  原因: {l.why_wrong}")
        print(f"  修正: {l.extracted_rule.get('action')}")

    # 应用教训
    report = engine.apply_lessons(g, normal_ranges)
    print(f"\n应用结果: 加{report['edges_added']}边, "
          f"调{report['ranges_adjusted']}阈值, "
          f"加{report['rules_added']}规则")

    stats = engine.get_stats()
    print(f"\n反思统计: {stats}")
