"""
根因分析引擎: 基于因果图进行异常检测和根因回溯

核心能力:
  1. 因果感知的异常检测: 不仅检测"哪个变量异常"，还追溯"为什么异常"
  2. 根因排序: 基于因果效应大小、路径距离、置信度对根因排序
  3. 因果链可视化: 输出从根因到异常变量的完整因果路径

与传统的区别:
  传统异常检测: "Reactor_Temp偏高" → 结束
  本方案: "Reactor_Temp偏高" → 回溯因果图 → 发现CW_Flow下降 →
          继续回溯 → CW_Valve卡滞 → 给出因果链 + 处置建议
"""

import numpy as np
import pandas as pd
import networkx as nx
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from collections import defaultdict


@dataclass
class AnomalyFinding:
    """异常发现"""
    variable: str
    observed_value: float
    normal_range: Tuple[float, float]
    deviation: float          # 偏离程度（标准差倍数）


@dataclass
class RootCause:
    """根因分析结果"""
    variable: str
    causal_effect: float      # 因果效应大小
    confidence: float
    path: List[str]           # 从根因到异常变量的因果路径
    path_length: int
    score: float              # 综合评分
    evidence: str = ""


@dataclass
class RootCauseReport:
    """完整的根因分析报告"""
    abnormal_variable: str
    observed_value: float
    normal_range: Tuple[float, float]
    root_causes: List[RootCause]
    recommended_actions: List[str]
    causal_paths_visualization: Dict


class RootCauseAnalyzer:
    """
    基于因果图的根因分析器

    算法思路:
      1. 检测异常变量（偏离正常范围）
      2. 对每个异常变量，在因果图上逆流回溯，找到所有祖先变量
      3. 检查哪些祖先变量也异常 → 这些是候选根因
      4. 对候选根因排序：考虑因果效应大小、路径距离、异常程度
      5. 输出根因排序 + 因果路径 + 处置建议
    """

    def __init__(self, causal_graph: nx.DiGraph):
        """
        Args:
            causal_graph: 融合后的因果图（含因果关系、置信度、机制信息）
        """
        self.causal_graph = causal_graph
        self._build_ancestor_cache()

    def _build_ancestor_cache(self):
        """预计算每个节点的所有祖先（用于快速回溯）"""
        self.ancestors = {}
        for node in self.causal_graph.nodes():
            self.ancestors[node] = set(nx.ancestors(self.causal_graph, node))

    def set_normal_ranges(self, normal_data: pd.DataFrame,
                          n_std: float = 3.0):
        """
        从正常工况数据自动计算各变量的正常范围

        Args:
            normal_data: 正常工况下的时序数据
            n_std: 几倍标准差视为异常
        """
        self.normal_ranges = {}
        for col in normal_data.columns:
            mean = normal_data[col].mean()
            std = normal_data[col].std()
            self.normal_ranges[col] = (mean - n_std * std, mean + n_std * std)

    def detect_anomalies(self, observation: Dict[str, float]
                         ) -> List[AnomalyFinding]:
        """检测当前观测中的异常变量"""
        anomalies = []
        for var, val in observation.items():
            if var in self.normal_ranges:
                lo, hi = self.normal_ranges[var]
                if val < lo or val > hi:
                    middle = (lo + hi) / 2
                    half_range = (hi - lo) / 2
                    deviation = (val - middle) / half_range if half_range > 0 else 0
                    anomalies.append(AnomalyFinding(
                        variable=var,
                        observed_value=val,
                        normal_range=(lo, hi),
                        deviation=deviation,
                    ))
        return anomalies

    def analyze(self, observation: Dict[str, float],
                top_k: int = 3) -> List[RootCauseReport]:
        """
        执行完整的根因分析

        Args:
            observation: 当前时刻的观测值 {变量名: 值}
            top_k: 每个异常变量返回的top-k根因

        Returns:
            根因分析报告列表（每个异常变量一个报告）
        """
        # Step 1: 检测异常
        anomalies = self.detect_anomalies(observation)
        if not anomalies:
            return []

        anomaly_var_names = {a.variable for a in anomalies}
        reports = []

        for anomaly in anomalies:
            # Step 2: 在因果图上逆流回溯，找祖先
            ancestors = self.ancestors.get(anomaly.variable, set())

            # Step 3: 祖先中哪些也异常？→ 候选根因
            candidate_root_causes = []
            for ancestor in ancestors:
                if ancestor in anomaly_var_names and ancestor != anomaly.variable:
                    # 检查是否有因果路径
                    if nx.has_path(self.causal_graph, ancestor, anomaly.variable):
                        path = nx.shortest_path(
                            self.causal_graph, ancestor, anomaly.variable
                        )
                        causal_effect = self._estimate_path_effect(
                            ancestor, anomaly.variable, path
                        )
                        confidence = self._get_path_confidence(path)

                        ancestor_anomaly = next(
                            (a for a in anomalies if a.variable == ancestor), None
                        )
                        ancestor_deviation = (ancestor_anomaly.deviation
                                             if ancestor_anomaly else 0)

                        score = self._compute_root_cause_score(
                            causal_effect, confidence, len(path),
                            ancestor_deviation
                        )

                        candidate_root_causes.append(RootCause(
                            variable=ancestor,
                            causal_effect=causal_effect,
                            confidence=confidence,
                            path=path,
                            path_length=len(path),
                            score=score,
                            evidence=self._get_edge_evidence(path),
                        ))

            # Step 4: 按评分排序
            candidate_root_causes.sort(key=lambda x: x.score, reverse=True)

            # Step 5: 生成处置建议
            top_causes = candidate_root_causes[:top_k]
            actions = self._generate_actions(top_causes, anomaly)

            reports.append(RootCauseReport(
                abnormal_variable=anomaly.variable,
                observed_value=anomaly.observed_value,
                normal_range=anomaly.normal_range,
                root_causes=top_causes,
                recommended_actions=actions,
                causal_paths_visualization={
                    "variable": anomaly.variable,
                    "paths": [
                        {"nodes": rc.path, "score": rc.score}
                        for rc in top_causes
                    ]
                }
            ))

        return reports

    def _estimate_path_effect(self, cause: str, effect: str,
                              path: List[str]) -> float:
        """
        估计因果路径上的总因果效应

        对于链式因果: X→Y→Z
        总效应 ≈ β_{X→Y} × β_{Y→Z}（简化：连续相乘）
        """
        total_effect = 1.0
        for i in range(len(path) - 1):
            u, v = path[i], path[i + 1]
            if self.causal_graph.has_edge(u, v):
                edge_data = self.causal_graph[u][v]
                # 使用边的confidence作为效应强度近似（如果有coefficient则用coefficient）
                edge_coef = edge_data.get("coefficient", edge_data.get("confidence", 0.5))
                total_effect *= abs(edge_coef)
        return total_effect

    def _get_path_confidence(self, path: List[str]) -> float:
        """计算因果路径的整体置信度"""
        confidences = []
        for i in range(len(path) - 1):
            u, v = path[i], path[i + 1]
            if self.causal_graph.has_edge(u, v):
                conf = self.causal_graph[u][v].get("confidence", 0.5)
                confidences.append(conf)
        if not confidences:
            return 0.0
        # 路径置信度 = 各边置信度的几何平均
        return np.exp(np.mean(np.log(confidences)))

    def _compute_root_cause_score(self, causal_effect: float,
                                  confidence: float, path_length: int,
                                  deviation: float) -> float:
        """
        综合评分（越高越可能是根因）

        因素:
          - 因果效应大
          - 置信度高
          - 路径短（越直接）
          - 自身偏离程度大
        """
        # path_penalty: 路径越长，是该异常的根本原因的可能性越低
        path_penalty = 1.0 / np.sqrt(path_length)

        score = (abs(causal_effect) * 0.30 +
                 confidence * 0.30 +
                 path_penalty * 0.20 +
                 min(abs(deviation) / 5.0, 1.0) * 0.20)
        return round(score, 4)

    def _get_edge_evidence(self, path: List[str]) -> str:
        """收集因果路径上的证据/机制描述"""
        evidences = []
        for i in range(len(path) - 1):
            u, v = path[i], path[i + 1]
            if self.causal_graph.has_edge(u, v):
                mech = self.causal_graph[u][v].get("mechanism", "")
                if mech:
                    evidences.append(f"{u}→{v}: {mech}")
        return "; ".join(evidences) if evidences else "无详细记录"

    def _generate_actions(self, root_causes: List[RootCause],
                          anomaly: AnomalyFinding) -> List[str]:
        """根据根因生成处置建议"""
        actions = []

        if not root_causes:
            actions.append(f"⚠ 未找到明确根因，建议人工排查 {anomaly.variable}")
            return actions

        top = root_causes[0]
        variable = top.variable

        # 基于变量名的启发式建议（实际使用中可接知识库）
        action_templates = {
            "CW_Valve": [
                f"检查{variable}是否存在机械卡滞",
                f"尝试增大{variable}开度指令,观察CW_Flow是否响应",
                f"如{variable}无响应,切换至备用冷却回路",
            ],
            "CW_Flow": [
                f"检查冷却水回路: 确认阀门开度→如阀门正常,检查泵运行状态",
                f"检查冷却水管路是否有堵塞或泄漏",
            ],
            "CW_Inlet_Temp": [
                f"检查冷却塔运行状态(风机/填料/布水器)",
                f"检查循环水补充量是否充足",
                f"如持续恶化,降低反应负荷运行",
            ],
            "Feed_Flow": [
                f"检查{variable}控制回路和调节阀",
                f"确认进料泵运行状态",
                f"如{variable}波动,切换备用泵",
            ],
            "Feed_Conc": [
                f"检查上游原料配比与品质",
                f"联系前道工序确认原料批次",
                f"如{variable}不达标,考虑适当降低Feed_Flow延长停留时间",
            ],
            "Reaction_Rate": [
                f"{top.path[0]}异常导致反应速率变化",
                f"优先处置上游根因: {top.path[0]}",
            ],
        }

        # 匹配模板
        for key, templates in action_templates.items():
            if key in variable or variable in key:
                actions.extend(templates)
                break
        else:
            actions.append(f"处置上游根因节点: {variable}")
            if len(top.path) > 1:
                actions.append(f"因果路径: {' → '.join(top.path)}")

        # 添加紧急停车条件提示
        if anomaly.variable in ["Reactor_Temp"]:
            hi = anomaly.normal_range[1]
            if anomaly.observed_value > hi * 1.15:
                actions.insert(0, "⚠⚠⚠ 反应器温度严重超标，建议立即紧急停车！")

        return actions


def create_action_knowledge_base() -> Dict[str, Dict]:
    """
    构建处置知识库（可扩展）

    实际落地时可以:
      - 从SOP文档中自动提取
      - 对接企业的应急预案系统
      - 用RAG检索历史处置记录
    """
    return {
        "CW_Valve": {
            "check_points": ["阀门机械卡滞", "气动/电动执行器", "控制信号"],
            "actions": ["增大开度指令", "切换备用回路", "手动操作"],
            "lead_time": "5-15分钟",
        },
        "CW_Inlet_Temp": {
            "check_points": ["冷却塔风机", "循环水补水", "环境温度"],
            "actions": ["启动备用风机", "补充循环水", "降低负荷"],
            "lead_time": "15-30分钟",
        },
        "Feed_Flow": {
            "check_points": ["进料泵", "调节阀", "流量计"],
            "actions": ["切换备用泵", "检修调节阀", "校准流量计"],
            "lead_time": "5-10分钟",
        },
    }


if __name__ == "__main__":
    # 测试根因分析
    import sys
    sys.path.insert(0, ".")

    print("=" * 60)
    print("根因分析引擎测试")
    print("=" * 60)

    # 用融合后的因果图（从graph_fusion测试中构建）
    from graph_fusion import CausalGraphFusion
    import networkx as nx

    kg = nx.DiGraph()
    kg.add_edge("CW_Valve", "CW_Flow", confidence=0.95)
    kg.add_edge("CW_Flow", "Reactor_Temp", confidence=0.90)
    kg.add_edge("Reactor_Temp", "Reaction_Rate", confidence=0.95)

    dg = nx.DiGraph()
    dg.add_edge("CW_Valve", "CW_Flow", significance=0.95)
    dg.add_edge("CW_Flow", "Reactor_Temp", significance=0.88)
    dg.add_edge("CW_Inlet_Temp", "Reactor_Temp", significance=0.82)
    dg.add_edge("Reactor_Temp", "Reaction_Rate", significance=0.90)

    fusion = CausalGraphFusion()
    fused = fusion.fuse(kg, dg)

    # 设置正常范围
    analyzer = RootCauseAnalyzer(fused)
    analyzer.normal_ranges = {
        "CW_Valve": (45, 75),
        "CW_Flow": (180, 320),
        "CW_Inlet_Temp": (22, 30),
        "Reactor_Temp": (150, 175),
        "Reaction_Rate": (0.7, 1.3),
    }

    # 模拟一个观测: 冷却水阀门卡滞导致的一系列异常
    observation = {
        "CW_Valve": 22,           # 异常低 (正常45-75)
        "CW_Flow": 120,           # 异常低 (正常180-320)
        "CW_Inlet_Temp": 26,      # 正常
        "Reactor_Temp": 192,      # 异常高 (正常150-175)
        "Reaction_Rate": 1.45,    # 异常高 (正常0.7-1.3)
    }

    reports = analyzer.analyze(observation, top_k=3)

    for report in reports:
        print(f"\n{'='*40}")
        print(f"[异常] {report.abnormal_variable} = {report.observed_value}"
              f" (正常范围: {report.normal_range[0]}-{report.normal_range[1]})")
        print(f"\n[根因分析]")
        for i, rc in enumerate(report.root_causes, 1):
            print(f"  {i}. {rc.variable} (评分: {rc.score:.3f})")
            print(f"     因果路径: {' → '.join(rc.path)}")
            print(f"     路径长度: {rc.path_length}, 因果效应: {rc.causal_effect:.4f}")
        print(f"\n[处置建议]")
        for action in report.recommended_actions:
            print(f"  • {action}")
