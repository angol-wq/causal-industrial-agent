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
        if not hasattr(self, 'normal_ranges') or not self.normal_ranges:
            raise ValueError("normal_ranges未设置。请先调用 set_normal_ranges() 或手动设置 self.normal_ranges。")
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
        """生成专业级诊断报告: 原因解释 + 物理机制 + 分步处置 + 预期效果"""
        actions = []

        if not root_causes:
            actions.append(f"⚠ 未找到明确根因。{anomaly.variable}偏离正常范围，"
                          f"但其上游因果链中无其他异常变量同时偏离，"
                          f"推测为独立的局部故障或传感器异常。建议人工巡检确认。")
            return actions

        top = root_causes[0]
        variable = top.variable

        # 起点: 根因解释
        root_explanations = {
            "CW_Valve": (
                f"**根因分析**: 冷却水阀门开度({variable})偏离正常值，"
                f"评分 {top.score:.3f}，因果效应系数 {abs(top.causal_effect):.3f}。\n\n"
                f"**物理机制**: 阀门作为冷却水回路的节流元件，其开度直接决定冷却水流量。"
                f"根据流体力学伯努利方程 Q = Cv × √(ΔP)，开度↓ → 流量系数Cv↓ → 流量Q↓。"
                f"流量不足导致反应器换热不充分，热量积累引起温度连锁上升。\n\n"
                f"**常见原因**: (1)阀芯结垢或异物卡滞 (2)气动/电动执行器故障 "
                f"(3)定位器反馈信号漂移 (4)阀座密封面磨损"
            ),
            "CW_Flow": (
                f"**根因分析**: 冷却水流量({variable})异常，评分 {top.score:.3f}。\n\n"
                f"**物理机制**: 冷却水是反应器的主要散热介质。"
                f"根据热平衡方程 Q_removed = ṁ_cw × Cp × (T_out - T_in)，"
                f"流量ṁ_cw↓ → 换热量Q_removed↓ → 反应器温度T_r↑。"
                f"同时，流量下降会导致换热器对数平均温差(LMTD)变化，"
                f"进一步降低换热效率。\n\n"
                f"**常见原因**: 上游阀门故障、泵性能下降、管路堵塞或泄漏、过滤器压差过大"
            ),
            "CW_Inlet_Temp": (
                f"**根因分析**: 冷却水入口温度({variable})升高，评分 {top.score:.3f}。\n\n"
                f"**物理机制**: 换热器的驱动力是对数平均温差(LMTD)。"
                f"入口温度↑ → 冷热流体温差↓ → LMTD↓ → 换热量↓ → 反应器温度↑。"
                f"根据传热方程 Q = U × A × LMTD，LMTD每下降1°C，换热量约减少2-5%。\n\n"
                f"**常见原因**: 冷却塔风机故障、填料老化、布水不均、环境湿球温度过高、循环水浓缩倍数超标"
            ),
            "Feed_Flow": (
                f"**根因分析**: 进料流量({variable})异常，评分 {top.score:.3f}。\n\n"
                f"**物理机制**: 进料流量变化直接影响反应器内的物料停留时间和空间速率。"
                f"停留时间 τ = V/Q，流量Q↑ → τ↓ → 反应不充分 → 转化率降低。"
                f"同时流量增大导致反应器内物料累积，压力升高（PV=nRT）。\n\n"
                f"**常见原因**: 进料泵转速漂移、调节阀故障、上游压力波动、流量计零点漂移"
            ),
            "Feed_Conc": (
                f"**根因分析**: 进料浓度({variable})偏离正常范围，评分 {top.score:.3f}。\n\n"
                f"**物理机制**: 对于一级反应 r = k × C，反应物浓度C是决定反应速率的直接因素。"
                f"浓度每下降10%，在相同温度下反应速率同比例下降10%。"
                f"对于级数>1的反应，浓度下降的影响会被放大。"
                f"产物浓度由反应速率与停留时间共同决定。\n\n"
                f"**常见原因**: 上游原料批次波动、配比误差、溶剂/稀释剂过量、原料储罐分层"
            ),
            "Reaction_Rate": (
                f"**根因分析**: 反应速率({variable})变化，但不一定是根本原因，"
                f"通常由上游变量(温度/浓度/压力)驱动。评分 {top.score:.3f}。\n\n"
                f"**物理机制**: 根据Arrhenius方程 k = A × exp(-Ea/RT)，"
                f"反应速率常数k随温度T指数变化。温度每升高10°C，速率约翻倍。"
                f"浓度和压力也通过质量作用定律影响速率。\n\n"
                f"**建议**: 优先处置上游根因 {top.path[0] if len(top.path) > 1 else variable}，"
                f"而非直接干预反应速率。"
            ),
        }

        # 找到匹配的根因解释
        explanation_added = False
        for key, explanation in root_explanations.items():
            if key in variable:
                actions.append(explanation)
                explanation_added = True
                break
        if not explanation_added:
            actions.append(f"**根因分析**: {variable}是导致{anomaly.variable}异常的最可能根因"
                          f"（评分 {top.score:.3f}），因果路径: {' → '.join(top.path)}。")

        # 因果关系链
        if len(top.path) > 1:
            chain = " → ".join(top.path)
            actions.append(f"\n**因果链**: `{chain}`\n"
                          f"该链路上每一步的因果关系均来自{len(top.path)-1}条因果边，"
                          f"整体置信度 {top.confidence:.3f}。")

        # 分步处置建议（按优先级排）
        step_templates = {
            "CW_Valve": [
                ("**🔴 紧急处置** (5分钟内)", [
                    "1. 在DCS上尝试远程增大阀门开度指令，观察CW_Flow是否响应",
                    "2. 若流量无变化，确认阀门定位器反馈信号是否正常",
                    "3. 若反应器温度持续上升(>185°C)，立即启动紧急停车程序",
                ]),
                ("**🟡 短期修复** (2小时内)", [
                    "4. 现场检查阀门执行机构(气动/电动)是否有异响、过热",
                    "5. 检查阀门定位器输入/输出信号，使用475手操器校准",
                    "6. 若阀芯卡滞，尝试手动盘动阀门手轮，确认机械灵活性",
                ]),
                ("**🟢 长期根治** (下次检修窗口)", [
                    "7. 拆卸阀体检修: 清理阀芯结垢、更换密封填料",
                    "8. 检查阀门选型是否满足工艺要求(Cv值是否匹配)",
                    "9. 建议加装阀门在线诊断系统，实现预测性维护",
                ]),
            ],
            "CW_Flow": [
                ("**🔴 紧急处置**", [
                    "1. 确认上游阀门(CW_Valve)状态是否正常",
                    "2. 若阀门正常，检查冷却水泵电流和出口压力",
                    "3. 切换至备用冷却水泵，观察流量是否恢复",
                ]),
                ("**🟡 短期修复**", [
                    "4. 检查冷却水管路压差: 进出口压差异常增大→管路堵塞; 压差过小→泵出力不足",
                    "5. 清洗Y型过滤器，检查是否有异物堵塞",
                    "6. 排放管路高点排气，排除气缚可能",
                ]),
                ("**🟢 长期根治**", [
                    "7. 建立冷却水流量趋势监测，设定流量下降5%预警",
                    "8. 定期检测冷却水水质(硬度/浊度/pH)，控制结垢趋势",
                ]),
            ],
            "CW_Inlet_Temp": [
                ("**🔴 紧急处置**", [
                    "1. 检查冷却塔风机运行电流，确认是否全部投入",
                    "2. 检查循环水补水阀是否正常开启",
                    "3. 若温度持续升高，降低反应负荷至设计值的70%",
                ]),
                ("**🟡 短期修复**", [
                    "4. 测量冷却塔进出水温差(逼近度)，逼近度>5°C→填料老化",
                    "5. 检查冷却塔布水器是否均匀分布，清理堵塞喷嘴",
                    "6. 检测循环水水质: 浓缩倍数是否超标，必要时加大排污",
                ]),
                ("**🟢 长期根治**", [
                    "7. 根据季节建立冷却塔性能基线，偏离10%即触发检修",
                    "8. 评估冷却塔扩容/改造需求，预留夏季高温余量",
                ]),
            ],
            "Feed_Flow": [
                ("**🔴 紧急处置**", [
                    "1. 立即检查进料泵运行电流和出口压力",
                    "2. 切换到备用泵，确认流量恢复正常",
                    "3. 检查进料调节阀实际开度与DCS指令是否一致",
                ]),
                ("**🟡 短期修复**", [
                    "4. 校准进料流量计(使用标准流量标定装置)",
                    "5. 检查上游储罐液位和氮封压力是否正常",
                    "6. 排查管路是否有内漏(关闭出口阀观察压力保持情况)",
                ]),
                ("**🟢 长期根治**", [
                    "7. 建立进料泵性能曲线(P-Q曲线)定期测试制度",
                    "8. 关键进料回路加装备用流量计，实现冗余检测",
                ]),
            ],
            "Feed_Conc": [
                ("**🔴 紧急处置**", [
                    "1. 取样分析当前进料浓度(实验室验证，排除在线分析仪漂移)",
                    "2. 联系前道工序确认原料配比和批次信息",
                    "3. 若浓度确实偏低，适当降低进料流量延长停留时间补偿",
                ]),
                ("**🟡 短期修复**", [
                    "4. 检查原料储罐是否有分层现象(密度梯度)",
                    "5. 校准在线浓度分析仪(使用标准溶液)",
                    "6. 检查配料系统的计量泵精度是否在规定范围内",
                ]),
                ("**🟢 长期根治**", [
                    "7. 建立原料品质追溯系统，浓度偏离立即溯源至供应商/批次",
                ]),
            ],
        }

        for key, steps in step_templates.items():
            if key in variable:
                for title, items in steps:
                    actions.append(f"\n{title}")
                    for item in items:
                        actions.append(item)
                break

        # 反事实预期效果
        if variable in self.normal_ranges and anomaly.variable in self.normal_ranges:
            actions.append(f"\n**📊 预期效果**: "
                          f"将{variable}恢复至正常值后，"
                          f"{anomaly.variable}预期恢复到正常范围。"
                          f"（具体改善量可使用反事实推理模块量化评估）")

        # 紧急停车条件
        danger_conditions = {
            "Reactor_Temp": (1.15, "反应器温度超过正常上限15%，存在热失控风险"),
            "Reactor_Press": (1.20, "反应器压力超过正常上限20%，存在超压破裂风险"),
        }
        if anomaly.variable in danger_conditions:
            threshold_pct, msg = danger_conditions[anomaly.variable]
            hi = anomaly.normal_range[1]
            if anomaly.observed_value > hi * threshold_pct:
                actions.insert(0, f"⚠⚠⚠ **紧急**: {msg}。建议立即启动紧急停车程序！")

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
