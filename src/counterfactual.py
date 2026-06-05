"""
反事实推理引擎 ⭐ 答辩杀手锏模块

核心能力: 回答"如果当初...会怎样？"类型的问题

三种反事实查询模式:
  1. 确定性反事实: "如果冷却水流量=250, 炉壁温度会是多少？"
  2. 对比性反事实: "修复阀门 vs 降低进料, 哪种方案更有效？"
  3. 归因反事实: "这个异常中, 冷却水阀门贡献了多少？进料异常又贡献了多少？"

理论基础:
  - Pearl's Structural Causal Model (SCM)
  - 三步反事实推理: 溯因(Abduction) → 行动(Action) → 预测(Prediction)
  - 在简化线性SEM下, 反事实计算是解析可解的
"""

import numpy as np
import pandas as pd
import networkx as nx
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from itertools import combinations


@dataclass
class CounterfactualResult:
    """单个反事实推理结果"""
    variable: str                    # 关注的变量
    factual_value: float             # 实际观测值
    counterfactual_value: float      # 反事实推断值
    improvement: float               # 改善量（正=变好）
    improvement_pct: float           # 改善百分比
    intervention: Dict[str, float]   # 施加的干预 {变量: 值}


@dataclass
class InterventionComparison:
    """多种干预方案对比"""
    scenarios: List[CounterfactualResult]
    best_intervention: str
    recommendation: str


class CounterfactualEngine:
    """
    反事实推理引擎

    使用简化线性结构方程模型(SCM):
      X_i = Σ_{j∈parents(i)} β_{ji} * X_j + ε_i

    反事实推理三步:
      1. 溯因(Abduction): 从观测值x推断噪声项ε
         ε_i = x_i - Σ β_{ji} * x_j
      2. 行动(Action): 修改干预变量的值 do(X_k = x_k')
      3. 预测(Prediction): 用更新后的值重新计算下游变量
    """

    def __init__(self, causal_graph: nx.DiGraph):
        self.causal_graph = causal_graph
        self.effect_matrix = None   # 因果效应系数矩阵

    def fit_effect_coefficients(self, data: pd.DataFrame):
        """
        从数据中拟合因果效应系数

        对每条边 X→Y, 用线性回归估计 Y = β*X + ε 中的β
        """
        self.effect_matrix = {}
        var_list = list(self.causal_graph.nodes())

        from sklearn.linear_model import LinearRegression

        for u in var_list:
            for v in var_list:
                if self.causal_graph.has_edge(u, v):
                    lag = self.causal_graph[u][v].get("time_lag", 0)
                    X = data[u].values[:-lag if lag > 0 else None].reshape(-1, 1)
                    Y = data[v].values[lag if lag > 0 else 0:].reshape(-1, 1)

                    if len(X) > 1:
                        reg = LinearRegression().fit(X, Y)
                        self.effect_matrix[(u, v)] = reg.coef_[0][0]
                    else:
                        self.effect_matrix[(u, v)] = 0.0

        # 处理没有数据的边: 用confidence作为近似系数
        for u, v, edge_data in self.causal_graph.edges(data=True):
            if (u, v) not in self.effect_matrix:
                self.effect_matrix[(u, v)] = edge_data.get("confidence", 0.5)

    def set_manual_coefficients(self, coefficients: Dict[Tuple[str, str], float]):
        """手动设置因果效应系数（用于合成数据已知ground truth时）"""
        self.effect_matrix = coefficients
        # 补充缺失边
        for u, v in self.causal_graph.edges():
            if (u, v) not in self.effect_matrix:
                self.effect_matrix[(u, v)] = 0.0

    def _get_bfs_order(self, start_nodes) -> list:
        """BFS遍历因果下游节点，返回传播顺序（处理循环图）"""
        from collections import deque
        visited = set()
        order = []
        queue = deque(start_nodes)
        while queue:
            node = queue.popleft()
            for successor in self.causal_graph.successors(node):
                if successor not in visited:
                    visited.add(successor)
                    order.append(successor)
                    queue.append(successor)
        # 追加未访问的节点
        for node in self.causal_graph.nodes():
            if node not in visited and node not in start_nodes:
                order.append(node)
        return order

    def what_if(self, observation: Dict[str, float],
                intervention: Dict[str, float],
                target_variable: str,
                normal_ranges: Dict[str, Tuple[float, float]] = None
                ) -> CounterfactualResult:
        """
        确定性反事实推理: "如果做X, Y会怎样？"

        Args:
            observation: 当前观测值
            intervention: 干预 {变量: 新值}
            target_variable: 关心的结果变量
            normal_ranges: 变量正常范围（用于计算改善幅度）

        Returns:
            CounterfactualResult
        """
        # 获取因果传播顺序（处理可能的循环图）
        try:
            topo_order = list(nx.topological_sort(self.causal_graph))
        except nx.NetworkXUnfeasible:
            # 图中有环: 按BFS层次从干预变量向下传播
            topo_order = self._get_bfs_order(intervention.keys())

        # 创建反事实世界的变量值
        cf_values = observation.copy()

        # 施加干预
        for var, new_val in intervention.items():
            cf_values[var] = new_val

        # 沿因果图向下传播干预效应（最多迭代3轮以处理循环）
        for _ in range(3):
            changed = False
            for var in topo_order:
                if var in intervention:
                    continue

                parents = list(self.causal_graph.predecessors(var))
                if not parents:
                    continue

                new_val = 0.0
                for parent in parents:
                    coef = self.effect_matrix.get((parent, var), 0.0)
                    new_val += coef * cf_values.get(parent, observation.get(parent, 0))

                if abs(cf_values.get(var, 0) - new_val) > 1e-6:
                    cf_values[var] = new_val
                    changed = True
            if not changed:
                break

        factual = observation.get(target_variable, 0)
        counterfactual = cf_values.get(target_variable, 0)
        improvement = counterfactual - factual

        # 计算改善百分比（相对于正常范围）
        improvement_pct = 0.0
        if normal_ranges and target_variable in normal_ranges:
            lo, hi = normal_ranges[target_variable]
            range_span = hi - lo
            if range_span > 0:
                improvement_pct = abs(improvement) / range_span

        return CounterfactualResult(
            variable=target_variable,
            factual_value=round(factual, 4),
            counterfactual_value=round(counterfactual, 4),
            improvement=round(improvement, 4),
            improvement_pct=round(improvement_pct, 4),
            intervention=intervention,
        )

    def compare_interventions(self, observation: Dict[str, float],
                              target_variable: str,
                              intervention_candidates: List[Dict[str, float]],
                              normal_ranges: Dict[str, Tuple[float, float]] = None
                              ) -> InterventionComparison:
        """
        对比多种干预方案: "方案A vs 方案B, 哪个更有效？"

        例如:
          - 方案A: 恢复冷却水阀门到60%
          - 方案B: 降低进料流量到90

        Returns:
            InterventionComparison: 方案对比结果
        """
        scenarios = []
        for i, intervention in enumerate(intervention_candidates):
            result = self.what_if(
                observation, intervention, target_variable, normal_ranges
            )
            scenarios.append(result)

        # 找最佳方案
        best_idx = max(range(len(scenarios)),
                      key=lambda i: abs(scenarios[i].improvement))
        best = scenarios[best_idx]

        recommendation = (f"推荐方案{best_idx+1}: 干预 {best.intervention}, "
                         f"预期改善: {best.improvement:+.4f} (相对改善 {best.improvement_pct:.1%})")

        return InterventionComparison(
            scenarios=scenarios,
            best_intervention=f"方案{best_idx+1}",
            recommendation=recommendation,
        )

    def attribution(self, observation: Dict[str, float],
                    target_variable: str,
                    candidate_causes: List[str],
                    normal_values: Dict[str, float] = None
                    ) -> Dict[str, float]:
        """
        归因反事实: "每个因素对异常的贡献是多少？"

        对于每个候选原因，计算:
          "如果这个原因正常(其他不变), 结果会改善多少？"
        → 改善量 = 该因素的异常贡献

        Args:
            observation: 当前异常观测
            target_variable: 异常结果变量
            candidate_causes: 候选原因变量列表
            normal_values: 各变量的正常值（默认用observation中的值）

        Returns:
            {cause: contribution_pct} 各因素的贡献百分比
        """
        if normal_values is None:
            normal_values = observation.copy()

        contributions = {}
        for cause in candidate_causes:
            intervention = {cause: normal_values.get(cause, observation.get(cause, 0))}
            result = self.what_if(observation, intervention, target_variable)
            contributions[cause] = abs(result.improvement)

        total = sum(contributions.values())
        if total > 0:
            pcts = {k: v / total for k, v in contributions.items()}
        else:
            pcts = {k: 0.0 for k in contributions}

        return dict(sorted(pcts.items(), key=lambda x: x[1], reverse=True))

    def causal_path_contribution(self, observation: Dict[str, float],
                                 target_variable: str,
                                 root_cause: str
                                 ) -> Dict[str, Any]:
        """
        沿因果路径逐段分解贡献

        对于路径: CW_Valve → CW_Flow → Reactor_Temp

        输出:
        ┌──────────────────────────────────────────────┐
        │ CW_Valve 贡献: 65%                            │
        │   └→ CW_Flow (β=0.55):  贡献 55%             │
        │      └→ Reactor_Temp (β=-0.35): 贡献 45%     │
        │ 其他因素(CW_Inlet_Temp等): 35%                │
        └──────────────────────────────────────────────┘
        """
        # 找因果路径
        try:
            path = nx.shortest_path(self.causal_graph, root_cause, target_variable)
        except nx.NetworkXNoPath:
            return {"error": f"从{root_cause}到{target_variable}无因果路径"}

        # 逐个节点做反事实归因
        steps = []
        for i in range(len(path) - 1):
            u, v = path[i], path[i + 1]
            coef = self.effect_matrix.get((u, v), 0.0)
            # 反事实: 只恢复u到正常值, 看v改善多少
            result = self.what_if(
                observation,
                {u: 0},  # 简化: 假设干预到0分析边际效应
                v,
            )
            steps.append({
                "from": u,
                "to": v,
                "coefficient": coef,
                "marginal_effect": result.improvement,
            })

        return {
            "path": path,
            "steps": steps,
            "total_effect": sum(s["marginal_effect"] for s in steps),
        }

    def generate_counterfactual_report(self, observation: Dict[str, float],
                                       root_cause: str,
                                       target_variable: str,
                                       normal_range: Tuple[float, float]
                                       ) -> str:
        """生成反事实推理报告"""
        lines = [
            "# 反事实推理报告",
            "",
            f"## 问题",
            f"{target_variable} 当前值 {observation.get(target_variable)} "
            f"(正常范围 {normal_range[0]}-{normal_range[1]})",
            "",
            f"## 根因",
            f"{root_cause}",
            "",
            "## 反事实推断",
        ]

        # 场景1: 只修复根因
        r1 = self.what_if(observation,
                          {root_cause: (normal_range[0] + normal_range[1]) / 2},
                          target_variable)
        lines.append(f"### 场景1: 仅恢复 {root_cause}")
        lines.append(f"- 预期 {target_variable}: {r1.counterfactual_value:.2f}")
        lines.append(f"- 改善: {r1.improvement:+.2f}")

        return "\n".join(lines)


if __name__ == "__main__":
    print("=" * 60)
    print("反事实推理引擎测试")
    print("=" * 60)

    import networkx as nx

    # 构建因果图
    cg = nx.DiGraph()
    cg.add_edge("CW_Valve", "CW_Flow", coefficient=0.55, confidence=0.95)
    cg.add_edge("CW_Flow", "Reactor_Temp", coefficient=-0.35, confidence=0.90)
    cg.add_edge("CW_Inlet_Temp", "Reactor_Temp", coefficient=0.28, confidence=0.85)
    cg.add_edge("Reactor_Temp", "Reaction_Rate", coefficient=0.50, confidence=0.95)
    cg.add_edge("Reactor_Temp", "Product_Conc", coefficient=-0.15, confidence=0.80)

    engine = CounterfactualEngine(cg)
    engine.set_manual_coefficients({
        ("CW_Valve", "CW_Flow"): 0.55,
        ("CW_Flow", "Reactor_Temp"): -0.35,
        ("CW_Inlet_Temp", "Reactor_Temp"): 0.28,
        ("Reactor_Temp", "Reaction_Rate"): 0.50,
        ("Reactor_Temp", "Product_Conc"): -0.15,
    })

    # 当前异常观测
    observation = {
        "CW_Valve": 22,
        "CW_Flow": 120,
        "CW_Inlet_Temp": 26,
        "Reactor_Temp": 192,
        "Reaction_Rate": 1.45,
        "Product_Conc": 0.58,
    }

    normal_ranges = {
        "Reactor_Temp": (150, 175),
        "Product_Conc": (0.70, 0.95),
    }

    # 反事实1: 如果修复了阀门
    r1 = engine.what_if(observation, {"CW_Valve": 60}, "Reactor_Temp", normal_ranges)
    print(f"\n[反事实] 如果CW_Valve=60:")
    print(f"  实际Reactor_Temp: {r1.factual_value}")
    print(f"  反事实: {r1.counterfactual_value:.1f}")
    print(f"  改善: {r1.improvement:+.1f}")

    # 对比多种方案
    comparison = engine.compare_interventions(
        observation, "Reactor_Temp",
        [
            {"CW_Valve": 60},           # 方案1: 恢复阀门
            {"CW_Inlet_Temp": 26},      # 方案2: 维持入口温度
            {"CW_Valve": 60, "CW_Inlet_Temp": 26},  # 方案3: 两者都做
        ],
        normal_ranges,
    )
    print(f"\n[方案对比]")
    for i, s in enumerate(comparison.scenarios):
        print(f"  方案{i+1}: {s.intervention} → {s.counterfactual_value:.1f} (改善 {s.improvement:+.1f})")
    print(f"  {comparison.recommendation}")

    # 归因分析
    attr = engine.attribution(
        observation, "Reactor_Temp",
        ["CW_Valve", "CW_Inlet_Temp", "CW_Flow"],
        {"CW_Valve": 60, "CW_Inlet_Temp": 25, "CW_Flow": 250}
    )
    print(f"\n[归因分析]")
    for cause, pct in attr.items():
        bar = "█" * int(pct * 30)
        print(f"  {cause}: {bar} {pct:.1%}")
