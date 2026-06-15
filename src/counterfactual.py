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

    def _abduct_noise(self, observation: Dict[str, float]) -> Dict[str, float]:
        """
        Step 1 — 溯因 (Abduction): 从观测值推断每个变量的外生噪声项 ε_i

        对于线性 SCM: X_i = Σ_{j∈parents(i)} β_{ji} × X_j + ε_i
        噪声项: ε_i = x_i - Σ β_{ji} × x_j

        噪声项代表"观测值中不能被父变量线性解释的部分"——
        包括未建模的外部因素、测量误差、非线性成分等。
        在反事实世界中这些噪声保持不变，因为我们只干预特定变量。
        """
        noise = {}
        for var in self.causal_graph.nodes():
            if var in observation:
                predicted = 0.0
                for parent in self.causal_graph.predecessors(var):
                    coef = self.effect_matrix.get((parent, var), 0.0)
                    if parent in observation:
                        predicted += coef * observation[parent]
                noise[var] = observation[var] - predicted
            else:
                noise[var] = 0.0
        return noise

    def what_if(self, observation: Dict[str, float],
                intervention: Dict[str, float],
                target_variable: str,
                normal_ranges: Dict[str, Tuple[float, float]] = None
                ) -> CounterfactualResult:
        """
        确定性反事实推理: "如果做X, Y会怎样？"

        Pearl SCM 三步法完整实现:
          1. 溯因(Abduction): 从观测值推断噪声项 ε_i = x_i - Σ β_ji × x_j
          2. 行动(Action):   施加干预 do(X_k = x_k')
          3. 预测(Prediction): 保持噪声不变，沿因果图重新计算下游变量
                                X_i' = Σ β_ji × X_j' + ε_i

        Args:
            observation: 当前观测值
            intervention: 干预 {变量: 新值}
            target_variable: 关心的结果变量
            normal_ranges: 变量正常范围（用于计算改善幅度）

        Returns:
            CounterfactualResult
        """
        if not self.effect_matrix:
            raise ValueError("因果效应系数未设置。请先调用 fit_effect_coefficients() "
                           "或 set_manual_coefficients()。")

        # Step 1 — 溯因: 推断噪声项
        noise = self._abduct_noise(observation)

        # 获取因果传播顺序（处理可能的循环图）
        try:
            topo_order = list(nx.topological_sort(self.causal_graph))
        except nx.NetworkXUnfeasible:
            topo_order = self._get_bfs_order(list(intervention.keys()))

        # Step 2 — 行动: 创建反事实世界的变量值，施加干预
        cf_values = {}
        for var in self.causal_graph.nodes():
            cf_values[var] = observation.get(var, 0.0)

        for var, new_val in intervention.items():
            cf_values[var] = new_val

        # Step 3 — 预测: 沿因果图向下游传播，保留噪声项
        # 拓扑序确保父节点在子节点之前被更新
        for var in topo_order:
            if var in intervention:
                continue  # 干预变量已固定，不重算

            parents = list(self.causal_graph.predecessors(var))
            if not parents:
                continue

            # X_i' = Σ β_ji × X_j' + ε_i
            new_val = noise.get(var, 0.0)
            for parent in parents:
                coef = self.effect_matrix.get((parent, var), 0.0)
                new_val += coef * cf_values.get(parent, 0.0)

            cf_values[var] = new_val

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

        使用因果顺序上的序列归因（Sequential Attribution）:
          按因果拓扑序（根因在前）依次恢复每个原因到正常值，
          每步的边际改善就是该原因的净贡献。
          这避免了同一条因果链上父子变量的贡献重复计算。

        例如: CW_Valve → CW_Flow → Reactor_Temp
          先恢复 CW_Valve → 改善最多（含下游传递效应）
          再恢复 CW_Flow → 改善已很小（CW_Valve已正常）

        Args:
            observation: 当前异常观测
            target_variable: 异常结果变量
            candidate_causes: 候选原因变量列表（将按拓扑序重新排列）
            normal_values: 各变量的正常值（默认使用observation中的值）

        Returns:
            {cause: contribution_pct} 各因素的贡献百分比
        """
        if normal_values is None:
            normal_values = observation.copy()

        # 按拓扑序排列candidate_causes（根因在前，下游在后）
        try:
            full_order = list(nx.topological_sort(self.causal_graph))
        except nx.NetworkXUnfeasible:
            full_order = list(self.causal_graph.nodes())

        ordered_causes = [c for c in full_order if c in candidate_causes]
        # 追加不在拓扑序中的变量
        for c in candidate_causes:
            if c not in ordered_causes:
                ordered_causes.append(c)

        # 序列归因: 逐步恢复，每一步的边际改善就是该因素的净贡献
        contributions = {}
        current_state = observation.copy()

        for cause in ordered_causes:
            normal_val = normal_values.get(cause, observation.get(cause, 0))
            # 在当前已部分恢复的状态上，恢复这个原因
            result = self.what_if(current_state, {cause: normal_val}, target_variable)
            contribution = abs(result.improvement)
            contributions[cause] = contribution

            # 更新状态: 将该原因设置为正常值（为下一步归因做准备）
            current_state[cause] = normal_val
            # 同步更新下游变量
            for var in self.causal_graph.nodes():
                if var in ordered_causes and var != cause:
                    continue  # 其他待归因的变量保持原观测值
                if var == cause:
                    continue  # 已手动设置
                parents = list(self.causal_graph.predecessors(var))
                if not parents:
                    continue
                new_val = 0.0
                for parent in parents:
                    coef = self.effect_matrix.get((parent, var), 0.0)
                    new_val += coef * current_state.get(parent, observation.get(parent, 0))
                current_state[var] = new_val

        total = sum(contributions.values())
        if total > 0:
            pcts = {k: v / total for k, v in contributions.items()}
        else:
            pcts = {k: 0.0 for k in contributions}

        return dict(sorted(pcts.items(), key=lambda x: x[1], reverse=True))

    def causal_path_contribution(self, observation: Dict[str, float],
                                 target_variable: str,
                                 root_cause: str,
                                 normal_values: Dict[str, float] = None
                                 ) -> Dict[str, Any]:
        """
        沿因果路径逐段分解贡献

        方法: 计算根因→目标的完整路径上每一步的传递效应。
        对于每条边 u→v，u 的异常通过该边对 v 的贡献 = β_uv × Δ_u
        (Δ_u = u的观测值 - u的正常值)。

        对于路径 CW_Valve → CW_Flow → Reactor_Temp:
          - CW_Valve→CW_Flow: β=0.55, Δ_Valve=-38 → CW_Flow下降 20.9
          - CW_Flow→Reactor_Temp: β=-0.35, Δ_Flow=X → Reactor_Temp变化 Y

        Args:
            observation: 当前观测值
            target_variable: 异常结果变量
            root_cause: 根因变量
            normal_values: 各变量的正常值。如不提供则用 observation 估算。

        Returns:
            {"path": [...], "steps": [...], "total_effect": float, "breakdown": str}
        """
        try:
            path = nx.shortest_path(self.causal_graph, root_cause, target_variable)
        except nx.NetworkXNoPath:
            return {"error": f"从{root_cause}到{target_variable}无因果路径"}

        if normal_values is None:
            normal_values = {}
            for var in path:
                normal_values[var] = observation.get(var, 0.0)

        # 沿路径计算偏差传播
        steps = []
        accumulated_deviation = observation.get(root_cause, 0) - normal_values.get(root_cause, 0)

        for i in range(len(path) - 1):
            u, v = path[i], path[i + 1]
            coef = self.effect_matrix.get((u, v), 0.0)

            # u 的偏差通过这条边传递给 v: Δ_v_from_u = β × Δ_u
            edge_contribution = coef * accumulated_deviation

            steps.append({
                "from": u,
                "to": v,
                "coefficient": round(coef, 6),
                "deviation_of_cause": round(accumulated_deviation, 4),
                "contribution_to_effect": round(edge_contribution, 4),
            })

            # 累积效应传递到下一步: v 的总偏差 = 来自u的贡献 + v自身的偏差
            v_deviation = observation.get(v, 0) - normal_values.get(v, 0)
            # u 的贡献经 v 传递到下游: 如果 v 还有其他父节点，做近似衰减
            accumulated_deviation = edge_contribution

        # 总效应: 目标变量的观测值与正常值之差
        total_deviation = observation.get(target_variable, 0) - normal_values.get(target_variable, 0)

        # 路径解释
        breakdown_parts = []
        for s in steps:
            sign = "↑" if s["contribution_to_effect"] > 0 else "↓"
            breakdown_parts.append(
                f"{s['from']}→{s['to']} (β={s['coefficient']:.3f}): "
                f"{s['contribution_to_effect']:+.3f}{sign}"
            )

        return {
            "path": path,
            "steps": steps,
            "total_effect_on_target": round(total_deviation, 4),
            "breakdown": " | ".join(breakdown_parts),
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
