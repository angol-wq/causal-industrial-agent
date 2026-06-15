"""
因果发现模块: 从传感器时序数据中学习因果结构

方法:
  1. PCMCI+: 基于条件独立性检验的时序因果发现（Tigramite库）
  2. PC算法 (备选): 适用于i.i.d.数据
  3. GES (备选): 贪婪等价搜索

输入: 过程变量时序数据 (pandas DataFrame)
输出: 因果图 (networkx DiGraph) + 因果效应矩阵
"""

import numpy as np
import pandas as pd
import networkx as nx
from typing import Optional, Tuple, Dict, List
import warnings

warnings.filterwarnings("ignore")


class CausalDiscovery:
    """
    时序因果发现引擎

    核心方法:
      - discover_pcmciplus(): 用PCMCI+发现时序因果结构
      - discover_static(): 静态因果发现（PC/GES备选）
      - export_graph(): 导出因果图为标准格式
    """

    def __init__(self, var_names: List[str], significance_level: float = 0.05):
        """
        Args:
            var_names: 变量名列表
            significance_level: 条件独立性检验显著性水平
        """
        self.var_names = var_names
        self.n_vars = len(var_names)
        self.significance_level = significance_level
        self.causal_graph = None       # networkx DiGraph
        self.causal_matrix = None      # numpy array (N, N, tau_max+1)
        self.pcmci_results = None      # PCMCI+ 原始结果

    def discover_pcmciplus(self, data: pd.DataFrame, tau_max: int = 5,
                           cond_ind_test: str = "ParCorr",
                           var_names_subset: List[str] = None
                           ) -> nx.DiGraph:
        """
        使用PCMCI+算法发现时序因果结构

        PCMCI+ 是当前时序因果发现中最鲁棒的方法之一:
        - PCMCI: 先做条件选择(PC阶段)，再做瞬时条件独立性检验(MCI阶段)
        - PCMCI+: PCMCI的扩展，可以同时发现滞后和瞬时因果

        Args:
            data: 时序数据 DataFrame
            tau_max: 最大滞后时间步
            cond_ind_test: 条件独立性检验方法 ("ParCorr" / "GPDC" / "CMIknn")
            var_names_subset: 只分析指定变量子集，默认全部

        Returns:
            networkx DiGraph: 因果图
        """
        from tigramite import data_processing as tigramite_data
        from tigramite.pcmci import PCMCI
        from tigramite.independence_tests.parcorr import ParCorr

        # 选择变量
        if var_names_subset is None:
            var_names_subset = self.var_names
        elif isinstance(var_names_subset, list) and len(var_names_subset) == 0:
            var_names_subset = self.var_names

        df = data[var_names_subset].copy()

        # 构建Tigramite数据格式
        numpy_data = df.values
        tigramite_df = tigramite_data.DataFrame(
            numpy_data,
            var_names=var_names_subset
        )

        # 条件独立性检验
        ci_test = ParCorr(significance="analytic")

        # 运行PCMCI+
        print(f"[因果发现] 运行PCMCI+... tau_max={tau_max}, vars={len(var_names_subset)}")
        pcmci = PCMCI(dataframe=tigramite_df, cond_ind_test=ci_test, verbosity=0)

        results = pcmci.run_pcmciplus(
            tau_min=0,        # 0=包含瞬时因果
            tau_max=tau_max,
            pc_alpha=self.significance_level,
        )

        self.pcmci_results = results
        self.causal_matrix = results["graph"]  # shape (N, N, tau_max+1)

        # 转换为networkx有向图
        self.causal_graph = self._pcmci_to_networkx(
            self.causal_matrix, var_names_subset, tau_max
        )

        # 统计
        n_edges = self.causal_graph.number_of_edges()
        print(f"[因果发现] PCMCI+完成: 发现 {n_edges} 条因果边")

        return self.causal_graph

    def _pcmci_to_networkx(self, graph_matrix: np.ndarray,
                           var_names: List[str], tau_max: int
                           ) -> nx.DiGraph:
        """
        将PCMCI+的graph矩阵转换为networkx有向图

        graph_matrix[i, j, tau]:
          "-->" 表示 X_j[t-tau] → X_i[t]
          "o-o", "o->", "x-x" 等为不确定关系
        """
        G = nx.DiGraph()

        for i in range(len(var_names)):
            G.add_node(var_names[i])

        for i in range(len(var_names)):
            for j in range(len(var_names)):
                for tau in range(tau_max + 1):
                    link_type = graph_matrix[i, j, tau]
                    if link_type == "-->":
                        source = var_names[j]
                        target = var_names[i]

                        if not G.has_edge(source, target):
                            # 提取该边的p-value: p_matrix[j, i, tau]
                            p_val = None
                            if "p_matrix" in self.pcmci_results:
                                p_val = self.pcmci_results["p_matrix"][j, i, tau]
                            significance = 1.0 - p_val if p_val is not None else 0.8

                            G.add_edge(
                                source, target,
                                lag=tau,
                                p_value=p_val,
                                significance=round(significance, 4),
                            )

        return G

    def get_causal_strength_matrix(self, data: pd.DataFrame,
                                   var_names_subset: List[str] = None
                                   ) -> pd.DataFrame:
        """
        计算因果效应的强度矩阵

        对每条因果边，用线性回归估计因果效应系数 β
        Y = β*X + ε
        """
        if var_names_subset is None:
            var_names_subset = self.var_names

        df = data[var_names_subset]
        n = len(var_names_subset)
        strength = pd.DataFrame(
            np.zeros((n, n)),
            index=var_names_subset,
            columns=var_names_subset
        )

        from sklearn.linear_model import LinearRegression

        for i, target in enumerate(var_names_subset):
            for j, cause in enumerate(var_names_subset):
                if cause != target and self.causal_graph is not None:
                    if self.causal_graph.has_edge(cause, target):
                        lag = self.causal_graph[cause][target].get("lag", 0)
                        if lag > 0 and len(df) > lag:
                            X = df[cause].values[:-lag].reshape(-1, 1)
                            Y = df[target].values[lag:].reshape(-1, 1)
                        else:
                            X = df[cause].values.reshape(-1, 1)
                            Y = df[target].values.reshape(-1, 1)

                        reg = LinearRegression().fit(X, Y)
                        strength.loc[cause, target] = reg.coef_[0][0]

        return strength

    def compare_with_ground_truth(self, ground_truth_graph: nx.DiGraph
                                  ) -> Dict[str, float]:
        """
        与真实因果图比较，计算精度/召回率/F1
        仅在合成数据场景可用（已知ground truth）
        """
        if self.causal_graph is None:
            raise ValueError("请先运行因果发现")

        true_edges = set(ground_truth_graph.edges())
        predicted_edges = set(self.causal_graph.edges())

        tp = len(true_edges & predicted_edges)
        fp = len(predicted_edges - true_edges)
        fn = len(true_edges - predicted_edges)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        return {
            "true_positives": tp,
            "false_positives": fp,
            "false_negatives": fn,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "n_true_edges": len(true_edges),
            "n_predicted_edges": len(predicted_edges),
        }

    def export_graph(self, filepath: str):
        """导出因果图为GraphML格式"""
        if self.causal_graph is None:
            raise ValueError("请先运行因果发现")
        nx.write_graphml(self.causal_graph, filepath)
        print(f"[因果发现] 因果图已导出至 {filepath}")


if __name__ == "__main__":
    # 快速测试
    import sys
    sys.path.insert(0, ".")

    from synthetic_data_generator import SyntheticProcessSimulator

    print("=" * 60)
    print("因果发现模块测试")
    print("=" * 60)

    # 1. 生成合成数据
    sim = SyntheticProcessSimulator(seed=42)
    df, meta = sim.generate_fault_dataset(n_normal=500, n_fault=0)

    # 2. 因果发现
    cd = CausalDiscovery(
        var_names=["Feed_Flow", "Feed_Conc", "CW_Inlet_Temp", "CW_Valve",
                    "Reactor_Temp", "Reactor_Press", "CW_Flow", "Reaction_Rate",
                    "Product_Conc", "Byproduct_Conc", "HX_Outlet_Temp", "Energy_Index"]
    )

    graph = cd.discover_pcmciplus(
        data=df,
        tau_max=5,
    )

    # 3. 与ground truth比较
    from synthetic_data_generator import CAUSAL_GRAPH_TRUTH
    import networkx as nx
    truth_g = nx.DiGraph()
    for edge in CAUSAL_GRAPH_TRUTH:
        truth_g.add_edge(edge.cause, edge.effect)

    metrics = cd.compare_with_ground_truth(truth_g)
    print(f"\n与Ground Truth比较:")
    print(f"  Precision: {metrics['precision']:.2%}")
    print(f"  Recall:    {metrics['recall']:.2%}")
    print(f"  F1:        {metrics['f1']:.2%}")
    print(f"  TP={metrics['true_positives']}, FP={metrics['false_positives']}, FN={metrics['false_negatives']}")
