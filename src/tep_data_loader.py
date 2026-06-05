"""
真实TEP数据加载器
将 Tennessee Eastman Process 数据集加载为与合成数据相同的接口格式，
无缝集成到现有因果推理Pipeline。

数据集来源: MIT Braatz Group (http://web.mit.edu/braatzgroup/TE_process.zip)
格式: 52个过程变量 (41 XMEAS + 11 XMV), 21种故障模式 + 1种正常

TEP故障模式对照表 (Downs & Vogel, 1993):
  IDV(0)  - 正常运行
  IDV(1)  - A/C进料比变化 (B成分恒定) → 进料组成阶跃
  IDV(2)  - B成分变化 (A/C比恒定) → 进料组成阶跃
  IDV(3)  - D进料温度变化 → 阶跃
  IDV(4)  - 反应器冷却水入口温度变化 → 阶跃
  IDV(5)  - 冷凝器冷却水入口温度变化 → 阶跃
  IDV(6)  - A进料损失 → 阶跃
  IDV(7)  - C集管压力损失 → 阶跃
  IDV(8)  - A/B/C进料成分变化 → 随机波动
  IDV(9)  - D进料温度变化 → 随机波动
  IDV(10) - C进料温度变化 → 随机波动
  IDV(11) - 反应器冷却水入口温度变化 → 随机波动
  IDV(12) - 冷凝器冷却水入口温度变化 → 随机波动
  IDV(13) - 反应动力学漂移 → 慢漂移
  IDV(14) - 反应器冷却水阀门卡滞 → 粘滞
  IDV(15) - 冷凝器冷却水阀门卡滞 → 粘滞
  IDV(16) - 未知故障
  IDV(17) - 未知故障
  IDV(18) - 未知故障
  IDV(19) - 未知故障
  IDV(20) - 未知故障
  IDV(21) - 阀门位置恒定 → 阀门卡死

变量名:
  XMEAS(1)  - A进料流量 (流11)
  XMEAS(2)  - D进料流量 (流12)
  XMEAS(3)  - E进料流量 (流13)
  XMEAS(4)  - A+C总进料流量 (流14)
  XMEAS(5)  - 循环流量 (流15)
  XMEAS(6)  - 反应器进料流量 (流16)
  XMEAS(7)  - 反应器压力
  XMEAS(8)  - 反应器液位
  XMEAS(9)  - 反应器温度
  XMEAS(10) - 排放流量 (流19)
  XMEAS(11) - 分离器温度
  XMEAS(12) - 分离器液位
  XMEAS(13) - 分离器压力
  XMEAS(14) - 分离器底部流量 (流10)
  XMEAS(15) - 汽提塔液位
  XMEAS(16) - 汽提塔压力
  XMEAS(17) - 汽提塔底部流量 (流11)
  XMEAS(18) - 汽提塔温度
  XMEAS(19) - 汽提塔蒸汽流量
  XMEAS(20) - 压缩机功率
  XMEAS(21) - 反应器冷却水出口温度
  XMEAS(22) - 分离器冷却水出口温度
  XMEAS(23-36) - 成分分析器A-F (流16/流19)
  XMEAS(37-41) - 成分分析器G-H (流10/流11)
  XMV(1)   - D进料流量阀
  XMV(2)   - E进料流量阀
  XMV(3)   - A进料流量阀
  XMV(4)   - A+C总进料流量阀
  XMV(5)   - 压缩机循环阀
  XMV(6)   - 排放阀
  XMV(7)   - 分离器罐液流量阀
  XMV(8)   - 汽提塔液产品流量阀
  XMV(9)   - 汽提塔蒸汽阀
  XMV(10)  - 反应器冷却水流量
  XMV(11)  - 冷凝器冷却水流量
  XMV(12)  - 搅拌速度
"""

import numpy as np
import pandas as pd
import os
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field


# TEP 变量名
TEP_VAR_NAMES = [
    # XMEAS(1-22): 连续过程测量
    "A_Feed_Flow",        # XMEAS(1)
    "D_Feed_Flow",        # XMEAS(2)
    "E_Feed_Flow",        # XMEAS(3)
    "AC_Total_Flow",      # XMEAS(4)
    "Recycle_Flow",       # XMEAS(5)
    "Reactor_Feed",       # XMEAS(6)
    "Reactor_Press",      # XMEAS(7)
    "Reactor_Level",      # XMEAS(8)
    "Reactor_Temp",       # XMEAS(9)
    "Purge_Flow",         # XMEAS(10)
    "Separator_Temp",     # XMEAS(11)
    "Separator_Level",    # XMEAS(12)
    "Separator_Press",    # XMEAS(13)
    "Sep_Underflow",      # XMEAS(14)
    "Stripper_Level",     # XMEAS(15)
    "Stripper_Press",     # XMEAS(16)
    "Stripper_Flow",      # XMEAS(17)
    "Stripper_Temp",      # XMEAS(18)
    "Stripper_Steam",     # XMEAS(19)
    "Compressor_Work",    # XMEAS(20)
    "Reactor_CW_Temp",    # XMEAS(21)
    "Separator_CW_Temp",  # XMEAS(22)
    # XMEAS(23-36): 成分分析器A-F
    "Comp_A_Feed",        # XMEAS(23)
    "Comp_B_Feed",        # XMEAS(24)
    "Comp_C_Feed",        # XMEAS(25)
    "Comp_D_Feed",        # XMEAS(26)
    "Comp_E_Feed",        # XMEAS(27)
    "Comp_F_Feed",        # XMEAS(28)
    "Comp_A_Purge",       # XMEAS(29)
    "Comp_B_Purge",       # XMEAS(30)
    "Comp_C_Purge",       # XMEAS(31)
    "Comp_D_Purge",       # XMEAS(32)
    "Comp_E_Purge",       # XMEAS(33)
    "Comp_F_Purge",       # XMEAS(34)
    "Comp_G_Feed",        # XMEAS(35)
    "Comp_H_Feed",        # XMEAS(36)
    # XMEAS(37-41): 成分分析器G-H
    "Comp_D_Product",     # XMEAS(37)
    "Comp_E_Product",     # XMEAS(38)
    "Comp_F_Product",     # XMEAS(39)
    "Comp_G_Product",     # XMEAS(40)
    "Comp_H_Product",     # XMEAS(41)
    # XMV(1-12): 操纵变量
    "Valve_D_Feed",       # XMV(1)
    "Valve_E_Feed",       # XMV(2)
    "Valve_A_Feed",       # XMV(3)
    "Valve_AC_Total",     # XMV(4)
    "Valve_Recycle",      # XMV(5)
    "Valve_Purge",        # XMV(6)
    "Valve_Sep_Flow",     # XMV(7)
    "Valve_Strip_Flow",   # XMV(8)
    "Valve_Strip_Steam",  # XMV(9)
    "Valve_Reactor_CW",   # XMV(10) ⭐ 对应故障14: 反应器冷却水阀门
    "Valve_Cond_CW",      # XMV(11) ⭐ 对应故障15: 冷凝器冷却水阀门
]

# 手动标注的关键变量（基于TEP的物理结构，用于因果图构建）
TEP_KEY_VARIABLES = {
    "root_vars": [
        "A_Feed_Flow", "D_Feed_Flow", "E_Feed_Flow",   # 进料
        "Valve_Reactor_CW", "Valve_Cond_CW",            # 冷却水阀门
        "Valve_A_Feed", "Valve_D_Feed", "Valve_E_Feed", # 进料阀门
    ],
    "process_vars": [
        "Reactor_Press", "Reactor_Temp", "Reactor_Level",
        "Separator_Temp", "Separator_Level", "Separator_Press",
        "Stripper_Temp", "Stripper_Level", "Stripper_Press",
        "Compressor_Work", "Reactor_CW_Temp", "Separator_CW_Temp",
    ],
    "quality_vars": [
        "Comp_A_Feed", "Comp_B_Feed", "Comp_D_Product", "Comp_E_Product",
        "Purge_Flow", "Stripper_Flow",
    ],
}

# TEP故障 → 根因映射 (用于验证，基于文献 Downs & Vogel 1993)
TEP_FAULT_ROOT_CAUSE = {
    "IDV(0)":  None,
    "IDV(1)":  "A/C进料比 — 进料组成阶跃",
    "IDV(2)":  "B成分 — 进料组成阶跃",
    "IDV(3)":  "D进料温度 — 阶跃",
    "IDV(4)":  "反应器冷却水入口温度 — 阶跃",
    "IDV(5)":  "冷凝器冷却水入口温度 — 阶跃",
    "IDV(6)":  "A进料损失 — 阶跃",
    "IDV(7)":  "C集管压力损失 — 阶跃",
    "IDV(8)":  "A/B/C进料成分 — 随机波动",
    "IDV(9)":  "D进料温度 — 随机波动",
    "IDV(10)": "C进料温度 — 随机波动",
    "IDV(11)": "反应器冷却水入口温度 — 随机波动",
    "IDV(12)": "冷凝器冷却水入口温度 — 随机波动",
    "IDV(13)": "反应动力学 — 慢漂移",
    "IDV(14)": "反应器冷却水阀门卡滞 ⭐",
    "IDV(15)": "冷凝器冷却水阀门卡滞 ⭐",
    "IDV(16)": "故障16 — 未知",
    "IDV(17)": "故障17 — 未知",
    "IDV(18)": "故障18 — 未知",
    "IDV(19)": "故障19 — 未知",
    "IDV(20)": "故障20 — 未知",
    "IDV(21)": "阀门位置恒定 — 阀门卡死",
}

# 精选用于演示的6种因果结构清晰的故障
DEMO_FAULTS = {
    "IDV(1)": {
        "name": "A/C进料比变化",
        "description": "A/C进料比例发生阶跃变化（B成分保持恒定），导致反应器内组成偏离设计值",
        "expected_root_cause_vars": ["A_Feed_Flow", "Comp_A_Feed", "Comp_C_Feed"],
        "causal_path": "进料组成变化 → 反应器组成变化 → 反应速率变化 → 产物浓度变化",
        "tep_causal_vars": ["A_Feed_Flow", "Reactor_Press", "Reactor_Temp", "Comp_A_Feed", "Comp_D_Product"],
    },
    "IDV(4)": {
        "name": "反应器冷却水入口温度阶跃",
        "description": "反应器冷却水入口温度突然升高，影响反应温度控制",
        "expected_root_cause_vars": ["Reactor_CW_Temp", "Valve_Reactor_CW"],
        "causal_path": "冷却水入口温度↑ → 换热效率↓ → 反应器温度↑ → 反应速率↑ → 压力↑",
        "tep_causal_vars": ["Reactor_CW_Temp", "Reactor_Temp", "Reactor_Press", "Valve_Reactor_CW"],
    },
    "IDV(6)": {
        "name": "A进料损失",
        "description": "A组分进料流量突然减少（阶跃损失），是最常见的化工过程异常之一",
        "expected_root_cause_vars": ["A_Feed_Flow"],
        "causal_path": "A进料↓ → 进料组成↓ → 反应速率↓ → 产物质量↓",
        "tep_causal_vars": ["A_Feed_Flow", "Reactor_Press", "Reactor_Level", "Comp_A_Feed", "Comp_D_Product"],
    },
    "IDV(7)": {
        "name": "C集管压力损失",
        "description": "C组分进料系统压力损失，导致C进料异常",
        "expected_root_cause_vars": ["Comp_C_Feed"],
        "causal_path": "C组分进料↓ → 反应物比例失衡 → 产物浓度变化",
        "tep_causal_vars": ["Comp_C_Feed", "Reactor_Press", "Reactor_Level", "Comp_D_Product"],
    },
    "IDV(13)": {
        "name": "反应动力学漂移",
        "description": "催化剂活性缓慢下降或反应动力学参数漂移，典型的缓慢故障",
        "expected_root_cause_vars": ["Reactor_Temp"],
        "causal_path": "催化剂活性↓ / 动力学参数变化 → 转化率↓ → 产物质量↓",
        "tep_causal_vars": ["Reactor_Temp", "Reactor_Press", "Comp_D_Product", "Separator_Temp"],
    },
    "IDV(14)": {
        "name": "反应器冷却水阀门卡滞 ⭐",
        "description": "反应器冷却水阀门出现机械卡滞，无法正常调节开度。此故障与合成数据的FAULT_COOLING_VALVE_STUCK直接对应",
        "expected_root_cause_vars": ["Valve_Reactor_CW"],
        "causal_path": "阀门卡滞 → 冷却水流量异常 → 反应温度失控 → 连锁反应",
        "tep_causal_vars": ["Valve_Reactor_CW", "Reactor_CW_Temp", "Reactor_Temp", "Reactor_Press", "Comp_D_Product"],
    },
}


class TEPDataLoader:
    """TEP真实数据集加载器"""

    def __init__(self, data_dir: str = "data/TEP/TE_process"):
        self.data_dir = data_dir
        self.var_names = TEP_VAR_NAMES
        self.n_vars = len(TEP_VAR_NAMES)
        self._cache = {}

    def load_file(self, fault_id: int) -> Tuple[np.ndarray, np.ndarray]:
        """加载指定故障ID的训练和测试数据"""
        if fault_id in self._cache:
            return self._cache[fault_id]

        train_file = os.path.join(self.data_dir, f"d{fault_id:02d}.dat")
        test_file = os.path.join(self.data_dir, f"d{fault_id:02d}_te.dat")

        train_raw = np.loadtxt(train_file)
        test_raw = np.loadtxt(test_file)

        # 统一格式: (samples, variables)
        if train_raw.shape[0] < train_raw.shape[1]:
            train_raw = train_raw.T
        if test_raw.shape[0] < test_raw.shape[1]:
            test_raw = test_raw.T

        self._cache[fault_id] = (train_raw, test_raw)
        return train_raw, test_raw

    def load_normal_data(self) -> pd.DataFrame:
        """加载正常运行数据"""
        train, test = self.load_file(0)
        data = np.vstack([train, test])
        df = pd.DataFrame(data, columns=self.var_names)
        df["fault"] = "NORMAL"
        df["fault_id"] = 0
        return df

    def load_fault_data(self, fault_id: int) -> Tuple[pd.DataFrame, Dict]:
        """加载单个故障数据，返回(DataFrame, metadata)"""
        train, test = self.load_file(fault_id)
        # 故障在测试集中引入（通常在160步后）
        n_train = len(train)
        n_test = len(test)

        # 正常部分
        df_normal = pd.DataFrame(train, columns=self.var_names)
        df_normal["label"] = 0
        df_normal["fault_id"] = fault_id
        df_normal["fault_phase"] = "train"

        # 故障部分
        df_fault = pd.DataFrame(test, columns=self.var_names)
        df_fault["label"] = 1
        df_fault["fault_id"] = fault_id
        df_fault["fault_phase"] = "test"

        df = pd.concat([df_normal, df_fault], ignore_index=True)
        df["time"] = range(len(df))

        fault_name = f"IDV({fault_id})"
        root_cause = TEP_FAULT_ROOT_CAUSE.get(fault_name, f"故障{fault_id}")

        metadata = {
            "fault_name": fault_name,
            "description": root_cause,
            "expected_root_cause": root_cause,
            "n_train": n_train,
            "n_test": n_test,
            "n_variables": self.n_vars,
            "var_names": self.var_names,
        }

        return df, metadata

    def load_all_faults(self) -> Dict[str, pd.DataFrame]:
        """加载所有21种故障"""
        result = {}
        for i in range(1, 22):
            df, meta = self.load_fault_data(i)
            result[f"IDV({i})"] = df
        return result

    def get_causal_subset(self, fault_id: int) -> List[str]:
        """获取该故障相关的因果变量子集"""
        fault_name = f"IDV({fault_id})"
        if fault_name in DEMO_FAULTS:
            return DEMO_FAULTS[fault_name]["tep_causal_vars"]
        # 默认返回关键过程变量
        subset = (TEP_KEY_VARIABLES["root_vars"][:3] +
                  TEP_KEY_VARIABLES["process_vars"][:6] +
                  TEP_KEY_VARIABLES["quality_vars"][:4])
        return subset

    def build_tep_causal_graph(self, variable_subset: List[str] = None
                               ) -> "nx.DiGraph":
        """根据TEP物理结构构建初始因果图（知识通道）"""
        import networkx as nx

        G = nx.DiGraph()

        if variable_subset is None:
            variable_subset = self.var_names
        for v in variable_subset:
            G.add_node(v)

        # 基于TEP物理结构的因果边
        tep_causal_edges = [
            # 进料 → 反应器
            ("A_Feed_Flow", "Reactor_Level", 0.15, "进料流量→反应器液位"),
            ("A_Feed_Flow", "Reactor_Press", 0.10, "进料→压力"),
            ("D_Feed_Flow", "Reactor_Temp", 0.12, "D进料温度→反应器温度"),
            # 冷却水 → 温度
            ("Valve_Reactor_CW", "Reactor_Temp", -0.25, "反应器冷却水阀→温度"),
            ("Valve_Cond_CW", "Separator_Temp", -0.22, "冷凝器冷却水阀→分离器温度"),
            ("Reactor_CW_Temp", "Reactor_Temp", 0.20, "冷却水温度→反应器温度"),
            # 反应器 → 下游
            ("Reactor_Temp", "Reactor_Press", 0.12, "温度→压力(气相膨胀)"),
            ("Reactor_Press", "Separator_Press", 0.30, "反应器压力→分离器压力"),
            ("Reactor_Level", "Separator_Level", 0.25, "反应器液位→分离器液位"),
            # 分离器 → 汽提塔
            ("Separator_Level", "Stripper_Level", 0.20, "分离器→汽提塔液位"),
            ("Separator_Press", "Stripper_Press", 0.18, "分离器→汽提塔压力"),
            # 压缩机
            ("Separator_Press", "Compressor_Work", 0.15, "分离器压力→压缩机功率"),
            # 成分
            ("A_Feed_Flow", "Comp_A_Feed", 0.30, "A进料→A成分"),
            ("Comp_A_Feed", "Comp_D_Product", 0.25, "成分→产物质量"),
            ("Reactor_Temp", "Comp_D_Product", -0.15, "高温→副反应→产物质量下降"),
            # 阀门
            ("Valve_Reactor_CW", "Reactor_CW_Temp", -0.30, "阀门→冷却水温度"),
            ("Valve_Purge", "Purge_Flow", 0.50, "阀门→排放流量"),
            # C成分相关
            ("Comp_C_Feed", "Reactor_Press", 0.08, "C进料→反应器压力"),
            ("Comp_C_Feed", "Comp_D_Product", 0.12, "C成分→D产物"),
            ("Reactor_Temp", "Separator_Temp", 0.40, "反应器温度→分离器温度"),
            ("Separator_Temp", "Comp_D_Product", -0.10, "分离温度→产物纯度"),
        ]

        for cause, effect, coef, mechanism in tep_causal_edges:
            if cause in variable_subset and effect in variable_subset:
                G.add_edge(cause, effect, coefficient=coef,
                          mechanism=mechanism, confidence=0.85,
                          source="domain_knowledge")

        return G


if __name__ == "__main__":
    loader = TEPDataLoader()

    # 测试加载
    print("=" * 60)
    print("TEP数据加载器测试")
    print("=" * 60)

    # 正常数据
    df_normal = loader.load_normal_data()
    print(f"\n正常数据: {df_normal.shape}")

    # 故障数据
    df, meta = loader.load_fault_data(14)  # 冷却水阀门卡滞
    print(f"故障IDV(14): {meta['description']}")
    print(f"  训练集: {meta['n_train']}步")
    print(f"  测试集: {meta['n_test']}步")
    print(f"  变量数: {meta['n_variables']}")

    # 因果变量子集
    causal_vars = loader.get_causal_subset(14)
    print(f"因果相关变量: {causal_vars}")

    # 构建因果图
    G = loader.build_tep_causal_graph(causal_vars)
    print(f"\nTEP因果图: {G.number_of_nodes()}节点, {G.number_of_edges()}条边")
    for u, v, d in G.edges(data=True):
        print(f"  {u} → {v} [{d.get('mechanism', '')}]")
