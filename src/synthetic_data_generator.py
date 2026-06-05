"""
合成工业过程数据生成器
模拟简化化工反应过程，因果关系完全已知，用于验证因果发现和根因分析算法。

模拟场景：连续搅拌釜式反应器（CSTR）+ 换热系统
变量之间具有已知的物理因果关系，可注入多种故障模式。

变量结构（12个关键变量）:
  层级1（外部条件/根因层）:
    - 进料流量 (Feed_Flow)
    - 进料浓度 (Feed_Conc)
    - 冷却水入口温度 (CW_Inlet_Temp)
    - 冷却水阀门开度 (CW_Valve)

  层级2（中间过程层）:
    - 反应器温度 (Reactor_Temp)
    - 反应器压力 (Reactor_Press)
    - 冷却水流量 (CW_Flow)
    - 反应速率 (Reaction_Rate)

  层级3（输出/结果层）:
    - 产物浓度 (Product_Conc)
    - 副产物浓度 (Byproduct_Conc)
    - 换热器出口温度 (HX_Outlet_Temp)
    - 能耗指标 (Energy_Index)
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
import json


@dataclass
class CausalEdge:
    """因果边定义"""
    cause: str
    effect: str
    coefficient: float  # 线性因果效应系数
    time_lag: int       # 滞后时间步
    mechanism: str      # 物理机制描述
    direction: str      # "positive" or "negative"


# ============================================================
# 真实的因果图定义（Ground Truth）
# ============================================================
CAUSAL_GRAPH_TRUTH: List[CausalEdge] = [
    # 进料 → 反应
    CausalEdge("Feed_Flow", "Reactor_Press", 0.15, 0,
               "进料流速增大 → 反应器内物料累积 → 压力升高", "positive"),
    CausalEdge("Feed_Conc", "Reaction_Rate", 0.42, 1,
               "进料浓度升高 → 反应物浓度梯度增大 → 反应速率加快", "positive"),
    CausalEdge("Feed_Flow", "Product_Conc", -0.08, 3,
               "进料流量过大 → 停留时间缩短 → 反应不充分 → 产物浓度下降", "negative"),

    # 冷却水 → 温度
    CausalEdge("CW_Valve", "CW_Flow", 0.55, 0,
               "阀门开度增大 → 冷却水流量增大", "positive"),
    CausalEdge("CW_Inlet_Temp", "Reactor_Temp", 0.28, 1,
               "冷却水入口温度升高 → 换热温差减小 → 反应器温度升高", "positive"),
    CausalEdge("CW_Flow", "Reactor_Temp", -0.35, 1,
               "冷却水流量增大 → 换热量增大 → 反应器温度降低", "negative"),

    # 温度 → 反应
    CausalEdge("Reactor_Temp", "Reaction_Rate", 0.50, 0,
               "温度升高 → 反应速率常数增大(Arrhenius) → 反应速率加快", "positive"),
    CausalEdge("Reactor_Temp", "Reactor_Press", 0.12, 0,
               "温度升高 → 气相膨胀 → 压力增大", "positive"),
    CausalEdge("Reactor_Temp", "Energy_Index", 0.40, 1,
               "温度升高 → 加热能耗增大", "positive"),

    # 压力 → 反应
    CausalEdge("Reactor_Press", "Reaction_Rate", 0.18, 0,
               "压力升高 → 气相反应物分压增大 → 反应速率加快", "positive"),

    # 反应速率 → 产物/副产物
    CausalEdge("Reaction_Rate", "Product_Conc", 0.60, 1,
               "反应速率加快 → 主反应产物生成增多", "positive"),
    CausalEdge("Reaction_Rate", "Byproduct_Conc", 0.22, 2,
               "反应速率过快 → 副反应加剧 → 副产物增多", "positive"),
    CausalEdge("Reaction_Rate", "Energy_Index", 0.30, 0,
               "反应速率加快 → 放热量增大 → 冷却能耗增大", "positive"),

    # 冷却水 → 后续影响
    CausalEdge("CW_Flow", "HX_Outlet_Temp", -0.45, 1,
               "冷却水流量增大 → 换热充分 → 出口温度降低", "negative"),
    CausalEdge("CW_Inlet_Temp", "HX_Outlet_Temp", 0.55, 0,
               "入口温度升高 → 换热基准温度升高 → 出口温度升高", "positive"),
    CausalEdge("Reactor_Temp", "HX_Outlet_Temp", 0.65, 0,
               "反应器温度升高 → 待冷却物料温度升高 → 出口温度升高", "positive"),

    # 产物浓度 → 间接影响
    CausalEdge("Product_Conc", "Energy_Index", -0.10, 2,
               "产物浓度达标 → 反应趋于稳态 → 能耗趋于稳定", "negative"),
]

# 变量列表
VAR_NAMES = [
    "Feed_Flow", "Feed_Conc", "CW_Inlet_Temp", "CW_Valve",
    "Reactor_Temp", "Reactor_Press", "CW_Flow", "Reaction_Rate",
    "Product_Conc", "Byproduct_Conc", "HX_Outlet_Temp", "Energy_Index"
]

# 变量分类
ROOT_VARS = ["Feed_Flow", "Feed_Conc", "CW_Inlet_Temp", "CW_Valve"]
INTERMEDIATE_VARS = ["Reactor_Temp", "Reactor_Press", "CW_Flow", "Reaction_Rate"]
OUTPUT_VARS = ["Product_Conc", "Byproduct_Conc", "HX_Outlet_Temp", "Energy_Index"]

# 正常工况参数
NORMAL_PARAMS = {
    "Feed_Flow": {"mean": 100, "std": 2, "unit": "L/min"},
    "Feed_Conc": {"mean": 0.85, "std": 0.02, "unit": "mol/L"},
    "CW_Inlet_Temp": {"mean": 25, "std": 1.5, "unit": "°C"},
    "CW_Valve": {"mean": 60, "std": 3, "unit": "%"},
}

# 故障模式定义
FAULT_MODES = {
    "FAULT_COOLING_VALVE_STUCK": {
        "id": 1,
        "name": "冷却水阀门卡滞",
        "description": "冷却水阀门逐渐卡滞在低开度位置，导致冷却不足",
        "root_cause": "CW_Valve",
        "causal_path": "CW_Valve → CW_Flow → Reactor_Temp → Reaction_Rate → Product_Conc",
        "injection": {"variable": "CW_Valve", "type": "drift",
                      "start": 1.0, "rate": -0.15, "min_val": 15},
    },
    "FAULT_FEED_CONC_DROP": {
        "id": 2,
        "name": "进料浓度异常下降",
        "description": "上游原料品质波动，进料浓度持续下降",
        "root_cause": "Feed_Conc",
        "causal_path": "Feed_Conc → Reaction_Rate → Product_Conc",
        "injection": {"variable": "Feed_Conc", "type": "drift",
                      "start": 1.0, "rate": -0.008, "min_val": 0.5},
    },
    "FAULT_CW_INLET_TEMP_HIGH": {
        "id": 3,
        "name": "冷却水入口温度升高",
        "description": "冷却塔故障导致循环水温度升高",
        "root_cause": "CW_Inlet_Temp",
        "causal_path": "CW_Inlet_Temp → Reactor_Temp → (多头影响) → HX_Outlet_Temp / Reaction_Rate / Energy_Index",
        "injection": {"variable": "CW_Inlet_Temp", "type": "step",
                      "step_value": 35, "step_time": 200},
    },
    "FAULT_FEED_FLOW_SURGE": {
        "id": 4,
        "name": "进料流量突增",
        "description": "进料泵控制异常，流量突然增大",
        "root_cause": "Feed_Flow",
        "causal_path": "Feed_Flow → Reactor_Press / Product_Conc (负向)",
        "injection": {"variable": "Feed_Flow", "type": "step",
                      "step_value": 130, "step_time": 150},
    },
    "FAULT_COOLING_PUMP_FAIL": {
        "id": 5,
        "name": "冷却水泵故障",
        "description": "冷却水泵叶轮磨损，冷却水流量逐渐下降",
        "root_cause": "CW_Valve",
        "causal_path": "CW_Valve → CW_Flow → Reactor_Temp → (多个变量)",
        "injection": {"variable": "CW_Valve", "type": "drift",
                      "start": 1.0, "rate": -0.10, "min_val": 20},
    },
    "FAULT_REACTOR_FOULING": {
        "id": 6,
        "name": "反应器结垢",
        "description": "反应器内壁结垢，换热效率缓慢下降",
        "root_cause": "Reactor_Temp",
        "causal_path": "换热效率↓ → Reactor_Temp↑ → Reaction_Rate↑ → Product_Conc/Byproduct_Conc变化",
        "injection": {"variable": "CW_Inlet_Temp", "type": "drift",
                      "start": 1.0, "rate": 0.04, "min_val": 32},
    },
    "FAULT_SENSOR_DRIFT_TEMP": {
        "id": 7,
        "name": "温度传感器漂移",
        "description": "反应器温度传感器零点漂移，读数持续偏高（实际温度正常）",
        "root_cause": "Reactor_Temp",
        "causal_path": "传感器漂移 → 温度读数偏高 → 误判反应异常",
        "injection": {"variable": "CW_Inlet_Temp", "type": "step",
                      "step_value": 27, "step_time": 300},
    },
    "FAULT_FEED_VALVE_STUCK": {
        "id": 8,
        "name": "进料阀门卡滞",
        "description": "进料调节阀卡滞在低开度，进料流量低于设定值",
        "root_cause": "Feed_Flow",
        "causal_path": "Feed_Flow↓ → Reactor_Press↓ → Reaction_Rate↓ → Product_Conc↓",
        "injection": {"variable": "Feed_Flow", "type": "drift",
                      "start": 1.0, "rate": -0.20, "min_val": 65},
    },
    "FAULT_CATALYST_DEACTIVATION": {
        "id": 9,
        "name": "催化剂失活",
        "description": "催化剂缓慢失活，反应速率持续下降，产物质量逐渐不达标",
        "root_cause": "Feed_Conc",
        "causal_path": "催化剂活性↓ → 等效浓度↓ → Reaction_Rate↓ → Product_Conc↓",
        "injection": {"variable": "Feed_Conc", "type": "drift",
                      "start": 1.0, "rate": -0.002, "min_val": 0.55},
    },
    "FAULT_COMBINED_VALVE_AND_TEMP": {
        "id": 10,
        "name": "复合故障：阀门卡滞 + 入口水温升高",
        "description": "冷却水阀门卡滞与冷却塔效率下降同时发生，多个根因并存",
        "root_cause": "CW_Valve",
        "causal_path": "CW_Valve↓ + CW_Inlet_Temp↑ → Reactor_Temp↑↑ → 连锁反应",
        "injection": {"variable": "CW_Valve", "type": "drift",
                      "start": 1.0, "rate": -0.12, "min_val": 25},
    },
}


class SyntheticProcessSimulator:
    """
    基于已知因果图的合成工业过程仿真器

    使用线性结构方程模型（Linear SEM），同时加入非线性项和噪声
    """

    def __init__(self, causal_edges: List[CausalEdge] = None, noise_level: float = 0.05,
                 seed: int = 42):
        self.causal_edges = causal_edges or CAUSAL_GRAPH_TRUTH
        self.noise_level = noise_level
        self.rng = np.random.RandomState(seed)

        # 构建变量索引
        self._build_var_index()
        # 构建因果结构
        self._build_causal_structure()

    def _build_var_index(self):
        """构建变量名到索引的映射"""
        self.var_index = {name: i for i, name in enumerate(VAR_NAMES)}

    def _build_causal_structure(self):
        """从因果边列表中构建计算图"""
        # 拓扑排序：确保计算变量时，其父节点已经算好
        import networkx as nx
        G = nx.DiGraph()
        for edge in self.causal_edges:
            G.add_edge(edge.cause, edge.effect)

        self.topological_order = list(nx.topological_sort(G))
        self.causal_graph_nx = G

        # 每个变量的入边列表（延迟展开）
        self.incoming_edges: Dict[str, List[CausalEdge]] = {v: [] for v in VAR_NAMES}
        for edge in self.causal_edges:
            self.incoming_edges[edge.effect].append(edge)

    def _compute_variable(self, var_name: str, current_state: np.ndarray,
                          history: np.ndarray, t: int) -> float:
        """根据因果父节点计算当前变量的值"""

        # 根变量：完全由外部设定决定（均值+噪声）
        if var_name in ROOT_VARS:
            params = NORMAL_PARAMS[var_name]
            base = params["mean"]
            noise_std = params["std"] * self.noise_level * 2
        else:
            # 非根变量：由因果父节点决定
            base = 0.0
            noise_std = 0.01

            for edge in self.incoming_edges[var_name]:
                cause_idx = self.var_index[edge.cause]
                lag = edge.time_lag

                if t >= lag:
                    cause_value = history[t - lag, cause_idx]
                else:
                    cause_value = current_state[cause_idx]

                # 非线性变换（模拟真实物理过程的非线性）
                if edge.direction == "negative":
                    nonlinear = -np.log1p(abs(cause_value) * abs(edge.coefficient) * 10)
                    base += edge.coefficient * cause_value + 0.05 * nonlinear
                else:
                    nonlinear = np.sqrt(abs(cause_value) * abs(edge.coefficient) + 1e-6)
                    base += edge.coefficient * cause_value + 0.03 * nonlinear

        # 添加噪声
        noise = self.rng.normal(0, noise_std)
        return base + noise

    def simulate(self, n_steps: int = 1000, fault_config: Optional[dict] = None,
                 fault_start: int = 300) -> pd.DataFrame:
        """
        运行仿真

        参数:
            n_steps: 总仿真步数
            fault_config: 故障注入配置（来自 FAULT_MODES）
            fault_start: 故障开始的时间步

        返回:
            DataFrame，每列一个变量
        """
        history = np.zeros((n_steps, len(VAR_NAMES)))

        # 初始化根变量的状态
        root_state = {v: NORMAL_PARAMS[v]["mean"] for v in ROOT_VARS}

        for t in range(n_steps):
            # 故障注入
            if fault_config and t >= fault_start:
                inj = fault_config["injection"]
                if inj["type"] == "drift":
                    drift_amount = inj["rate"] * (t - fault_start)
                    new_val = NORMAL_PARAMS[inj["variable"]]["mean"] * inj["start"] + drift_amount
                    root_state[inj["variable"]] = max(inj.get("min_val", -np.inf), new_val)
                elif inj["type"] == "step":
                    root_state[inj["variable"]] = inj["step_value"]
                elif inj["type"] == "spike":
                    if t == fault_start:
                        root_state[inj["variable"]] = inj["spike_value"]
            else:
                # 正常状态：根变量围绕均值波动
                for v in ROOT_VARS:
                    params = NORMAL_PARAMS[v]
                    root_state[v] = params["mean"] + self.rng.normal(0, params["std"] * 0.3)

            # 将根变量写入history
            for v in ROOT_VARS:
                history[t, self.var_index[v]] = root_state[v]

            # 按拓扑顺序计算所有非根变量
            for var_name in self.topological_order:
                if var_name in ROOT_VARS:
                    continue
                val = self._compute_variable(var_name, history[t], history, t)
                history[t, self.var_index[var_name]] = val

        # 构建DataFrame
        df = pd.DataFrame(history, columns=VAR_NAMES)

        # 添加标准化的变量值（方便异常检测）
        # 使用前100步正常数据做标准化
        normal_data = df.iloc[:100]
        for col in VAR_NAMES:
            mean = normal_data[col].mean()
            std = normal_data[col].std()
            df[f"{col}_norm"] = (df[col] - mean) / (std + 1e-8)

        return df

    def generate_fault_dataset(self, n_normal: int = 500, n_fault: int = 500,
                               fault_name: str = "FAULT_COOLING_VALVE_STUCK"
                               ) -> Tuple[pd.DataFrame, dict]:
        """
        生成包含正常+故障的完整数据集

        返回:
            df: 完整数据（前n_normal步正常，后n_fault步故障）
            metadata: 数据集元信息
        """
        fault_config = FAULT_MODES[fault_name]

        # 正常阶段
        df_normal = self.simulate(n_normal, fault_config=None)
        df_normal["label"] = 0  # 正常
        df_normal["fault"] = "NORMAL"
        df_normal["fault_root_cause"] = ""

        # 故障阶段
        df_fault = self.simulate(n_fault, fault_config=fault_config, fault_start=0)
        df_fault["label"] = 1  # 故障
        df_fault["fault"] = fault_name
        df_fault["fault_root_cause"] = fault_config["root_cause"]

        # 合并
        df = pd.concat([df_normal, df_fault], ignore_index=True)
        df["time"] = range(len(df))

        metadata = {
            "fault_name": fault_name,
            "description": fault_config["description"],
            "root_cause": fault_config["root_cause"],
            "causal_path": fault_config["causal_path"],
            "fault_start_step": n_normal,
            "n_variables": len(VAR_NAMES),
            "var_names": VAR_NAMES,
            "ground_truth_causal_edges": [
                {"cause": e.cause, "effect": e.effect, "coefficient": e.coefficient,
                 "lag": e.time_lag, "mechanism": e.mechanism}
                for e in self.causal_edges
            ]
        }

        return df, metadata

    def export_ground_truth(self, filepath: str):
        """导出真实因果图到JSON"""
        truth = {
            "variables": VAR_NAMES,
            "causal_edges": [
                {
                    "cause": e.cause,
                    "effect": e.effect,
                    "coefficient": e.coefficient,
                    "time_lag": e.time_lag,
                    "mechanism": e.mechanism,
                    "direction": e.direction,
                }
                for e in self.causal_edges
            ]
        }
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(truth, f, indent=2, ensure_ascii=False)
        return filepath


def generate_all_datasets(output_dir: str = "data/synthetic"):
    """生成所有演示数据集"""
    import os
    os.makedirs(output_dir, exist_ok=True)

    simulator = SyntheticProcessSimulator(seed=42)

    # 导出ground truth因果图
    simulator.export_ground_truth(f"{output_dir}/ground_truth_causal_graph.json")
    print(f"✓ Ground truth因果图已导出")

    # 生成正常工况数据（用于因果发现训练）
    df_normal, _ = simulator.generate_fault_dataset(
        n_normal=1000, n_fault=0, fault_name="FAULT_COOLING_VALVE_STUCK"
    )
    df_normal = df_normal[df_normal["fault"] == "NORMAL"]
    df_normal.drop(columns=["label", "fault", "fault_root_cause"], inplace=True)
    df_normal.to_csv(f"{output_dir}/normal_operation.csv", index=False)
    print(f"✓ 正常工况数据: {len(df_normal)} 步")

    # 生成各故障数据
    for fault_name in FAULT_MODES:
        df, meta = simulator.generate_fault_dataset(
            n_normal=300, n_fault=500, fault_name=fault_name
        )
        df.to_csv(f"{output_dir}/{fault_name}.csv", index=False)

        # 导出metadata
        with open(f"{output_dir}/{fault_name}_metadata.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)

        print(f"✓ 故障 [{fault_name}]: 正常300步 + 故障500步, "
              f"根因={meta['root_cause']}")

    # 生成工艺文档（模拟）
    generate_process_documentation(output_dir)

    print(f"\n所有数据已生成至 {output_dir}/")
    return simulator


def generate_process_documentation(output_dir: str):
    """生成模拟的工业过程文档，供LLM因果抽取使用"""
    doc = """
# 连续搅拌釜式反应器（CSTR）操作规程

## 第一章 设备概述
本装置为连续搅拌釜式反应器系统，用于液相催化反应生产目标产物。
系统包括：进料系统、反应器本体、换热系统、产物分离系统。

## 第二章 过程变量

### 2.1 进料系统
- 进料流量(Feed_Flow)：控制范围 80-120 L/min，正常设定值 100 L/min
  异常升高会导致反应器内压力上升，同时缩短物料停留时间，使反应不充分。
- 进料浓度(Feed_Conc)：正常范围 0.75-0.95 mol/L
  进料浓度直接决定反应速率。经验表明浓度每下降0.1 mol/L，反应速率降低约17%。

### 2.2 换热系统
- 冷却水入口温度(CW_Inlet_Temp)：正常 ≤28°C
  当入口温度超过32°C时，换热效率显著下降，反应器温度将异常升高。
  应定期检查冷却塔运行状态。
- 冷却水阀门开度(CW_Valve)：正常范围 45-75%
  阀门开度与冷却水流量呈正比关系。阀门卡滞是常见故障，一旦发生，
  冷却水流量立即下降。

### 2.3 反应器
- 反应器温度(Reactor_Temp)：正常范围 150-175°C
  温度是影响反应速率的关键因素，每升高10°C，反应速率约增加一倍（Arrhenius定律）。
  温度异常通常源于冷却水系统问题或进料异常。
- 反应器压力(Reactor_Press)：正常范围 2.5-4.0 MPa
  压力受进料流量和反应器温度双重影响。

### 2.4 产物
- 产物浓度(Product_Conc)：目标 ≥0.70 mol/L
  产物浓度由反应速率和停留时间共同决定。反应速率不足或进料过快都会导致产物不达标。
- 副产物浓度(Byproduct_Conc)：控制 ≤0.12 mol/L
  反应速率过快时副反应加剧，副产物浓度上升。

## 第三章 常见故障处理

### 故障1：冷却水阀门卡滞
现象：冷却水流量(CW_Flow)持续下降 → 反应器温度(Reactor_Temp)上升 →
      反应速率短暂加快后因副反应加剧而产物质量下降 →
      副产物浓度(Byproduct_Conc)上升。
处理：1) 尝试增大阀门开度指令 2) 如无效，切换至备用冷却回路
      3) 如反应器温度超过185°C，紧急停车

### 故障2：进料浓度异常
现象：进料浓度(Feed_Conc)下降 → 反应速率(Reaction_Rate)降低 →
      产物浓度(Product_Conc)不达标。
处理：1) 检查上游原料配比 2) 适当降低进料流量以延长停留时间

### 故障3：冷却塔效率下降
现象：冷却水入口温度(CW_Inlet_Temp)升高 → 换热器出口温度(HX_Outlet_Temp)升高 →
      反应器温度(Reactor_Temp)升高 → 能耗指标(Energy_Index)增大
处理：1) 检查冷却塔风机运行 2) 补充循环水 3) 如持续恶化，减负荷运行

### 故障4：进料泵控制异常
现象：进料流量(Feed_Flow)突增 → 反应器压力(Reactor_Press)快速上升 →
      停留时间不足 → 产物浓度(Product_Conc)下降
处理：1) 立即切换至备用泵 2) 检修进料调节阀
"""

    with open(f"{output_dir}/process_documentation.txt", "w", encoding="utf-8") as f:
        f.write(doc)

    # 同时生成预提取的因果对（保证Demo在没有LLM API时也能跑通）
    pre_extracted_pairs = {
        "causal_pairs": [
            {"cause": "Feed_Flow", "effect": "Reactor_Press", "direction": "positive",
             "mechanism": "进料流量增大→反应器内物料累积→压力升高", "quantitative_relation": "",
             "time_lag": "0", "confidence": 0.90, "evidence": "异常升高会导致反应器内压力上升——操作手册2.1节"},
            {"cause": "Feed_Conc", "effect": "Reaction_Rate", "direction": "positive",
             "mechanism": "进料浓度升高→反应速率常数增大", "quantitative_relation": "浓度每下降0.1mol/L,反应速率降低约17%",
             "time_lag": "1", "confidence": 0.95, "evidence": "进料浓度直接决定反应速率——操作手册2.1节"},
            {"cause": "Feed_Flow", "effect": "Product_Conc", "direction": "negative",
             "mechanism": "进料流量过大→停留时间缩短→反应不充分→产物浓度下降", "quantitative_relation": "",
             "time_lag": "3", "confidence": 0.85, "evidence": "缩短物料停留时间,使反应不充分——操作手册2.1节"},
            {"cause": "CW_Valve", "effect": "CW_Flow", "direction": "positive",
             "mechanism": "阀门开度与冷却水流量呈正比", "quantitative_relation": "",
             "time_lag": "0", "confidence": 0.95, "evidence": "阀门开度与冷却水流量呈正比关系——操作手册2.2节"},
            {"cause": "CW_Inlet_Temp", "effect": "Reactor_Temp", "direction": "positive",
             "mechanism": "冷却水入口温度升高→换热温差减小→反应器温度升高", "quantitative_relation": "入口温度超过32°C时,换热效率显著下降",
             "time_lag": "1", "confidence": 0.90, "evidence": "当入口温度超过32°C时,换热效率显著下降——操作手册2.2节"},
            {"cause": "CW_Flow", "effect": "Reactor_Temp", "direction": "negative",
             "mechanism": "冷却水流量增大→换热量增大→反应器温度降低", "quantitative_relation": "",
             "time_lag": "1", "confidence": 0.95, "evidence": "冷却水流量(CW_Flow)持续下降→反应器温度(Reactor_Temp)上升——故障1"},
            {"cause": "Reactor_Temp", "effect": "Reaction_Rate", "direction": "positive",
             "mechanism": "温度升高→反应速率常数增大(Arrhenius定律)", "quantitative_relation": "每升高10°C,反应速率约增加一倍",
             "time_lag": "0", "confidence": 0.95, "evidence": "温度是影响反应速率的关键因素,每升高10°C,反应速率约增加一倍——操作手册2.3节"},
            {"cause": "Reactor_Temp", "effect": "Reactor_Press", "direction": "positive",
             "mechanism": "温度升高→气相膨胀→压力增大", "quantitative_relation": "",
             "time_lag": "0", "confidence": 0.85, "evidence": "压力受进料流量和反应器温度双重影响——操作手册2.3节"},
            {"cause": "Reactor_Temp", "effect": "Energy_Index", "direction": "positive",
             "mechanism": "温度升高→加热/冷却能耗增大", "quantitative_relation": "",
             "time_lag": "1", "confidence": 0.80, "evidence": "反应器温度(Reactor_Temp)升高→能耗指标(Energy_Index)增大——故障3"},
            {"cause": "Reactor_Temp", "effect": "HX_Outlet_Temp", "direction": "positive",
             "mechanism": "反应器温度升高→待冷却物料温度升高→出口温度升高", "quantitative_relation": "",
             "time_lag": "0", "confidence": 0.85, "evidence": "反应器温度(Reactor_Temp)升高→换热器出口温度(HX_Outlet_Temp)升高——故障3"},
            {"cause": "Reactor_Press", "effect": "Reaction_Rate", "direction": "positive",
             "mechanism": "压力升高→气相反应物分压增大→反应速率加快", "quantitative_relation": "",
             "time_lag": "0", "confidence": 0.80, "evidence": "压力受进料流量和反应器温度双重影响——操作手册2.3节"},
            {"cause": "Reaction_Rate", "effect": "Product_Conc", "direction": "positive",
             "mechanism": "反应速率加快→主反应产物生成增多", "quantitative_relation": "",
             "time_lag": "1", "confidence": 0.90, "evidence": "产物浓度由反应速率和停留时间共同决定——操作手册2.4节"},
            {"cause": "Reaction_Rate", "effect": "Byproduct_Conc", "direction": "positive",
             "mechanism": "反应速率过快→副反应加剧→副产物增多", "quantitative_relation": "",
             "time_lag": "2", "confidence": 0.85, "evidence": "反应速率过快时副反应加剧,副产物浓度上升——操作手册2.4节"},
            {"cause": "CW_Flow", "effect": "HX_Outlet_Temp", "direction": "negative",
             "mechanism": "冷却水流量增大→换热充分→出口温度降低", "quantitative_relation": "",
             "time_lag": "1", "confidence": 0.85, "evidence": "冷却水流量增大→换热充分→出口温度降低——操作手册2.2节"},
            {"cause": "Reaction_Rate", "effect": "Energy_Index", "direction": "positive",
             "mechanism": "反应速率加快→反应放热量增大→冷却能耗增大", "quantitative_relation": "",
             "time_lag": "0", "confidence": 0.80, "evidence": "反应速率加快→反应放热量增大——操作手册2.4节"},
        ]
    }
    with open(f"{output_dir}/knowledge_pairs.json", "w", encoding="utf-8") as f:
        json.dump(pre_extracted_pairs, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    generate_all_datasets()
