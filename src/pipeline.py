"""
完整演示流程: 从数据到因果推理

串联所有模块的端到端Pipeline，用于:
  1. 开发调试: 快速验证所有模块正常工作
  2. Demo演示: 生成用于答辩的完整输出
  3. 对比实验: 与baseline方法对比
"""

import numpy as np
import pandas as pd
import networkx as nx
import json
import os
from typing import Dict, Tuple, Optional

from src.synthetic_data_generator import (
    SyntheticProcessSimulator, CAUSAL_GRAPH_TRUTH, VAR_NAMES,
    ROOT_VARS, FAULT_MODES, generate_process_documentation
)
from src.causal_discovery import CausalDiscovery
from src.llm_causal_extract import LLMCausalExtractor, extract_from_synthetic_doc
from src.graph_fusion import CausalGraphFusion
from src.root_cause_analysis import RootCauseAnalyzer, RootCauseReport
from src.counterfactual import CounterfactualEngine


class CausalAgentPipeline:
    """
    因果增强工业智能体完整流水线

    流程:
      1. 数据准备: 生成/加载过程数据
      2. 因果图构建:
         a. 知识通道: LLM从工艺文档提取因果对 → 知识因果图
         b. 数据通道: PCMCI+从传感器数据发现因果结构 → 数据因果图
         c. 融合: 双通道融合 → 最终因果图
      3. 根因分析: 基于融合因果图进行异常检测和根因回溯
      4. 反事实推理: 评估干预方案效果
    """

    def __init__(self, data_dir: str = "data/synthetic"):
        self.data_dir = data_dir
        self.simulator: Optional[SyntheticProcessSimulator] = None
        self.causal_discovery: Optional[CausalDiscovery] = None
        self.fusion: Optional[CausalGraphFusion] = None
        self.analyzer: Optional[RootCauseAnalyzer] = None
        self.cf_engine: Optional[CounterfactualEngine] = None

        # 结果存储
        self.knowledge_graph: Optional[nx.DiGraph] = None
        self.data_graph: Optional[nx.DiGraph] = None
        self.fused_graph: Optional[nx.DiGraph] = None
        self.normal_range: Dict = {}
        self.ground_truth_graph: Optional[nx.DiGraph] = None

    def step1_prepare_data(self, seed: int = 42):
        """
        Step 1: 准备数据
          - 生成合成工业过程数据
          - 生成正常工况数据（用于建立normal range）
          - 生成多种故障数据（用于验证根因分析）
        """
        print("=" * 60)
        print("Step 1: 数据准备")
        print("=" * 60)

        self.simulator = SyntheticProcessSimulator(seed=seed)

        # 保存ground truth
        self.ground_truth_graph = nx.DiGraph()
        for edge in CAUSAL_GRAPH_TRUTH:
            self.ground_truth_graph.add_edge(
                edge.cause, edge.effect,
                coefficient=edge.coefficient,
                lag=edge.time_lag,
                mechanism=edge.mechanism,
            )

        os.makedirs(self.data_dir, exist_ok=True)
        self.simulator.export_ground_truth(
            f"{self.data_dir}/ground_truth_causal_graph.json"
        )

        # 生成正常工况数据
        df_normal = self.simulator.simulate(n_steps=1000, fault_config=None)
        df_normal.to_csv(f"{self.data_dir}/normal_operation.csv", index=False)

        # 计算各变量正常范围
        self.normal_range = {}
        for col in VAR_NAMES:
            mean = df_normal[col].mean()
            std = df_normal[col].std()
            self.normal_range[col] = (mean - 3 * std, mean + 3 * std)

        # 生成故障数据
        self.fault_datasets = {}
        for fault_name, fault_config in FAULT_MODES.items():
            df, meta = self.simulator.generate_fault_dataset(
                n_normal=300, n_fault=500, fault_name=fault_name
            )
            df.to_csv(f"{self.data_dir}/{fault_name}.csv", index=False)
            self.fault_datasets[fault_name] = {"df": df, "meta": meta}

        # 生成工艺文档
        generate_process_documentation(self.data_dir)

        print(f"  ✓ 正常工况: 1000步")
        print(f"  ✓ 故障场景: {len(FAULT_MODES)}种")
        print(f"  ✓ 正常范围: {len(self.normal_range)}个变量")
        print(f"  ✓ Ground truth: {len(CAUSAL_GRAPH_TRUTH)}条因果边")

        return df_normal

    def step2_build_causal_graph(self, df_normal: pd.DataFrame,
                                 use_llm_api: bool = False):
        """
        Step 2: 构建因果图（双通道融合）

        通道1: LLM从工艺文档提取因果知识
        通道2: PCMCI+从传感器数据发现因果结构
        """
        print(f"\n{'='*60}")
        print("Step 2: 因果图构建（双通道融合）")
        print("=" * 60)

        # ---- 通道1: 知识驱动 ----
        print("\n[通道1] LLM知识提取...")
        doc_path = f"{self.data_dir}/process_documentation.txt"
        if os.path.exists(doc_path):
            knowledge_pairs = extract_from_synthetic_doc(
                doc_path, VAR_NAMES, use_llm=use_llm_api
            )
            print(f"  提取到 {len(knowledge_pairs)} 个因果对")
            # 显示部分结果
            for p in knowledge_pairs[:5]:
                print(f"    {p.cause} → {p.effect} [{p.direction}] (conf={p.confidence:.2f})")
        else:
            print("  ⚠ 工艺文档不存在，跳过知识通道")
            knowledge_pairs = []

        self.knowledge_graph = LLMCausalExtractor.pairs_to_graph(
            knowledge_pairs, VAR_NAMES
        )
        print(f"  知识因果图: {self.knowledge_graph.number_of_nodes()}节点, "
              f"{self.knowledge_graph.number_of_edges()}条边")

        # ---- 通道2: 数据驱动 ----
        print("\n[通道2] PCMCI+因果发现...")
        self.causal_discovery = CausalDiscovery(VAR_NAMES)

        # 用正常工况数据的前500步做因果发现
        data_subset = df_normal[VAR_NAMES].iloc[:500]
        self.data_graph = self.causal_discovery.discover_pcmciplus(
            data=data_subset,
            tau_max=5,
        )

        # 与ground truth比较
        metrics = self.causal_discovery.compare_with_ground_truth(
            self.ground_truth_graph
        )
        print(f"  数据因果图 vs Ground Truth:")
        print(f"    Precision: {metrics['precision']:.1%}")
        print(f"    Recall:    {metrics['recall']:.1%}")
        print(f"    F1:        {metrics['f1']:.1%}")

        # ---- 融合 ----
        print("\n[融合] 双通道融合...")
        self.fusion = CausalGraphFusion()

        # 转换knowledge_pairs为融合所需的格式
        kp_formatted = []
        for p in knowledge_pairs:
            # time_lag 从 str 转换为 int
            try:
                t_lag = int(p.time_lag) if p.time_lag else 0
            except (ValueError, TypeError):
                t_lag = 0

            kp_formatted.append({
                "cause": p.cause,
                "effect": p.effect,
                "confidence": p.confidence,
                "mechanism": p.mechanism,
                "evidence": p.evidence,
                "time_lag": t_lag,
            })

        self.fused_graph = self.fusion.fuse(
            self.knowledge_graph, self.data_graph, kp_formatted
        )

        # 保存
        nx.write_graphml(self.fused_graph,
                        f"{self.data_dir}/fused_causal_graph.graphml")
        with open(f"{self.data_dir}/fused_graph.json", "w", encoding="utf-8") as f:
            json.dump(self.fusion.to_dict(), f, indent=2, ensure_ascii=False)

        print(self.fusion.generate_fusion_report())

        # 显示新发现
        discoveries = self.fusion.get_new_discoveries()
        if discoveries:
            print(f"\n  ⚡ 数据驱动新发现 {len(discoveries)} 条:")
            for d in discoveries:
                print(f"    {d.cause} → {d.effect} (conf={d.confidence:.3f})")

        return self.fused_graph

    def step3_root_cause_analysis(self, fault_name: str = "FAULT_COOLING_VALVE_STUCK"):
        """
        Step 3: 根因分析

        用真实故障场景验证因果推理的准确性
        """
        print(f"\n{'='*60}")
        print(f"Step 3: 根因分析 [{fault_name}]")
        print("=" * 60)

        # 加载故障数据
        if fault_name not in self.fault_datasets:
            print(f"  ⚠ 故障数据不存在, 先生成...")
            df, meta = self.simulator.generate_fault_dataset(
                n_normal=300, n_fault=500, fault_name=fault_name
            )
        else:
            df = self.fault_datasets[fault_name]["df"]
            meta = self.fault_datasets[fault_name]["meta"]

        # 设置根因分析器
        self.analyzer = RootCauseAnalyzer(self.fused_graph)

        # 从正常数据设置normal range
        normal_data = df[df["fault"] == "NORMAL"][VAR_NAMES]
        self.analyzer.set_normal_ranges(normal_data, n_std=3.0)

        # 取故障阶段的一个观测点
        fault_data = df[df["fault"] == fault_name]
        sample = fault_data.iloc[200]  # 故障中期的一个点

        observation = {v: sample[v] for v in VAR_NAMES}

        print(f"\n[观测] 故障中期(t=200)各变量状态:")
        for v in VAR_NAMES:
            lo, hi = self.analyzer.normal_ranges.get(v, (-np.inf, np.inf))
            val = observation[v]
            status = "⚠ 异常" if (val < lo or val > hi) else "✓ 正常"
            print(f"  {v:20s}: {val:10.3f} [{lo:.1f}, {hi:.1f}] {status}")

        # 执行根因分析
        reports = self.analyzer.analyze(observation, top_k=3)

        print(f"\n[根因分析结果]")
        for report in reports:
            print(f"\n  异常变量: {report.abnormal_variable}")
            print(f"  观测值: {report.observed_value:.3f} "
                  f"(正常: {report.normal_range[0]:.1f}-{report.normal_range[1]:.1f})")
            print(f"  根因排序:")
            for i, rc in enumerate(report.root_causes):
                print(f"    {i+1}. {rc.variable} (评分={rc.score:.3f})")
                print(f"       因果路径: {' → '.join(rc.path)}")
            print(f"  处置建议:")
            for action in report.recommended_actions[:3]:
                print(f"    • {action}")

        # 验证根因是否正确
        true_root_cause = meta["root_cause"]
        print(f"\n[验证] Ground Truth根因: {true_root_cause}")
        found_correct = any(
            true_root_cause in rc.variable
            for report in reports
            for rc in report.root_causes
        )
        print(f"  ✓ 根因分析正确!" if found_correct else "  ✗ 根因分析不匹配")

        return reports

    def step4_counterfactual(self, fault_name: str = "FAULT_COOLING_VALVE_STUCK"):
        """
        Step 4: 反事实推理

        评估不同干预方案的效果
        """
        print(f"\n{'='*60}")
        print("Step 4: 反事实推理")
        print("=" * 60)

        # 获取故障观测
        if fault_name in self.fault_datasets:
            df = self.fault_datasets[fault_name]["df"]
            fault_data = df[df["fault"] == fault_name]
            sample = fault_data.iloc[200]
            observation = {v: sample[v] for v in VAR_NAMES}
        else:
            observation = {
                "CW_Valve": 22, "CW_Flow": 120, "CW_Inlet_Temp": 26,
                "Reactor_Temp": 192, "Reactor_Press": 3.8, "Reaction_Rate": 1.45,
                "Feed_Flow": 101, "Feed_Conc": 0.84,
                "Product_Conc": 0.58, "Byproduct_Conc": 0.15,
                "HX_Outlet_Temp": 45, "Energy_Index": 1.8,
            }

        # 初始化反事实引擎
        self.cf_engine = CounterfactualEngine(self.fused_graph)

        # 从ground truth获取真实系数（如果是合成数据）
        coeffs = {}
        for edge in CAUSAL_GRAPH_TRUTH:
            coeffs[(edge.cause, edge.effect)] = edge.coefficient
        self.cf_engine.set_manual_coefficients(coeffs)

        # 场景1: 单一干预
        print("\n[场景1] 单一干预效果评估")
        r1 = self.cf_engine.what_if(
            observation,
            {"CW_Valve": 60},
            "Product_Conc",
            self.normal_range,
        )
        print(f"  干预: CW_Valve → 60%")
        print(f"  Product_Conc: {r1.factual_value:.3f} → {r1.counterfactual_value:.3f}")
        print(f"  改善: {r1.improvement:+.3f} ({r1.improvement_pct:.1%})")

        # 场景2: 方案对比
        print("\n[场景2] 多方案对比")
        comparison = self.cf_engine.compare_interventions(
            observation,
            "Product_Conc",
            [
                {"CW_Valve": 60},                          # 方案1
                {"Feed_Flow": 95},                         # 方案2
                {"CW_Valve": 60, "CW_Inlet_Temp": 25},    # 方案3
            ],
            self.normal_range,
        )

        for i, s in enumerate(comparison.scenarios):
            print(f"  方案{i+1}: {s.intervention}")
            print(f"    Product_Conc → {s.counterfactual_value:.3f} "
                  f"(改善 {s.improvement:+.3f}, {s.improvement_pct:.1%})")
        print(f"  {comparison.recommendation}")

        # 场景3: 归因分析
        print("\n[场景3] 异常归因分解")
        attr = self.cf_engine.attribution(
            observation,
            "Product_Conc",
            ["CW_Valve", "Feed_Flow", "CW_Inlet_Temp"],
            {"CW_Valve": 60, "Feed_Flow": 100, "CW_Inlet_Temp": 25},
        )
        for cause, pct in attr.items():
            bar = "█" * int(pct * 30)
            print(f"  {cause:20s}: {bar} {pct:.1%}")

        return comparison

    def run_demo(self, fault_name: str = "FAULT_COOLING_VALVE_STUCK",
                 use_llm_api: bool = False):
        """运行完整演示流程"""
        print("\n" + "=" * 70)
        print("  因果增强工业智能体 — 完整演示")
        print("  创智青山AI智能体创新大赛 · 技术挑战赛道")
        print("=" * 70)

        df_normal = self.step1_prepare_data(seed=42)
        self.step2_build_causal_graph(df_normal, use_llm_api=use_llm_api)

        # 测试所有故障模式
        for fault_name in FAULT_MODES.keys():
            self.step3_root_cause_analysis(fault_name)

        self.step4_counterfactual(fault_name)

        # 生成综合报告
        print(f"\n{'='*70}")
        print("  演示完成！所有结果已保存至 data/synthetic/")
        print("=" * 70)

        self._save_summary_report(fault_name)

    def _save_summary_report(self, fault_name: str):
        """保存综合报告"""
        report = {
            "pipeline": "因果增强工业智能体",
            "ground_truth_edges": len(CAUSAL_GRAPH_TRUTH),
            "knowledge_edges": (self.knowledge_graph.number_of_edges()
                               if self.knowledge_graph else 0),
            "data_graph_edges": (self.data_graph.number_of_edges()
                                if self.data_graph else 0),
            "fused_graph_edges": (self.fused_graph.number_of_edges()
                                 if self.fused_graph else 0),
            "fusion_stats": self.fusion.fusion_stats if self.fusion else {},
            "tested_faults": list(FAULT_MODES.keys()),
        }

        with open(f"{self.data_dir}/demo_summary.json", "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    pipeline = CausalAgentPipeline()
    pipeline.run_demo(use_llm_api=False)
