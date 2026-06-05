"""
TEP真实数据 + 因果推理 集成演示
使用真实的Tennessee Eastman过程数据，替代合成数据进行因果推理

运行: python run_tep_demo.py
"""

import sys
import os
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))

from src.tep_data_loader import (
    TEPDataLoader, TEP_FAULT_ROOT_CAUSE, DEMO_FAULTS, TEP_VAR_NAMES
)
from src.causal_discovery import CausalDiscovery
from src.graph_fusion import CausalGraphFusion
from src.root_cause_analysis import RootCauseAnalyzer
from src.counterfactual import CounterfactualEngine


def main():
    print("""
╔══════════════════════════════════════════════════════════════╗
║  基于真实TEP数据的因果增强工业智能体                            ║
║  Tennessee Eastman Process — Causal Agent Demo              ║
║  创智青山AI智能体创新大赛 · 技术挑战赛道                        ║
╚══════════════════════════════════════════════════════════════╝
""")

    # ==========================================================
    # Step 1: 加载真实TEP数据
    # ==========================================================
    print("=" * 60)
    print("Step 1: 加载TEP真实数据集")
    print("=" * 60)

    loader = TEPDataLoader()
    df_normal = loader.load_normal_data()
    print(f"  正常工况: {len(df_normal)} 采样点, {loader.n_vars} 过程变量")

    # ==========================================================
    # Step 2: 对每种演示故障进行因果推理
    # ==========================================================
    for fault_id_str, fault_info in DEMO_FAULTS.items():
        fault_id = int(fault_id_str.replace("IDV(", "").replace(")", ""))

        print(f"\n{'='*70}")
        print(f"Step 2: {fault_id_str} — {fault_info['name']}")
        print(f"{'='*70}")
        print(f"  描述: {fault_info['description']}")
        print(f"  预期根因: {fault_info['expected_root_cause_vars']}")
        print(f"  预期因果路径: {fault_info['causal_path']}")

        # 加载故障数据
        df_fault, meta = loader.load_fault_data(fault_id)
        print(f"  训练集: {meta['n_train']}步, 测试集: {meta['n_test']}步")

        # 选取因果相关变量子集
        causal_vars = fault_info["tep_causal_vars"]
        print(f"  因果推理变量: {causal_vars}")

        # ---- 通道1: 领域知识因果图 ----
        knowledge_graph = loader.build_tep_causal_graph(causal_vars)
        print(f"  知识因果图: {knowledge_graph.number_of_nodes()}节点, "
              f"{knowledge_graph.number_of_edges()}条边")

        # ---- 通道2: PCMCI+ 因果发现 ----
        cd = CausalDiscovery(causal_vars)
        # 用正常数据做因果发现
        df_discovery = df_normal[causal_vars].iloc[:500]
        try:
            data_graph = cd.discover_pcmciplus(
                data=df_discovery,
                tau_max=3,
            )
            print(f"  PCMCI+数据因果图: {data_graph.number_of_nodes()}节点, "
                  f"{data_graph.number_of_edges()}条边")
        except Exception as e:
            print(f"  PCMCI+失败: {e}，使用知识因果图作为fallback")
            data_graph = knowledge_graph  # Fallback

        # ---- 融合 ----
        fusion = CausalGraphFusion()
        fused_graph = fusion.fuse(knowledge_graph, data_graph)
        stats = fusion.fusion_stats
        print(f"  融合图: {stats['total_edges']}条边 "
              f"(双重验证: {stats['dual_verified']}, "
              f"知识独有: {stats['knowledge_only']}, "
              f"数据新发现: {stats['data_discovered']})")

        # ---- 根因分析 ----
        print(f"\n  [根因分析]")
        analyzer = RootCauseAnalyzer(fused_graph)

        # 用正常数据建立baseline
        normal_subset = df_normal[causal_vars].iloc[:300]
        analyzer.set_normal_ranges(normal_subset, n_std=3.0)

        # 取故障阶段的观测（测试集后半部分，故障已充分发展）
        fault_phase = df_fault[df_fault["fault_phase"] == "test"]
        sample = fault_phase.iloc[len(fault_phase) // 2]  # 故障中期
        observation = {v: float(sample[v]) for v in causal_vars}

        print(f"  故障中期观测:")
        for v in causal_vars:
            lo, hi = analyzer.normal_ranges.get(v, (-np.inf, np.inf))
            val = observation[v]
            status = "⚠ 异常" if (val < lo or val > hi) else " 正常"
            print(f"    {v:22s}: {val:12.6f} [{lo:.4f}, {hi:.4f}] {status}")

        reports = analyzer.analyze(observation, top_k=3)
        if reports:
            for report in reports:
                print(f"\n  ▶ 异常变量: {report.abnormal_variable}")
                if report.root_causes:
                    for i, rc in enumerate(report.root_causes):
                        print(f"    #{i+1} 根因: {rc.variable} (评分={rc.score:.3f})")
                        print(f"       因果路径: {' → '.join(rc.path)}")
                print(f"   处置建议:")
                for action in report.recommended_actions[:3]:
                    print(f"     • {action}")
        else:
            print(f"  （未检测到显著异常）")

        # ---- 反事实推理 ----
        print(f"\n  [反事实推理]")
        cf_engine = CounterfactualEngine(fused_graph)

        # 从知识图估算因果效应系数
        coeffs = {}
        for u, v, data in fused_graph.edges(data=True):
            coeffs[(u, v)] = data.get("coefficient", data.get("confidence", 0.5))
        cf_engine.set_manual_coefficients(coeffs)

        # 对根因变量做what-if
        if reports and reports[0].root_causes:
            top_rc = reports[0].root_causes[0].variable
            target = reports[0].abnormal_variable

            # 干预: 将根因恢复到正常范围中点
            if top_rc in analyzer.normal_ranges:
                lo, hi = analyzer.normal_ranges[top_rc]
                normal_val = (lo + hi) / 2
                result = cf_engine.what_if(
                    observation,
                    {top_rc: normal_val},
                    target,
                    analyzer.normal_ranges,
                )
                print(f"  干预: {top_rc} → {normal_val:.4f}")
                print(f"  {target}: {result.factual_value:.6f} → "
                      f"{result.counterfactual_value:.6f}")
                print(f"  改善: {result.improvement:+.6f}")

    # ==========================================================
    # Step 3: 基准对比总结
    # ==========================================================
    print(f"\n{'='*70}")
    print("  演示完成！")
    print(f"  使用数据集: Tennessee Eastman Process (Braatz Group, MIT)")
    print(f"  数据规模: 52变量, 21种故障, 每故障约1440采样点")
    print(f"  因果推理: 双通道融合 + 根因分析 + 反事实推理")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
