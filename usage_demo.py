"""
因果增强工业智能体 — 实际使用示例

演示三种使用场景:
  1. 离线分析: 用历史数据文件批量分析
  2. 单步诊断: 给一个观测，返回根因+建议
  3. 实时监控: 模拟逐秒读取传感器数据并分析（核心用法）
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd
from src.synthetic_data_generator import SyntheticProcessSimulator, VAR_NAMES, FAULT_MODES, CAUSAL_GRAPH_TRUTH
from src.llm_causal_extract import extract_from_synthetic_doc, LLMCausalExtractor
from src.graph_fusion import CausalGraphFusion
from src.root_cause_analysis import RootCauseAnalyzer
from src.counterfactual import CounterfactualEngine
from src.causal_discovery import CausalDiscovery
import networkx as nx


# ============================================================
# 初始化阶段（只做一次，启动时完成）
# ============================================================
def initialize_system():
    """
    系统初始化：构建因果图（离线，只做一次）

    这就像给AI装了一个"工业过程的大脑模型"——
    之后无论来多少传感器数据，都用这个大脑模型来推理
    """
    print("=" * 60)
    print("系统初始化：构建因果知识库...")
    print("=" * 60)

    # Step A: 知识通道 — 从工艺文档提取因果知识
    print("[1/3] 加载工艺知识...")
    os.makedirs("data/synthetic", exist_ok=True)
    from src.synthetic_data_generator import generate_process_documentation
    generate_process_documentation("data/synthetic")

    knowledge_pairs = extract_from_synthetic_doc(
        "data/synthetic/process_documentation.txt", VAR_NAMES
    )
    knowledge_graph = LLMCausalExtractor.pairs_to_graph(knowledge_pairs, VAR_NAMES)
    print(f"  知识因果图: {knowledge_graph.number_of_edges()} 条边")

    # Step B: 数据通道 — 用历史正常数据跑因果发现
    print("[2/3] 数据驱动因果发现...")
    sim = SyntheticProcessSimulator(seed=42)
    df_normal = sim.simulate(n_steps=500)
    cd = CausalDiscovery(VAR_NAMES)
    data_graph = cd.discover_pcmciplus(df_normal[VAR_NAMES].iloc[:500], tau_max=5)
    print(f"  数据因果图: {data_graph.number_of_edges()} 条边")

    # Step C: 双通道融合 → 最终因果图（大脑）
    print("[3/3] 双通道融合...")
    fusion = CausalGraphFusion()
    kp_formatted = [
        {"cause": p.cause, "effect": p.effect, "confidence": p.confidence,
         "mechanism": p.mechanism, "time_lag": 0}
        for p in knowledge_pairs
    ]
    fused_graph = fusion.fuse(knowledge_graph, data_graph, kp_formatted)
    print(f"  融合因果图: {fused_graph.number_of_edges()} 条边")

    # 初始化分析器
    analyzer = RootCauseAnalyzer(fused_graph)
    analyzer.set_normal_ranges(df_normal[VAR_NAMES].iloc[:300])

    cf_engine = CounterfactualEngine(fused_graph)
    coeffs = {(e.cause, e.effect): e.coefficient for e in CAUSAL_GRAPH_TRUTH}
    cf_engine.set_manual_coefficients(coeffs)

    print(f"\n✅ 系统初始化完成！因果图已加载，可以开始诊断。")
    return analyzer, cf_engine, fused_graph


# ============================================================
# 场景1: 离线批量分析 — 用历史故障数据跑一遍
# ============================================================
def scenario1_batch_analysis(analyzer, cf_engine):
    """
    场景1: 拿一段历史故障数据，分析整个过程，找出故障的起始时刻和演化过程

    适用场景:
      - 事故复盘: "上个月那次停机的根本原因是什么？"
      - 定期巡检: 每天对前一天的数据做一遍分析
    """
    print("\n" + "=" * 60)
    print("场景1: 离线批量分析")
    print("=" * 60)

    # 加载一段故障数据
    sim = SyntheticProcessSimulator(seed=42)
    df, meta = sim.generate_fault_dataset(
        n_normal=100, n_fault=200, fault_name="FAULT_COOLING_VALVE_STUCK"
    )

    # 逐点分析，记录根因随时间变化
    results = []
    for t in range(0, len(df), 20):  # 每20步分析一次
        observation = {v: df.iloc[t][v] for v in VAR_NAMES}
        reports = analyzer.analyze(observation, top_k=1)

        if reports and reports[0].root_causes:
            top_rc = reports[0].root_causes[0]
            results.append({
                "time": t,
                "anomaly_count": len(reports),
                "top_root_cause": top_rc.variable,
                "score": top_rc.score,
                "abnormal_var": reports[0].abnormal_variable,
            })

    if results:
        print(f"\n分析完成！共检测 {len(results)} 个时间窗口的异常。")
        print(f"\nTop根因统计:")
        from collections import Counter
        rc_counts = Counter(r["top_root_cause"] for r in results)
        for rc, count in rc_counts.most_common(3):
            print(f"  {rc}: {count} 次 (占比 {count/len(results):.0%})")

        first_anomaly = results[0]
        print(f"\n首次异常出现在 t={first_anomaly['time']}")
        print(f"  根因: {first_anomaly['top_root_cause']}")
        print(f"  异常变量: {first_anomaly['abnormal_var']}")

    return results


# ============================================================
# 场景2: 单步诊断 — "现在这个读数，是什么问题？"
# ============================================================
def scenario2_single_diagnosis(analyzer, cf_engine):
    """
    场景2: 一句话诊断 — 输入当前读数，输出根因+建议

    这是最直观的使用方式，相当于一个"AI诊断医生"

    适用场景:
      - 操作员看到报警，想知道根因
      - 交接班时快速了解当前状态
      - 新人培训时学习故障因果关系
    """
    print("\n" + "=" * 60)
    print("场景2: 单步诊断")
    print("=" * 60)

    # 模拟：操作员发现 DCS 上几个参数不对劲，手动输入查询
    # 实际使用中，这些值从传感器/PLC/DCS 自动获取
    current_readings = {
        "Feed_Flow": 101.0,
        "Feed_Conc": 0.84,
        "CW_Inlet_Temp": 25.5,
        "CW_Valve": 22.0,        # ⚠ 异常低（正常约60）
        "Reactor_Temp": 192.0,   # ⚠ 异常高
        "Reactor_Press": 3.82,   # ⚠ 异常高
        "CW_Flow": 118.0,        # ⚠ 异常低
        "Reaction_Rate": 1.48,   # ⚠ 异常高
        "Product_Conc": 0.57,    # ⚠ 异常低
        "Byproduct_Conc": 0.16,  # ⚠ 异常高
        "HX_Outlet_Temp": 46.0,  # ⚠ 异常高
        "Energy_Index": 1.85,    # ⚠ 异常高
    }

    # 一键诊断
    reports = analyzer.analyze(current_readings, top_k=3)

    print("\n📊 诊断结果:")
    print("-" * 40)

    for report in reports:
        print(f"\n⚠  {report.abnormal_variable} = {report.observed_value:.1f}"
              f" (正常: {report.normal_range[0]:.1f} ~ {report.normal_range[1]:.1f})")

        if report.root_causes:
            print(f"   🔍 根因排序:")
            for i, rc in enumerate(report.root_causes):
                print(f"      #{i+1} {rc.variable} (置信度: {rc.score:.1%})")
                print(f"         因果链: {' → '.join(rc.path)}")

        if report.recommended_actions:
            print(f"   💡 处置建议:")
            for action in report.recommended_actions[:3]:
                print(f"      • {action}")

    # 反事实: 如果修复阀门会怎样
    print(f"\n🔄 反事实推理:")
    r = cf_engine.what_if(
        current_readings,
        {"CW_Valve": 60.0},  # 干预：恢复阀门到正常开度
        "Reactor_Temp",
        analyzer.normal_ranges,
    )
    print(f"   干预方案: 恢复 CW_Valve → 60.0%")
    print(f"   Reactor_Temp: {r.factual_value:.1f} → {r.counterfactual_value:.1f}°C")
    print(f"   预期改善: {r.improvement:+.1f}°C")

    return reports


# ============================================================
# 场景3: 实时监控 — 模拟逐秒读取传感器，检测到异常就报警
# ============================================================
def scenario3_realtime_monitoring(analyzer, cf_engine):
    """
    场景3: 模拟实时监控（核心生产场景）

    每来一个数据点就分析一次，检测到异常立即输出根因

    实际部署时：
      df.iloc[t] 替换为 read_sensor() → 从PLC/MQTT/OPC-UA读取
    """
    print("\n" + "=" * 60)
    print("场景3: 实时监控模拟")
    print("=" * 60)

    # 加载故障数据（实际环境换成实时数据流）
    sim = SyntheticProcessSimulator(seed=42)
    df, meta = sim.generate_fault_dataset(
        n_normal=100, n_fault=200, fault_name="FAULT_COOLING_VALVE_STUCK"
    )
    print(f"模拟数据流: {len(df)} 个时间步")
    print(f"  t=0~99:   正常运行")
    print(f"  t=100~299: 冷却水阀门卡滞故障")
    print(f"\n开始监控...\n")

    alert_count = 0
    last_root_cause = None

    for t in range(len(df)):
        # ═══════════════════════════════════════════════
        # 实际环境：这里换成传感器读数
        # observation = read_sensors()  # 从PLC/SCADA读取
        # ═══════════════════════════════════════════════
        observation = {v: float(df.iloc[t][v]) for v in VAR_NAMES}
        reports = analyzer.analyze(observation, top_k=1)

        if reports and reports[0].root_causes:
            current_root_cause = reports[0].root_causes[0].variable

            # 只在根因变化时报警（避免重复告警轰炸）
            if current_root_cause != last_root_cause:
                alert_count += 1
                last_root_cause = current_root_cause

                print(f"\n🚨 t={t} 报警!")
                rc = reports[0].root_causes[0]
                print(f"   异常: {reports[0].abnormal_variable} = "
                      f"{observation[reports[0].abnormal_variable]:.1f}")
                print(f"   根因: {rc.variable} (评分: {rc.score:.1%})")
                print(f"   路径: {' → '.join(rc.path)}")

                # 给出处置建议
                if rc.variable in analyzer.normal_ranges:
                    lo, hi = analyzer.normal_ranges[rc.variable]
                    target_val = (lo + hi) / 2
                    result = cf_engine.what_if(
                        observation,
                        {rc.variable: target_val},
                        reports[0].abnormal_variable,
                    )
                    print(f"   建议: 恢复 {rc.variable} → {target_val:.1f}, "
                          f"预期改善 {result.improvement:+.1f}")

    print(f"\n监控结束。共触发 {alert_count} 次报警。")
    return alert_count


# ============================================================
# 主程序
# ============================================================
if __name__ == "__main__":
    # 1. 初始化系统（只做一次）
    analyzer, cf_engine, fused_graph = initialize_system()

    # 2. 选择运行场景
    print("\n" + "=" * 60)
    print("选择运行场景:")
    print("  1 = 离线批量分析（事故复盘）")
    print("  2 = 单步诊断（一问一答）")
    print("  3 = 实时监控（逐秒分析）")
    print("  0 = 全部运行")
    print("=" * 60)

    # 默认跑场景2（最有演示价值）
    scenario2_single_diagnosis(analyzer, cf_engine)
