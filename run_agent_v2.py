"""
因果增强工业智能体 V2 — 完整使用指南

三大模式:
  [1] 快速诊断模式 — 加载现有因果图，输入观测数据，秒级返回根因
  [2] 文献进化模式 — 爬取最新文献 → AI精读 → 更新因果图 → 自测验证
  [3] 全链路自主模式 — 诊断+进化同时运行，越用越准

用法:
  python run_agent_v2.py          # 交互式菜单
  python run_agent_v2.py 1        # 直接进入快速诊断模式
  python run_agent_v2.py 2        # 直接进入文献进化模式
  python run_agent_v2.py 3        # 直接进入全链路自主模式
"""

import sys, os, json, time
sys.path.insert(0, os.path.dirname(__file__))

from src.agent_v2 import CausalAgentV2
from src.synthetic_data_generator import (
    SyntheticProcessSimulator, VAR_NAMES, FAULT_MODES, CAUSAL_GRAPH_TRUTH
)
from src.root_cause_analysis import RootCauseAnalyzer
from src.counterfactual import CounterfactualEngine
from src.graph_fusion import CausalGraphFusion
from src.llm_causal_extract import extract_from_synthetic_doc, LLMCausalExtractor
import networkx as nx


def init_knowledge_graph():
    """从工艺文档构建初始因果图"""
    os.makedirs("data/synthetic", exist_ok=True)
    from src.synthetic_data_generator import generate_process_documentation
    generate_process_documentation("data/synthetic")

    pairs = extract_from_synthetic_doc(
        "data/synthetic/process_documentation.txt", VAR_NAMES
    )
    return LLMCausalExtractor.pairs_to_graph(pairs, VAR_NAMES)


# ================================================================
# 模式1: 快速诊断
# ================================================================
def mode_quick_diagnosis():
    """加载已有因果图，输入观测数据，秒级返回根因诊断"""
    print("\n" + "=" * 60)
    print("🔍 快速诊断模式")
    print("=" * 60)

    # 构建因果图
    print("[1/3] 加载因果图...")
    kg = init_knowledge_graph()
    print(f"  因果图: {kg.number_of_nodes()}节点, {kg.number_of_edges()}条边")

    # 初始化分析器
    print("[2/3] 初始化分析器...")
    sim = SyntheticProcessSimulator(seed=42)
    df_normal = sim.simulate(n_steps=500)
    analyzer = RootCauseAnalyzer(kg)
    analyzer.set_normal_ranges(df_normal[VAR_NAMES].iloc[:300])
    print(f"  正常范围: {len(analyzer.normal_ranges)}变量")

    # 默认运行全部10种故障，各取一条诊断结果
    fault_list = list(FAULT_MODES.items())
    print(f"\n[3/3] 运行诊断（{len(fault_list)}种故障）:")
    for idx, (fault_name, fault_config) in enumerate(fault_list[:5], 1):  # 显示前5种

        # 生成故障数据
        df, meta = sim.generate_fault_dataset(n_normal=300, n_fault=200,
                                               fault_name=fault_name)
        fault_data = df[df["fault"] == fault_name]
        # 取故障后期样本（漂移故障需要时间累积偏差）
        sample_idx = min(len(fault_data) - 1, int(len(fault_data) * 0.7))
        sample = fault_data.iloc[sample_idx]
        observation = {v: float(sample[v]) for v in VAR_NAMES}

        # 诊断
        reports = analyzer.analyze(observation, top_k=2)
        if reports:
            # 找第一个有根因的报告（跳过无上游祖先的根变量）
            best = None
            for r in reports:
                if r.root_causes:
                    best = r; break
            if not best:
                best = reports[0]  # 根变量自身是根因

            abnormal_count = len(reports)
            if best.root_causes:
                rc = best.root_causes[0]
                print(f"  [{idx}] {fault_config['name']}: "
                      f"{abnormal_count}个异常 → 根因={rc.variable} "
                      f"(评分:{rc.score:.3f}) 路径={'→'.join(rc.path[:3])}")
            else:
                print(f"  [{idx}] {fault_config['name']}: "
                      f"{abnormal_count}个异常 → 根因={best.abnormal_variable} "
                      f"(根变量，无更上游原因)")
        else:
            print(f"  [{idx}] {fault_config['name']}: 未检出异常")

    print(f"\n✅ 诊断完成。共分析 {len(fault_list[:5])} 种故障场景。")


# ================================================================
# 模式2: 文献进化
# ================================================================
def mode_literature_evolution():
    """爬取文献 → AI精读 → 更新因果图 → 自测验证"""
    print("\n" + "=" * 60)
    print("📚 文献进化模式")
    print("=" * 60)

    print("\n此模式将:")
    print("  1. 爬取钢铁/冶金相关SCI论文")
    print("  2. AI自动精读提取因果机理")
    print("  3. 新增因果关系自动合并到因果图")
    print("  4. 提取仿真参数并校准模型")
    print("  5. 用文献基准自测模型正确性")
    print("\n⚠ 需要联网，首次运行约1-2分钟")
    print("⚠ 如需LLM增强，请设置环境变量 ANTHROPIC_API_KEY")

    # 初始化V2 Agent
    api_key = os.environ.get("ANTHROPIC_API_KEY", None)
    agent = CausalAgentV2(api_key=api_key)

    # 先加载初始因果图
    print("\n[准备] 加载初始因果图...")
    kg = init_knowledge_graph()
    agent.causal_graph = kg
    print(f"  初始因果图: {kg.number_of_edges()}条边")

    # 执行进化
    report = agent.run_evolution_cycle()

    # 展示结果
    print("\n" + "=" * 40)
    print("📊 进化结果")
    print("=" * 40)
    print(f"  文献数: {report['steps'].get('literature', {}).get('papers_found', 0)}")
    print(f"  新因果边: {report['knowledge_growth'].get('new_edges', 0)}")
    print(f"  因果图: {report['knowledge_growth'].get('edges_before', 0)} → "
          f"{report['knowledge_growth'].get('edges_after', 0)} 条边")
    print(f"  自测评分: {report['final_validation_score']:.1%}")
    print(f"  收敛: {'✅ 是' if report['converged'] else '❌ 否'}")

    print(f"\n进化日志: data/evolution/")


# ================================================================
# 模式3: 全链路自主模式
# ================================================================
def mode_autonomous():
    """诊断 + 进化双线程，越用越准"""
    print("\n" + "=" * 60)
    print("🧬 全链路自主模式")
    print("=" * 60)

    print("\n此模式模拟真实运行场景:")
    print("  1. 系统启动时加载因果图")
    print("  2. 收到传感器数据 → 实时诊断")
    print("  3. 后台定期爬取文献 → 进化因果图")
    print("  4. 积累现场数据 → 校准仿真参数")
    print("  5. 每次进化后自测 → 通过则更新部署")
    print("\n⚠ 此模式会持续运行，按 Ctrl+C 停止")

    confirm = input("\n按回车开始: ").strip()

    api_key = os.environ.get("ANTHROPIC_API_KEY", None)
    agent = CausalAgentV2(api_key=api_key)
    kg = init_knowledge_graph()
    agent.causal_graph = kg

    sim = SyntheticProcessSimulator(seed=42)
    df_normal = sim.simulate(n_steps=500)
    analyzer = RootCauseAnalyzer(agent.causal_graph)
    analyzer.set_normal_ranges(df_normal[VAR_NAMES].iloc[:300])

    cycle = 0
    diagnosis_count = 0
    evolution_interval = 20  # 每20次诊断触发一次进化

    try:
        while True:
            cycle += 1

            # 模拟: 随机选一个故障场景作为"实时数据"
            import random
            fault_name = random.choice(list(FAULT_MODES.keys()))
            df, _ = sim.generate_fault_dataset(n_normal=100, n_fault=100,
                                                fault_name=fault_name)
            fault_data = df[df["fault"] == fault_name]
            sample = fault_data.iloc[50]
            observation = {v: float(sample[v]) for v in VAR_NAMES}

            # 诊断
            reports = analyzer.analyze(observation, top_k=1)
            diagnosis_count += 1

            if reports and reports[0].root_causes:
                rc = reports[0].root_causes[0]
                print(f"[诊断 #{diagnosis_count}] {fault_name}: "
                      f"根因={rc.variable} 评分={rc.score:.3f}")

            # 每N次诊断触发一次进化
            if diagnosis_count % evolution_interval == 0:
                print(f"\n{'~'*40}")
                print(f"触发进化周期 (累计{diagnosis_count}次诊断)")
                agent.run_evolution_cycle()
                # 更新分析器的因果图
                analyzer = RootCauseAnalyzer(agent.causal_graph)
                analyzer.set_normal_ranges(df_normal[VAR_NAMES].iloc[:300])
                print(f"{'~'*40}\n")

            time.sleep(0.1)  # 模拟实时间隔

    except KeyboardInterrupt:
        print(f"\n\n⏹ 停止。共执行 {diagnosis_count} 次诊断, "
              f"{agent.evolution_generation} 次进化")


# ================================================================
# 主菜单
# ================================================================
def main():
    print("""
╔══════════════════════════════════════════════════════════════╗
║  因果增强工业智能体 V2 — 全链路自主循环进化                    ║
║  Causal Enhanced Industrial Agent v2.0                      ║
╚══════════════════════════════════════════════════════════════╝
    """)

    mode = sys.argv[1] if len(sys.argv) > 1 else "1"

    if mode not in ("1", "2", "3"):
        print("用法: python run_agent_v2.py [1|2|3]")
        print("  1 = 快速诊断  2 = 文献进化  3 = 全链路自主")
        print("  默认运行模式1（快速诊断）")
        mode = "1"

    if mode == "1":
        mode_quick_diagnosis()
    elif mode == "2":
        confirm = os.environ.get("AUTO_RUN") or "y"
        if confirm.lower() != "n":
            mode_literature_evolution()
    elif mode == "3":
        mode_autonomous()


if __name__ == "__main__":
    main()
