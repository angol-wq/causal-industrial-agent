"""
因果增强工业智能体 V2 — 全链路自主循环进化

新增能力:
  文献自主迭代: 爬虫定向抓取 → AI精读提取 → 因果知识库增长
  仿真自进化:   现场/文献数据双向回喂 → 参数自动微调 → 模型迭代
  文献自测:     仿真 vs 文献预期 → 验证模型 → 回馈进化

主循环:
  ┌──────────────────────────────────────────────┐
  │          全链路自主循环进化主引擎               │
  │                                              │
  │  ① 文献爬取 → ② AI精读 → ③ 知识库更新       │
  │       ↑                          ↓           │
  │  ⑥ 自测验证 ← ⑤ 仿真重跑 ← ④ 参数进化      │
  │       │                          │           │
  │       └──── 通过? → 否 → 回到① ──┘           │
  │              → 是 → 部署更新                  │
  └──────────────────────────────────────────────┘

用法:
  agent = CausalAgentV2(api_key="...")
  agent.run_evolution_cycle()  # 执行一个全链路进化周期
"""

import sys, os, json, time
from typing import Dict, List, Optional
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.literature_crawler import LiteratureCrawler
from src.literature_extractor import LiteratureExtractor, NewKnowledge
from src.simulation_evolution import (
    SimulationEvolutionEngine, SimParameters, EvolutionRecord
)
from src.self_validator import SelfValidator
from src.synthetic_data_generator import (
    SyntheticProcessSimulator, VAR_NAMES, CAUSAL_GRAPH_TRUTH, FAULT_MODES,
)
from src.causal_discovery import CausalDiscovery
from src.graph_fusion import CausalGraphFusion
from src.root_cause_analysis import RootCauseAnalyzer
from src.counterfactual import CounterfactualEngine

import networkx as nx
import numpy as np


class CausalAgentV2:
    """因果增强工业智能体 V2 — 具备自主进化能力"""

    def __init__(self, api_key: str = None):
        # 核心模块
        self.crawler = LiteratureCrawler()
        self.extractor = LiteratureExtractor(api_key=api_key)
        self.evolution_engine = SimulationEvolutionEngine()
        self.validator = SelfValidator()

        # 仿真器
        self.simulator = SyntheticProcessSimulator(seed=42)
        self.evolution_engine.set_simulator(self.simulator)

        # 因果图（动态增长）
        self.causal_graph = nx.DiGraph()
        self.knowledge_base: List[Dict] = []  # 累积知识

        # 进化状态
        self.evolution_generation = 0
        self.evolution_log = []

    # ================================================================
    # 全链路自主循环进化
    # ================================================================
    def run_evolution_cycle(self,
                            keyword_set: List[str] = None,
                            field_data: Dict = None,
                            max_new_papers: int = 20,
                            convergence_threshold: float = 0.001,
                            max_generations: int = 10
                            ) -> Dict:
        """
        执行一个完整的全链路自主进化周期

        周期流程:
          Step 1: 文献爬取 — 抓取最新文献
          Step 2: AI精读 — 提取因果知识+仿真参数
          Step 3: 知识库合并 — 新旧因果边去重融合
          Step 4: 参数进化 — 文献/现场数据双向校准
          Step 5: 仿真重跑 — 用新参数生成仿真数据
          Step 6: 自测验证 — 仿真 vs 文献基准
          Step 7: 判定 — 通过则更新，否则回到Step 1
        """
        print("\n" + "█" * 70)
        print("█  因果增强工业智能体 V2 — 全链路自主循环进化")
        print(f"█  第 {self.evolution_generation + 1} 周期启动")
        print("█" * 70)

        cycle_report = {
            "cycle_id": self.evolution_generation + 1,
            "timestamp": None,
            "steps": {},
            "converged": False,
            "knowledge_growth": {},
            "final_validation_score": 0.0,
        }

        # ============================================================
        # Step 1+2: 文献爬取 + AI精读
        # ============================================================
        print("\n[Step 1+2] 文献获取与精读...")
        papers = self.crawler.search_all(keyword_set or
                                          self.crawler.DEFAULT_KEYWORDS[:5])
        print(f"  抓取: {len(papers)} 篇")

        new_knowledge = self.extractor.extract_from_papers(
            papers, self.causal_graph
        )
        print(f"  新因果边: {len(new_knowledge.causal_edges)}")
        print(f"  新故障场景: {len(new_knowledge.fault_scenarios)}")
        print(f"  新仿真参数: {len(new_knowledge.sim_params)}")

        cycle_report["steps"]["literature"] = {
            "papers_found": len(papers),
            "new_causal_edges": len(new_knowledge.causal_edges),
            "new_fault_scenarios": len(new_knowledge.fault_scenarios),
        }

        # ============================================================
        # Step 3: 知识库合并
        # ============================================================
        print("\n[Step 3] 知识库更新...")
        before_count = self.causal_graph.number_of_edges()
        added_count = self._merge_knowledge(new_knowledge)
        after_count = self.causal_graph.number_of_edges()

        print(f"  因果图: {before_count} → {after_count} 条边 "
              f"(+{added_count} 新增)")
        print(f"  知识库: {len(self.knowledge_base)} 条记录")

        cycle_report["knowledge_growth"] = {
            "edges_before": before_count,
            "edges_after": after_count,
            "new_edges": added_count,
        }

        # ============================================================
        # Step 4: 仿真参数进化
        # ============================================================
        print("\n[Step 4] 仿真参数进化...")

        # 通道A: 文献数据 → 扩展参数库
        if new_knowledge.sim_params:
            self.evolution_engine.expand_from_literature(
                {}, new_knowledge.sim_params,
                description=f"文献通道-第{self.evolution_generation+1}周期"
            )

        # 通道B: 现场数据 → 反向校准
        if field_data:
            self.evolution_engine.calibrate_from_field_data(
                np.array(field_data.get("data", [])),
                field_data.get("variables", VAR_NAMES[:6]),
                field_data.get("observation", {}),
                description=f"现场数据-第{self.evolution_generation+1}周期"
            )

        # ============================================================
        # Step 5: 仿真重跑
        # ============================================================
        print("\n[Step 5] 用进化后参数重新仿真...")
        sim_results = self._rerun_simulation()

        cycle_report["steps"]["evolution"] = {
            "params_updated": len(self.evolution_engine.params.to_dict()),
            "evolution_generation": self.evolution_engine.evolution_generation,
        }

        # ============================================================
        # Step 6: 文献自测
        # ============================================================
        print("\n[Step 6] 文献自测验证...")
        validation_report = self.validator.run_full_validation(
            causal_graph=self.causal_graph,
            simulation_output=sim_results.get("output", {}),
            simulation_fault_trace=sim_results.get("fault_trace", {}),
            literature_knowledge=new_knowledge,
        )

        cycle_report["final_validation_score"] = validation_report["overall_score"]

        # ============================================================
        # Step 7: 判定
        # ============================================================
        cycle_report["converged"] = validation_report["overall_score"] >= 0.75
        cycle_report["timestamp"] = str(time.time())

        self.evolution_generation += 1
        self.evolution_log.append(cycle_report)

        # 保存周期报告
        os.makedirs("data/evolution", exist_ok=True)
        report_path = f"data/evolution/cycle_{self.evolution_generation:04d}.json"
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(cycle_report, f, indent=2, ensure_ascii=False)

        print(f"\n{'='*60}")
        print(f"周期 {self.evolution_generation} 完成")
        print(f"  验证评分: {validation_report['overall_score']:.1%}")
        print(f"  收敛: {'✅ 是' if cycle_report['converged'] else '❌ 否'}")
        if not cycle_report["converged"]:
            print(f"  建议: 运行下一周期以改进模型")
        print(f"  报告: {report_path}")

        return cycle_report

    # ================================================================
    # 自动多周期进化
    # ================================================================
    def auto_evolve_until_converge(self, max_cycles: int = 10) -> Dict:
        """重复执行进化周期直到收敛"""
        final_report = None

        for cycle in range(max_cycles):
            report = self.run_evolution_cycle(max_new_papers=20)
            final_report = report
            if report["converged"]:
                print(f"\n✅ 第{cycle+1}周期收敛，进化完成！")
                break
            print(f"\n⏳ 第{cycle+1}周期未收敛，继续下一周期...")
            time.sleep(2)  # 礼貌间隔

        # 生成总结
        self._save_evolution_summary()
        return final_report

    # ================================================================
    # 内部方法
    # ================================================================
    def _merge_knowledge(self, new_knowledge: NewKnowledge) -> int:
        """将新知识合并到因果图中，去重"""
        added = 0
        existing_edges = set(self.causal_graph.edges())

        for edge in new_knowledge.causal_edges:
            cause = edge.get("cause", "")
            effect = edge.get("effect", "")
            if (cause, effect) not in existing_edges:
                self.causal_graph.add_edge(
                    cause, effect,
                    mechanism=edge.get("mechanism", ""),
                    confidence=edge.get("confidence", 0.6),
                    source="literature_extracted",
                )
                added += 1

        # 保存到知识库
        self.knowledge_base.extend(new_knowledge.causal_edges)

        return added

    def _rerun_simulation(self) -> Dict:
        """用当前进化后的参数重新运行仿真"""
        # 更新仿真器参数
        params = self.evolution_engine.params
        # 将进化后的参数注入仿真器
        self.simulator.noise_level = params.noise_level

        # 运行正常工况
        df_normal = self.simulator.simulate(n_steps=500, fault_config=None)

        # 运行故障工况
        fault_results = {}
        for fault_name in list(FAULT_MODES.keys())[:3]:
            df, meta = self.simulator.generate_fault_dataset(
                n_normal=100, n_fault=200, fault_name=fault_name
            )
            fault_results[fault_name] = {
                "data": df,
                "root_cause": meta.get("root_cause", ""),
            }

        return {
            "output": {v: float(df_normal[v].mean()) for v in VAR_NAMES},
            "fault_trace": fault_results,
            "params_used": params.to_dict(),
        }

    def _save_evolution_summary(self):
        """保存进化总结报告"""
        summary = {
            "total_cycles": self.evolution_generation,
            "final_causal_graph_edges": self.causal_graph.number_of_edges(),
            "knowledge_base_size": len(self.knowledge_base),
            "parameter_evolution": self.evolution_engine.get_evolution_summary(),
            "cycle_history": self.evolution_log,
        }
        path = "data/evolution/evolution_summary.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        print(f"\n进化总结已保存: {path}")

    def get_status(self) -> Dict:
        """获取当前状态"""
        return {
            "agent_version": "2.0",
            "evolution_generation": self.evolution_generation,
            "causal_graph_size": self.causal_graph.number_of_edges(),
            "knowledge_base_size": len(self.knowledge_base),
            "current_params": self.evolution_engine.params.to_dict(),
            "last_validation_score": (
                self.evolution_log[-1]["final_validation_score"]
                if self.evolution_log else 0.0
            ),
        }


if __name__ == "__main__":
    print("因果增强工业智能体 V2")
    print("全链路自主循环进化引擎就绪")

    agent = CausalAgentV2()
    print(f"状态: {json.dumps(agent.get_status(), indent=2, ensure_ascii=False)}")

    print("\n调用 run_evolution_cycle() 执行一个完整进化周期")
    print("调用 auto_evolve_until_converge() 自动进化至收敛")
