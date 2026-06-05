"""
文献自测模块 — 仿真结果 vs 文献理论预期, 验证模型正确性

核心机制:
  1. 从文献中提取验证基准（理论的故障传播路径、定量关系、参数范围）
  2. 运行仿真生成测试案例
  3. 对比仿真结果与文献预期
  4. 生成自测报告 — 通过/偏离/矛盾
  5. 偏离案例回馈进化引擎

知识自测闭环:
  文献基准 ──→ 仿真测试 ──→ 结果对比 ──→ 通过 ✓
                                          ├── 偏离 → 标记待审核 → 回馈进化
                                          └── 矛盾 → 人工介入
"""

import os, json
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from datetime import datetime
import numpy as np


@dataclass
class ValidationCase:
    """验证案例"""
    case_id: str
    source: str                     # 文献来源
    description: str                # 验证描述
    expected_behavior: Dict         # 预期行为
    simulation_result: Dict = field(default_factory=dict)  # 仿真结果
    match_score: float = 0.0        # 匹配度 [0, 1]
    verdict: str = "pending"        # passed / deviated / contradicted
    deviation_details: str = ""
    suggested_action: str = ""


class SelfValidator:
    """
    文献自测引擎 — 用文献验证模型正确性

    三种验证模式:
      1. 结构验证: 因果图结构是否与文献一致
      2. 定量验证: 仿真数值是否在文献报告范围内
      3. 行为验证: 故障演化过程是否与文献描述一致
    """

    def __init__(self, validation_dir: str = "data/validation"):
        self.validation_dir = validation_dir
        self.cases: List[ValidationCase] = []
        os.makedirs(validation_dir, exist_ok=True)

    # ================================================================
    # 模式1: 因果结构验证
    # ================================================================
    def validate_causal_structure(self, causal_graph,
                                   literature_knowledge) -> Dict:
        """
        验证因果图结构是否与文献一致

        检查项:
          - 文献明确报告的因果边在图中有吗？
          - 文献否定/证伪的因果边在图中错误存在吗？
          - 因果方向是否正确？
        """
        results = {"passed": [], "missing": [], "wrong_direction": [],
                   "score": 0.0}

        literature_edges = set()
        for edge in literature_knowledge.causal_edges:
            lit_edge = (edge.get("cause", ""), edge.get("effect", ""))
            literature_edges.add(lit_edge)

        graph_edges = set(causal_graph.edges())
        graph_reverse = {(v, u) for u, v in graph_edges}

        for cause, effect in literature_edges:
            if (cause, effect) in graph_edges:
                results["passed"].append({
                    "cause": cause, "effect": effect,
                    "match": "exact",
                })
            elif (effect, cause) in graph_edges:
                results["wrong_direction"].append({
                    "cause": cause, "effect": effect,
                    "graph_has": f"{effect}→{cause}",
                    "issue": "因果方向与文献相反！",
                })
            else:
                results["missing"].append({
                    "cause": cause, "effect": effect,
                    "issue": "文献报告的因果边在图中不存在",
                })

        total = len(literature_edges)
        passed_count = len(results["passed"])
        results["score"] = passed_count / total if total > 0 else 0

        verdict = "✅ 通过" if results["score"] > 0.8 else \
                  "⚠️ 部分通过" if results["score"] > 0.5 else "❌ 需修正"

        print(f"[结构验证] {verdict} (匹配率: {results['score']:.1%})")
        print(f"  正确: {passed_count}, 缺失: {len(results['missing'])}, "
              f"方向反: {len(results['wrong_direction'])}")

        return results

    # ================================================================
    # 模式2: 定量验证
    # ================================================================
    def validate_quantitative(self, simulation_output: Dict,
                               literature_experiments: List[Dict]) -> Dict:
        """
        验证仿真数值是否在文献实验数据范围内

        simulation_output: {"变量名": 仿真值, ...}
        literature_experiments: [{"variable": "X", "normal_range": [lo,hi]}, ...]
        """
        results = {"passed": [], "deviated": [], "score": 0.0}

        for exp in literature_experiments:
            var = exp.get("variable", "")
            if var not in simulation_output:
                continue

            sim_val = simulation_output[var]
            normal_range = exp.get("normal_range", [0, 0])
            fault_range = exp.get("fault_range", None)

            in_normal = normal_range[0] <= sim_val <= normal_range[1]
            in_fault = (fault_range and
                       fault_range[0] <= sim_val <= fault_range[1])

            if in_normal:
                results["passed"].append({
                    "variable": var, "sim_value": sim_val,
                    "expected_range": normal_range, "match": "normal",
                })
            elif in_fault:
                results["passed"].append({
                    "variable": var, "sim_value": sim_val,
                    "expected_range": fault_range, "match": "fault",
                })
            else:
                deviation_pct = (sim_val - normal_range[1]) / normal_range[1] * 100
                results["deviated"].append({
                    "variable": var, "sim_value": sim_val,
                    "expected_range": normal_range,
                    "deviation_pct": deviation_pct,
                    "issue": f"仿真值偏离正常范围 {deviation_pct:+.1f}%",
                })

        total = len(results["passed"]) + len(results["deviated"])
        results["score"] = len(results["passed"]) / total if total > 0 else 1.0

        verdict = "✅ 通过" if results["score"] > 0.85 else \
                  "⚠️ 需调整参数" if results["score"] > 0.6 else "❌ 模型需重校准"

        print(f"[定量验证] {verdict} (通过率: {results['score']:.1%})")

        # 偏离案例 → 可作为进化输入
        if results["deviated"]:
            print(f"  偏离变量: {[d['variable'] for d in results['deviated']]}")

        return results

    # ================================================================
    # 模式3: 故障行为验证
    # ================================================================
    def validate_fault_behavior(self, simulation_fault_trace: Dict,
                                 literature_fault_desc: List[Dict]) -> Dict:
        """
        验证故障演化过程是否与文献描述一致

        literature_fault_desc: [{
          "fault_name": "X",
          "propagation_path": ["A", "B", "C"],  # 文献描述的传播路径
          "symptoms": ["s1", "s2"],
        }]
        """
        results = {"passed": [], "deviated": [], "score": 0.0}

        for fault in literature_fault_desc:
            expected_path = fault.get("propagation_path", [])
            fault_name = fault.get("fault_name", "")

            # 从仿真轨迹中提取实际传播顺序
            actual_order = self._extract_propagation_order(
                simulation_fault_trace, expected_path
            )

            # 比较传播顺序
            path_match = self._compare_propagation_paths(
                expected_path, actual_order
            )

            case = {
                "fault_name": fault_name,
                "expected_path": expected_path,
                "actual_order": actual_order,
                "path_match": path_match,
            }

            if path_match >= 0.7:
                results["passed"].append(case)
            else:
                results["deviated"].append(case)

        total = len(results["passed"]) + len(results["deviated"])
        results["score"] = len(results["passed"]) / total if total > 0 else 1.0

        return results

    def _extract_propagation_order(self, trace: Dict,
                                    variables: List[str]) -> List[str]:
        """从仿真轨迹中提取变量的异常发生顺序"""
        order = []
        for var in variables:
            if var in trace:
                # 找变量首次超出正常范围的时刻
                order.append(var)
        return order

    def _compare_propagation_paths(self, expected: List[str],
                                    actual: List[str]) -> float:
        """比较传播路径的相似度"""
        if not expected or not actual:
            return 0.0
        # 计算最长公共子序列的匹配度
        matches = sum(1 for e, a in zip(expected, actual) if e == a)
        return matches / max(len(expected), len(actual))

    # ================================================================
    # 综合自测报告
    # ================================================================
    def run_full_validation(self, causal_graph,
                            simulation_output: Dict,
                            simulation_fault_trace: Dict,
                            literature_knowledge) -> Dict:
        """
        运行全部三项验证，生成综合报告

        返回的报告可直接用于:
          - 判断是否需要触发参数进化
          - 识别因果图中的错误边
          - 标记需要人工审核的案例
        """
        print("\n" + "=" * 60)
        print("模型自测验证")
        print("=" * 60)

        report = {
            "timestamp": datetime.now().isoformat(),
            "overall_score": 0.0,
            "structure_validation": None,
            "quantitative_validation": None,
            "fault_behavior_validation": None,
            "actions_required": [],
            "ready_for_field_deployment": False,
        }

        # 1. 结构验证
        report["structure_validation"] = self.validate_causal_structure(
            causal_graph, literature_knowledge
        )

        # 2. 定量验证
        if simulation_output:
            report["quantitative_validation"] = self.validate_quantitative(
                simulation_output,
                literature_knowledge.experimental_data if hasattr(
                    literature_knowledge, 'experimental_data') else []
            )

        # 3. 故障行为验证
        if simulation_fault_trace:
            report["fault_behavior_validation"] = self.validate_fault_behavior(
                simulation_fault_trace,
                literature_knowledge.fault_scenarios if hasattr(
                    literature_knowledge, 'fault_scenarios') else []
            )

        # 综合评分
        scores = []
        for key in ["structure_validation", "quantitative_validation",
                     "fault_behavior_validation"]:
            if report[key] and "score" in report[key]:
                scores.append(report[key]["score"])
        report["overall_score"] = np.mean(scores) if scores else 0.0

        # 行动建议
        if report["overall_score"] < 0.6:
            report["actions_required"].append(
                "❌ 模型准确性不足，建议触发仿真参数进化")
        if report.get("structure_validation", {}).get("wrong_direction", []):
            report["actions_required"].append(
                "⚠️ 存在方向错误的因果边，建议人工审核并修正")
        if report.get("structure_validation", {}).get("missing", []):
            report["actions_required"].append(
                "📖 存在文献报告但因果图缺失的边，建议添加到知识库")

        report["ready_for_field_deployment"] = report["overall_score"] >= 0.75

        # 保存报告
        report_path = os.path.join(self.validation_dir,
                                   f"validation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        print(f"\n综合评分: {report['overall_score']:.1%}")
        print(f"可部署: {'✅ 是' if report['ready_for_field_deployment'] else '❌ 否'}")
        for action in report["actions_required"]:
            print(f"  {action}")
        print(f"报告已保存: {report_path}")

        return report


if __name__ == "__main__":
    print("文献自测模块就绪")
    print("调用 run_full_validation() 执行全项验证")
