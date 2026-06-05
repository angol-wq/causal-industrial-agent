"""
双通道因果图融合算法 ⭐ 核心创新模块

融合策略:
  通道1（知识驱动）: LLM从工艺文档提取因果对 → 初始因果图
  通道2（数据驱动）: PCMCI+从传感器数据发现因果结构 → 数据因果图

  融合层:
    1. 图结构对齐: 将两个图的节点/边映射到统一空间
    2. 冲突消解:
       - 知识有+数据有 → 双重验证, 最高置信度
       - 知识有+数据无 → 降权（可能是慢因果/非线性因果）
       - 知识无+数据有 → 标记"新发现"，需专家审核
       - 知识无+数据无 → 无边
    3. 置信度加权: 综合两个通道的证据强度

论文/答辩叙事:
  这不是简单的取并集，而是利用了知识通道和数据通道各自的互补优势:
  - 知识通道覆盖"慢因果"（如年久腐蚀）、"稀有事件因果"（如地震响应）
  - 数据通道覆盖"快因果"（秒/分钟级动态）、"数据隐含因果"（工程师经验未总结的）
"""

import json
import networkx as nx
import numpy as np
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass, field
from enum import Enum


class EdgeSource(Enum):
    """因果边的来源"""
    DUAL_VERIFIED = "dual_verified"       # 两通道都确认
    KNOWLEDGE_ONLY = "knowledge_only"     # 仅知识通道
    DATA_ONLY = "data_discovered"         # 仅数据通道（新发现！）


@dataclass
class FusedCausalEdge:
    """融合后的因果边"""
    cause: str
    effect: str
    source: EdgeSource
    confidence: float          # 综合置信度 [0, 1]
    knowledge_confidence: float = 0.0   # 知识通道置信度
    data_significance: float = 0.0      # 数据通道显著性 (1 - p_value)
    mechanism: str = ""         # 物理机制说明（来自知识）
    evidence: str = ""          # 文档证据（来自知识）
    time_lag: int = 0
    note: str = ""


class CausalGraphFusion:
    """
    双通道因果图融合引擎

    用法:
      fusion = CausalGraphFusion()
      fused_graph = fusion.fuse(knowledge_graph, data_graph, knowledge_pairs)
      report = fusion.generate_fusion_report()
    """

    def __init__(self,
                 knowledge_weight: float = 0.5,
                 data_weight: float = 0.5,
                 knowledge_only_penalty: float = 0.3,
                 data_only_penalty: float = 0.4):
        """
        Args:
            knowledge_weight: 知识通道总体权重
            data_weight: 数据通道总体权重
            knowledge_only_penalty: 仅知识通道出现的边的惩罚系数
            data_only_penalty: 仅数据通道出现的边的惩罚系数
        """
        self.knowledge_weight = knowledge_weight
        self.data_weight = data_weight
        self.knowledge_only_penalty = knowledge_only_penalty
        self.data_only_penalty = data_only_penalty

        self.fused_graph: Optional[nx.DiGraph] = None
        self.fused_edges: List[FusedCausalEdge] = []
        self.fusion_stats: Dict = {}

    def fuse(self,
             knowledge_graph: nx.DiGraph,
             data_graph: nx.DiGraph,
             knowledge_pairs: List[Dict] = None
             ) -> nx.DiGraph:
        """
        融合知识因果图和数据因果图

        Args:
            knowledge_graph: 从工艺文档提取的因果图（LLM提取）
            data_graph: 从PCMCI+等算法发现的因果图
            knowledge_pairs: LLM提取的原始因果对列表（含机制、证据等）

        Returns:
            fused_graph: 融合后的因果图
        """
        if knowledge_pairs is None:
            knowledge_pairs = []

        # Step 0: 收集所有节点
        all_nodes = set(knowledge_graph.nodes()) | set(data_graph.nodes())

        # 为知识因果对建立快速检索索引
        kp_index = {}
        for kp in knowledge_pairs:
            key = (kp.get("cause", ""), kp.get("effect", ""))
            kp_index[key] = kp

        # Step 1: 收集所有候选因果边
        all_edges = set()
        knowledge_edges = set()
        data_edges = set()

        for u, v in knowledge_graph.edges():
            knowledge_edges.add((u, v))
            all_edges.add((u, v))

        for u, v in data_graph.edges():
            data_edges.add((u, v))
            all_edges.add((u, v))

        # Step 2: 对每条候选边进行融合判定
        fused_graph = nx.DiGraph()
        for node in all_nodes:
            fused_graph.add_node(node)

        self.fused_edges = []

        for cause, effect in all_edges:
            in_knowledge = (cause, effect) in knowledge_edges
            in_data = (cause, effect) in data_edges

            kp = kp_index.get((cause, effect), {})
            k_confidence = float(kp.get("confidence", 0.6) if isinstance(kp.get("confidence"), (int, float)) else 0.6)

            # 数据通道: 从tigramite p-value推断显著性
            d_significance = 0.8  # 默认
            if in_data and data_graph.has_edge(cause, effect):
                edge_data = data_graph[cause][effect]
                d_significance = edge_data.get("significance", 0.8)

            # 判定来源 & 计算综合置信度
            if in_knowledge and in_data:
                source = EdgeSource.DUAL_VERIFIED
                confidence = (self.knowledge_weight * k_confidence +
                             self.data_weight * d_significance)
                note = "双重验证 — 最高置信度"
            elif in_knowledge and not in_data:
                source = EdgeSource.KNOWLEDGE_ONLY
                confidence = k_confidence * (1 - self.knowledge_only_penalty)
                note = ("数据通道未检出 — 可能原因: ①因果效应慢于数据窗口 "
                       "②非线性关系 ③数据不足 ④噪声掩盖")
            elif not in_knowledge and in_data:
                source = EdgeSource.DATA_ONLY
                confidence = d_significance * (1 - self.data_only_penalty)
                note = "数据驱动新发现 — 建议领域专家审核此因果假设"
            else:
                continue  # 不会到这里

            edge = FusedCausalEdge(
                cause=cause,
                effect=effect,
                source=source,
                confidence=round(min(confidence, 1.0), 4),
                knowledge_confidence=k_confidence,
                data_significance=d_significance,
                mechanism=kp.get("mechanism", ""),
                evidence=kp.get("evidence", ""),
                time_lag=int(kp.get("time_lag", 0)),
                note=note,
            )

            self.fused_edges.append(edge)

            # 添加到融合图
            fused_graph.add_edge(
                cause, effect,
                source=source.value,
                confidence=edge.confidence,
                mechanism=edge.mechanism,
                evidence=edge.evidence,
                time_lag=edge.time_lag,
                note=note,
            )

        # Step 3: 统计
        self.fused_graph = fused_graph
        self.fusion_stats = self._compute_stats()

        return fused_graph

    def _compute_stats(self) -> Dict:
        """计算融合统计信息"""
        stats = {
            "total_edges": len(self.fused_edges),
            "dual_verified": sum(1 for e in self.fused_edges
                                if e.source == EdgeSource.DUAL_VERIFIED),
            "knowledge_only": sum(1 for e in self.fused_edges
                                 if e.source == EdgeSource.KNOWLEDGE_ONLY),
            "data_discovered": sum(1 for e in self.fused_edges
                                  if e.source == EdgeSource.DATA_ONLY),
            "avg_confidence": np.mean([e.confidence for e in self.fused_edges]),
            "high_confidence_edges": len([e for e in self.fused_edges if e.confidence >= 0.8]),
            "medium_confidence_edges": len([e for e in self.fused_edges if 0.5 <= e.confidence < 0.8]),
            "low_confidence_edges": len([e for e in self.fused_edges if e.confidence < 0.5]),
        }
        return stats

    def get_high_confidence_graph(self, threshold: float = 0.7) -> nx.DiGraph:
        """获取只包含高置信度边的因果图"""
        if self.fused_graph is None:
            raise ValueError("请先运行fuse()")

        G = nx.DiGraph()
        for node in self.fused_graph.nodes():
            G.add_node(node)

        for u, v, data in self.fused_graph.edges(data=True):
            if data.get("confidence", 0) >= threshold:
                G.add_edge(u, v, **data)

        return G

    def get_new_discoveries(self) -> List[FusedCausalEdge]:
        """获取数据通道新发现的因果边（知识库中不存在的）"""
        return [e for e in self.fused_edges if e.source == EdgeSource.DATA_ONLY]

    def generate_fusion_report(self) -> str:
        """生成融合报告（Markdown格式）"""
        if self.fusion_stats is None:
            return "尚未执行融合"

        lines = [
            "# 因果图融合报告",
            "",
            "## 统计概览",
            f"- 总因果边数: {self.fusion_stats['total_edges']}",
            f"- 双重验证边: {self.fusion_stats['dual_verified']}",
            f"- 仅知识通道: {self.fusion_stats['knowledge_only']}",
            f"- 数据新发现: {self.fusion_stats['data_discovered']}",
            f"- 平均置信度: {self.fusion_stats['avg_confidence']:.3f}",
            f"- 高置信度边(≥0.8): {self.fusion_stats['high_confidence_edges']}",
            f"- 中置信度边(0.5-0.8): {self.fusion_stats['medium_confidence_edges']}",
            f"- 低置信度边(<0.5): {self.fusion_stats['low_confidence_edges']}",
            "",
            "## 数据驱动新发现",
        ]

        discoveries = self.get_new_discoveries()
        if discoveries:
            for i, d in enumerate(discoveries, 1):
                lines.append(f"{i}. **{d.cause} → {d.effect}** "
                           f"(置信度: {d.confidence:.3f})")
                lines.append(f"   - {d.note}")
        else:
            lines.append("（无）")

        lines.extend([
            "",
            "## 高置信度因果边",
        ])
        high_edges = sorted(
            [e for e in self.fused_edges if e.confidence >= 0.8],
            key=lambda x: x.confidence, reverse=True
        )
        for i, e in enumerate(high_edges[:20], 1):
            lines.append(
                f"{i}. **{e.cause} → {e.effect}** "
                f"[{e.source.value}] 置信度={e.confidence:.3f}"
            )
            if e.mechanism:
                lines.append(f"   机制: {e.mechanism}")

        return "\n".join(lines)

    def to_dict(self) -> Dict:
        """序列化为字典，供前端可视化"""
        return {
            "nodes": list(self.fused_graph.nodes()),
            "edges": [
                {
                    "source": e.cause,
                    "target": e.effect,
                    "source_type": e.source.value,
                    "confidence": e.confidence,
                    "mechanism": e.mechanism,
                    "time_lag": e.time_lag,
                    "note": e.note,
                }
                for e in self.fused_edges
            ],
            "stats": self.fusion_stats,
        }


if __name__ == "__main__":
    # 测试双通道融合
    print("=" * 60)
    print("双通道因果图融合测试")
    print("=" * 60)

    # 模拟知识通道结果
    kg = nx.DiGraph()
    kg.add_edge("CW_Valve", "CW_Flow", confidence=0.95)
    kg.add_edge("CW_Flow", "Reactor_Temp", confidence=0.90)
    kg.add_edge("Feed_Flow", "Reactor_Press", confidence=0.85)
    kg.add_edge("Reactor_Temp", "Reaction_Rate", confidence=0.95)

    knowledge_pairs = [
        {"cause": "CW_Valve", "effect": "CW_Flow", "confidence": 0.95,
         "mechanism": "阀门开度 → 冷却水流量", "evidence": "操作手册2.2节"},
        {"cause": "CW_Flow", "effect": "Reactor_Temp", "confidence": 0.90,
         "mechanism": "冷却水 → 反应器温度", "evidence": "操作手册2.3节"},
        {"cause": "Feed_Flow", "effect": "Reactor_Press", "confidence": 0.85,
         "mechanism": "进料 → 反应器压力", "evidence": "操作手册2.1节"},
        {"cause": "Reactor_Temp", "effect": "Reaction_Rate", "confidence": 0.95,
         "mechanism": "温度 → 反应速率(Arrhenius)", "evidence": "操作手册2.3节"},
    ]

    # 模拟数据通道结果
    dg = nx.DiGraph()
    dg.add_edge("CW_Valve", "CW_Flow", significance=0.95)
    dg.add_edge("CW_Flow", "Reactor_Temp", significance=0.88)
    dg.add_edge("Feed_Flow", "Reactor_Press", significance=0.91)
    # 数据通道额外发现了一个边
    dg.add_edge("CW_Inlet_Temp", "Reactor_Temp", significance=0.82)
    dg.add_edge("Reactor_Temp", "Reaction_Rate", significance=0.90)

    # 融合
    fusion = CausalGraphFusion()
    fused = fusion.fuse(kg, dg, knowledge_pairs)

    print(fusion.generate_fusion_report())

    # 展示新发现
    print("\n[数据新发现]:")
    for d in fusion.get_new_discoveries():
        print(f"  ⚡ {d.cause} → {d.effect} (conf={d.confidence:.3f}) — {d.note}")
