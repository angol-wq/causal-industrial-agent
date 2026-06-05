"""
预计算脚本 — 在部署前运行一次，生成所有缓存数据
Streamlit Cloud 启动时直接加载缓存，秒开

运行: python precompute.py
输出: cache/ 目录下的预计算数据
"""

import sys, os, json, pickle
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import pandas as pd
import networkx as nx

from src.synthetic_data_generator import (
    SyntheticProcessSimulator, CAUSAL_GRAPH_TRUTH, VAR_NAMES,
    FAULT_MODES, ROOT_VARS, INTERMEDIATE_VARS, OUTPUT_VARS,
    generate_process_documentation,
)
from src.causal_discovery import CausalDiscovery
from src.llm_causal_extract import extract_from_synthetic_doc, LLMCausalExtractor
from src.graph_fusion import CausalGraphFusion


def clean_graph_for_serialize(g):
    """清理图中不可序列化的属性（numpy array等）"""
    clean = nx.DiGraph()
    clean.add_nodes_from(g.nodes())
    for u, v, data in g.edges(data=True):
        safe = {}
        for k, val in data.items():
            if isinstance(val, np.ndarray):
                continue  # 跳过numpy数组
            if isinstance(val, (str, int, float, bool)):
                safe[k] = val
        clean.add_edge(u, v, **safe)
    return clean

os.makedirs("cache", exist_ok=True)
os.makedirs("data/synthetic", exist_ok=True)

print("=" * 50)
print("预计算所有缓存数据...")
print("=" * 50)

# 1. 生成工艺文档和知识对
print("[1/5] 生成工艺文档和知识对...")
generate_process_documentation("data/synthetic")
knowledge_pairs = extract_from_synthetic_doc(
    "data/synthetic/process_documentation.txt", VAR_NAMES, use_llm=False
)
knowledge_graph = LLMCausalExtractor.pairs_to_graph(knowledge_pairs, VAR_NAMES)
with open("cache/knowledge_pairs.json", "w") as f:
    json.dump([{
        "cause": p.cause, "effect": p.effect,
        "direction": p.direction, "mechanism": p.mechanism,
        "confidence": p.confidence, "evidence": p.evidence,
        "time_lag": p.time_lag,
    } for p in knowledge_pairs], f)
print(f"  知识对: {len(knowledge_pairs)} 条")

# 2. 生成正常工况数据
print("[2/5] 生成正常工况数据...")
sim = SyntheticProcessSimulator(seed=42)
df_normal = sim.simulate(n_steps=1000, fault_config=None)
df_normal.to_csv("cache/normal_operation.csv", index=False)

# 正常范围
normal_range = {}
for col in VAR_NAMES:
    mean = df_normal[col].mean()
    std = df_normal[col].std()
    normal_range[col] = (float(mean - 3*std), float(mean + 3*std))
with open("cache/normal_range.json", "w") as f:
    json.dump(normal_range, f)
print(f"  正常范围: {len(normal_range)} 变量")

# 3. PCMCI+ 因果发现
print("[3/5] PCMCI+ 因果发现（可能需要半分钟）...")
cd = CausalDiscovery(VAR_NAMES)
data_subset = df_normal[VAR_NAMES].iloc[:500]
data_graph = cd.discover_pcmciplus(data=data_subset, tau_max=5)
nx.write_graphml(clean_graph_for_serialize(data_graph), "cache/data_graph.graphml")
print(f"  数据因果图: {data_graph.number_of_edges()} 条边")

# 4. 双通道融合
print("[4/5] 双通道融合...")
fusion = CausalGraphFusion()
kp_fmt = [{
    "cause": p.cause, "effect": p.effect, "confidence": p.confidence,
    "mechanism": p.mechanism, "evidence": p.evidence, "time_lag": 0
} for p in knowledge_pairs]
fused_graph = fusion.fuse(knowledge_graph, data_graph, kp_fmt)
nx.write_graphml(clean_graph_for_serialize(fused_graph), "cache/fused_graph.graphml")
with open("cache/fusion_dict.json", "w") as f:
    json.dump(fusion.to_dict(), f)
print(f"  融合图: {fused_graph.number_of_edges()} 条边")

# 5. 生成各故障数据
print("[5/5] 生成故障场景数据...")
fault_datasets = {}
for name, config in FAULT_MODES.items():
    df, meta = sim.generate_fault_dataset(n_normal=300, n_fault=500, fault_name=name)
    df.to_csv(f"cache/{name}.csv", index=False)
    fault_datasets[name] = {
        "description": config["description"],
        "root_cause": config["root_cause"],
        "causal_path": config["causal_path"],
    }
with open("cache/fault_metadata.json", "w") as f:
    json.dump(fault_datasets, f)

print(f"  故障场景: {len(fault_datasets)} 种")
print(f"\n✅ 全部缓存已生成到 cache/ 目录")
print("现在可以推送到 GitHub 并部署了。")
