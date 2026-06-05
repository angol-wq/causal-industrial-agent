"""
快速演示脚本 — 命令行版本
不需要Streamlit也能跑通完整流程，输出文本版结果

用法: python run_demo.py
"""

import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from src.pipeline import CausalAgentPipeline

if __name__ == "__main__":
    print("""
╔══════════════════════════════════════════════════════════════╗
║     因果增强工业智能体 - 快速演示                              ║
║     Causal Enhanced Industrial Agent                        ║
║     创智青山AI智能体创新大赛 · 技术挑战赛道                    ║
╚══════════════════════════════════════════════════════════════╝
""")

    pipeline = CausalAgentPipeline(data_dir="data/synthetic")
    pipeline.run_demo(fault_name="FAULT_COOLING_VALVE_STUCK", use_llm_api=False)
