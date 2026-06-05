"""
因果增强工业智能体 — Streamlit 演示界面

运行: streamlit run app.py
"""

import streamlit as st
import numpy as np
import pandas as pd
import networkx as nx
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from src.synthetic_data_generator import (
    SyntheticProcessSimulator, CAUSAL_GRAPH_TRUTH, VAR_NAMES,
    FAULT_MODES, ROOT_VARS, INTERMEDIATE_VARS, OUTPUT_VARS,
)
from src.causal_discovery import CausalDiscovery
from src.llm_causal_extract import extract_from_synthetic_doc, LLMCausalExtractor
from src.graph_fusion import CausalGraphFusion
from src.root_cause_analysis import RootCauseAnalyzer
from src.counterfactual import CounterfactualEngine

st.set_page_config(
    page_title="因果增强工业智能体",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# 初始化 Session State
# ============================================================
if "simulator" not in st.session_state:
    st.session_state.simulator = SyntheticProcessSimulator(seed=42)
if "pipeline_ready" not in st.session_state:
    st.session_state.pipeline_ready = False


@st.cache_resource
def build_causal_graph():
    """构建因果图（缓存以避免重复计算）"""
    sim = SyntheticProcessSimulator(seed=42)

    # 生成正常数据
    df_normal = sim.simulate(n_steps=500, fault_config=None)

    # 通道1: 知识驱动
    doc_path = "data/synthetic/process_documentation.txt"
    if not os.path.exists(doc_path):
        os.makedirs("data/synthetic", exist_ok=True)
        from src.synthetic_data_generator import generate_process_documentation
        generate_process_documentation("data/synthetic")

    knowledge_pairs = extract_from_synthetic_doc(doc_path, VAR_NAMES, use_llm=False)
    knowledge_graph = LLMCausalExtractor.pairs_to_graph(knowledge_pairs, VAR_NAMES)

    # 通道2: 数据驱动
    cd = CausalDiscovery(VAR_NAMES)
    data_subset = df_normal[VAR_NAMES].iloc[:500]
    data_graph = cd.discover_pcmciplus(data=data_subset, tau_max=5)

    # 融合
    fusion = CausalGraphFusion()
    kp_formatted = [
        {"cause": p.cause, "effect": p.effect, "confidence": p.confidence,
         "mechanism": p.mechanism, "evidence": p.evidence, "time_lag": 0}
        for p in knowledge_pairs
    ]
    fused_graph = fusion.fuse(knowledge_graph, data_graph, kp_formatted)

    # Ground truth
    truth_graph = nx.DiGraph()
    for edge in CAUSAL_GRAPH_TRUTH:
        truth_graph.add_edge(edge.cause, edge.effect)

    # 计算正常范围
    normal_range = {}
    for col in VAR_NAMES:
        mean = df_normal[col].mean()
        std = df_normal[col].std()
        normal_range[col] = (mean - 3 * std, mean + 3 * std)

    return {
        "sim": sim,
        "fused_graph": fused_graph,
        "truth_graph": truth_graph,
        "knowledge_graph": knowledge_graph,
        "data_graph": data_graph,
        "fusion": fusion,
        "normal_range": normal_range,
        "df_normal": df_normal,
    }


def plot_causal_graph(causal_graph, title="因果图", highlight_path=None):
    """用Plotly绘制交互式因果图"""
    pos = nx.spring_layout(causal_graph, k=2, iterations=50, seed=42)

    edge_x, edge_y = [], []
    edge_colors = []
    edge_texts = []

    for u, v, data in causal_graph.edges(data=True):
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

        source_type = data.get("source", "unknown")
        if source_type == "dual_verified":
            edge_colors.append("green")
        elif source_type == "knowledge_only":
            edge_colors.append("orange")
        elif source_type == "data_discovered":
            edge_colors.append("red")
        else:
            edge_colors.append("gray")

        conf = data.get("confidence", 0)
        edge_texts.append(f"{u} → {v}<br>置信度: {conf:.2f}")

    # 高亮路径
    highlight_edge_x, highlight_edge_y = [], []
    if highlight_path:
        for i in range(len(highlight_path) - 1):
            u, v = highlight_path[i], highlight_path[i + 1]
            if u in pos and v in pos:
                x0, y0 = pos[u]
                x1, y1 = pos[v]
                highlight_edge_x.extend([x0, x1, None])
                highlight_edge_y.extend([y0, y1, None])

    # 节点
    node_x, node_y = [], []
    node_colors = []
    node_texts = []
    node_sizes = []

    for node in causal_graph.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)

        if node in ROOT_VARS:
            node_colors.append("#FF6B6B")
            node_sizes.append(25)
        elif node in INTERMEDIATE_VARS:
            node_colors.append("#4ECDC4")
            node_sizes.append(20)
        elif node in OUTPUT_VARS:
            node_colors.append("#45B7D1")
            node_sizes.append(22)
        else:
            node_colors.append("#96CEB4")
            node_sizes.append(18)

        node_texts.append(node)

    fig = go.Figure()

    # 边
    fig.add_trace(go.Scatter(
        x=edge_x, y=edge_y,
        mode='lines',
        line=dict(width=1.5, color='rgba(150,150,150,0.6)'),
        hoverinfo='text',
        text=edge_texts,
        name='因果边',
    ))

    # 高亮路径
    if highlight_edge_x:
        fig.add_trace(go.Scatter(
            x=highlight_edge_x, y=highlight_edge_y,
            mode='lines',
            line=dict(width=4, color='red'),
            name='根因路径',
        ))

    # 节点
    fig.add_trace(go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        marker=dict(size=node_sizes, color=node_colors,
                   line=dict(width=2, color='white')),
        text=node_texts,
        textposition="top center",
        textfont=dict(size=10),
        hoverinfo='text',
        name='变量',
    ))

    fig.update_layout(
        title=title,
        showlegend=False,
        hovermode='closest',
        margin=dict(b=20, l=20, r=20, t=40),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        plot_bgcolor='rgba(0,0,0,0)',
        height=500,
    )

    return fig


def main():
    st.title("🏭 因果增强工业智能体")
    st.markdown("**创智青山AI智能体创新大赛 · 技术挑战赛道**")
    st.markdown("---")

    # 侧边栏: 控制面板
    with st.sidebar:
        st.header("⚙️ 控制面板")

        st.subheader("1. 选择故障场景")
        fault_name = st.selectbox(
            "故障模式",
            list(FAULT_MODES.keys()),
            format_func=lambda x: f"{FAULT_MODES[x]['name']} ({x})",
        )

        st.subheader("2. 推理设置")
        top_k = st.slider("显示Top-K根因", 1, 5, 3)
        confidence_threshold = st.slider("置信度阈值", 0.0, 1.0, 0.5, 0.05)

        st.subheader("3. 运行")
        if st.button("▶ 运行因果推理", type="primary", use_container_width=True):
            st.session_state.run_pipeline = True
        run_pipeline = st.session_state.get("run_pipeline", False)

        st.markdown("---")
        st.markdown("### 📊 图例")
        st.markdown("🔴 根变量 (外部条件)")
        st.markdown("🟢 中间变量 (过程状态)")
        st.markdown("🔵 输出变量 (结果)")
        st.markdown("---")
        st.markdown("### 🔗 边颜色")
        st.markdown("🟢 绿色: 双重验证")
        st.markdown("🟠 橙色: 仅知识通道")
        st.markdown("🔴 红色: 数据新发现")

    # 主界面
    if not run_pipeline:
        # 初始状态: 展示架构图
        st.info("👈 从左侧选择故障场景并点击 **运行因果推理** 开始演示")

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("系统架构")
            st.markdown("""
            ```
            传感器数据 ──→ 因果发现(PCMCI+) ──┐
                                              ├──→ 融合因果图 ──→ 根因分析
            工艺文档 ──→ LLM因果抽取 ─────────┘         │
                                                        ├──→ 反事实推理
                                                        └──→ 处置建议
            ```
            """)
        with col2:
            st.subheader("核心创新")
            st.markdown("""
            ✅ **双通道因果图构建**
            知识驱动 + 数据驱动交叉验证

            ✅ **因果增强异常检测**
            不仅告诉你"什么异常",还告诉你"为什么"

            ✅ **反事实推理**
            评估干预方案效果，辅助操作决策
            """)

        return

    # 运行Pipeline
    with st.spinner("正在运行因果推理流水线..."):
        # 构建因果图
        resources = build_causal_graph()
        sim = resources["sim"]
        fused_graph = resources["fused_graph"]
        truth_graph = resources["truth_graph"]
        fusion = resources["fusion"]
        normal_range = resources["normal_range"]

        # 生成故障数据
        fault_config = FAULT_MODES[fault_name]
        df, meta = sim.generate_fault_dataset(
            n_normal=300, n_fault=500, fault_name=fault_name
        )

        # 初始化分析器
        analyzer = RootCauseAnalyzer(fused_graph)
        normal_data = df[df["fault"] == "NORMAL"][VAR_NAMES]
        analyzer.set_normal_ranges(normal_data, n_std=3.0)

        # 获取故障观测
        fault_data = df[df["fault"] == fault_name]
        sample = fault_data.iloc[200]
        observation = {v: sample[v] for v in VAR_NAMES}

        # 根因分析
        reports = analyzer.analyze(observation, top_k=top_k)

        # 反事实推理
        cf_engine = CounterfactualEngine(fused_graph)
        coeffs = {}
        for edge in CAUSAL_GRAPH_TRUTH:
            coeffs[(edge.cause, edge.effect)] = edge.coefficient
        cf_engine.set_manual_coefficients(coeffs)

        # 确定根因变量和关注的结果变量
        root_cause_var = fault_config["root_cause"]
        target_var = reports[0].abnormal_variable if reports else "Product_Conc"

    # ============================================================
    # 展示结果
    # ============================================================

    st.success(f"✅ 推理完成 | 故障: {fault_config['name']}")

    # Tab布局
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 概览面板", "🔍 因果图", "🎯 根因分析",
        "🔄 反事实推理", "📋 技术报告"
    ])

    # ---- Tab1: 概览面板 ----
    with tab1:
        st.subheader(f"故障场景: {fault_config['name']}")
        st.markdown(f"**描述**: {fault_config['description']}")
        st.markdown(f"**Ground Truth根因**: `{fault_config['root_cause']}`")
        st.markdown(f"**因果路径**: {fault_config['causal_path']}")

        # 变量状态一览
        st.subheader("当前变量状态")
        cols = st.columns(4)
        anomaly_count = 0
        for i, var in enumerate(VAR_NAMES):
            col = cols[i % 4]
            val = observation[var]
            lo, hi = normal_range.get(var, (-np.inf, np.inf))
            is_normal = lo <= val <= hi

            if is_normal:
                col.metric(var, f"{val:.3f}", delta=None)
            else:
                anomaly_count += 1
                deviation = (val - (lo + hi) / 2) / ((hi - lo) / 2)
                delta_str = f"{deviation:+.1f}σ"
                col.metric(var, f"{val:.3f}", delta=delta_str,
                          delta_color="inverse")

        st.metric("异常变量数", anomaly_count, delta=f"/ {len(VAR_NAMES)} 总变量",
                 delta_color="off")

        # 时序趋势
        st.subheader("关键变量时序趋势")
        key_vars = [fault_config["root_cause"]] if fault_config["root_cause"] in VAR_NAMES else []
        # 找2-3个相关变量
        for edge in CAUSAL_GRAPH_TRUTH:
            if edge.cause == fault_config["root_cause"] and edge.effect not in key_vars:
                key_vars.append(edge.effect)
            if len(key_vars) >= 4:
                break

        fig = go.Figure()
        for var in key_vars[:4]:
            vals = df[var].values[-200:]  # 最后200步
            fig.add_trace(go.Scatter(
                y=vals, mode='lines', name=var,
                line=dict(width=1.5),
            ))

        # 标注故障开始
        fig.add_vline(x=100, line_dash="dash", line_color="red",
                     annotation_text="故障开始")
        fig.update_layout(
            title="最近200步关键变量趋势",
            xaxis_title="步数",
            height=350,
            hovermode='x unified',
        )
        st.plotly_chart(fig, use_container_width=True)

    # ---- Tab2: 因果图 ----
    with tab2:
        col1, col2 = st.columns([2, 1])

        with col1:
            # 选择展示的因果图
            graph_choice = st.radio(
                "选择因果图",
                ["融合因果图 (最终)", "知识因果图", "数据因果图", "Ground Truth"],
                horizontal=True,
                key="causal_graph_selector",
            )

            graph_map = {
                "融合因果图 (最终)": fused_graph,
                "知识因果图": resources["knowledge_graph"],
                "数据因果图": resources["data_graph"],
                "Ground Truth": truth_graph,
            }
            selected_graph = graph_map[graph_choice]

            # 获取根因路径用于高亮
            highlight = None
            if reports and reports[0].root_causes:
                highlight = reports[0].root_causes[0].path

            fig = plot_causal_graph(selected_graph, graph_choice, highlight)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.subheader("融合统计")
            stats = fusion.fusion_stats
            st.metric("总因果边", stats.get("total_edges", 0))
            st.metric("双重验证", stats.get("dual_verified", 0),
                     delta="高置信度")
            st.metric("知识通道独有", stats.get("knowledge_only", 0),
                     delta="待数据验证")
            st.metric("数据新发现", stats.get("data_discovered", 0),
                     delta="⚡ 新发现")

            st.markdown("---")
            st.subheader("新发现的因果边")
            for d in fusion.get_new_discoveries()[:5]:
                st.markdown(f"⚡ **{d.cause}** → **{d.effect}**")
                st.caption(f"置信度: {d.confidence:.3f}")

    # ---- Tab3: 根因分析 ----
    with tab3:
        if reports:
            st.subheader("🎯 根因分析结果")

            for report in reports:
                lo, hi = report.normal_range

                col1, col2 = st.columns([1, 1])
                with col1:
                    st.metric(
                        f"⚠ 异常变量: {report.abnormal_variable}",
                        f"{report.observed_value:.3f}",
                        delta=f"正常范围 [{lo:.1f}, {hi:.1f}]",
                        delta_color="off",
                    )

                with col2:
                    if report.root_causes:
                        top_rc = report.root_causes[0]
                        st.metric(
                            f"🔍 Top-1根因: {top_rc.variable}",
                            f"评分: {top_rc.score:.3f}",
                            delta=f"因果路径长度: {top_rc.path_length}",
                        )

                # 根因排序表
                st.markdown("#### 根因排序")
                rc_data = []
                for rc in report.root_causes:
                    rc_data.append({
                        "排名": report.root_causes.index(rc) + 1,
                        "根因变量": rc.variable,
                        "综合评分": f"{rc.score:.4f}",
                        "因果效应": f"{rc.causal_effect:.4f}",
                        "置信度": f"{rc.confidence:.3f}",
                        "路径": " → ".join(rc.path),
                        "路径长度": rc.path_length,
                    })
                st.dataframe(pd.DataFrame(rc_data), use_container_width=True,
                            hide_index=True)

                # 因果路径可视化
                st.markdown("#### 因果路径追溯")
                for i, rc in enumerate(report.root_causes[:2]):
                    st.markdown(f"**路径{i+1}**: {' ➔ '.join(rc.path)}")
                    st.caption(f"证据: {rc.evidence}")
                    st.progress(rc.score, text=f"评分: {rc.score:.3f}")

                # 处置建议
                st.markdown("#### 💡 处置建议")
                for i, action in enumerate(report.recommended_actions[:5]):
                    if "⚠⚠⚠" in action:
                        st.error(action.replace("⚠⚠⚠ ", ""))
                    elif "⚠" in action:
                        st.warning(action.replace("⚠ ", ""))
                    else:
                        st.info(f"{i+1}. {action}")
        else:
            st.warning("未检测到异常变量")

    # ---- Tab4: 反事实推理 ----
    with tab4:
        st.subheader("🔄 反事实推理: 如果...会怎样？")

        # 干预方案对比
        target_for_cf = st.selectbox(
            "关注结果变量",
            OUTPUT_VARS + INTERMEDIATE_VARS,
            index=0,
        )

        col1, col2, col3 = st.columns(3)

        interventions = [
            {root_cause_var: normal_range[root_cause_var][0] +
             (normal_range[root_cause_var][1] - normal_range[root_cause_var][0]) / 2},
            {"Feed_Flow": 95},
            {root_cause_var: normal_range[root_cause_var][0] +
             (normal_range[root_cause_var][1] - normal_range[root_cause_var][0]) / 2,
             "Feed_Flow": 95},
        ]

        intervention_labels = [
            f"方案1: 恢复 {root_cause_var}",
            "方案2: 调整 Feed_Flow",
            "方案3: 综合干预",
        ]

        for i, (col, intervention, label) in enumerate(
            zip([col1, col2, col3], interventions, intervention_labels)
        ):
            with col:
                result = cf_engine.what_if(
                    observation, intervention, target_for_cf, normal_range
                )
                st.markdown(f"**{label}**")
                st.markdown(f"干预: {intervention}")
                delta_val = f"{result.improvement:+.4f}"
                st.metric(
                    f"{target_for_cf} 预期值",
                    f"{result.counterfactual_value:.4f}",
                    delta=delta_val,
                )
                st.caption(f"实际值: {result.factual_value:.4f}")

        # 归因分析
        st.markdown("---")
        st.subheader("📊 异常贡献归因分解")

        candidate_causes = [root_cause_var]
        # 根据因果图添加更多候选
        for node in fused_graph.predecessors(target_for_cf):
            if node not in candidate_causes:
                candidate_causes.append(node)
        candidate_causes = candidate_causes[:5]

        normal_values = {
            v: (normal_range[v][0] + normal_range[v][1]) / 2
            for v in candidate_causes if v in normal_range
        }

        attribution = cf_engine.attribution(
            observation, target_for_cf,
            candidate_causes, normal_values,
        )

        fig = px.bar(
            x=list(attribution.keys()),
            y=list(attribution.values()),
            labels={"x": "变量", "y": "贡献比例"},
            title=f"各因素对 {target_for_cf} 异常的贡献",
            text_auto='.1%',
        )
        st.plotly_chart(fig, use_container_width=True)

    # ---- Tab5: 技术报告 ----
    with tab5:
        st.subheader("📋 技术报告摘要")

        st.markdown(f"""
        ### 故障案例: {fault_config['name']}

        **问题描述**: {fault_config['description']}

        **因果推理结果**:
        - 检测到 {len([r for r in reports])} 个异常变量
        """)

        if reports and reports[0].root_causes:
            rc = reports[0].root_causes[0]
            st.markdown(f"""
        **根因识别**:
        - 主要根因: `{rc.variable}`
        - 因果路径: `{' → '.join(rc.path)}`
        - 置信度评分: {rc.score:.4f}
        """)

        st.markdown("---")
        st.subheader("方法对比")

        comparison_data = {
            "方法": ["传统异常检测 (PCA/OCSVM)", "纯数据驱动因果发现", "大模型直接推理", "本方案 (因果增强智能体)"],
            "异常检测": ["✓", "✓", "✓", "✓"],
            "根因分析": ["✗", "△ (能发现因果边但无解释)", "△ (有幻觉风险)", "✓ (双通道验证)"],
            "反事实推理": ["✗", "△ (仅线性SEM)", "✗", "✓"],
            "可解释性": ["✗", "△", "中", "高"],
            "处置建议": ["✗", "✗", "△ (不保证正确)", "✓ (因果约束)"],
        }
        st.dataframe(pd.DataFrame(comparison_data), use_container_width=True,
                    hide_index=True)

        st.markdown("---")
        st.subheader("技术栈")
        st.markdown("""
        | 组件 | 技术 |
        |------|------|
        | 因果发现 | PCMCI+ (Tigramite) |
        | 因果图融合 | 双通道置信度加权融合算法 (原创) |
        | 因果推断 | DoWhy + 后门调整 + 反事实推理 |
        | 知识抽取 | LLM (Claude) + 规则匹配 |
        | 可视化 | Plotly + NetworkX + Streamlit |
        """)

        st.markdown("---")
        st.caption("因果增强工业智能体 · 创智青山AI智能体创新大赛 · 技术挑战赛道")


if __name__ == "__main__":
    main()
