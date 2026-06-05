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
import sys, os, json

sys.path.insert(0, os.path.dirname(__file__))

from src.synthetic_data_generator import (
    SyntheticProcessSimulator, CAUSAL_GRAPH_TRUTH, VAR_NAMES,
    FAULT_MODES, ROOT_VARS, INTERMEDIATE_VARS, OUTPUT_VARS,
)
from src.root_cause_analysis import RootCauseAnalyzer
from src.counterfactual import CounterfactualEngine

st.set_page_config(
    page_title='因果增强工业智能体',
    page_icon='🏭',
    layout='wide',
    initial_sidebar_state='expanded',
)

# ============================================================
# 预计算缓存加载（秒开，不做实时PCMCI+）
# ============================================================
@st.cache_resource
def load_cache():
    """加载预计算的所有数据"""
    data = {}
    cache_dir = os.path.join(os.path.dirname(__file__), 'cache')

    # 如果没有缓存，运行预计算
    if not os.path.exists(os.path.join(cache_dir, 'fused_graph.graphml')):
        st.info('正在生成缓存数据（仅首次需要，约30秒）...')
        import subprocess
        subprocess.run([sys.executable, 'precompute.py'],
                       cwd=os.path.dirname(__file__))
        st.success('缓存生成完成！刷新页面。')
        st.stop()

    # 加载融合因果图
    data['fused_graph'] = nx.read_graphml(os.path.join(cache_dir, 'fused_graph.graphml'))
    data['data_graph'] = nx.read_graphml(os.path.join(cache_dir, 'data_graph.graphml'))

    # 加载知识对 → 知识图
    with open(os.path.join(cache_dir, 'knowledge_pairs.json')) as f:
        kps = json.load(f)
    kg = nx.DiGraph()
    for v in VAR_NAMES:
        kg.add_node(v)
    for kp in kps:
        kg.add_edge(kp['cause'], kp['effect'], confidence=kp.get('confidence', 0.5),
                     mechanism=kp.get('mechanism', ''))
    data['knowledge_graph'] = kg

    # Ground truth
    truth = nx.DiGraph()
    for e in CAUSAL_GRAPH_TRUTH:
        truth.add_edge(e.cause, e.effect)
    data['truth_graph'] = truth

    # 融合统计
    with open(os.path.join(cache_dir, 'fusion_dict.json')) as f:
        fusion = json.load(f)
    data['fusion'] = type('Fusion', (), {'fusion_stats': fusion['stats'], 'to_dict': lambda: fusion})()

    # 正常范围
    with open(os.path.join(cache_dir, 'normal_range.json')) as f:
        data['normal_range'] = json.load(f)

    # 故障数据
    data['fault_datasets'] = {}
    with open(os.path.join(cache_dir, 'fault_metadata.json')) as f:
        data['fault_meta'] = json.load(f)

    # 普通正常数据
    data['df_normal'] = pd.read_csv(os.path.join(cache_dir, 'normal_operation.csv'))

    # 所有变量名
    data['var_names'] = VAR_NAMES
    data['root_vars'] = ROOT_VARS
    data['intermediate_vars'] = INTERMEDIATE_VARS
    data['output_vars'] = OUTPUT_VARS

    return data


def load_fault_df(fault_name):
    """按需加载故障数据"""
    cache_dir = os.path.join(os.path.dirname(__file__), 'cache')
    path = os.path.join(cache_dir, f'{fault_name}.csv')
    if os.path.exists(path):
        return pd.read_csv(path)
    # fallback: 实时生成
    sim = SyntheticProcessSimulator(seed=42)
    df, _ = sim.generate_fault_dataset(n_normal=300, n_fault=500, fault_name=fault_name)
    return df


@st.cache_resource
def make_sim():
    return SyntheticProcessSimulator(seed=42)


# ============================================================
# 因果图绘制
# ============================================================
def plot_causal_graph(causal_graph, title='因果图', highlight_path=None):
    pos = nx.spring_layout(causal_graph, k=2, iterations=50, seed=42)
    edge_x, edge_y, edge_texts = [], [], []

    for u, v, data in causal_graph.edges(data=True):
        x0, y0 = pos[u]; x1, y1 = pos[v]
        edge_x.extend([x0, x1, None]); edge_y.extend([y0, y1, None])
        stype = data.get('source', 'unknown')
        conf = data.get('confidence', 0)
        edge_texts.append(f'{u} → {v}<br>置信度: {conf:.2f}')

    highlight_ex, highlight_ey = [], []
    if highlight_path:
        for i in range(len(highlight_path) - 1):
            u, v = highlight_path[i], highlight_path[i + 1]
            if u in pos and v in pos:
                highlight_ex.extend([pos[u][0], pos[v][0], None])
                highlight_ey.extend([pos[u][1], pos[v][1], None])

    node_x, node_y, node_cols, node_texts = [], [], [], []
    for node in causal_graph.nodes():
        x, y = pos[node]
        node_x.append(x); node_y.append(y)
        node_texts.append(node)
        if node in ROOT_VARS: node_cols.append('#FF6B6B')
        elif node in INTERMEDIATE_VARS: node_cols.append('#4ECDC4')
        elif node in OUTPUT_VARS: node_cols.append('#45B7D1')
        else: node_cols.append('#96CEB4')

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=edge_x, y=edge_y, mode='lines',
        line=dict(width=1.5, color='rgba(150,150,150,0.6)'),
        hoverinfo='text', text=edge_texts, name='因果边'))
    if highlight_ex:
        fig.add_trace(go.Scatter(x=highlight_ex, y=highlight_ey, mode='lines',
            line=dict(width=4, color='red'), name='根因路径'))
    fig.add_trace(go.Scatter(x=node_x, y=node_y, mode='markers+text',
        marker=dict(size=22, color=node_cols, line=dict(width=2, color='white')),
        text=node_texts, textposition='top center', textfont=dict(size=10),
        hoverinfo='text', name='变量'))
    fig.update_layout(title=title, showlegend=False, hovermode='closest',
        margin=dict(b=20, l=20, r=20, t=40),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        plot_bgcolor='rgba(0,0,0,0)', height=500)
    return fig


# ============================================================
# 主界面
# ============================================================
def main():
    st.title('🏭 因果增强工业智能体')
    st.markdown('**创智青山AI智能体创新大赛 · 技术挑战赛道 · 溯因智工**')
    st.markdown('---')

    # 加载缓存
    with st.spinner('加载数据...'):
        cache = load_cache()

    # ---- 侧边栏 ----
    with st.sidebar:
        st.header('⚙️ 控制面板')
        st.subheader('1. 选择故障场景')
        fault_name = st.selectbox('故障模式', list(FAULT_MODES.keys()),
                                  format_func=lambda x: f'{FAULT_MODES[x]["name"]} ({x})')
        st.subheader('2. 推理设置')
        top_k = st.slider('显示Top-K根因', 1, 5, 3, key='top_k_slider')
        st.subheader('3. 运行')
        if st.button('▶ 运行因果推理', type='primary', use_container_width=True):
            st.session_state.run_pipeline = True
        run_pipeline = st.session_state.get('run_pipeline', False)
        st.markdown('---')
        st.markdown('### 📊 图例')
        st.markdown('🔴 根变量  🟢 中间变量  🔵 输出变量')
        st.markdown('---')
        st.markdown('### 🔗 边颜色')
        st.markdown('🟢 双重验证  🟠 仅知识  🔴 数据新发现')

    # ---- 初始状态 ----
    if not run_pipeline:
        st.info('👈 从左侧选择故障场景并点击 **运行因果推理** 开始演示')
        c1, c2 = st.columns(2)
        with c1:
            st.subheader('系统架构')
            st.code('传感器数据 → 因果发现(PCMCI+) ──┐\n'
                    '                                ├→ 融合因果图 → 根因分析\n'
                    '工艺文档 → LLM因果抽取 ──────────┘       │\n'
                    '                                      ├→ 反事实推理\n'
                    '                                      └→ 处置建议')
        with c2:
            st.subheader('核心创新')
            st.markdown('✅ **双通道因果图构建** — 知识+数据交叉验证\n'
                       '✅ **因果增强异常检测** — 不仅"什么异常"，更知"为什么"\n'
                       '✅ **反事实推理** — 量化干预方案效果')
        return

    # ---- 运行推理 ----
    fault_config = FAULT_MODES[fault_name]
    fused_graph = cache['fused_graph']
    truth_graph = cache['truth_graph']
    normal_range = cache['normal_range']
    root_cause_var = fault_config['root_cause']

    # 加载故障数据
    df = load_fault_df(fault_name)
    fault_data = df[df['fault'] == fault_name]
    sample = fault_data.iloc[min(200, len(fault_data) - 1)]
    observation = {v: float(sample[v]) for v in VAR_NAMES}

    # 根因分析
    analyzer = RootCauseAnalyzer(fused_graph)
    normal_data = df[df['fault'] == 'NORMAL'][VAR_NAMES]
    analyzer.set_normal_ranges(normal_data, n_std=3.0)
    reports = analyzer.analyze(observation, top_k=top_k)

    # 反事实引擎
    cf_engine = CounterfactualEngine(fused_graph)
    coeffs = {(e.cause, e.effect): e.coefficient for e in CAUSAL_GRAPH_TRUTH}
    cf_engine.set_manual_coefficients(coeffs)
    target_var = reports[0].abnormal_variable if reports else VAR_NAMES[4]

    st.success(f'✅ 推理完成 | 故障: {fault_config["name"]}')

    # ============ Tabs ============
    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        ['📊 概览面板', '🔍 因果图', '🎯 根因分析', '🔄 反事实推理', '📋 技术报告'])

    # ---- Tab1: 概览 ----
    with tab1:
        st.subheader(f'故障场景: {fault_config["name"]}')
        st.markdown(f'**描述**: {fault_config["description"]}')
        st.markdown(f'**Ground Truth根因**: `{fault_config["root_cause"]}`')
        st.markdown(f'**因果路径**: {fault_config["causal_path"]}')

        st.subheader('当前变量状态')
        cols = st.columns(4)
        anomaly_count = 0
        for i, var in enumerate(VAR_NAMES):
            val = observation[var]
            lo, hi = normal_range.get(var, (-np.inf, np.inf))
            if lo <= val <= hi:
                cols[i % 4].metric(var, f'{val:.3f}')
            else:
                anomaly_count += 1
                dev = (val - (lo + hi) / 2) / ((hi - lo) / 2)
                cols[i % 4].metric(var, f'{val:.3f}', delta=f'{dev:+.1f}σ', delta_color='inverse')
        st.metric('异常变量数', anomaly_count, delta=f'/ {len(VAR_NAMES)} 总变量', delta_color='off')

    # ---- Tab2: 因果图 ----
    with tab2:
        c1, c2 = st.columns([2, 1])
        with c1:
            graph_choice = st.radio('选择因果图',
                ['融合因果图 (最终)', '知识因果图', '数据因果图', 'Ground Truth'],
                horizontal=True, key='causal_graph_selector')
            graph_map = {
                '融合因果图 (最终)': fused_graph,
                '知识因果图': cache['knowledge_graph'],
                '数据因果图': cache['data_graph'],
                'Ground Truth': truth_graph,
            }
            highlight = reports[0].root_causes[0].path if reports and reports[0].root_causes else None
            fig = plot_causal_graph(graph_map[graph_choice], graph_choice, highlight)
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            st.subheader('融合统计')
            stats = cache['fusion'].fusion_stats
            st.metric('总因果边', stats.get('total_edges', 0))
            st.metric('双重验证', stats.get('dual_verified', 0), delta='高置信度')
            st.metric('知识通道独有', stats.get('knowledge_only', 0), delta='待数据验证')
            st.metric('数据新发现', stats.get('data_discovered', 0), delta='⚡ 新发现')

    # ---- Tab3: 根因分析 ----
    with tab3:
        if reports:
            for report in reports:
                lo, hi = report.normal_range
                c1, c2 = st.columns(2)
                with c1:
                    st.metric(f'⚠ 异常: {report.abnormal_variable}',
                             f'{report.observed_value:.3f}', delta=f'正常 [{lo:.1f}, {hi:.1f}]', delta_color='off')
                with c2:
                    if report.root_causes:
                        st.metric(f'🔍 Top-1根因: {report.root_causes[0].variable}',
                                 f'评分: {report.root_causes[0].score:.3f}')

                st.markdown('#### 根因排序')
                rc_data = [{'排名': i+1, '根因变量': rc.variable, '评分': f'{rc.score:.4f}',
                           '因果路径': ' → '.join(rc.path), '路径长度': rc.path_length}
                          for i, rc in enumerate(report.root_causes)]
                st.dataframe(pd.DataFrame(rc_data), use_container_width=True, hide_index=True)

                st.markdown('#### 因果路径追溯')
                for i, rc in enumerate(report.root_causes[:2]):
                    st.markdown(f'**路径{i+1}**: {" ➔ ".join(rc.path)}')
                    st.caption(f'置信度: {rc.confidence:.3f}')
                    st.progress(min(rc.score, 1.0), text=f'评分: {rc.score:.3f}')

                st.markdown('#### 💡 处置建议')
                for i, action in enumerate(report.recommended_actions[:5]):
                    if '⚠⚠⚠' in action: st.error(action.replace('⚠⚠⚠ ', ''))
                    elif '⚠' in action: st.warning(action.replace('⚠ ', ''))
                    else: st.info(f'{i+1}. {action}')
        else:
            st.warning('未检测到异常')

    # ---- Tab4: 反事实推理 ----
    with tab4:
        st.subheader('🔄 反事实推理: 如果...会怎样？')
        target_for_cf = st.selectbox('关注结果变量', OUTPUT_VARS + INTERMEDIATE_VARS, index=0)
        c1, c2, c3 = st.columns(3)
        interventions = [
            {root_cause_var: (normal_range[root_cause_var][0] + normal_range[root_cause_var][1]) / 2},
            {'Feed_Flow': 95},
            {root_cause_var: (normal_range[root_cause_var][0] + normal_range[root_cause_var][1]) / 2, 'Feed_Flow': 95},
        ]
        labels = [f'方案1: 恢复 {root_cause_var}', '方案2: 调整 Feed_Flow', '方案3: 综合干预']
        for col, inter, label in zip([c1, c2, c3], interventions, labels):
            with col:
                result = cf_engine.what_if(observation, inter, target_for_cf, normal_range)
                st.markdown(f'**{label}**')
                st.metric(f'{target_for_cf} 预期', f'{result.counterfactual_value:.4f}',
                         delta=f'{result.improvement:+.4f}')
                st.caption(f'实际: {result.factual_value:.4f}')

        st.markdown('---')
        st.subheader('📊 异常贡献归因分解')
        candidate = [root_cause_var]
        for node in list(fused_graph.predecessors(target_for_cf))[:4]:
            if node not in candidate: candidate.append(node)
        normal_vals = {v: (normal_range[v][0] + normal_range[v][1]) / 2 for v in candidate if v in normal_range}
        attr = cf_engine.attribution(observation, target_for_cf, candidate[:5], normal_vals)
        fig = px.bar(x=list(attr.keys()), y=list(attr.values()),
                    labels={'x': '变量', 'y': '贡献比例'},
                    title=f'各因素对 {target_for_cf} 异常的贡献', text_auto='.1%')
        st.plotly_chart(fig, use_container_width=True)

    # ---- Tab5: 技术报告 ----
    with tab5:
        st.subheader('📋 技术报告摘要')
        st.markdown(f'### 故障案例: {fault_config["name"]}\n**描述**: {fault_config["description"]}\n')
        if reports and reports[0].root_causes:
            rc = reports[0].root_causes[0]
            st.markdown(f'**根因识别**: `{rc.variable}` | **因果路径**: `{" → ".join(rc.path)}` | **评分**: {rc.score:.4f}')
        st.markdown('---'); st.subheader('方法对比')
        comp = pd.DataFrame({
            '方法': ['传统异常检测', '纯大模型方案', '纯数据因果发现', '本方案'],
            '根因分析': ['✗', '△ (幻觉)', '△ (无解释)', '✓ (双通道)'],
            '反事实推理': ['✗', '✗', '△', '✓'],
            '处置建议': ['✗', '△ (不保证)', '✗', '✓ (因果约束)'],
            '可解释性': ['低', '中', '中', '高'],
        })
        st.dataframe(comp, use_container_width=True, hide_index=True)
        st.markdown('---'); st.subheader('技术栈')
        st.markdown('| 组件 | 技术 |\n|------|------|\n| 因果发现 | PCMCI+ (Tigramite) |'
                   '\n| 因果图融合 | 双通道置信度加权融合算法 (原创) |'
                   '\n| 因果推断 | Pearl SCM + 反事实推理 |'
                   '\n| 知识抽取 | LLM (Claude) + 预提取知识库 |'
                   '\n| 可视化 | Plotly + NetworkX + Streamlit |')
        st.caption('因果增强工业智能体 · 创智青山AI智能体创新大赛 · 溯因智工')


if __name__ == '__main__':
    main()
