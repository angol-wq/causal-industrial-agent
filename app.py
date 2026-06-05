"""
因果增强工业智能体 V2 — Streamlit 可视化界面
新增: 31种故障 | 自主文献进化 | 进化状态面板
"""
import streamlit as st
import numpy as np
import pandas as pd
import networkx as nx
import plotly.graph_objects as go
import plotly.express as px
import sys, os, json, time
from datetime import datetime

sys.path.insert(0, os.path.dirname(__file__))

from src.synthetic_data_generator import (VAR_NAMES, ROOT_VARS, INTERMEDIATE_VARS, OUTPUT_VARS, CAUSAL_GRAPH_TRUTH)
from src.root_cause_analysis import RootCauseAnalyzer
from src.counterfactual import CounterfactualEngine
from src.unified_fault_registry import UnifiedFaultRegistry
from src.llm_causal_extract import extract_from_synthetic_doc, LLMCausalExtractor
from src.agent_v2 import CausalAgentV2

st.set_page_config(page_title='因果增强工业智能体 V2', page_icon='🧬', layout='wide',
                   initial_sidebar_state='expanded')

# ============================================================
# 初始化
# ============================================================
@st.cache_resource
def init_system():
    """初始化: 统一故障注册中心 + 因果图 + Agent V2"""
    registry = UnifiedFaultRegistry()
    # 初始因果图
    os.makedirs('data/synthetic', exist_ok=True)
    from src.synthetic_data_generator import generate_process_documentation
    generate_process_documentation('data/synthetic')
    pairs = extract_from_synthetic_doc('data/synthetic/process_documentation.txt', VAR_NAMES)
    kg = LLMCausalExtractor.pairs_to_graph(pairs, VAR_NAMES)
    # Agent V2
    agent = CausalAgentV2()
    agent.causal_graph = kg
    return registry, kg, agent


@st.cache_resource
def load_cache():
    """加载预计算缓存"""
    cache_dir = os.path.join(os.path.dirname(__file__), 'cache')
    if not os.path.exists(os.path.join(cache_dir, 'normal_range.json')):
        st.warning('缓存数据不存在，请先运行: python precompute.py')
        st.stop()
    data = {}
    data['fused_graph'] = nx.read_graphml(os.path.join(cache_dir, 'fused_graph.graphml'))
    with open(os.path.join(cache_dir, 'normal_range.json')) as f:
        data['normal_range'] = json.load(f)
    with open(os.path.join(cache_dir, 'knowledge_pairs.json')) as f:
        data['knowledge_pairs'] = json.load(f)
    data['df_normal'] = pd.read_csv(os.path.join(cache_dir, 'normal_operation.csv'))
    with open(os.path.join(cache_dir, 'fusion_dict.json')) as f:
        fusion = json.load(f)
    data['fusion_stats'] = fusion['stats']
    return data


# ============================================================
# 可视化
# ============================================================
def plot_causal_graph(causal_graph, title='因果图', highlight_path=None):
    pos = nx.spring_layout(causal_graph, k=2, iterations=50, seed=42)
    edge_x, edge_y, edge_texts = [], [], []
    for u, v, data in causal_graph.edges(data=True):
        x0, y0 = pos[u]; x1, y1 = pos[v]
        edge_x.extend([x0, x1, None]); edge_y.extend([y0, y1, None])
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
        x, y = pos[node]; node_x.append(x); node_y.append(y); node_texts.append(node)
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
        text=node_texts, textposition='top center', textfont=dict(size=9),
        hoverinfo='text', name='变量'))
    fig.update_layout(title=title, showlegend=False, hovermode='closest',
        margin=dict(b=20, l=20, r=20, t=40), height=480,
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        plot_bgcolor='rgba(0,0,0,0)')
    return fig


# ============================================================
# 主界面
# ============================================================
def main():
    st.title('🧬 因果增强工业智能体 V2')
    st.markdown('**全链路自主循环进化 | 创智青山AI智能体创新大赛 · 溯因智工**')
    st.markdown('---')

    registry, kg, agent = init_system()
    cache = load_cache()
    fused_graph = cache['fused_graph']
    normal_range = cache['normal_range']

    # ---- 侧边栏 ----
    with st.sidebar:
        st.header('⚙️ 控制面板')

        st.subheader('1. 选择故障')
        fault_categories = ['全部(31种)', '合成数据(10种)', 'TEP真实数据(21种)']
        cat_choice = st.selectbox('数据源', fault_categories)

        # 按类别筛选故障列表
        if '合成' in cat_choice:
            faults = [f for f in registry.list_all() if f.category == 'synthetic']
        elif 'TEP' in cat_choice:
            faults = [f for f in registry.list_all() if f.category == 'tep']
        else:
            faults = registry.list_all()

        fault_names = [f'{f.fault_id}: {f.name}' for f in faults]
        fault_choice = st.selectbox('故障模式', fault_names,
                                     format_func=lambda x: x)
        selected_id = fault_choice.split(':')[0].strip()

        st.subheader('2. 推理设置')
        top_k = st.slider('Top-K根因', 1, 5, 3, key='topk')

        st.subheader('3. 运行')
        c1, c2 = st.columns(2)
        with c1:
            if st.button('▶ 运行诊断', type='primary', use_container_width=True):
                st.session_state.run_diag = True
        with c2:
            if st.button('📚 文献进化', use_container_width=True):
                st.session_state.run_evo = True

        run_diag = st.session_state.get('run_diag', False)
        run_evo = st.session_state.get('run_evo', False)

        st.markdown('---')
        st.metric('📊 故障总数', registry.get_stats()['total'])
        st.metric('🧬 因果边', kg.number_of_edges())

    # ================================================================
    # 文献进化
    # ================================================================
    if run_evo:
        st.info('📚 正在爬取钢铁/冶金/石化领域最新文献...')
        with st.spinner('文献爬取 + AI精读 + 知识库更新中 (~60秒)...'):
            try:
                report = agent.run_evolution_cycle(max_new_papers=10)
                kg = agent.causal_graph  # 更新因果图
                growth = report.get('knowledge_growth', {})
                st.success(
                    f'✅ 进化完成！因果图从 {growth.get("edges_before", 0)} '
                    f'→ {growth.get("edges_after", 0)} 条边 '
                    f'(+{growth.get("new_edges", 0)} 新增) | '
                    f'自测评分: {report.get("final_validation_score", 0):.1%}'
                )
            except Exception as e:
                st.warning(f'文献爬取受限（网络/API）: {e}。使用离线知识库。')
        st.session_state.run_evo = False

    # ---- 初始状态 ----
    if not run_diag and not run_evo:
        st.info('👈 从左侧选择故障，点击 **运行诊断** 或 **文献进化**')
        c1, c2, c3 = st.columns(3)
        with c1:
            st.subheader('🔍 31种故障覆盖')
            stats = registry.get_stats()
            st.markdown(f'**合成数据**: {stats["by_category"].get("synthetic", 0)}种\n'
                       f'**TEP真实数据**: {stats["by_category"].get("tep", 0)}种\n'
                       f'**覆盖行业**: 钢铁、石化、通用化工')
        with c2:
            st.subheader('📚 文献自主进化')
            st.markdown('启动时自动爬取钢铁/冶金SCI论文\n'
                       'AI精读提取新因果机理\n'
                       '知识库持续增长，越用越准')
        with c3:
            st.subheader('🔄 仿真自进化')
            st.markdown('现场数据 + 文献数据双向回喂\n'
                       '仿真参数自动微调校准\n'
                       '文献自测验证模型正确性')
        return

    # ================================================================
    # 运行诊断
    # ================================================================
    fault_entry = registry.get(selected_id)
    if not fault_entry:
        st.error(f'未知故障: {selected_id}')
        return

    st.success(f'✅ 诊断: {fault_entry.fault_id} — {fault_entry.name} | '
               f'数据源: {fault_entry.data_source}')

    # 加载数据
    with st.spinner('加载故障数据...'):
        df, meta = registry.load_fault(selected_id)
        if fault_entry.category == 'synthetic':
            fault_data = df[df['fault'] == selected_id]
            var_list = VAR_NAMES
        else:
            fault_data = df[df['fault_phase'] == 'test']
            var_list = [v for v in TEP_VAR_NAMES[:8] if v in df.columns]

        sample_idx = min(len(fault_data) - 1, int(len(fault_data) * 0.7))
        sample = fault_data.iloc[sample_idx]
        observation = {v: float(sample[v]) for v in var_list if v in df.columns}

    # 根因分析
    use_graph = fused_graph if fault_entry.category == 'synthetic' else kg
    analyzer = RootCauseAnalyzer(use_graph)
    normal_subset = df[df.get('fault', df.get('fault_phase')) == 'NORMAL'][var_list]
    if len(normal_subset) > 50:
        analyzer.set_normal_ranges(normal_subset.iloc[:300])
    else:
        analyzer.normal_ranges = {v: normal_range.get(v, (-np.inf, np.inf)) for v in var_list}
    reports = analyzer.analyze(observation, top_k=top_k)

    # 反事实引擎
    cf_engine = CounterfactualEngine(use_graph)
    coeffs = {(e.cause, e.effect): e.coefficient for e in CAUSAL_GRAPH_TRUTH}
    cf_engine.set_manual_coefficients(coeffs)

    # ============ Tabs ============
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
        ['📊 概览', '🔍 因果图', '🎯 根因分析', '🔄 反事实',
         '📚 知识库', '🧬 进化状态'])

    with tab1:
        st.subheader(f'故障: {fault_entry.name}')
        st.markdown(f'**描述**: {fault_entry.description}')
        st.markdown(f'**分类**: {fault_entry.category} | **行业**: {fault_entry.industry} | '
                   f'**严重度**: {fault_entry.severity} | **数据源**: {fault_entry.data_source}')
        if fault_entry.causal_path:
            st.markdown(f'**因果路径**: {fault_entry.causal_path}')
        st.subheader('变量状态')
        cols = st.columns(4)
        anomaly_count = 0
        for i, var in enumerate(var_list[:12]):
            if var in observation:
                val = observation[var]
                lo, hi = analyzer.normal_ranges.get(var, (-np.inf, np.inf))
                if lo <= val <= hi:
                    cols[i % 4].metric(var[:15], f'{val:.3f}')
                else:
                    anomaly_count += 1
                    dev = (val - (lo + hi) / 2) / max((hi - lo) / 2, 1e-6)
                    cols[i % 4].metric(var[:15], f'{val:.3f}', delta=f'{dev:+.1f}σ',
                                       delta_color='inverse')
        st.metric('异常变量', anomaly_count, delta=f'/ {len(var_list)} 总变量', delta_color='off')

    with tab2:
        c1, c2 = st.columns([2, 1])
        with c1:
            highlight = reports[0].root_causes[0].path if reports and reports[0].root_causes else None
            fig = plot_causal_graph(use_graph, f'因果图 — {fault_entry.name}', highlight)
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            st.subheader('图统计')
            st.metric('节点', use_graph.number_of_nodes())
            st.metric('因果边', use_graph.number_of_edges())
            if 'fusion_stats' in cache:
                fs = cache['fusion_stats']
                st.metric('双重验证', fs.get('dual_verified', 0), delta='高置信')
                st.metric('知识独有', fs.get('knowledge_only', 0))
                st.metric('数据发现', fs.get('data_discovered', 0), delta='⚡')

    with tab3:
        if reports:
            for report in reports[:6]:
                lo, hi = report.normal_range
                c1, c2 = st.columns(2)
                with c1:
                    st.metric(f'⚠ {report.abnormal_variable}', f'{report.observed_value:.3f}',
                             delta=f'正常 [{lo:.1f}, {hi:.1f}]', delta_color='off')
                with c2:
                    if report.root_causes:
                        rc = report.root_causes[0]
                        st.metric(f'🔍 根因: {rc.variable}', f'评分: {rc.score:.3f}')
                if report.root_causes:
                    st.markdown('**根因排序**')
                    rc_df = pd.DataFrame([{
                        '排名': i+1, '根因': rc.variable, '评分': f'{rc.score:.4f}',
                        '路径': ' → '.join(rc.path[:4])
                    } for i, rc in enumerate(report.root_causes)])
                    st.dataframe(rc_df, use_container_width=True, hide_index=True)
                if report.recommended_actions:
                    for act in report.recommended_actions[:3]:
                        if '⚠⚠⚠' in act: st.error(act.replace('⚠⚠⚠ ', ''))
                        elif '⚠' in act: st.warning(act.replace('⚠ ', ''))
                        else: st.info(act)
                st.markdown('---')
        else:
            st.warning('未检测到异常')

    with tab4:
        st.subheader('🔄 反事实推理')
        if reports:
            target = st.selectbox('关注变量', var_list[:8], index=0)
            c1, c2, c3 = st.columns(3)
            top_rc = None
            for r in reports:
                if r.root_causes:
                    top_rc = r.root_causes[0].variable; break
            if top_rc and top_rc in analyzer.normal_ranges:
                lo, hi = analyzer.normal_ranges[top_rc]
                normal_val = (lo + hi) / 2
                interventions = [
                    {top_rc: normal_val},
                    {'Feed_Flow': 95} if 'Feed_Flow' in var_list else {var_list[0]: 0},
                    {top_rc: normal_val, var_list[0]: float(observation.get(var_list[0], 0))},
                ]
                labels = [f'方案1: 恢复{top_rc}', '方案2: 其他干预', '方案3: 综合']
                for col, inter, label in zip([c1, c2, c3], interventions, labels):
                    with col:
                        try:
                            res = cf_engine.what_if(observation, inter, target, analyzer.normal_ranges)
                            st.markdown(f'**{label}**')
                            st.metric(f'{target} 预期', f'{res.counterfactual_value:.3f}',
                                     delta=f'{res.improvement:+.3f}')
                        except Exception:
                            st.caption(f'{label}: 参数不足')

    with tab5:
        st.subheader('📚 知识库')
        pairs = cache.get('knowledge_pairs', [])
        st.markdown(f'**因果知识条目**: {len(pairs)} 条')
        if pairs:
            kp_df = pd.DataFrame([{
                '原因': p.get('cause', ''), '结果': p.get('effect', ''),
                '方向': p.get('direction', ''), '置信度': f'{p.get("confidence", 0):.2f}',
                '机制': p.get('mechanism', '')[:40]
            } for p in pairs[:20]])
            st.dataframe(kp_df, use_container_width=True, hide_index=True)

        st.markdown('---')
        st.subheader('📖 文献爬取状态')
        if st.button('🔄 手动触发文献爬取', use_container_width=True):
            with st.spinner('爬取中...'):
                try:
                    from src.literature_crawler import LiteratureCrawler
                    crawler = LiteratureCrawler()
                    papers = crawler.search_all()
                    st.success(f'获取 {len(papers)} 篇文献')
                    papers_df = pd.DataFrame([{
                        '标题': p.title[:60], '年份': p.year,
                        '来源': p.source_type, '引用': p.citation_count
                    } for p in papers[:20]])
                    st.dataframe(papers_df, use_container_width=True, hide_index=True)
                except Exception as e:
                    st.warning(f'爬取受限: {e}')

    with tab6:
        st.subheader('🧬 自主进化状态')
        status = agent.get_status()
        c1, c2, c3, c4 = st.columns(4)
        with c1: st.metric('Agent版本', status['agent_version'])
        with c2: st.metric('进化代数', status['evolution_generation'])
        with c3: st.metric('因果边', status['causal_graph_size'])
        with c4: st.metric('知识库', status['knowledge_base_size'])

        st.markdown('---')
        st.subheader('📈 仿真参数进化轨迹')
        params = status['current_params']
        param_df = pd.DataFrame([{
            '参数': k, '当前值': f'{v:.4f}',
            '类别': '热力学' if 'temp' in k or 'heat' in k else ('换热' if 'transfer' in k or 'fouling' in k else ('侵蚀' if 'erosion' in k or 'wear' in k else '动力学'))
        } for k, v in params.items()])
        st.dataframe(param_df, use_container_width=True, hide_index=True)

        st.markdown('---')
        st.subheader('🔄 文献→仿真 全链路循环')
        st.markdown('''
        ```
        文献爬取 → AI精读 → 因果知识库 ──→ 仿真参数校准
            ↑                                  ↓
        自测验证 ←────────────────────── 仿真重跑
            │
        通过? ── 否 → 回到文献爬取
            ── 是 → 部署更新 ✓
        ```
        ''')

        if st.button('▶ 执行一个进化周期', type='primary', use_container_width=True):
            with st.spinner('全链路进化中 (~60秒)...'):
                try:
                    report = agent.run_evolution_cycle(max_new_papers=10)
                    growth = report.get('knowledge_growth', {})
                    st.success(
                        f'进化完成！因果图 {growth.get("edges_before", 0)} → '
                        f'{growth.get("edges_after", 0)} 条边 '
                        f'(+{growth.get("new_edges", 0)}) | '
                        f'自测评分: {report.get("final_validation_score", 0):.1%}'
                    )
                except Exception as e:
                    st.error(f'进化失败: {e}')


if __name__ == '__main__':
    main()
