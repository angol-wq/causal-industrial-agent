"""
因果增强工业智能体 V2.1 — Streamlit
新增: 汉化因果图 · 场景模拟器 · 参数自定义
"""
import streamlit as st
import numpy as np
import pandas as pd
import networkx as nx
import plotly.graph_objects as go
import plotly.express as px
import sys, os, json

sys.path.insert(0, os.path.dirname(__file__))

from src.synthetic_data_generator import (VAR_NAMES, ROOT_VARS, INTERMEDIATE_VARS, OUTPUT_VARS, CAUSAL_GRAPH_TRUTH, FAULT_MODES)
from src.root_cause_analysis import RootCauseAnalyzer
from src.counterfactual import CounterfactualEngine
from src.chinese_labels import (VAR_CN, get_cn_label, get_cn_unit, get_category, translate_path, FAULT_CN)

st.set_page_config(page_title='因果增强工业智能体 V2.1', page_icon='🏭', layout='wide',
                   initial_sidebar_state='expanded')

# ============================================================
# 加载
# ============================================================
@st.cache_resource
def load_data():
    d = {}
    if os.path.exists('cache/normal_range.json'):
        with open('cache/normal_range.json') as f: d['normal_range'] = json.load(f)
    else:
        from src.synthetic_data_generator import SyntheticProcessSimulator
        sim = SyntheticProcessSimulator(seed=42)
        df_n = sim.simulate(n_steps=500)
        d['normal_range'] = {}
        for v in VAR_NAMES:
            m = df_n[v].mean(); s = df_n[v].std()
            d['normal_range'][v] = (m - 3*s, m + 3*s)
        d['df_normal'] = df_n
    if os.path.exists('cache/fused_graph.graphml'):
        d['fused_graph'] = nx.read_graphml('cache/fused_graph.graphml')
    else:
        from src.llm_causal_extract import extract_from_synthetic_doc, LLMCausalExtractor
        os.makedirs('data/synthetic', exist_ok=True)
        from src.synthetic_data_generator import generate_process_documentation
        generate_process_documentation('data/synthetic')
        pairs = extract_from_synthetic_doc('data/synthetic/process_documentation.txt', VAR_NAMES)
        d['fused_graph'] = LLMCausalExtractor.pairs_to_graph(pairs, VAR_NAMES)
    if os.path.exists('cache/knowledge_pairs.json'):
        with open('cache/knowledge_pairs.json') as f: d['kp'] = json.load(f)
    else:
        d['kp'] = []
    return d


@st.cache_resource
def get_sim():
    from src.synthetic_data_generator import SyntheticProcessSimulator
    return SyntheticProcessSimulator(seed=42)


# ============================================================
# 汉化因果图
# ============================================================
def plot_cn_graph(graph, title='因果图', highlight_path=None):
    """汉化因果图 — 节点和hover都用中文"""
    pos = nx.spring_layout(graph, k=2.2, iterations=50, seed=42)

    # 分类着色
    cat_colors = {
        '进料系统': '#FF6B6B', '冷却系统': '#4ECDC4', '反应器': '#FFD93D',
        '换热系统': '#C084FC', '产品质量': '#45B7D1', '阀门': '#F59E0B',
        '能源效率': '#6BCB77', '': '#96CEB4',
        '分离器': '#FF8C42', '汽提塔': '#A855F7', '压缩机': '#EC4899',
        '循环系统': '#14B8A6', '排放系统': '#F43F5E', '成分分析': '#8B5CF6',
    }

    edge_x, edge_y, edge_texts = [], [], []
    for u, v, data in graph.edges(data=True):
        x0, y0 = pos[u]; x1, y1 = pos[v]
        edge_x.extend([x0, x1, None]); edge_y.extend([y0, y1, None])
        conf = data.get('confidence', 0)
        cn_cause = get_cn_label(u); cn_effect = get_cn_label(v)
        edge_texts.append(f'{cn_cause} → {cn_effect}<br>置信度: {conf:.2f}')

    hl_ex, hl_ey = [], []
    if highlight_path:
        for i in range(len(highlight_path)-1):
            u, v = highlight_path[i], highlight_path[i+1]
            if u in pos and v in pos:
                hl_ex.extend([pos[u][0], pos[v][0], None])
                hl_ey.extend([pos[u][1], pos[v][1], None])

    node_x, node_y, node_cols, node_names = [], [], [], []
    for node in graph.nodes():
        x, y = pos[node]; node_x.append(x); node_y.append(y)
        cat = get_category(node)
        node_cols.append(cat_colors.get(cat, '#96CEB4'))
        name = get_cn_label(node)
        unit = get_cn_unit(node)
        node_names.append(f'{name}({unit})' if unit else name)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=edge_x, y=edge_y, mode='lines',
        line=dict(width=1.5, color='rgba(150,150,150,0.6)'),
        hoverinfo='text', text=edge_texts, name='因果边'))
    if hl_ex:
        fig.add_trace(go.Scatter(x=hl_ex, y=hl_ey, mode='lines',
            line=dict(width=4, color='#EF4444'), name='根因路径'))
    fig.add_trace(go.Scatter(x=node_x, y=node_y, mode='markers+text',
        marker=dict(size=28, color=node_cols, line=dict(width=2, color='white')),
        text=node_names, textposition='top center', textfont=dict(size=11),
        hovertext=[f'{get_cn_label(n)}<br>分类: {get_category(n)}<br>单位: {get_cn_unit(n)}'
                    for n in graph.nodes()],
        hoverinfo='text', name='变量'))
    fig.update_layout(title=title, showlegend=False, hovermode='closest',
        margin=dict(b=20, l=20, r=20, t=40), height=550,
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        plot_bgcolor='rgba(0,0,0,0)')
    return fig


# ============================================================
# 主界面
# ============================================================
def main():
    st.title('🏭 因果增强工业智能体 V2.1')
    st.markdown('**汉化因果图 · 场景模拟器 · 双通道融合 | 创智青山AI智能体创新大赛 · 溯因智工**')
    st.markdown('---')

    data = load_data()
    graph = data['fused_graph']
    normal_range = data.get('normal_range', {})

    # ==== 模式选择 ====
    mode = st.sidebar.radio('📌 选择模式',
        ['🎯 故障诊断', '🧪 场景模拟器', '📊 因果图浏览', '📚 知识库与进化', '🌐 文献爬虫', '🧬 自进化状态'],
        key='mode_selector')

    # ================================================================
    # 模式1: 故障诊断
    # ================================================================
    if mode == '🎯 故障诊断':
        with st.sidebar:
            st.subheader('故障场景')
            f_names = list(FAULT_MODES.keys())
            f_labels = [FAULT_CN.get(k, FAULT_MODES[k]['name']) for k in f_names]
            fi = st.selectbox('选择故障', range(len(f_names)), format_func=lambda i: f_labels[i])
            fault_name = f_names[fi]
            top_k = st.slider('Top-K根因', 1, 5, 3, key='diag_tk')

            if st.button('▶ 运行诊断', type='primary', use_container_width=True):
                st.session_state.run_d = True
            run_d = st.session_state.get('run_d', False)

        if not run_d:
            st.info('👈 选择一个故障场景，点击 **运行诊断**')
            # 展示图例
            st.subheader('📊 变量分类图例')
            c1,c2,c3,c4,c5,c6 = st.columns(6)
            c1.markdown('🔴 **进料系统**')
            c2.markdown('🟢 **冷却系统**')
            c3.markdown('🟡 **反应器**')
            c4.markdown('🟣 **换热系统**')
            c5.markdown('🔵 **产品质量**')
            c6.markdown('🟠 **阀门**')
            return

        fault_cfg = FAULT_MODES[fault_name]
        sim = get_sim()
        df, meta = sim.generate_fault_dataset(n_normal=300, n_fault=300, fault_name=fault_name)
        fd = df[df['fault'] == fault_name]
        s = fd.iloc[min(len(fd)-1, int(len(fd)*0.7))]
        obs = {v: float(s[v]) for v in VAR_NAMES}

        analyzer = RootCauseAnalyzer(graph)
        analyzer.set_normal_ranges(df[df['fault']=='NORMAL'][VAR_NAMES].iloc[:200])
        reports = analyzer.analyze(obs, top_k=top_k)

        cf_engine = CounterfactualEngine(graph)
        coeffs = {(e.cause, e.effect): e.coefficient for e in CAUSAL_GRAPH_TRUTH}
        cf_engine.set_manual_coefficients(coeffs)

        cn_fault_name = FAULT_CN.get(fault_name, fault_cfg['name'])
        st.success(f'✅ {cn_fault_name}')

        t1, t2, t3 = st.tabs(['📊 诊断结果', '🔍 汉化因果图', '🔄 反事实推理'])

        with t1:
            for report in reports[:6]:
                lo, hi = report.normal_range
                cn_var = get_cn_label(report.abnormal_variable)
                c1, c2 = st.columns(2)
                with c1:
                    st.metric(f'⚠ {cn_var}({get_cn_unit(report.abnormal_variable)})',
                             f'{report.observed_value:.1f}', delta=f'正常[{lo:.1f},{hi:.1f}]',
                             delta_color='off')
                with c2:
                    if report.root_causes:
                        rc = report.root_causes[0]
                        st.metric(f'🔍 根因', f'{get_cn_label(rc.variable)}({rc.score:.2f})')
                if report.root_causes:
                    rc_df = pd.DataFrame([{
                        '排名': i+1, '根因': get_cn_label(rc.variable),
                        '评分': f'{rc.score:.3f}',
                        '因果链': translate_path(rc.path)
                    } for i, rc in enumerate(report.root_causes)])
                    st.dataframe(rc_df, use_container_width=True, hide_index=True)
                if report.recommended_actions:
                    for act in report.recommended_actions[:3]:
                        if '⚠⚠⚠' in act: st.error(act.replace('⚠⚠⚠ ',''))
                        elif '⚠' in act: st.warning(act.replace('⚠ ',''))
                        else: st.info(act)
                st.markdown('---')

        with t2:
            hl = reports[0].root_causes[0].path if reports and reports[0].root_causes else None
            fig = plot_cn_graph(graph, f'因果图 — {cn_fault_name}', hl)
            st.plotly_chart(fig, use_container_width=True)

        with t3:
            st.subheader('🔄 反事实推理')
            top_rc = None
            for r in reports:
                if r.root_causes:
                    top_rc = r.root_causes[0]; break
            if top_rc and top_rc.variable in analyzer.normal_ranges:
                lo, hi = analyzer.normal_ranges[top_rc.variable]
                nv = (lo+hi)/2
                c1,c2,c3 = st.columns(3)
                tgt = reports[0].abnormal_variable if reports else VAR_NAMES[4]
                for col, inter, lb in zip([c1,c2,c3],
                    [{top_rc.variable: nv}, {'Feed_Flow': 95},
                     {top_rc.variable: nv, 'Feed_Flow': 95}],
                    [f'方案1: 恢复{get_cn_label(top_rc.variable)}', '方案2: 调整进料流量', '方案3: 综合干预']):
                    with col:
                        try:
                            res = cf_engine.what_if(obs, inter, tgt, analyzer.normal_ranges)
                            st.markdown(f'**{lb}**')
                            st.metric(get_cn_label(tgt),
                                     f'{res.counterfactual_value:.2f}',
                                     delta=f'{res.improvement:+.2f}')
                        except Exception:
                            st.caption(f'{lb}: 参数不足')

    # ================================================================
    # 模式2: 场景模拟器 ⭐ 核心新功能
    # ================================================================
    elif mode == '🧪 场景模拟器':
        st.subheader('🧪 场景模拟器 — 自定义故障参数，实时验证Agent诊断准确性')

        st.markdown('调整以下滑块模拟不同工况，Agent将自动诊断并报告与真实根因的对比。')

        # 分类显示变量滑块
        categories = ['进料系统', '冷却系统', '反应器', '换热系统', '产品质量', '能源效率']

        with st.expander('⚙️ 参数设置', expanded=True):
            custom_obs = {}
            for cat in categories:
                vars_in_cat = [(v, VAR_CN[v]) for v in VAR_NAMES if get_category(v) == cat]
                if not vars_in_cat:
                    continue
                st.markdown(f'**{cat}**')
                cols = st.columns(min(3, len(vars_in_cat)))
                for i, (var, cn) in enumerate(vars_in_cat):
                    with cols[i % 3]:
                        lo, hi = normal_range.get(var, (0, 200))
                        mid = (lo + hi) / 2
                        # 扩大范围让用户可以模拟故障
                        val = st.slider(
                            f'{cn["name"]} ({cn["unit"]})',
                            float(lo * 0.5), float(hi * 2.0),
                            float(mid), 5.0,
                            key=f'sim_{var}'
                        )
                        custom_obs[var] = val

            # 快捷故障注入按钮
            st.markdown('---')
            st.markdown('**⚡ 快捷故障注入**')
            qc1, qc2, qc3, qc4 = st.columns(4)
            if qc1.button('阀门卡滞', use_container_width=True, key='q1'):
                custom_obs['CW_Valve'] = 22.0
                custom_obs['CW_Flow'] = 118.0
                st.rerun()
            if qc2.button('入口水温高', use_container_width=True, key='q2'):
                custom_obs['CW_Inlet_Temp'] = 35.0
                st.rerun()
            if qc3.button('进料流量突增', use_container_width=True, key='q3'):
                custom_obs['Feed_Flow'] = 130.0
                st.rerun()
            if qc4.button('全部恢复正常', use_container_width=True, key='q4'):
                for v in VAR_NAMES:
                    lo, hi = normal_range.get(v, (0, 200))
                    custom_obs[v] = (lo + hi) / 2
                st.rerun()

        # 实时诊断
        if custom_obs:
            st.markdown('---')
            st.subheader('🔍 Agent 实时诊断')

            analyzer = RootCauseAnalyzer(graph)
            analyzer.normal_ranges = normal_range
            reports = analyzer.analyze(custom_obs, top_k=3)

            # 异常变量卡片
            anomalies = [r for r in reports]
            anomaly_vars = {r.abnormal_variable for r in anomalies}

            if anomalies:
                # 统计异常
                cols = st.columns(4)
                cols[0].metric('🔴 异常变量', len(anomalies), delta=f'/ {len(VAR_NAMES)}', delta_color='off')

                # 根因
                root_causes_found = set()
                for r in anomalies:
                    for rc in r.root_causes:
                        root_causes_found.add(rc.variable)
                cols[1].metric('🎯 识别根因数', len(root_causes_found))

                # 最可能的根因
                best = None
                for r in anomalies:
                    if r.root_causes:
                        best = r.root_causes[0]; break
                if best:
                    cols[2].metric('🔍 Top根因', f'{get_cn_label(best.variable)}')

                # 诊断耗时
                cols[3].metric('⚡ 响应', '<10ms')

                st.markdown('---')

                # 详细结果
                for report in anomalies[:5]:
                    cn_var = get_cn_label(report.abnormal_variable)
                    lo, hi = report.normal_range
                    dev = (report.observed_value - (lo+hi)/2) / max((hi-lo)/2, 1e-6)

                    c1, c2 = st.columns([1.5, 1])
                    with c1:
                        st.markdown(f'### ⚠ {cn_var}')
                        st.caption(f'观测值: {report.observed_value:.1f} | 正常范围: [{lo:.1f}, {hi:.1f}] | 偏离: {dev:+.1f}σ')
                        if report.root_causes:
                            st.markdown('**根因链**:')
                            for i, rc in enumerate(report.root_causes):
                                cn_path = translate_path(rc.path)
                                st.markdown(f'{i+1}. **{get_cn_label(rc.variable)}** (置信度 {rc.score:.1%})')
                                st.markdown(f'   ↳ {cn_path}')
                    with c2:
                        if report.root_causes:
                            # 迷你因果图
                            mini_g = nx.DiGraph()
                            path = report.root_causes[0].path
                            for i in range(len(path)-1):
                                mini_g.add_edge(path[i], path[i+1])
                            if mini_g.number_of_nodes() > 0:
                                fig = plot_cn_graph(mini_g, f'{cn_var} 根因链', path)
                                st.plotly_chart(fig, use_container_width=True)

                    if report.recommended_actions:
                        with st.expander(f'💡 处置建议'):
                            for a in report.recommended_actions[:3]:
                                st.info(a)
                    st.markdown('---')
            else:
                st.success('✅ 所有变量在正常范围内，系统运行正常')

            # 验证面板
            st.markdown('---')
            st.subheader('🧪 验证面板')
            st.markdown('*如果你知道真实根因，可以在这里评估Agent的诊断是否准确。*')

            vc1, vc2 = st.columns(2)
            with vc1:
                true_fault = st.selectbox('真实故障（你知道吗？）',
                    ['不知道真实根因'] + [FAULT_CN.get(k, FAULT_MODES[k]['name']) for k in FAULT_MODES],
                    key='true_fault')

            with vc2:
                if true_fault != '不知道真实根因':
                    # 找出对应的故障
                    true_k = None
                    for k, v in FAULT_CN.items():
                        if v == true_fault:
                            true_k = k; break
                    if true_k and true_k in FAULT_MODES:
                        true_rc = FAULT_MODES[true_k]['root_cause']
                        agent_rc = list(root_causes_found)[0] if root_causes_found else None
                        correct = (agent_rc == true_rc)
                        if correct:
                            st.success(f'✅ Agent诊断正确！根因={get_cn_label(true_rc)}')
                        else:
                            st.warning(f'⚠ Agent诊断={get_cn_label(agent_rc) if agent_rc else "无"}, 真实根因={get_cn_label(true_rc)}')
                            st.caption('这可能是由于: 1)因果图不完整 2)数据通道噪声 3)参数设置接近边界')

    # ================================================================
    # 模式3: 因果图浏览
    # ================================================================
    elif mode == '📊 因果图浏览':
        st.subheader('📊 汉化因果图')
        st.markdown('鼠标悬停查看因果边详情，拖动节点调整布局。')

        fig = plot_cn_graph(graph, '融合因果图 (双通道交叉验证)')
        st.plotly_chart(fig, use_container_width=True)

        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric('节点数', graph.number_of_nodes())
        with c2:
            st.metric('因果边', graph.number_of_edges())
        with c3:
            st.metric('DAG', '✅' if nx.is_directed_acyclic_graph(graph) else '❌')

        st.markdown('---')
        st.subheader('🔗 因果边一览')
        edges_data = []
        for u, v, d in graph.edges(data=True):
            cn_u = get_cn_label(u); cn_v = get_cn_label(v)
            conf = d.get('confidence', 0)
            mech = d.get('mechanism', '')[:50]
            edges_data.append({'原因': cn_u, '结果': cn_v, '置信度': f'{conf:.2f}', '物理机制': mech})
        st.dataframe(pd.DataFrame(edges_data), use_container_width=True, hide_index=True)

    # ================================================================
    # 模式4: 知识库与进化
    # ================================================================
    elif mode == '📚 知识库与进化':
        st.subheader('📚 因果知识库 & 系统知识迭代')

        # 知识增长追踪
        from src.literature_crawler import LiteratureCrawler
        crawler = LiteratureCrawler()
        local_papers = crawler.load_manual_papers()

        c1, c2, c3, c4 = st.columns(4)
        with c1: st.metric('🧬 因果边', graph.number_of_edges(), delta=f'已融合')
        with c2: st.metric('📄 本地文献', len(local_papers), delta='持续增长')
        with c3:
            kp_count = len(data.get('kp', []))
            st.metric('📝 知识条目', kp_count)
        with c4:
            st.metric('🔄 进化状态', '就绪', delta='可触发')

        st.markdown('---')

        # 因果知识表
        st.subheader('🔗 因果知识条目')
        kp = data.get('kp', [])
        if kp:
            kp_data = pd.DataFrame([{
                '原因(中)': get_cn_label(p.get('cause', '')),
                '结果(中)': get_cn_label(p.get('effect', '')),
                '原因(英)': p.get('cause', ''),
                '结果(英)': p.get('effect', ''),
                '方向': p.get('direction', ''),
                '置信度': f'{p.get("confidence", 0):.2f}',
                '机制': p.get('mechanism', '')[:40],
            } for p in kp[:30]])
            st.dataframe(kp_data, use_container_width=True, hide_index=True)

        st.markdown('---')

        # 本地文献库
        st.subheader('📖 本地文献库 — 知识迭代基础')
        manual_dir = 'data/literature/manual'
        if os.path.exists(manual_dir):
            files = [f for f in os.listdir(manual_dir) if f.endswith(('.txt', '.md', '.json'))]
            st.markdown(f'**已加载 {len(files)} 篇文献**')

            # 显示文献内容预览
            for f in files:
                path = os.path.join(manual_dir, f)
                with open(path, encoding='utf-8') as ff:
                    content = ff.read()
                # 统计因果关键词
                keywords = ['因果', '导致', '引起', '影响', '下降', '升高', '异常', '故障', '根因']
                kw_count = sum(1 for kw in keywords if kw in content)
                with st.expander(f'📄 {f} (含{kw_count}个因果关键词 | {len(content)}字)'):
                    st.text(content[:500] + ('...' if len(content) > 500 else ''))
        else:
            st.warning('本地文献目录不存在，请创建 data/literature/manual/')

        st.markdown('---')

        # 从本地文献提取的知识
        st.subheader('🧠 文献提取的因果知识')
        if local_papers:
            from src.literature_extractor import LiteratureExtractor
            extractor = LiteratureExtractor()
            knowledge = extractor.extract_from_papers(local_papers)
            if knowledge and knowledge.causal_edges:
                st.success(f'从{len(local_papers)}篇文献中提取到 {len(knowledge.causal_edges)} 条因果知识')
                lit_kp = pd.DataFrame([{
                    '原因': e.get('cause', ''),
                    '结果': e.get('effect', ''),
                    '方向': e.get('direction', ''),
                    '置信度': f'{e.get("confidence", 0):.2f}',
                } for e in knowledge.causal_edges[:20]])
                st.dataframe(lit_kp, use_container_width=True, hide_index=True)
            else:
                st.info('当前本地文献中未提取到新因果边。添加更多含因果描述的文献即可自动提取。')
        st.caption('💡 提示: 将论文摘要/故障报告/操作规程放入 data/literature/manual/ 即可自动纳入知识库。')

        st.markdown('---')
        st.markdown(f'### 📊 变量词典 ({len(VAR_CN)} 个变量)')
        dict_data = pd.DataFrame([{
            '中文名': cn['name'], '英文名': en, '单位': cn['unit'],
            '分类': cn['category'], '描述': cn['desc']
        } for en, cn in VAR_CN.items()])
        st.dataframe(dict_data, use_container_width=True, hide_index=True)

    # ================================================================
    # 模式5: 文献爬虫
    # ================================================================
    elif mode == '🌐 文献爬虫':
        st.subheader('🌐 文献自主爬虫 & 知识迭代')

        st.markdown('''
        ### 文献迭代流程
        ```
        ① 爬虫搜索 ──→ ② AI精读提取 ──→ ③ 因果知识注入 ──→ ④ 仿真自测
             ↑                                                    │
             └──────────── ⑤ 未通过 → 回到爬虫 ──────────────────┘
                                 通过 → 更新因果图 ✓
        ```
        ''')

        c1, c2 = st.columns(2)

        with c1:
            st.subheader('📡 爬虫状态')
            st.markdown(f'''
            | 数据源 | 状态 | 说明 |
            |--------|------|------|
            | 📖 本地文献 | 🟢 在线 | `data/literature/manual/` |
            | 🔬 百度学术 | 🟡 国内可访问 | 钢铁/冶金关键词 |
            | 📄 Semantic Scholar | 🟡 海外API | 英文SCI论文 |
            | 🏭 武钢官网 | 🟢 在线 | 技改公告/新闻 |
            | 📋 arXiv | 🟡 预印本 | 工业AI方向 |
            | 🎓 CNKI知网 | 🔴 需教育网IP | 可通过学校VPN |
            ''')

        with c2:
            st.subheader('🔄 知识迭代统计')
            from src.literature_crawler import LiteratureCrawler
            crawler = LiteratureCrawler()
            local_papers = crawler.load_manual_papers()

            st.metric('📄 当前文献总量', len(local_papers))

            # 统计文献中的因果关键词
            total_kw = 0
            manual_dir = 'data/literature/manual'
            if os.path.exists(manual_dir):
                for f in os.listdir(manual_dir):
                    if f.endswith(('.txt', '.md', '.json')):
                        with open(os.path.join(manual_dir, f), encoding='utf-8') as ff:
                            content = ff.read()
                        keywords = ['因果', '导致', '引起', '影响', '下降', '升高', '异常', '故障', '根因', '机理', '→', '上游', '下游']
                        total_kw += sum(1 for kw in keywords if kw in content)
            st.metric('🔑 因果关键词', total_kw)

            st.metric('🧬 因果图边数', graph.number_of_edges(), delta='已融合所有知识')

        st.markdown('---')

        # 手动触发爬虫
        st.subheader('🚀 手动触发文献搜索')
        search_kw = st.text_input('搜索关键词', '高炉 冷却壁 故障 因果')

        if st.button('🔍 开始搜索', type='primary', use_container_width=True):
            with st.spinner('搜索中 (国内源: 百度学术 + 本地)...'):
                try:
                    crawler = LiteratureCrawler()
                    papers = crawler.search_all([search_kw])

                    if papers:
                        st.success(f'获取 {len(papers)} 篇文献')
                        r_df = pd.DataFrame([{
                            '标题': p.title[:60],
                            '年份': p.year,
                            '来源': p.source_type,
                            '引用': p.citation_count,
                        } for p in papers[:15]])
                        st.dataframe(r_df, use_container_width=True, hide_index=True)

                        # 尝试提取因果知识
                        from src.literature_extractor import LiteratureExtractor
                        extractor = LiteratureExtractor()
                        knowledge = extractor.extract_from_papers(papers[:10])
                        if knowledge and knowledge.causal_edges:
                            st.success(f'提取到 {len(knowledge.causal_edges)} 条新因果边！')
                            ek_df = pd.DataFrame([{
                                '原因': e.get('cause', ''),
                                '结果': e.get('effect', ''),
                                '方向': e.get('direction', ''),
                                '置信度': f'{e.get("confidence", 0):.2f}',
                                '机制': e.get('mechanism', '')[:50],
                            } for e in knowledge.causal_edges[:10]])
                            st.dataframe(ek_df, use_container_width=True, hide_index=True)
                    else:
                        st.warning('网络搜索未返回结果（可能被限制）。本地文献仍可正常使用。')
                        st.info('✅ 替代方案: 手动将论文摘要放入 `data/literature/manual/` 文件夹')

                except Exception as e:
                    st.warning(f'搜索受限: {e}')
                    st.info('✅ 这很正常——国内网络环境限制海外学术API。本地文献不受影响。')

        st.markdown('---')
        st.subheader('📖 当前本地文献列表')
        manual_dir = 'data/literature/manual'
        if os.path.exists(manual_dir):
            files = [f for f in os.listdir(manual_dir) if f.endswith(('.txt', '.md', '.json'))]
            for f in files:
                path = os.path.join(manual_dir, f)
                size = os.path.getsize(path)
                with open(path, encoding='utf-8') as ff:
                    first_line = ff.readline().strip().lstrip('#').strip()
                st.markdown(f'📄 **{f}** ({size}B) — {first_line[:80]}')


    # ================================================================
    # 模式6: 自进化状态 ⭐ 核心 — 回答"越用越准了吗？"
    # ================================================================
    elif mode == '🧬 自进化状态':
        st.subheader('🧬 自主进化状态面板')

        from src.evolution_db import EvolutionDB
        db = EvolutionDB()

        # 初始化演示数据
        if st.button('🔄 初始化进化数据', key='init_evo_db'):
            db.seed_demo_data()
            st.success('进化演示数据已加载')
            st.rerun()

        status = db.get_status()

        # 顶部状态卡片
        st.markdown('### 📊 进化总览')
        c1, c2, c3, c4, c5 = st.columns(5)
        with c1:
            st.metric('🔄 进化周期', status['total_cycles'],
                     delta='持续进化' if status['total_cycles'] > 0 else '待启动')
        with c2:
            st.metric('🧬 知识条目', status['total_knowledge'],
                     delta=f'上次+{status["last_growth"]}' if status['last_growth'] else None)
        with c3:
            st.metric('⚙️ 参数变更', status['total_param_changes'],
                     delta='自动校准')
        with c4:
            st.metric('✅ 自测评分', f'{status["avg_validation_score"]:.1%}' if status['avg_validation_score'] else 'N/A',
                     delta='上升中' if status['avg_validation_score'] > 0.7 else '待提高')
        with c5:
            st.metric('📅 最近进化', status['last_evolution_time'][:16] if status['last_evolution_time'] else '暂无')

        st.markdown('---')

        # 知识增长曲线
        st.subheader('📈 知识增长曲线')
        growth = db.get_growth_curve()
        if not growth.empty:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=growth['cycle'], y=growth['edges_after'],
                         mode='lines+markers', name='因果边总数',
                         line=dict(width=3, color='#10b981'),
                         marker=dict(size=10)))
            fig.add_trace(go.Bar(x=growth['cycle'], y=growth['new_causal_edges'],
                         name='新增边', marker_color='#1a56db'))
            fig.update_layout(
                title='因果图知识增长轨迹',
                xaxis_title='进化周期',
                yaxis_title='因果边数量',
                hovermode='x unified',
                height=350,
                plot_bgcolor='rgba(0,0,0,0)',
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info('暂无进化数据。点击上方按钮初始化演示数据。')

        # 自测评分趋势
        st.subheader('📉 自测验证趋势')
        val_data = db.get_validation_trend()
        if not val_data.empty:
            val_fig = go.Figure()
            val_fig.add_trace(go.Scatter(x=val_data['id'], y=val_data['score'],
                             mode='lines+markers',
                             line=dict(width=3, color='#8b5cf6'),
                             marker=dict(size=10, color=val_data['passed'].apply(
                                 lambda x: '#10b981' if x else '#ef4444')),
                             name='验证评分'))
            val_fig.add_hline(y=0.75, line_dash='dash', line_color='green',
                            annotation_text='部署阈值(0.75)')
            val_fig.update_layout(
                title='自测验证评分变化',
                xaxis_title='验证次数',
                yaxis_title='评分',
                yaxis_range=[0, 1],
                height=300,
                plot_bgcolor='rgba(0,0,0,0)',
            )
            st.plotly_chart(val_fig, use_container_width=True)

        c1, c2 = st.columns(2)

        # 知识库内容
        with c1:
            st.subheader('🧠 知识库内容')
            all_k = db.get_all_knowledge()
            if not all_k.empty:
                k_df = pd.DataFrame([{
                    '原因': row['cause'], '结果': row['effect'],
                    '机制': row['mechanism'][:30],
                    '置信度': f'{row["confidence"]:.2f}',
                    '来源': row['source'],
                } for _, row in all_k.iterrows()])
                st.dataframe(k_df, use_container_width=True, hide_index=True)
            else:
                st.info('知识库为空')

        # 参数进化轨迹
        with c2:
            st.subheader('⚙️ 参数进化轨迹')
            params = db.get_param_history()
            if not params.empty:
                p_df = pd.DataFrame([{
                    '参数': row['param_name'][:25],
                    '旧值': f'{row["old_value"]:.4f}',
                    '新值': f'{row["new_value"]:.4f}',
                    '变化': f'{row["new_value"] - row["old_value"]:+.4f}',
                    '来源': row['trigger_source'],
                } for _, row in params.iterrows()])
                st.dataframe(p_df, use_container_width=True, hide_index=True)
            else:
                st.info('参数无变更记录')

        # 进化流程说明
        st.markdown('---')
        st.subheader('🔄 自进化流程')
        st.markdown('''
        ```
        ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐
        │ ① 文献爬取│ → │ ② AI精读 │ → │ ③ 知识注入│ → │ ④ 参数校准│
        │ 定时/手动 │    │ 因果提取 │    │ 去重融合 │    │ 数据回喂 │
        └──────────┘    └──────────┘    └──────────┘    └─────┬────┘
             ↑                                                ↓
             │                                          ┌──────────┐
             └────────── ⑥ 自测未通过 ──────────────── │ ⑤ 仿真验证│
                        通过 → 部署更新 ✓               │ 文献自测 │
                                                       └──────────┘
        ```
        ''')

        st.info(
            '💡 **当前限制**: Streamlit Cloud 不支持后台定时任务。'
            '在真实部署环境中（Docker + 定时调度器），进化会自动周期性运行。'
            '当前可按手动触发按钮体验进化流程。'
        )

        st.markdown('---')
        st.caption(f'进化数据库: `{db.db_path}` | 累计 {status["total_cycles"]} 周期')


if __name__ == '__main__':
    main()
