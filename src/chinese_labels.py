"""
中文标签映射 — 工业过程变量中文名 + 设备/行业术语

用法:
  from src.chinese_labels import VAR_CN, EDGE_CN, get_cn_label
  cn_name = VAR_CN["CW_Valve"]  # "冷却水阀门开度"
"""

# ================================================================
# 合成数据变量 (12个) — 中文名 + 单位 + 描述
# ================================================================
VAR_CN = {
    # 根变量 (外部条件)
    "Feed_Flow": {"name": "进料流量", "unit": "L/min", "desc": "反应器主进料管道流量",
                  "category": "进料系统"},
    "Feed_Conc": {"name": "进料浓度", "unit": "mol/L", "desc": "进料中反应物的摩尔浓度",
                  "category": "进料系统"},
    "CW_Inlet_Temp": {"name": "冷却水入口温度", "unit": "°C", "desc": "冷却塔来水温度",
                      "category": "冷却系统"},
    "CW_Valve": {"name": "冷却水阀门开度", "unit": "%", "desc": "冷却水调节阀开度百分比",
                 "category": "冷却系统"},

    # 中间变量 (过程状态)
    "Reactor_Temp": {"name": "反应器温度", "unit": "°C", "desc": "反应器内部实时温度",
                     "category": "反应器"},
    "Reactor_Press": {"name": "反应器压力", "unit": "MPa", "desc": "反应器内部压力",
                      "category": "反应器"},
    "CW_Flow": {"name": "冷却水流量", "unit": "m³/h", "desc": "冷却水回路实际流量",
                "category": "冷却系统"},
    "Reaction_Rate": {"name": "反应速率", "unit": "mol/(L·h)", "desc": "化学反应速率",
                      "category": "反应器"},

    # 输出变量 (结果)
    "Product_Conc": {"name": "产物浓度", "unit": "mol/L", "desc": "目标产物出口浓度",
                     "category": "产品质量"},
    "Byproduct_Conc": {"name": "副产物浓度", "unit": "mol/L", "desc": "副反应产物浓度",
                       "category": "产品质量"},
    "HX_Outlet_Temp": {"name": "换热器出口温度", "unit": "°C", "desc": "换热器物料出口温度",
                       "category": "换热系统"},
    "Energy_Index": {"name": "能耗指标", "unit": "kW", "desc": "综合能耗指数",
                     "category": "能源效率"},
}

# ================================================================
# TEP数据变量 (52个) — 精选关键变量中文名
# ================================================================
TEP_VAR_CN = {
    "A_Feed_Flow": {"name": "A进料流量", "unit": "kmol/h", "category": "进料系统"},
    "D_Feed_Flow": {"name": "D进料流量", "unit": "kmol/h", "category": "进料系统"},
    "E_Feed_Flow": {"name": "E进料流量", "unit": "kmol/h", "category": "进料系统"},
    "AC_Total_Flow": {"name": "总进料流量(A+C)", "unit": "kmol/h", "category": "进料系统"},
    "Recycle_Flow": {"name": "循环流量", "unit": "kmol/h", "category": "循环系统"},
    "Reactor_Feed": {"name": "反应器进料", "unit": "kmol/h", "category": "反应器"},
    "Reactor_Press": {"name": "反应器压力", "unit": "kPa", "category": "反应器"},
    "Reactor_Level": {"name": "反应器液位", "unit": "%", "category": "反应器"},
    "Reactor_Temp": {"name": "反应器温度", "unit": "°C", "category": "反应器"},
    "Purge_Flow": {"name": "排放流量", "unit": "kmol/h", "category": "排放系统"},
    "Separator_Temp": {"name": "分离器温度", "unit": "°C", "category": "分离器"},
    "Separator_Level": {"name": "分离器液位", "unit": "%", "category": "分离器"},
    "Separator_Press": {"name": "分离器压力", "unit": "kPa", "category": "分离器"},
    "Stripper_Temp": {"name": "汽提塔温度", "unit": "°C", "category": "汽提塔"},
    "Stripper_Level": {"name": "汽提塔液位", "unit": "%", "category": "汽提塔"},
    "Compressor_Work": {"name": "压缩机功率", "unit": "kW", "category": "压缩机"},
    "Reactor_CW_Temp": {"name": "反应器冷却水出口温度", "unit": "°C", "category": "冷却系统"},
    "Separator_CW_Temp": {"name": "分离器冷却水出口温度", "unit": "°C", "category": "冷却系统"},
    "Comp_A_Feed": {"name": "进料A组分", "unit": "mol%", "category": "成分分析"},
    "Comp_D_Product": {"name": "产物D组分", "unit": "mol%", "category": "成分分析"},
    "Valve_Reactor_CW": {"name": "反应器冷却水阀", "unit": "%", "category": "阀门"},
    "Valve_Cond_CW": {"name": "冷凝器冷却水阀", "unit": "%", "category": "阀门"},
    "Valve_Purge": {"name": "排放阀", "unit": "%", "category": "阀门"},
}

# ================================================================
# 因果边中文描述
# ================================================================
EDGE_CN = {
    ("冷却水阀门开度", "冷却水流量"): "阀门开度↑ → 冷却水流量↑",
    ("冷却水流量", "反应器温度"): "冷却水流量↑ → 换热量↑ → 反应器温度↓",
    ("冷却水流量", "换热器出口温度"): "冷却水流量↑ → 换热充分 → 出口温度↓",
    ("冷却水入口温度", "反应器温度"): "入口水温↑ → 换热温差↓ → 反应器温度↑",
    ("反应器温度", "反应速率"): "温度↑ → Arrhenius效应 → 反应速率↑",
    ("反应器温度", "反应器压力"): "温度↑ → 气相膨胀 → 压力↑",
    ("反应器温度", "能耗指标"): "温度↑ → 冷却负荷↑ → 能耗↑",
    ("反应器温度", "换热器出口温度"): "反应器温度↑ → 待冷却物料温度↑ → 出口温度↑",
    ("进料流量", "反应器压力"): "进料↑ → 物料累积 → 压力↑",
    ("进料流量", "产物浓度"): "进料↑ → 停留时间↓ → 反应不充分 → 产物浓度↓",
    ("进料浓度", "反应速率"): "浓度↑ → 反应物浓度梯度↑ → 反应速率↑",
    ("反应速率", "产物浓度"): "反应速率↑ → 主反应产物↑",
    ("反应速率", "副产物浓度"): "反应速率↑↑ → 副反应加剧 → 副产物↑",
    ("反应速率", "能耗指标"): "反应速率↑ → 放热量↑ → 冷却能耗↑",
    ("反应器压力", "反应速率"): "压力↑ → 气相分压↑ → 反应速率↑",
}

# ================================================================
# 故障模式中文描述
# ================================================================
FAULT_CN = {
    "FAULT_COOLING_VALVE_STUCK": "冷却水阀门卡滞 — 阀门逐渐卡在低位，冷却不足导致温度失控",
    "FAULT_FEED_CONC_DROP": "进料浓度下降 — 上游原料品质波动，反应速率持续降低",
    "FAULT_CW_INLET_TEMP_HIGH": "冷却水入口温度升高 — 冷却塔效率下降，换热能力不足",
    "FAULT_FEED_FLOW_SURGE": "进料流量突增 — 进料泵控制异常，流量瞬间增大",
    "FAULT_COOLING_PUMP_FAIL": "冷却水泵故障 — 叶轮磨损，冷却水流量逐渐下降",
    "FAULT_REACTOR_FOULING": "反应器结垢 — 内壁结垢导致换热效率缓慢下降",
    "FAULT_SENSOR_DRIFT_TEMP": "温度传感器漂移 — 读数持续偏高，实际温度正常",
    "FAULT_FEED_VALVE_STUCK": "进料阀门卡滞 — 调节阀卡在低位，进料流量不足",
    "FAULT_CATALYST_DEACTIVATION": "催化剂失活 — 活性缓慢下降，产物质量不达标",
    "FAULT_COMBINED_VALVE_AND_TEMP": "复合故障 — 阀门卡滞+冷却水温升高，多根因并存",
}


def get_cn_label(en_name: str) -> str:
    """获取变量中文名"""
    cn = VAR_CN.get(en_name, TEP_VAR_CN.get(en_name, {}))
    return cn.get("name", en_name)

def get_cn_unit(en_name: str) -> str:
    """获取变量单位"""
    cn = VAR_CN.get(en_name, TEP_VAR_CN.get(en_name, {}))
    return cn.get("unit", "")

def get_category(en_name: str) -> str:
    """获取变量所属设备类别"""
    cn = VAR_CN.get(en_name, TEP_VAR_CN.get(en_name, {}))
    return cn.get("category", "其他")

def translate_path(path) -> str:
    """翻译因果路径为中文"""
    return " → ".join(get_cn_label(v) for v in path)
