"""
真实钢铁工业数据加载器

两个数据源:
  1. 工业数字孪生轧钢数据 (Kaggle) — 5台轧机, 1年, 10分钟间隔, 26万行
  2. UCI钢板缺陷数据 — 1941行, 27个特征, 7种缺陷类型

用法:
  loader = SteelDataLoader()
  telemetry, failures = loader.load_rolling_mill()
  df_steel = loader.load_plate_defects()
"""

import sys, os, json, glob
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple


class SteelDataLoader:
    """钢铁工业真实数据加载器"""

    # 轧机传感器变量 → 中文描述
    SENSOR_CN = {
        "bearing_temp_c": {"name": "轴承温度", "unit": "°C", "desc": "主轴承实时温度"},
        "vibration_rms": {"name": "振动RMS", "unit": "mm/s", "desc": "振动均方根值"},
        "motor_torque_nm": {"name": "电机转矩", "unit": "Nm", "desc": "驱动电机输出转矩"},
        "load_index": {"name": "负载指数", "unit": "-", "desc": "标准化负载(0-2)"},
        "bearing_health": {"name": "轴承健康度", "unit": "-", "desc": "0=失效, 1=完好"},
        "lube_health": {"name": "润滑健康度", "unit": "-", "desc": "0=干磨, 1=正常"},
        "align_health": {"name": "对中健康度", "unit": "-", "desc": "0=严重偏斜, 1=对中良好"},
        "motor_health": {"name": "电机健康度", "unit": "-", "desc": "0=故障, 1=正常"},
        "cumulative_op_hours": {"name": "累计运行小时", "unit": "h", "desc": "总运行时长"},
    }

    # 故障根因类型
    FAILURE_ROOT_CAUSES = [
        "Motor_or_Drive_Issue",       # 电机/驱动故障
        "Overheat_Bearing_or_Lube",   # 轴承过热/润滑不良
        "Bearing_Wear_or_Misalignment", # 轴承磨损/不对中
        "Lubrication_Failure",        # 润滑失效
        "Alignment_Drift",            # 对中漂移
        "Overload_or_Jam",            # 过载/卡料
    ]

    def load_rolling_mill(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        加载工业数字孪生轧钢数据

        数据集: 5台轧机(Mill_01~Mill_05), 2025全年, 10分钟采样
        26万行 × 38列传感器数据 + 139次真实故障记录 + 552次维护记录

        Returns:
            (telemetry, failures, maintenance)
        """
        kaggle_path = os.path.expanduser(
            "~/.cache/kagglehub/datasets/aiwithcagri/"
            "industrial-digital-twin-dataset-1-year/versions/1"
        )

        telemetry = pd.read_parquet(os.path.join(kaggle_path, "telemetry_2025.parquet"))
        failures = pd.read_parquet(os.path.join(kaggle_path, "failures_2025.parquet"))
        maintenance = pd.read_parquet(os.path.join(kaggle_path, "maintenance_2025.parquet"))

        # 时间解析
        telemetry["timestamp"] = pd.to_datetime(telemetry["timestamp"])
        failures["timestamp_occurrence"] = pd.to_datetime(failures["timestamp_occurrence"])

        print(f"轧钢数据加载完成: {len(telemetry)}行遥测, {len(failures)}次故障, "
              f"{len(maintenance)}次维护")

        return telemetry, failures, maintenance

    def extract_causal_pairs_from_failures(self, failures: pd.DataFrame) -> List[Dict]:
        """
        从故障记录中提取因果关系（注入Agent因果图）

        故障记录格式:
          machine_id | failure_category | root_cause | bearing_temp_c_at_failure | vibration_rms_at_failure | ...

        提取逻辑:
          每种root_cause → 观察故障时刻的传感器异常模式 → 形成因果边
        """
        causal_pairs = []

        # 统计每种根因在故障时刻的传感器异常模式
        for cause in self.FAILURE_ROOT_CAUSES:
            subset = failures[failures["root_cause"] == cause]
            if len(subset) < 3:
                continue

            # 故障时刻 vs 正常运行时的传感器差异
            avg_temp = subset["bearing_temp_c_at_failure"].mean()
            avg_vib = subset["vibration_rms_at_failure"].mean()

            if cause == "Overheat_Bearing_or_Lube":
                if avg_temp > 75:  # 温度显著偏高
                    causal_pairs.append({
                        "cause": "润滑不足/润滑脂劣化",
                        "effect": "轴承温度",
                        "direction": "negative",
                        "mechanism": f"润滑不良→摩擦增大→温升(故障时均温{avg_temp:.0f}°C)",
                        "confidence": min(0.9, len(subset) / 20),
                        "evidence": f"2025年{len(subset)}次故障记录",
                    })

            elif cause == "Bearing_Wear_or_Misalignment":
                if avg_vib > 2.5:
                    causal_pairs.append({
                        "cause": "轴承磨损/不对中",
                        "effect": "振动幅值",
                        "direction": "positive",
                        "mechanism": f"磨损/不对中→振动增大(故障时均值{avg_vib:.1f}mm/s)",
                        "confidence": min(0.9, len(subset) / 20),
                        "evidence": f"2025年{len(subset)}次故障记录",
                    })

            elif cause == "Motor_or_Drive_Issue":
                causal_pairs.append({
                    "cause": "电机/驱动故障",
                    "effect": "电机转矩",
                    "direction": "positive",
                    "mechanism": f"电机异常→转矩波动/过载(累计{len(subset)}次故障)",
                    "confidence": min(0.85, len(subset) / 20),
                })

            elif cause == "Lubrication_Failure":
                causal_pairs.append({
                    "cause": "润滑失效",
                    "effect": "轴承温度",
                    "direction": "negative",
                    "mechanism": f"润滑失效→干摩擦→急剧温升",
                    "confidence": min(0.9, len(subset) / 20),
                })

        print(f"从{len(failures)}次故障中提取{len(causal_pairs)}条因果知识")
        return causal_pairs

    def get_fault_evolution_trace(self, telemetry: pd.DataFrame,
                                   failure_id: str,
                                   failures: pd.DataFrame
                                   ) -> pd.DataFrame:
        """
        获取某次故障前后的传感器演化轨迹

        用于分析: 故障是突然发生还是逐渐退化?
                  哪些传感器先异常? (因果传播方向)
        """
        fault_row = failures[failures["failure_id"] == failure_id]
        if fault_row.empty:
            return pd.DataFrame()

        fault_time = fault_row.iloc[0]["timestamp_occurrence"]
        machine = fault_row.iloc[0]["machine_id"]

        # 故障前24小时 + 故障后6小时的数据
        window = telemetry[
            (telemetry["machine_id"] == machine) &
            (telemetry["timestamp"] >= fault_time - pd.Timedelta(hours=24)) &
            (telemetry["timestamp"] <= fault_time + pd.Timedelta(hours=6))
        ].copy()

        window["hours_to_failure"] = (
            window["timestamp"] - fault_time
        ).dt.total_seconds() / 3600

        return window

    def load_plate_defects(self) -> pd.DataFrame:
        """加载UCI钢板缺陷数据集"""
        path = "data/steel_industry/raw/Faults.NNA"
        if not os.path.exists(path):
            print("钢板缺陷数据不存在，请先下载")
            return pd.DataFrame()

        # UCI格式: tab分隔, 无header
        cols = [
            "X_Minimum", "X_Maximum", "Y_Minimum", "Y_Maximum",
            "Pixels_Areas", "X_Perimeter", "Y_Perimeter",
            "Sum_of_Luminosity", "Minimum_of_Luminosity", "Maximum_of_Luminosity",
            "Length_of_Conveyer", "TypeOfSteel_A300", "TypeOfSteel_A400",
            "Steel_Plate_Thickness", "Edges_Index", "Empty_Index",
            "Square_Index", "Outside_X_Index", "Edges_X_Index", "Edges_Y_Index",
            "Outside_Global_Index", "LogOfAreas", "Log_X_Index",
            "Log_Y_Index", "Orientation_Index", "Luminosity_Index",
            "SigmoidOfAreas",
            "Pastry", "Z_Scratch", "K_Scratch", "Stains",
            "Dirtiness", "Bumps", "Other_Faults",
        ]
        df = pd.read_csv(path, sep="\t", names=cols, skiprows=1)
        print(f"钢板缺陷数据: {len(df)}行, {len(cols)}列, "
              f"{sum(df[['Pastry','Z_Scratch','K_Scratch','Stains','Dirtiness','Bumps','Other_Faults']].sum().sum())}个缺陷标记")
        return df


if __name__ == "__main__":
    loader = SteelDataLoader()

    # 测试轧钢数据
    tel, fail, maint = loader.load_rolling_mill()
    print(f"\n轧机总览:")
    for m in sorted(tel["machine_id"].unique()):
        m_data = tel[tel["machine_id"] == m]
        m_fail = fail[fail["machine_id"] == m]
        print(f"  {m}: {len(m_data)}条遥测, {len(m_fail)}次故障, "
              f"振动均值={m_data['vibration_rms'].mean():.1f}mm/s, "
              f"温度均值={m_data['bearing_temp_c'].mean():.0f}°C")

    # 提取因果知识
    causal = loader.extract_causal_pairs_from_failures(fail)
    print(f"\n因果知识: {len(causal)} 条")
    for c in causal:
        print(f"  {c['cause']} → {c['effect']} ({c['mechanism'][:50]})")
