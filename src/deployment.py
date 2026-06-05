"""
生产环境部署模块 — 从 Demo 到工厂的实际部署方案

三种部署模式:
  1. 离线模式: CSV → 批量诊断 → 报告
  2. 在线模式: MQTT/OPC-UA → 实时流 → 报警
  3. 全链路: 在线 + 文献进化 + 仿真自校准

使用示例:
  from src.deployment import DeployConfig, deploy_offline, deploy_online

  # 离线模式
  deploy_offline(csv_path="武钢故障数据.csv", output_dir="诊断报告/")

  # 在线模式 (需要 MQTT broker)
  deploy_online(mqtt_host="10.0.0.1", mqtt_topic="factory/sensors/#")
"""

import sys, os, json, time, threading
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))


@dataclass
class DeployConfig:
    """部署配置"""
    # 数据源
    data_source: str = "csv"        # csv / mqtt / opcua / api
    csv_path: str = ""
    mqtt_host: str = "localhost"
    mqtt_port: int = 1883
    mqtt_topic: str = "factory/#"
    opcua_url: str = ""

    # 输出
    alert_webhook: str = ""         # 钉钉/企业微信 Webhook
    log_dir: str = "data/logs"
    report_dir: str = "data/reports"

    # 进化
    auto_evolve: bool = True       # 自动文献进化
    evolve_interval_hours: int = 168  # 每周一次

    # 数据库
    db_type: str = "sqlite"         # sqlite / influxdb / postgres
    db_path: str = "data/agent.db"


# ================================================================
# 模式1: 离线批量诊断
# ================================================================
def deploy_offline(csv_path: str, output_dir: str = "data/reports",
                   config: DeployConfig = None):
    """
    离线模式: 拿历史CSV数据批量诊断

    适用场景:
      - 事故复盘: 昨天那起停机的根因是什么？
      - 周度巡检: 每周对上周数据做一次全面诊断
      - 数据迁移: 老系统导出的历史数据批量分析

    用法:
      deploy_offline("武钢/高炉数据_2026年5月.csv")
    """
    import pandas as pd
    from src.root_cause_analysis import RootCauseAnalyzer
    from src.agent_v2 import CausalAgentV2
    import networkx as nx

    print(f"[离线模式] 加载 {csv_path}...")
    df = pd.read_csv(csv_path)
    print(f"  数据: {len(df)} 行 × {len(df.columns)} 列")

    agent = CausalAgentV2()
    # 从工艺文档加载初始因果图
    from src.llm_causal_extract import extract_from_synthetic_doc, LLMCausalExtractor
    from src.synthetic_data_generator import VAR_NAMES, generate_process_documentation
    os.makedirs("data/synthetic", exist_ok=True)
    generate_process_documentation("data/synthetic")
    pairs = extract_from_synthetic_doc("data/synthetic/process_documentation.txt",
                                        VAR_NAMES)
    kg = LLMCausalExtractor.pairs_to_graph(pairs, VAR_NAMES)
    agent.causal_graph = kg

    analyzer = RootCauseAnalyzer(kg)
    # 用前100行建立正常基线
    analyzer.set_normal_ranges(df.iloc[:100])

    # 逐窗口分析
    results = []
    window_size = 50
    os.makedirs(output_dir, exist_ok=True)

    print(f"[离线模式] 开始诊断 (窗口={window_size}步)...")
    for start in range(0, len(df) - window_size, window_size):
        window = df.iloc[start:start + window_size]
        obs = {col: float(window[col].iloc[-1])
               for col in df.columns
               if col in kg.nodes()}

        reports = analyzer.analyze(obs, top_k=3)

        if reports:
            for r in reports:
                if r.root_causes:
                    results.append({
                        "time_window": f"{start}-{start+window_size}",
                        "abnormal_var": r.abnormal_variable,
                        "observed": r.observed_value,
                        "root_cause": r.root_causes[0].variable,
                        "score": r.root_causes[0].score,
                        "path": " → ".join(r.root_causes[0].path),
                    })

    # 生成报告
    report_path = os.path.join(output_dir,
                               f"diagnosis_{datetime.now().strftime('%Y%m%d_%H%M')}.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump({
            "source": csv_path,
            "total_windows": (len(df) - window_size) // window_size + 1,
            "anomalous_windows": len(results),
            "findings": results,
            "summary": _summarize_findings(results),
        }, f, ensure_ascii=False, indent=2)

    print(f"\n[离线模式] 完成！")
    print(f"  诊断窗口: {(len(df)-window_size)//window_size+1} 个")
    print(f"  发现异常: {len(results)} 个窗口")
    print(f"  报告: {report_path}")

    # 打印摘要
    summary = _summarize_findings(results)
    print(f"\n📊 根因分布:")
    for rc, count in list(summary.get("top_root_causes", {}).items())[:5]:
        print(f"  {rc}: {count} 次")

    return report_path


def _summarize_findings(results: List[Dict]) -> Dict:
    """汇总诊断发现"""
    from collections import Counter
    rc_counts = Counter(r["root_cause"] for r in results)
    abn_counts = Counter(r["abnormal_var"] for r in results)
    return {
        "top_root_causes": dict(rc_counts.most_common(10)),
        "top_abnormal_vars": dict(abn_counts.most_common(10)),
        "avg_score": sum(r["score"] for r in results) / len(results) if results else 0,
    }


# ================================================================
# 模式2: 在线实时监控
# ================================================================
def deploy_online(config: DeployConfig):
    """
    在线模式: 接MQTT/OPC-UA实时数据流，逐条诊断

    适用场景:
      - 工厂7×24实时监控
      - 异常秒级报警

    实际部署需要:
      pip install paho-mqtt    (MQTT)
      pip install opcua        (OPC-UA)
    """
    print(f"[在线模式] 连接 {config.mqtt_host}:{config.mqtt_port}...")

    try:
        import paho.mqtt.client as mqtt
        mqtt_available = True
    except ImportError:
        print("  ⚠ paho-mqtt 未安装。MQTT模式需要: pip install paho-mqtt")
        mqtt_available = False

    if not mqtt_available:
        print("  切换到模拟模式（用合成数据模拟实时流）...")
        _simulate_realtime(config)
        return

    # 真实MQTT模式
    from src.agent_v2 import CausalAgentV2
    from src.root_cause_analysis import RootCauseAnalyzer

    agent = CausalAgentV2()
    analyzer = RootCauseAnalyzer(agent.causal_graph)

    def on_message(client, userdata, msg):
        """收到传感器数据 → 即时诊断"""
        try:
            payload = json.loads(msg.payload)
            reports = analyzer.analyze(payload, top_k=2)
            if reports:
                for r in reports:
                    if r.root_causes:
                        alert = {
                            "timestamp": datetime.now().isoformat(),
                            "abnormal_var": r.abnormal_variable,
                            "root_cause": r.root_causes[0].variable,
                            "score": r.root_causes[0].score,
                            "recommendation": r.recommended_actions[0] if r.recommended_actions else "",
                        }
                        print(f"🚨 {alert['abnormal_var']} 异常 → {alert['root_cause']}")
                        # 发送报警
                        if config.alert_webhook:
                            _send_alert(config.alert_webhook, alert)
        except Exception as e:
            print(f"诊断异常: {e}")

    client = mqtt.Client()
    client.on_message = on_message
    client.connect(config.mqtt_host, config.mqtt_port, 60)
    client.subscribe(config.mqtt_topic)
    print(f"[在线模式] 监听 {config.mqtt_topic}...")
    client.loop_forever()


def _simulate_realtime(config: DeployConfig):
    """模拟实时数据流（演示用）"""
    from src.agent_v2 import CausalAgentV2
    from src.root_cause_analysis import RootCauseAnalyzer
    from src.synthetic_data_generator import SyntheticProcessSimulator, VAR_NAMES, FAULT_MODES
    import random

    agent = CausalAgentV2()
    from src.llm_causal_extract import extract_from_synthetic_doc, LLMCausalExtractor
    from src.synthetic_data_generator import generate_process_documentation
    os.makedirs("data/synthetic", exist_ok=True)
    generate_process_documentation("data/synthetic")
    pairs = extract_from_synthetic_doc("data/synthetic/process_documentation.txt", VAR_NAMES)
    kg = LLMCausalExtractor.pairs_to_graph(pairs, VAR_NAMES)
    agent.causal_graph = kg

    sim = SyntheticProcessSimulator(seed=42)
    df_normal = sim.simulate(n_steps=500)
    analyzer = RootCauseAnalyzer(kg)
    analyzer.set_normal_ranges(df_normal.iloc[:300])

    print("[模拟实时] 开始监控 (Ctrl+C 停止)...")
    count = 0
    last_alert = None

    try:
        while True:
            count += 1
            fault_name = random.choice(list(FAULT_MODES.keys()))
            df, _ = sim.generate_fault_dataset(n_normal=100, n_fault=100, fault_name=fault_name)
            fault_data = df[df['fault'] == fault_name]
            obs = {v: float(fault_data.iloc[50][v]) for v in VAR_NAMES}

            reports = analyzer.analyze(obs, top_k=1)
            if reports and reports[0].root_causes:
                rc = reports[0].root_causes[0]
                current_alert = rc.variable
                if current_alert != last_alert:
                    print(f"\n🚨 [{count}] {fault_name}")
                    print(f"   异常: {reports[0].abnormal_variable}")
                    print(f"   根因: {rc.variable} (评分:{rc.score:.3f})")
                    print(f"   路径: {' → '.join(rc.path[:3])}")
                    last_alert = current_alert

            time.sleep(0.5)

    except KeyboardInterrupt:
        print(f"\n[模拟实时] 停止。监控 {count} 条数据。")


def _send_alert(webhook_url: str, alert_data: Dict):
    """发送报警到钉钉/企业微信"""
    try:
        import requests
        msg = {
            "msgtype": "text",
            "text": {
                "content": (f"🚨 工业智能体报警\n"
                           f"异常变量: {alert_data['abnormal_var']}\n"
                           f"根因: {alert_data['root_cause']}\n"
                           f"评分: {alert_data['score']:.3f}\n"
                           f"建议: {alert_data.get('recommendation', '')}")
            }
        }
        requests.post(webhook_url, json=msg, timeout=5)
    except Exception as e:
        print(f"报警发送失败: {e}")


# ================================================================
# 快速启动
# ================================================================
if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description='因果增强工业智能体 — 部署工具')
    p.add_argument('--mode', choices=['offline', 'online', 'simulate'], default='simulate')
    p.add_argument('--csv', help='离线模式的CSV文件路径')
    p.add_argument('--mqtt-host', default='localhost')
    p.add_argument('--webhook', help='报警Webhook URL')

    args = p.parse_args()
    config = DeployConfig(
        csv_path=args.csv or "",
        mqtt_host=args.mqtt_host,
        alert_webhook=args.webhook or "",
    )

    if args.mode == 'offline' and args.csv:
        deploy_offline(args.csv, config=config)
    elif args.mode == 'online':
        deploy_online(config)
    else:
        # 默认: 模拟实时监控
        _simulate_realtime(config)
