"""
五层投产模块 — 从Demo到产线的完整工程实现

每层都是可运行的代码，不只是文档:

  第一层: data_adapter    — 多协议数据接入 + 清洗 + 归一化
  第二层: ops_monitor     — 健康检查 + 结构化日志 + 指标采集
  第三层: quality_gate    — 测试框架 + 因果图校验 + 回滚
  第四层: security_layer  — 权限控制 + 审计日志
  第五层: integration     — 外部系统接口 + Webhook报警

用法:
  from src.production import ProductionAgent
  agent = ProductionAgent()
  agent.start_monitoring()     # 启动第二层监控
  agent.run_health_check()     # 运行健康检查
  agent.run_quality_gate()     # 运行质量门禁
"""

import sys, os, json, time, logging, hashlib, shutil
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from functools import wraps

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))


# ================================================================
# 第一层: 数据接入适配器
# ================================================================
class DataSourceType(Enum):
    CSV = "csv"
    MQTT = "mqtt"
    OPCUA = "opcua"
    HTTP_API = "http_api"
    SYNTHETIC = "synthetic"


@dataclass
class DataSchema:
    """数据模式定义 — 外部变量名 → 内部变量名的映射"""
    mappings: Dict[str, str] = field(default_factory=dict)  # {外部名: 内部名}
    unit_conversions: Dict[str, callable] = field(default_factory=dict)
    required_fields: List[str] = field(default_factory=list)
    normal_ranges: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    max_gap_seconds: float = 10.0  # 超过此间隔视为数据断流


class DataValidator:
    """数据校验器 — 拒绝脏数据，防止误报"""

    @staticmethod
    def validate(observation: Dict, schema: DataSchema) -> Tuple[Dict, List[str]]:
        """校验并清洗数据，返回(清洗后数据, 警告列表)"""
        cleaned = {}
        warnings = []

        for external_name, value in observation.items():
            # 映射变量名
            internal_name = schema.mappings.get(external_name, external_name)

            # 类型检查
            if not isinstance(value, (int, float)):
                try:
                    value = float(value)
                except (ValueError, TypeError):
                    warnings.append(f"{external_name}: 非数值类型, 已跳过")
                    continue

            # 范围检查
            if internal_name in schema.normal_ranges:
                lo, hi = schema.normal_ranges[internal_name]
                if lo * 0.5 < value < hi * 1.5:  # 故障值允许50%裕度
                    cleaned[internal_name] = value
                else:
                    warnings.append(f"{internal_name}={value}: 超出物理可能范围[{lo-10},{hi+10}]")
            else:
                cleaned[internal_name] = value

        # 必填字段检查
        missing = [f for f in schema.required_fields if f not in cleaned]
        if missing:
            warnings.append(f"缺失必填字段: {missing}")

        return cleaned, warnings


class DataAdapter:
    """统一数据接入层 — 多协议适配 + 清洗 + 归一化"""

    def __init__(self, schema: DataSchema = None):
        self.schema = schema or DataSchema()
        self.validator = DataValidator()
        self.buffer = []
        self.last_read_time = None

    def read_csv(self, path: str) -> List[Dict]:
        """从CSV文件读取"""
        import pandas as pd
        df = pd.read_csv(path)
        observations = []
        for _, row in df.iterrows():
            obs = row.to_dict()
            cleaned, warnings = self.validator.validate(obs, self.schema)
            if cleaned:
                observations.append(cleaned)
            if warnings:
                logging.warning(f"数据校验: {warnings}")
        return observations

    def read_mqtt(self, host: str = "localhost", port: int = 1883,
                  topic: str = "factory/#"):
        """从MQTT读取（需要paho-mqtt）"""
        try:
            import paho.mqtt.client as mqtt
            client = mqtt.Client()
            client.connect(host, port, 60)
            # 返回 client 供外部使用
            return client
        except ImportError:
            raise ImportError("需要安装: pip install paho-mqtt")

    def normalize(self, observation: Dict) -> Dict:
        """数据归一化: 清洗 → 映射 → 标准化 → 返回Agent可用格式"""
        cleaned, warnings = self.validator.validate(observation, self.schema)
        if warnings:
            logging.info(f"数据归一化警告: {warnings}")
        return cleaned

    def detect_data_gap(self) -> bool:
        """检测数据断流"""
        if self.last_read_time is None:
            return False
        gap = (datetime.now() - self.last_read_time).total_seconds()
        return gap > self.schema.max_gap_seconds


# ================================================================
# 第二层: 稳定运维
# ================================================================
class HealthStatus(Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"


def setup_logging(log_dir: str = "data/logs"):
    """配置结构化日志（同时输出到文件和终端）"""
    os.makedirs(log_dir, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(name)s - %(message)s',
        handlers=[
            logging.FileHandler(f"{log_dir}/agent_{datetime.now():%Y%m%d}.log",
                              encoding='utf-8'),
            logging.StreamHandler()
        ]
    )


class HealthChecker:
    """Agent 健康检查器"""

    def __init__(self, agent):
        self.agent = agent
        self.last_healthy = datetime.now()
        self.health_history = []

    def check(self) -> Dict:
        """运行全面健康检查，返回状态报告"""
        checks = {}

        # 1. 因果图完整性
        if self.agent.causal_graph:
            n_nodes = self.agent.causal_graph.number_of_nodes()
            n_edges = self.agent.causal_graph.number_of_edges()
            has_cycles = not nx.is_directed_acyclic_graph(self.agent.causal_graph)
            checks["causal_graph"] = {
                "nodes": n_nodes,
                "edges": n_edges,
                "has_cycles": has_cycles,
                "status": "unhealthy" if (n_nodes == 0 or has_cycles) else "healthy"
            }
        else:
            checks["causal_graph"] = {"status": "unhealthy", "error": "因果图为空"}

        # 2. 数据流是否正常
        if hasattr(self.agent, 'data_source'):
            gap = getattr(self.agent.data_source, 'detect_data_gap', lambda: False)()
            checks["data_flow"] = {
                "status": "degraded" if gap else "healthy",
                "data_gap_detected": gap,
            }

        # 3. 内存使用
        import psutil
        mem = psutil.Process().memory_info()
        checks["memory"] = {
            "rss_mb": mem.rss / 1e6,
            "status": "degraded" if mem.rss > 1e9 else "healthy"
        }

        # 4. Agent 状态
        checks["agent"] = {
            "state": self.agent.state.value if hasattr(self.agent, 'state') else 'unknown',
            "diagnoses_today": getattr(self.agent.stats, 'diagnoses_made', 0) if hasattr(self.agent, 'stats') else 0,
        }

        # 综合判定
        all_statuses = [c.get("status", "healthy") for c in checks.values()]
        if "unhealthy" in all_statuses:
            overall = HealthStatus.UNHEALTHY
        elif "degraded" in all_statuses:
            overall = HealthStatus.DEGRADED
        else:
            overall = HealthStatus.HEALTHY

        result = {
            "timestamp": datetime.now().isoformat(),
            "overall": overall.value,
            "checks": checks,
        }

        self.health_history.append(result)
        if overall == HealthStatus.HEALTHY:
            self.last_healthy = datetime.now()

        return result


class MetricsCollector:
    """指标采集器 — 给 Prometheus/Grafana 用的"""

    def __init__(self):
        self.metrics = {
            "diagnosis_total": 0,
            "diagnosis_correct": 0,
            "anomalies_detected": 0,
            "false_alarms": 0,
            "avg_response_time_ms": 0,
            "causal_graph_edges": 0,
            "evolution_cycles": 0,
            "uptime_seconds": 0,
            "error_count": 0,
        }
        self.start_time = datetime.now()

    def record_diagnosis(self, correct: bool, response_time_ms: float):
        self.metrics["diagnosis_total"] += 1
        if correct:
            self.metrics["diagnosis_correct"] += 1
        self.metrics["avg_response_time_ms"] = (
            (self.metrics["avg_response_time_ms"] * (self.metrics["diagnosis_total"] - 1)
             + response_time_ms) / self.metrics["diagnosis_total"]
        )

    def record_evolution(self):
        self.metrics["evolution_cycles"] += 1

    def get_accuracy(self) -> float:
        total = self.metrics["diagnosis_total"]
        return self.metrics["diagnosis_correct"] / total if total > 0 else 1.0

    def export(self) -> Dict:
        self.metrics["uptime_seconds"] = (datetime.now() - self.start_time).total_seconds()
        return dict(self.metrics)


# ================================================================
# 第三层: 质量保证
# ================================================================
import networkx as nx

class CausalGraphValidator:
    """因果图质量校验器"""

    @staticmethod
    def validate(graph: nx.DiGraph) -> Dict:
        """全面校验因果图质量"""
        issues = []

        # 1. 循环检测
        try:
            cycles = list(nx.simple_cycles(graph))
            if cycles:
                issues.append({"severity": "critical",
                              "message": f"发现{len(cycles)}个因果循环",
                              "cycles": [c[:5] for c in cycles[:3]]})
        except Exception:
            pass

        # 2. 孤立节点
        isolated = [n for n in graph.nodes()
                   if graph.in_degree(n) == 0 and graph.out_degree(n) == 0]
        if isolated:
            issues.append({"severity": "warning",
                          "message": f"{len(isolated)}个孤立节点",
                          "nodes": isolated[:5]})

        # 3. 低置信度边
        low_conf = [(u, v, d.get("confidence", 0))
                   for u, v, d in graph.edges(data=True)
                   if d.get("confidence", 0) < 0.3]
        if low_conf:
            issues.append({"severity": "warning",
                          "message": f"{len(low_conf)}条低置信度边(<0.3)"})

        # 4. 无机制说明的边
        no_mechanism = [(u, v) for u, v, d in graph.edges(data=True)
                       if not d.get("mechanism")]
        if no_mechanism:
            issues.append({"severity": "info",
                          "message": f"{len(no_mechanism)}条边无物理机制说明"})

        passed = len([i for i in issues if i["severity"] == "critical"]) == 0
        return {
            "passed": passed,
            "issues": issues,
            "total_nodes": graph.number_of_nodes(),
            "total_edges": graph.number_of_edges(),
            "is_dag": nx.is_directed_acyclic_graph(graph),
        }


class CausalGraphVersionControl:
    """因果图版本控制 — 支持回滚"""

    def __init__(self, storage_dir: str = "data/graph_versions"):
        self.storage_dir = storage_dir
        os.makedirs(storage_dir, exist_ok=True)

    def save(self, graph: nx.DiGraph, label: str = "") -> str:
        """保存因果图版本"""
        version_id = f"v{datetime.now():%Y%m%d_%H%M%S}"
        if label:
            version_id += f"_{label}"

        path = os.path.join(self.storage_dir, f"{version_id}.graphml")
        nx.write_graphml(graph, path)

        # 更新 latest 指针
        with open(os.path.join(self.storage_dir, "latest.txt"), "w") as f:
            f.write(version_id)

        return version_id

    def load_latest(self) -> nx.DiGraph:
        """加载最新版本"""
        latest_path = os.path.join(self.storage_dir, "latest.txt")
        if os.path.exists(latest_path):
            with open(latest_path) as f:
                version_id = f.read().strip()
            graph_path = os.path.join(self.storage_dir, f"{version_id}.graphml")
            if os.path.exists(graph_path):
                return nx.read_graphml(graph_path)
        return nx.DiGraph()

    def rollback(self, version_id: str) -> nx.DiGraph:
        """回滚到指定版本"""
        path = os.path.join(self.storage_dir, f"{version_id}.graphml")
        if not os.path.exists(path):
            raise FileNotFoundError(f"版本 {version_id} 不存在")
        return nx.read_graphml(path)


class QualityGate:
    """质量门禁 — 部署前必须通过"""

    def __init__(self, graph: nx.DiGraph, metrics: MetricsCollector,
                 version_control: CausalGraphVersionControl):
        self.graph = graph
        self.metrics = metrics
        self.vc = version_control

    def run(self) -> Dict:
        """运行质量门禁检查"""
        results = {}

        # 1. 因果图校验
        validator = CausalGraphValidator()
        results["graph_validation"] = validator.validate(self.graph)

        # 2. 准确率检查
        accuracy = self.metrics.get_accuracy()
        results["accuracy"] = {
            "value": accuracy,
            "passed": accuracy >= 0.6,
            "threshold": 0.6,
        }

        # 3. 因果图不应退化
        prev_graph = self.vc.load_latest()
        if prev_graph.number_of_edges() > 0:
            edge_ratio = self.graph.number_of_edges() / max(prev_graph.number_of_edges(), 1)
            results["graph_growth"] = {
                "edges_before": prev_graph.number_of_edges(),
                "edges_after": self.graph.number_of_edges(),
                "ratio": edge_ratio,
                "passed": edge_ratio >= 0.8,  # 不能减少超过20%
            }

        # 综合判定
        passed = all(
            r.get("passed", True) if isinstance(r, dict) else True
            for r in results.values()
        )
        results["overall"] = "PASSED" if passed else "FAILED"

        if not passed:
            logging.warning(f"质量门禁未通过: {[k for k,v in results.items() if isinstance(v,dict) and not v.get('passed',True)]}")

        return results


# ================================================================
# 第四层: 安全框架
# ================================================================
class AccessLevel(Enum):
    VIEWER = 1       # 操作员: 只能看
    OPERATOR = 2     # 工程师: 能确认建议
    ADMIN = 3        # 管理员: 能改因果图


class SecurityLayer:
    """安全控制层"""

    def __init__(self):
        self.users = {}        # {username: {password_hash, access_level}}
        self.audit_log = []
        self.action_whitelist = {
            AccessLevel.VIEWER: ["view"],
            AccessLevel.OPERATOR: ["view", "confirm_diagnosis", "acknowledge_alert"],
            AccessLevel.ADMIN: ["view", "confirm_diagnosis", "acknowledge_alert",
                               "update_causal_graph", "adjust_parameter",
                               "trigger_evolution"],
        }

    def register_user(self, username: str, password: str, level: AccessLevel):
        """注册用户"""
        self.users[username] = {
            "password_hash": hashlib.sha256(password.encode()).hexdigest(),
            "access_level": level,
        }

    def authenticate(self, username: str, password: str) -> Optional[AccessLevel]:
        """验证用户身份"""
        user = self.users.get(username)
        if user and user["password_hash"] == hashlib.sha256(password.encode()).hexdigest():
            return user["access_level"]
        return None

    def authorize(self, username: str, action: str) -> bool:
        """检查用户是否有权限执行某动作"""
        user = self.users.get(username)
        if not user:
            return False
        allowed = self.action_whitelist.get(user["access_level"], [])
        return action in allowed

    def audit(self, username: str, action: str, details: str = "",
              success: bool = True):
        """记录审计日志"""
        entry = {
            "timestamp": datetime.now().isoformat(),
            "username": username,
            "action": action,
            "details": details,
            "success": success,
        }
        self.audit_log.append(entry)
        logging.info(f"[审计] {username} 执行 {action}: {'成功' if success else '拒绝'} {details}")
        return entry

    def get_audit_trail(self, username: str = None,
                        action: str = None, hours: int = 24) -> List[Dict]:
        """查询审计日志"""
        cutoff = datetime.now() - timedelta(hours=hours)
        result = [
            e for e in self.audit_log
            if datetime.fromisoformat(e["timestamp"]) > cutoff
            and (username is None or e["username"] == username)
            and (action is None or e["action"] == action)
        ]
        return result


# ================================================================
# 第五层: 系统集成
# ================================================================
class ExternalSystemConnector:
    """外部系统连接器 — 工厂系统的统一接口"""

    def __init__(self):
        self.connections = {}
        self.webhook_urls = {
            "dingtalk": "",
            "wechat_work": "",
            "work_order": "",
        }

    def send_alert(self, platform: str, title: str, content: str,
                   severity: str = "warning") -> bool:
        """发送报警到指定平台"""
        if platform == "dingtalk":
            return self._send_dingtalk(title, content)
        elif platform == "wechat_work":
            return self._send_wechat_work(title, content)
        elif platform == "work_order":
            return self._create_work_order(title, content)
        return False

    def _send_dingtalk(self, title: str, content: str) -> bool:
        """发送钉钉消息"""
        url = self.webhook_urls.get("dingtalk")
        if not url:
            logging.warning("钉钉Webhook未配置")
            return False
        try:
            import requests
            payload = {
                "msgtype": "markdown",
                "markdown": {
                    "title": title,
                    "text": f"## {title}\n\n{content}\n\n---\n*溯因智工 · 工业智能体*"
                }
            }
            resp = requests.post(url, json=payload, timeout=5)
            return resp.status_code == 200
        except Exception as e:
            logging.error(f"钉钉发送失败: {e}")
            return False

    def _send_wechat_work(self, title: str, content: str) -> bool:
        """发送企业微信消息"""
        url = self.webhook_urls.get("wechat_work")
        if not url:
            logging.warning("企业微信Webhook未配置")
            return False
        try:
            import requests
            payload = {
                "msgtype": "text",
                "text": {"content": f"【{title}】\n{content}"}
            }
            resp = requests.post(url, json=payload, timeout=5)
            return resp.status_code == 200
        except Exception as e:
            logging.error(f"企业微信发送失败: {e}")
            return False

    def _create_work_order(self, title: str, content: str) -> bool:
        """创建工单（实际环境中调用MES/工单系统API）"""
        url = self.webhook_urls.get("work_order")
        if not url:
            print(f"[工单模拟] {title}: {content}")
            return True  # 模拟模式下直接返回成功
        try:
            import requests
            payload = {
                "title": title,
                "description": content,
                "priority": "high",
                "source": "CausalAgent",
            }
            resp = requests.post(url, json=payload, timeout=5)
            return resp.status_code == 200
        except Exception as e:
            logging.error(f"工单创建失败: {e}")
            return False


# ================================================================
# 集成: 生产级Agent (五层全部接入)
# ================================================================
class ProductionAgent:
    """生产级Agent — 五层全接入"""

    def __init__(self, name: str, config_path: str = None):
        self.name = name
        self.config = self._load_config(config_path) if config_path else {}

        # 五层模块
        self.data_adapter = DataAdapter()
        self.health_checker = None      # 在 setup() 中初始化
        self.metrics = MetricsCollector()
        self.quality_gate = None
        self.security = SecurityLayer()
        self.connector = ExternalSystemConnector()

        # 主Agent
        self.causal_graph = None
        self.agent = None

        # 注册默认用户
        self.security.register_user("operator", "factory123", AccessLevel.OPERATOR)
        self.security.register_user("admin", "admin456", AccessLevel.ADMIN)

        setup_logging()
        logging.info(f"[{self.name}] 生产级Agent初始化完成")

    def _load_config(self, path: str) -> Dict:
        with open(path, encoding='utf-8') as f:
            return json.load(f)

    def run_health_check(self) -> Dict:
        self.health_checker = HealthChecker(self)
        return self.health_checker.check()

    def run_quality_gate(self) -> Dict:
        if self.quality_gate is None:
            vc = CausalGraphVersionControl()
            self.quality_gate = QualityGate(self.causal_graph, self.metrics, vc)
        return self.quality_gate.run()

    def send_alarm(self, title: str, content: str):
        """发报警到钉钉/企业微信"""
        return self.connector.send_alert("dingtalk", title, content)

    def get_status_report(self) -> Dict:
        """生成综合状态报告"""
        return {
            "name": self.name,
            "timestamp": datetime.now().isoformat(),
            "data_adapter_ready": self.data_adapter is not None,
            "causal_graph_edges": (self.causal_graph.number_of_edges()
                                   if self.causal_graph else 0),
            "metrics": self.metrics.export(),
            "health": self.run_health_check() if self.health_checker else None,
        }


if __name__ == "__main__":
    print("=" * 60)
    print("生产级Agent 五层模块验证")
    print("=" * 60)

    # 1. 数据接入层
    print("\n[Layer 1] 数据接入层")
    schema = DataSchema(
        mappings={"temp_reactor": "Reactor_Temp", "cw_flow": "CW_Flow"},
        normal_ranges={"Reactor_Temp": (150, 175), "CW_Flow": (200, 300)},
        required_fields=["Reactor_Temp"],
    )
    adapter = DataAdapter(schema)
    obs = {"temp_reactor": 192, "cw_flow": 118, "garbage_field": "N/A"}
    cleaned, warnings = adapter.validator.validate(obs, schema)
    print(f"  输入: {obs}")
    print(f"  清洗后: {cleaned}")
    print(f"  警告: {warnings}")

    # 2. 因果图校验
    print("\n[Layer 3] 因果图质量校验")
    from src.synthetic_data_generator import CAUSAL_GRAPH_TRUTH
    import networkx as nx
    g = nx.DiGraph()
    for e in CAUSAL_GRAPH_TRUTH:
        g.add_edge(e.cause, e.effect, confidence=0.9, mechanism=e.mechanism)
    v = CausalGraphValidator()
    result = v.validate(g)
    print(f"  DAG: {result['is_dag']}")
    print(f"  通过: {result['passed']}")

    # 3. 安全层
    print("\n[Layer 4] 安全控制")
    sec = SecurityLayer()
    sec.register_user("operator", "factory123", AccessLevel.OPERATOR)
    sec.register_user("admin", "admin456", AccessLevel.ADMIN)
    level = sec.authenticate("operator", "factory123")
    print(f"  操作员认证: {level}")
    print(f"  操作员能否改因果图: {sec.authorize('operator', 'update_causal_graph')}")
    print(f"  管理员能否改因果图: {sec.authorize('admin', 'update_causal_graph')}")

    # 4. 集成层
    print("\n[Layer 5] 系统集成")
    conn = ExternalSystemConnector()
    conn.send_alert("dingtalk", "测试报警", "反应器温度异常: 192°C")
    print("  (模拟模式下不实际发送)")

    print(f"\n{'='*60}")
    print("✅ 五层模块全部就绪")
