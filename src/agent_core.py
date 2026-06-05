"""
真正的自主工业Agent核心引擎

从"故障查询器"升级为"自主Agent"的四个关键模块:

  感知层: 自主监听数据流，主动发现异常（不再等人来问）
  决策层: 基于因果图 + 历史经验，自主选择最优干预方案
  执行层: 发出工单/调整参数/通知操作员（不只是建议，而是行动）
  进化层: 根据执行结果反馈，自我优化（越用越准）

Agent 状态机:
  IDLE → MONITORING → ANOMALY_DETECTED → DIAGNOSING
    → ACTION_PROPOSED → ACTION_EXECUTING → RESULT_EVALUATING
    → LEARNING → MONITORING (循环)

用法:
  agent = AutonomousIndustrialAgent("武钢-高炉-3号线")
  agent.start()  # 启动自主运行
  # Agent 会自己: 监听数据 → 发现异常 → 诊断 → 建议 → 学习
"""

import sys, os, json, time, threading, queue
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from collections import deque

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))


# ================================================================
# Agent 状态机
# ================================================================
class AgentState(Enum):
    IDLE = "idle"
    MONITORING = "monitoring"
    ANOMALY_DETECTED = "anomaly_detected"
    DIAGNOSING = "diagnosing"
    ACTION_PROPOSED = "action_proposed"
    ACTION_EXECUTING = "action_executing"
    RESULT_EVALUATING = "result_evaluating"
    LEARNING = "learning"
    EVOLVING = "evolving"
    ERROR = "error"


@dataclass
class AgentMemory:
    """Agent 的记忆 — 过去的经验让未来更好"""
    # 短期记忆: 最近N次诊断
    recent_diagnoses: deque = field(default_factory=lambda: deque(maxlen=100))
    # 长期记忆: 成功/失败的干预模式
    action_outcomes: List[Dict] = field(default_factory=list)
    # 知识记忆: 累积的因果边
    causal_knowledge: Dict = field(default_factory=dict)
    # 性能记忆: 准确率/响应时间等
    performance_history: List[Dict] = field(default_factory=list)

    def remember(self, event_type: str, data: Dict):
        """记录事件到记忆"""
        entry = {"timestamp": datetime.now().isoformat(), "type": event_type, **data}
        self.recent_diagnoses.append(entry)
        if event_type == "action_result":
            self.action_outcomes.append(entry)
        if event_type == "performance":
            self.performance_history.append(entry)

    def get_successful_patterns(self, root_cause: str) -> List[Dict]:
        """检索某类根因的成功干预模式"""
        return [a for a in self.action_outcomes
                if a.get("root_cause") == root_cause and a.get("success")]

    def get_accuracy_trend(self) -> float:
        """计算近期诊断准确率趋势"""
        recent = list(self.recent_diagnoses)[-20:]
        if not recent:
            return 0.0
        correct = sum(1 for r in recent if r.get("verified_correct"))
        return correct / len(recent)


@dataclass
class Action:
    """Agent 可执行的动作"""
    action_id: str
    action_type: str            # "work_order" / "parameter_adjust" / "alert" / "query"
    target_system: str          # 目标系统: "DCS" / "MES" / "工单系统" / "钉钉"
    target_variable: str        # 目标变量
    proposed_value: Any = None  # 建议值
    priority: int = 1           # 1-5, 5最高
    expected_effect: str = ""   # 预期效果
    risk_level: str = "low"     # low / medium / high
    requires_approval: bool = True  # 是否需要人工确认


@dataclass
class Goal:
    """Agent 的工作目标"""
    goal_id: str
    description: str
    kpi: str                    # 关键指标
    target_value: float
    current_value: float
    priority: int = 1


# ================================================================
# 真正的自主Agent
# ================================================================
class AutonomousIndustrialAgent:
    """自主工业Agent — 感知 → 决策 → 执行 → 进化 全闭环"""

    def __init__(self, name: str, config: Dict = None):
        self.name = name
        self.config = config or {}
        self.state = AgentState.IDLE
        self.memory = AgentMemory()
        self.goals: List[Goal] = []
        self.action_queue = queue.Queue()
        self.event_queue = queue.Queue()
        self._running = False
        self._threads: List[threading.Thread] = []

        # 核心组件（启动时注入）
        self.causal_graph = None
        self.analyzer = None
        self.cf_engine = None
        self.evolution_engine = None
        self.data_source = None       # 数据接入层
        self.action_executor = None   # 执行层

        # 统计
        self.stats = {
            "anomalies_detected": 0,
            "diagnoses_made": 0,
            "actions_executed": 0,
            "evolutions_completed": 0,
            "uptime_seconds": 0,
            "start_time": None,
        }

    # ================================================================
    # 1. 自主感知层 — 不等人来问，自己看
    # ================================================================
    def _perception_loop(self):
        """持续监听数据流，主动发现异常"""
        print(f"[感知] {self.name} 开始监听数据流...")
        buffer = deque(maxlen=100)

        while self._running:
            try:
                # 从数据源获取最新观测
                observation = self._read_sensors()

                if observation:
                    buffer.append(observation)

                    # 每10个数据点做一次异常检测
                    if len(buffer) >= 10:
                        anomalies = self.analyzer.detect_anomalies(observation)
                        if anomalies:
                            self.event_queue.put({
                                "type": "anomaly",
                                "anomalies": anomalies,
                                "observation": observation,
                                "timestamp": datetime.now().isoformat(),
                            })
                            self.stats["anomalies_detected"] += 1

                time.sleep(self.config.get("monitor_interval", 1.0))

            except Exception as e:
                print(f"[感知] 异常: {e}")
                time.sleep(5)

    def _read_sensors(self) -> Optional[Dict]:
        """读取传感器/数据源"""
        if self.data_source is None:
            # 模拟: 从合成数据读取
            if hasattr(self, '_sim_data') and self._sim_data:
                return self._sim_data.pop()
            return None

        # 真实数据源: MQTT / OPC-UA / API
        try:
            return self.data_source.read()
        except Exception:
            return None

    # ================================================================
    # 2. 自主决策层 — 不只是诊断，而是选择最优行动
    # ================================================================
    def _decision_loop(self):
        """事件驱动的决策循环"""
        print(f"[决策] {self.name} 决策引擎就绪...")

        while self._running:
            try:
                event = self.event_queue.get(timeout=1)

                if event["type"] == "anomaly":
                    self.state = AgentState.DIAGNOSING
                    self._handle_anomaly(event)

                elif event["type"] == "action_result":
                    self.state = AgentState.RESULT_EVALUATING
                    self._evaluate_result(event)

                elif event["type"] == "evolution_trigger":
                    self.state = AgentState.EVOLVING
                    self._trigger_evolution()

                self.event_queue.task_done()

            except queue.Empty:
                if self.state == AgentState.DIAGNOSING:
                    self.state = AgentState.MONITORING
                continue
            except Exception as e:
                self.state = AgentState.ERROR
                print(f"[决策] 异常: {e}")

    def _handle_anomaly(self, event: Dict):
        """处理异常事件: 诊断 → 生成候选方案 → 选择最优动作"""
        observation = event["observation"]
        anomalies = event["anomalies"]

        # Step 1: 根因诊断
        reports = self.analyzer.analyze(observation, top_k=3)
        self.stats["diagnoses_made"] += 1

        # Step 2: 对每个根因，检索历史成功经验
        actions = []
        for report in reports:
            if not report.root_causes:
                continue

            for rc in report.root_causes:
                # 查记忆: 这类根因以前怎么处理的？
                past_success = self.memory.get_successful_patterns(rc.variable)

                # 反事实推理: 如果做了X，效果如何？
                if rc.variable in self.analyzer.normal_ranges:
                    lo, hi = self.analyzer.normal_ranges[rc.variable]
                    normal_val = (lo + hi) / 2
                    try:
                        cf_result = self.cf_engine.what_if(
                            observation,
                            {rc.variable: normal_val},
                            report.abnormal_variable,
                            self.analyzer.normal_ranges,
                        )
                        expected_effect = (f"恢复{rc.variable}至{normal_val:.1f}, "
                                         f"{report.abnormal_variable}预期改善{cf_result.improvement:+.1f}")
                    except Exception:
                        expected_effect = f"恢复{rc.variable}至正常范围"

                else:
                    expected_effect = f"调查{rc.variable}异常原因"

                # 生成动作
                action = Action(
                    action_id=f"act_{datetime.now().strftime('%Y%m%d%H%M%S')}_{rc.variable}",
                    action_type="alert" if rc.score < 0.7 else "work_order",
                    target_system="工单系统" if rc.score >= 0.7 else "钉钉",
                    target_variable=rc.variable,
                    proposed_value=normal_val if rc.variable in self.analyzer.normal_ranges else None,
                    priority=min(5, int(rc.score * 5) + 1),
                    expected_effect=expected_effect,
                    risk_level="high" if rc.score < 0.5 else ("medium" if rc.score < 0.7 else "low"),
                    requires_approval=rc.score < 0.8,
                )

                # 有历史成功经验的 → 提升优先级
                if past_success:
                    action.priority = min(5, action.priority + 1)
                    action.requires_approval = False

                actions.append(action)

        # Step 3: 按优先级排序，选择最优动作
        actions.sort(key=lambda a: a.priority, reverse=True)

        # Step 4: 记录到记忆
        self.memory.remember("diagnosis", {
            "anomalies": [a.variable for a in anomalies],
            "root_causes": [(a.target_variable, a.priority) for a in actions[:3]],
            "top_action": actions[0].action_id if actions else None,
        })

        # Step 5: 投入执行队列
        for action in actions[:2]:  # 只执行Top-2
            self.action_queue.put(action)

        self.state = AgentState.ACTION_PROPOSED
        print(f"[决策] 异常处理完成 → {len(actions)}个候选动作")

    def _evaluate_result(self, event: Dict):
        """评估执行结果，更新记忆"""
        action_id = event.get("action_id")
        success = event.get("success", False)
        observation_after = event.get("observation_after", {})

        self.memory.remember("action_result", {
            "action_id": action_id,
            "success": success,
            "observation_after": observation_after,
        })

        # 绩效追踪
        self.memory.remember("performance", {
            "accuracy": self.memory.get_accuracy_trend(),
            "anomalies_today": self.stats["anomalies_detected"],
            "diagnoses_today": self.stats["diagnoses_made"],
        })

        self.state = AgentState.LEARNING
        print(f"[评估] 动作{action_id}: {'✅ 成功' if success else '❌ 失败'}, "
              f"近期准确率: {self.memory.get_accuracy_trend():.1%}")

    def _trigger_evolution(self):
        """触发自主进化"""
        print(f"[进化] 开始第{self.stats['evolutions_completed']+1}次自主进化...")
        if self.evolution_engine:
            try:
                # 文献爬取 + 知识注入 + 仿真校准 + 自测
                self.evolution_engine.auto_evolve(
                    lambda: self._get_literature_knowledge(),
                    lambda: self._get_field_data(),
                    max_generations=3,
                )
                self.stats["evolutions_completed"] += 1
                print(f"[进化] ✅ 完成 (累计{self.stats['evolutions_completed']}次)")
            except Exception as e:
                print(f"[进化] ❌ 失败: {e}")
        self.state = AgentState.MONITORING

    def _get_literature_knowledge(self):
        """从文献获取新知识"""
        try:
            from src.literature_crawler import LiteratureCrawler
            from src.literature_extractor import LiteratureExtractor
            crawler = LiteratureCrawler()
            papers = crawler.search_all()[:5]
            extractor = LiteratureExtractor()
            return extractor.extract_from_papers(papers, self.causal_graph)
        except Exception:
            return None

    def _get_field_data(self):
        """获取现场反馈数据"""
        # 实际环境中从数据库读取最近的故障记录
        return None

    # ================================================================
    # 3. 自主执行层 — 不只是建议，而是行动
    # ================================================================
    def _execution_loop(self):
        """消费动作队列，执行行动"""
        print(f"[执行] {self.name} 执行引擎就绪...")

        while self._running:
            try:
                action = self.action_queue.get(timeout=1)
                self.state = AgentState.ACTION_EXECUTING
                self._execute_action(action)
                self.action_queue.task_done()
                self.stats["actions_executed"] += 1
            except queue.Empty:
                continue
            except Exception as e:
                print(f"[执行] 异常: {e}")

    def _execute_action(self, action: Action):
        """执行具体动作"""
        # 需要人工确认的动作
        if action.requires_approval:
            print(f"[执行] ⚠ 需确认: {action.action_id} - "
                  f"将{action.target_variable}恢复至{action.proposed_value}")
            # 实际环境: 发钉钉/企业微信审批
            return

        # 自动执行
        if action.action_type == "work_order":
            print(f"[执行] 📋 生成工单: {action.action_id} → {action.target_system}")
            # 实际: 调用MES/工单系统API

        elif action.action_type == "parameter_adjust":
            print(f"[执行] ⚙️ 调整参数: {action.target_variable} → {action.proposed_value}")
            # 实际: 通过OPC-UA写入DCS

        elif action.action_type == "alert":
            print(f"[执行] 🚨 发送报警: {action.target_variable}异常, "
                  f"预期效果: {action.expected_effect}")
            # 实际: 调用钉钉/企业微信Webhook

    # ================================================================
    # 4. 自主进化调度 — 定时的自我提升
    # ================================================================
    def _evolution_scheduler(self):
        """定时触发进化事件"""
        print(f"[调度] 进化计划: 每{self.config.get('evolve_interval_hours', 168)}小时")
        while self._running:
            time.sleep(self.config.get("evolve_interval_hours", 168) * 3600)
            self.event_queue.put({
                "type": "evolution_trigger",
                "timestamp": datetime.now().isoformat(),
            })
            print(f"[调度] 触发定期进化")

    # ================================================================
    # Agent 生命周期
    # ================================================================
    def setup(self, causal_graph, analyzer, cf_engine, evolution_engine=None,
              data_source=None, action_executor=None):
        """注入核心组件"""
        self.causal_graph = causal_graph
        self.analyzer = analyzer
        self.cf_engine = cf_engine
        self.evolution_engine = evolution_engine
        self.data_source = data_source
        self.action_executor = action_executor

        # 设置目标
        self.goals = [
            Goal("g1", "减少非计划停机", "downtime_hours", 0, 0, priority=5),
            Goal("g2", "提高根因诊断准确率", "accuracy", 0.9, 0, priority=4),
            Goal("g3", "缩短故障响应时间", "response_time_min", 5, 0, priority=3),
        ]
        print(f"[{self.name}] 初始化完成, {len(self.goals)}个目标")

    def start(self):
        """启动Agent — 开始自主运行"""
        self._running = True
        self.stats["start_time"] = datetime.now().isoformat()
        self.state = AgentState.MONITORING

        # 启动四个核心线程
        self._threads = [
            threading.Thread(target=self._perception_loop, name="perception", daemon=True),
            threading.Thread(target=self._decision_loop, name="decision", daemon=True),
            threading.Thread(target=self._execution_loop, name="execution", daemon=True),
            threading.Thread(target=self._evolution_scheduler, name="scheduler", daemon=True),
        ]
        for t in self._threads:
            t.start()

        print(f"\n{'='*60}")
        print(f"🤖 {self.name} 已启动 — 自主运行中")
        print(f"   状态: {self.state.value}")
        print(f"   目标: {[g.description for g in self.goals]}")
        print(f"   按 Ctrl+C 停止")
        print(f"{'='*60}\n")

    def stop(self):
        """停止Agent"""
        self._running = False
        for t in self._threads:
            t.join(timeout=5)
        self.stats["uptime_seconds"] = (
            datetime.now() - datetime.fromisoformat(self.stats["start_time"])
        ).total_seconds() if self.stats["start_time"] else 0
        self.state = AgentState.IDLE
        print(f"\n[{self.name}] 已停止。运行{self.stats['uptime_seconds']:.0f}秒")

    def get_status(self) -> Dict:
        """获取Agent当前状态"""
        return {
            "name": self.name,
            "state": self.state.value,
            "uptime_minutes": self.stats["uptime_seconds"] / 60,
            "anomalies_detected": self.stats["anomalies_detected"],
            "diagnoses_made": self.stats["diagnoses_made"],
            "actions_executed": self.stats["actions_executed"],
            "evolutions_completed": self.stats["evolutions_completed"],
            "memory_size": len(self.memory.recent_diagnoses),
            "accuracy_trend": self.memory.get_accuracy_trend(),
            "pending_actions": self.action_queue.qsize(),
            "goals": [{"description": g.description, "target": g.target_value,
                       "current": g.current_value} for g in self.goals],
        }


# ================================================================
# 测试: 启动Agent
# ================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("自主工业Agent 核心引擎")
    print("=" * 60)

    # 初始化依赖组件
    from src.synthetic_data_generator import (SyntheticProcessSimulator, VAR_NAMES,
                                               generate_process_documentation)
    from src.root_cause_analysis import RootCauseAnalyzer
    from src.counterfactual import CounterfactualEngine
    from src.llm_causal_extract import extract_from_synthetic_doc, LLMCausalExtractor
    import networkx as nx

    os.makedirs("data/synthetic", exist_ok=True)
    generate_process_documentation("data/synthetic")
    pairs = extract_from_synthetic_doc("data/synthetic/process_documentation.txt", VAR_NAMES)
    kg = LLMCausalExtractor.pairs_to_graph(pairs, VAR_NAMES)

    sim = SyntheticProcessSimulator(seed=42)
    df_normal = sim.simulate(n_steps=500)
    analyzer = RootCauseAnalyzer(kg)
    analyzer.set_normal_ranges(df_normal.iloc[:300])

    cf_engine = CounterfactualEngine(kg)
    from src.synthetic_data_generator import FAULT_MODES, CAUSAL_GRAPH_TRUTH
    coeffs = {(e.cause, e.effect): e.coefficient for e in CAUSAL_GRAPH_TRUTH}
    cf_engine.set_manual_coefficients(coeffs)

    # 准备模拟数据
    import random
    sim_data = []
    for fault_name in list(FAULT_MODES.keys())[:3]:
        df, _ = sim.generate_fault_dataset(n_normal=100, n_fault=50, fault_name=fault_name)
        fault_rows = df[df['fault'] == fault_name]
        for i in range(10):
            obs = {v: float(fault_rows.iloc[i*5][v]) for v in VAR_NAMES}
            sim_data.append(obs)

    # 创建并启动Agent
    agent = AutonomousIndustrialAgent("武钢-高炉3号线-Agent", {
        "monitor_interval": 0.5,
        "evolve_interval_hours": 24,
    })
    agent._sim_data = sim_data  # 注入模拟数据
    agent.setup(kg, analyzer, cf_engine)

    try:
        agent.start()
        # 运行10秒展示Agent自主工作
        for i in range(10):
            time.sleep(1)
            status = agent.get_status()
            if i % 3 == 0:
                print(f"  [状态] {status['state']} | "
                      f"检测{status['anomalies_detected']}次异常 | "
                      f"诊断{status['diagnoses_made']}次 | "
                      f"执行{status['actions_executed']}个动作")
    except KeyboardInterrupt:
        pass
    finally:
        agent.stop()
