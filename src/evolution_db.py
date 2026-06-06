"""
进化状态持久化 — SQLite 数据库 + Streamlit 可视化

解决了自进化"看不见"的问题:
  - 每次进化结果写入SQLite → 刷新页面不丢失
  - 记录知识增长曲线 → 可视化为图表
  - 记录参数进化轨迹 → 可回溯每次变更
  - 记录自测评分变化 → 证明"越用越准"

用法:
  db = EvolutionDB()
  db.record_evolution(cycle_report)  # 记录一次进化
  db.get_growth_curve()              # 获取知识增长曲线
  db.get_latest_status()             # 获取最新进化状态
"""

import sqlite3, json, os, time
from datetime import datetime
from typing import Dict, List, Optional
import pandas as pd


class EvolutionDB:
    """进化状态数据库 — 让自进化可见、可追溯、可证明"""

    def __init__(self, db_path: str = "data/evolution.db"):
        self.db_path = db_path
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self._init_tables()

    def _init_tables(self):
        """建表"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS evolution_cycles (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    papers_found INTEGER DEFAULT 0,
                    new_causal_edges INTEGER DEFAULT 0,
                    edges_before INTEGER DEFAULT 0,
                    edges_after INTEGER DEFAULT 0,
                    knowledge_base_size INTEGER DEFAULT 0,
                    validation_score REAL DEFAULT 0,
                    simulation_params TEXT DEFAULT '{}',
                    status TEXT DEFAULT 'completed'
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS knowledge_growth (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    cause TEXT NOT NULL,
                    effect TEXT NOT NULL,
                    mechanism TEXT DEFAULT '',
                    confidence REAL DEFAULT 0.5,
                    source TEXT DEFAULT 'manual',
                    cycle_id INTEGER
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS simulation_params (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    param_name TEXT NOT NULL,
                    old_value REAL,
                    new_value REAL,
                    trigger_source TEXT DEFAULT 'manual',
                    improvement_metric REAL DEFAULT 0
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS validation_results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    test_type TEXT NOT NULL,
                    score REAL DEFAULT 0,
                    passed INTEGER DEFAULT 0,
                    details TEXT DEFAULT '{}'
                )
            """)
            conn.commit()

    # ================================================================
    # 进化周期记录
    # ================================================================
    def record_evolution(self, papers: int = 0, new_edges: int = 0,
                         edges_before: int = 0, edges_after: int = 0,
                         kb_size: int = 0, score: float = 0,
                         params: Dict = None, status: str = "completed"):
        """记录一次进化周期"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO evolution_cycles
                (timestamp, papers_found, new_causal_edges, edges_before, edges_after,
                 knowledge_base_size, validation_score, simulation_params, status)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                datetime.now().isoformat(),
                papers, new_edges, edges_before, edges_after,
                kb_size, score,
                json.dumps(params or {}, ensure_ascii=False),
                status
            ))
            conn.commit()

    def get_evolution_history(self, limit: int = 20) -> pd.DataFrame:
        """获取进化历史"""
        with sqlite3.connect(self.db_path) as conn:
            return pd.read_sql_query(
                "SELECT * FROM evolution_cycles ORDER BY id DESC LIMIT ?",
                conn, params=(limit,)
            )

    def get_growth_curve(self) -> pd.DataFrame:
        """获取知识增长曲线（用于图表）"""
        with sqlite3.connect(self.db_path) as conn:
            return pd.read_sql_query(
                "SELECT id as cycle, timestamp, edges_before, edges_after, "
                "new_causal_edges, validation_score "
                "FROM evolution_cycles ORDER BY id",
                conn
            )

    # ================================================================
    # 知识增长
    # ================================================================
    def add_knowledge(self, cause: str, effect: str, mechanism: str = "",
                      confidence: float = 0.5, source: str = "manual"):
        """添加一条新因果知识"""
        with sqlite3.connect(self.db_path) as conn:
            # 去重
            existing = conn.execute(
                "SELECT id FROM knowledge_growth WHERE cause=? AND effect=?",
                (cause, effect)
            ).fetchone()
            if existing:
                # 更新置信度（取更高值）
                conn.execute(
                    "UPDATE knowledge_growth SET confidence=MAX(confidence,?), timestamp=? WHERE id=?",
                    (confidence, datetime.now().isoformat(), existing[0])
                )
            else:
                conn.execute("""
                    INSERT INTO knowledge_growth (timestamp, cause, effect, mechanism, confidence, source)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (datetime.now().isoformat(), cause, effect, mechanism, confidence, source))
            conn.commit()

    def get_knowledge_stats(self) -> Dict:
        """知识统计"""
        with sqlite3.connect(self.db_path) as conn:
            total = conn.execute("SELECT COUNT(*) FROM knowledge_growth").fetchone()[0]
            by_source = conn.execute(
                "SELECT source, COUNT(*) FROM knowledge_growth GROUP BY source"
            ).fetchall()
            avg_conf = conn.execute(
                "SELECT AVG(confidence) FROM knowledge_growth"
            ).fetchone()[0] or 0
            return {
                "total": total,
                "by_source": dict(by_source),
                "avg_confidence": avg_conf,
            }

    def get_all_knowledge(self) -> pd.DataFrame:
        """获取全部知识"""
        with sqlite3.connect(self.db_path) as conn:
            return pd.read_sql_query(
                "SELECT * FROM knowledge_growth ORDER BY confidence DESC",
                conn
            )

    # ================================================================
    # 参数进化轨迹
    # ================================================================
    def record_param_change(self, param_name: str, old_val: float,
                            new_val: float, trigger: str = "manual",
                            improvement: float = 0):
        """记录一次参数变更"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO simulation_params
                (timestamp, param_name, old_value, new_value, trigger_source, improvement_metric)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (datetime.now().isoformat(), param_name, old_val, new_val, trigger, improvement))
            conn.commit()

    def get_param_history(self, param_name: str = None) -> pd.DataFrame:
        """获取参数变更历史"""
        with sqlite3.connect(self.db_path) as conn:
            if param_name:
                return pd.read_sql_query(
                    "SELECT * FROM simulation_params WHERE param_name=? ORDER BY id",
                    conn, params=(param_name,)
                )
            return pd.read_sql_query(
                "SELECT * FROM simulation_params ORDER BY id DESC LIMIT 50",
                conn
            )

    # ================================================================
    # 自测验证
    # ================================================================
    def record_validation(self, test_type: str, score: float,
                          passed: bool, details: Dict = None):
        """记录一次自测验证"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO validation_results
                (timestamp, test_type, score, passed, details)
                VALUES (?, ?, ?, ?, ?)
            """, (
                datetime.now().isoformat(), test_type, score,
                1 if passed else 0,
                json.dumps(details or {}, ensure_ascii=False)
            ))
            conn.commit()

    def get_validation_trend(self) -> pd.DataFrame:
        """获取验证趋势"""
        with sqlite3.connect(self.db_path) as conn:
            return pd.read_sql_query(
                "SELECT * FROM validation_results ORDER BY id",
                conn
            )

    # ================================================================
    # 综合状态
    # ================================================================
    def get_status(self) -> Dict:
        """获取综合进化状态"""
        with sqlite3.connect(self.db_path) as conn:
            cycles = conn.execute("SELECT COUNT(*) FROM evolution_cycles").fetchone()[0]
            last = conn.execute(
                "SELECT * FROM evolution_cycles ORDER BY id DESC LIMIT 1"
            ).fetchone()

            knowledge = conn.execute("SELECT COUNT(*) FROM knowledge_growth").fetchone()[0]
            params = conn.execute("SELECT COUNT(*) FROM simulation_params").fetchone()[0]
            validations = conn.execute(
                "SELECT AVG(score) FROM validation_results WHERE test_type='full'"
            ).fetchone()[0] or 0

            # 最近一次进化的增长
            last_growth = 0
            if last:
                last_growth = (last[5] or 0) - (last[4] or 0)  # edges_after - edges_before

            return {
                "total_cycles": cycles,
                "total_knowledge": knowledge,
                "total_param_changes": params,
                "avg_validation_score": validations,
                "last_evolution_time": last[1] if last else None,
                "last_edges_after": last[5] if last else 0,
                "last_growth": last_growth,
                "is_evolving": cycles > 0,
            }

    # ================================================================
    # 初始化演示数据（让Streamlit首次打开就有内容可看）
    # ================================================================
    def seed_demo_data(self):
        """注入演示进化数据 — 展示'越用越准'的效果"""
        status = self.get_status()
        if status["total_cycles"] > 0:
            return  # 已有数据，不重复注入

        print("[进化DB] 注入演示数据...")

        # 模拟3个进化周期
        demo_cycles = [
            {"papers": 5, "edges_before": 15, "edges_after": 17, "score": 0.65},
            {"papers": 8, "edges_before": 17, "edges_after": 20, "score": 0.72},
            {"papers": 6, "edges_before": 20, "edges_after": 21, "score": 0.78},
        ]
        for i, dc in enumerate(demo_cycles):
            # 把时间错开
            t = datetime.now().replace(hour=10+i*2)
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT INTO evolution_cycles (timestamp, papers_found, new_causal_edges,
                    edges_before, edges_after, knowledge_base_size, validation_score, status)
                    VALUES (?, ?, ?, ?, ?, ?, ?, 'completed')
                """, (
                    t.isoformat(),
                    dc["papers"],
                    dc["edges_after"] - dc["edges_before"],
                    dc["edges_before"], dc["edges_after"],
                    dc["edges_after"],
                    dc["score"],
                ))

        # 注入知识增长记录
        demo_knowledge = [
            ("冷却水阀门开度", "冷却水流量", "阀门开度↑→流量↑", 0.95, "工艺文档"),
            ("冷却水流量", "反应器温度", "流量↑→换热↑→温度↓", 0.90, "工艺文档"),
            ("反应器温度", "反应速率", "Arrhenius效应", 0.95, "工艺文档"),
            ("进料浓度", "反应速率", "浓度梯度驱动", 0.88, "文献提取"),
            ("轴承磨损", "振动幅值", "磨损加剧→振动增大", 0.80, "文献提取"),
            ("换热器结垢", "换热效率", "污垢热阻增大", 0.75, "文献提取"),
            ("催化剂活性", "反应速率", "活性↓→速率↓", 0.82, "文献提取"),
        ]
        for cause, effect, mech, conf, src in demo_knowledge:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT INTO knowledge_growth (timestamp, cause, effect, mechanism, confidence, source)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (datetime.now().isoformat(), cause, effect, mech, conf, src))

        # 注入参数进化
        demo_params = [
            ("heat_transfer_coeff_water", 3500, 3550, "literature", 0.02),
            ("fouling_resistance", 0.0001, 0.00012, "field_data", 0.05),
            ("activation_energy", 85.0, 83.5, "literature", 0.03),
        ]
        for name, old, new, trigger, imp in demo_params:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT INTO simulation_params (timestamp, param_name, old_value, new_value, trigger_source, improvement_metric)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (datetime.now().isoformat(), name, old, new, trigger, imp))

        # 注入自测验证
        demo_val = [
            ("structure", 0.85, True), ("quantitative", 0.72, True),
            ("behavior", 0.68, False), ("full", 0.78, True),
        ]
        for tt, sc, ps in demo_val:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT INTO validation_results (timestamp, test_type, score, passed)
                    VALUES (?, ?, ?, ?)
                """, (datetime.now().isoformat(), tt, sc, 1 if ps else 0))

        conn.commit()
        print(f"[进化DB] 演示数据已注入: {len(demo_cycles)}周期, {len(demo_knowledge)}知识, {len(demo_params)}参数变更")


if __name__ == "__main__":
    db = EvolutionDB()
    db.seed_demo_data()
    status = db.get_status()
    print(f"状态: {json.dumps(status, indent=2, ensure_ascii=False)}")
    print(f"知识增长: {len(db.get_growth_curve())} 条记录")
