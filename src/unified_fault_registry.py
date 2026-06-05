"""
统一故障注册中心 — 合并合成数据10种 + TEP真实数据21种 = 31种故障

用法:
  registry = UnifiedFaultRegistry()
  all_faults = registry.list_all()       # 31种故障
  df, meta = registry.load_fault("IDV(14)")  # 加载任意故障
"""

import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field

from src.synthetic_data_generator import (
    SyntheticProcessSimulator, FAULT_MODES, VAR_NAMES as SYNTH_VARS,
    CAUSAL_GRAPH_TRUTH,
)
from src.tep_data_loader import (
    TEPDataLoader, DEMO_FAULTS, TEP_FAULT_ROOT_CAUSE, TEP_VAR_NAMES,
)


@dataclass
class FaultEntry:
    """统一故障条目"""
    fault_id: str              # "FAULT_COOLING_VALVE_STUCK" / "IDV(14)"
    name: str                  # 故障名
    description: str           # 描述
    category: str              # "synthetic" / "tep"
    root_cause: str            # 预期根因变量
    causal_path: str           # 因果路径描述
    var_names: List[str]       # 相关变量
    severity: str = "中等"     # 轻微/中等/严重/致命
    industry: str = "通用化工"  # 钢铁/石化/通用化工
    data_source: str = ""      # 数据来源


class UnifiedFaultRegistry:
    """统一故障注册中心 — 31种故障"""

    def __init__(self):
        self.faults: Dict[str, FaultEntry] = {}
        self._synth_sim = None
        self._tep_loader = None
        self._register_all()

    @property
    def synth_sim(self):
        if self._synth_sim is None:
            self._synth_sim = SyntheticProcessSimulator(seed=42)
        return self._synth_sim

    @property
    def tep_loader(self):
        if self._tep_loader is None:
            self._tep_loader = TEPDataLoader()
        return self._tep_loader

    def _register_all(self):
        """注册全部31种故障"""
        self._register_synthetic_faults()
        self._register_tep_faults()

    def _register_synthetic_faults(self):
        """注册合成数据10种故障"""
        for fid, cfg in FAULT_MODES.items():
            self.faults[fid] = FaultEntry(
                fault_id=fid,
                name=cfg["name"],
                description=cfg["description"],
                category="synthetic",
                root_cause=cfg["root_cause"],
                causal_path=cfg["causal_path"],
                var_names=SYNTH_VARS,
                severity="严重" if "复合" in cfg["name"] else "中等",
                industry="通用化工",
                data_source="CSTR合成仿真器",
            )

    def _register_tep_faults(self):
        """注册TEP真实数据21种故障"""
        for i in range(1, 22):
            fid = f"IDV({i})"
            root_cause = TEP_FAULT_ROOT_CAUSE.get(fid, f"故障{i}")

            # 从DEMO_FAULTS获取增强信息（如有）
            if fid in DEMO_FAULTS:
                df_info = DEMO_FAULTS[fid]
                name = df_info["name"]
                desc = df_info["description"]
                path = df_info["causal_path"]
                severity = "严重" if "阀门" in name else "中等"
                industry = "钢铁" if "冷却水" in name or "温度" in name else "石化"
                var_subset = df_info.get("tep_causal_vars", TEP_VAR_NAMES[:8])
            else:
                name = root_cause.split("—")[0].strip() if "—" in root_cause else f"故障{i}"
                desc = root_cause
                path = ""
                severity = "中等"
                industry = "石化"
                var_subset = TEP_VAR_NAMES[:8]

            self.faults[fid] = FaultEntry(
                fault_id=fid,
                name=name,
                description=desc,
                category="tep",
                root_cause="",
                causal_path=path,
                var_names=var_subset,
                severity=severity,
                industry=industry,
                data_source="TEP真实工业数据 (MIT Braatz Group)",
            )

    def list_all(self, category: str = None, industry: str = None) -> List[FaultEntry]:
        """列出所有故障（可按分类/行业筛选）"""
        result = list(self.faults.values())
        if category:
            result = [f for f in result if f.category == category]
        if industry:
            result = [f for f in result if f.industry == industry]
        return result

    def list_by_industry(self) -> Dict[str, List[FaultEntry]]:
        """按行业分组"""
        grouped = {}
        for f in self.faults.values():
            grouped.setdefault(f.industry, []).append(f)
        return grouped

    def get(self, fault_id: str) -> Optional[FaultEntry]:
        return self.faults.get(fault_id)

    def load_fault(self, fault_id: str) -> Tuple[pd.DataFrame, Dict]:
        """根据故障ID自动选择合成或TEP数据加载"""
        entry = self.faults.get(fault_id)
        if not entry:
            raise ValueError(f"未知故障: {fault_id}")

        if entry.category == "synthetic":
            return self._load_synthetic(fault_id)
        else:
            return self._load_tep(fault_id)

    def _load_synthetic(self, fault_id: str) -> Tuple[pd.DataFrame, Dict]:
        df, meta = self.synth_sim.generate_fault_dataset(
            n_normal=300, n_fault=500, fault_name=fault_id
        )
        return df, meta

    def _load_tep(self, fault_id: str) -> Tuple[pd.DataFrame, Dict]:
        fault_num = int(fault_id.replace("IDV(", "").replace(")", ""))
        df, meta = self.tep_loader.load_fault_data(fault_num)
        return df, meta

    def get_stats(self) -> Dict:
        """统计信息"""
        by_cat = {}
        for f in self.faults.values():
            by_cat[f.category] = by_cat.get(f.category, 0) + 1
        by_ind = {}
        for f in self.faults.values():
            by_ind[f.industry] = by_ind.get(f.industry, 0) + 1
        return {
            "total": len(self.faults),
            "by_category": by_cat,
            "by_industry": by_ind,
        }


if __name__ == "__main__":
    registry = UnifiedFaultRegistry()
    stats = registry.get_stats()
    print(f"故障注册中心: {stats['total']}种故障")
    print(f"  分类: {stats['by_category']}")
    print(f"  行业: {stats['by_industry']}")
    print(f"\n前10种:")
    for f in registry.list_all()[:10]:
        print(f"  [{f.category}] {f.fault_id}: {f.name}")
