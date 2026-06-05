"""
AI精读提取 — LLM从文献中提取因果机理、故障数据、仿真参数

核心能力:
  1. 因果机理提取: 从摘要/全文识别新的变量因果关系
  2. 故障实验数据提取: 从论文表格/描述中抽取定量故障数据
  3. 仿真参数提取: 从方法论章节提取热力学/换热/侵蚀参数
  4. 新旧知识去重融合: 新提取的因果关系与现有因果图比对，去重、补新、修正

用法:
  extractor = LiteratureExtractor()
  new_knowledge = extractor.extract_from_papers(papers, existing_causal_graph)
  # new_knowledge.causal_edges  → 新因果边
  # new_knowledge.fault_data    → 新故障数据
  # new_knowledge.sim_params    → 新仿真参数
"""

import os, json, re
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
import hashlib


@dataclass
class NewKnowledge:
    """从文献中提取的新知识"""
    causal_edges: List[Dict] = field(default_factory=list)     # 新因果边
    corrected_edges: List[Dict] = field(default_factory=list)  # 修正现有边
    fault_scenarios: List[Dict] = field(default_factory=list)  # 新故障场景
    sim_params: Dict = field(default_factory=dict)             # 仿真参数
    experimental_data: List[Dict] = field(default_factory=list)  # 实验数据
    validation_cases: List[Dict] = field(default_factory=list)   # 验证案例


class LiteratureExtractor:
    """AI精读引擎 — 从文献中提取可融入因果图的知识"""

    # 提取Prompt模板
    EXTRACT_CAUSAL_PROMPT = """你是钢铁/冶金工业过程专家。仔细阅读以下文献摘要，提取其中的因果机理。

文献标题: {title}
文献来源: {source} ({year})
文献摘要: {abstract}

请以JSON格式提取以下内容（只提取文献中明确提及的，不要编造）:

{{
  "causal_relationships": [
    {{
      "cause": "原因变量（使用标准术语）",
      "effect": "结果变量（使用标准术语）",
      "mechanism": "物理/化学/冶金机理描述",
      "direction": "positive / negative / nonlinear",
      "quantitative": "定量关系描述（如有）",
      "confidence": 0.0-1.0,
      "context": "文献中的具体描述场景（温度范围/设备类型/工况）",
      "evidence_quote": "原文直接引用"
    }}
  ],
  "fault_descriptions": [
    {{
      "fault_name": "故障名称",
      "symptoms": ["症状1", "症状2"],
      "root_cause": "根本原因",
      "propagation_path": ["变量A", "变量B", "变量C"],
      "detection_method": "检测方法",
      "severity": "轻微/中等/严重/致命"
    }}
  ],
  "simulation_parameters": {{
    "parameter_name": {{
      "value": 数值,
      "unit": "单位",
      "source": "文献出处",
      "applicable_range": "适用范围"
    }}
  }},
  "experimental_data": [
    {{
      "variable": "变量名",
      "normal_range": [最小值, 最大值],
      "fault_range": [最小值, 最大值],
      "sample_rate": "采样频率",
      "test_condition": "实验条件描述"
    }}
  ],
  "has_causal_graph": true/false,
  "has_experimental_data": true/false,
  "novelty_score": 0.0-1.0
}}

要求:
1. 只提取文献中明确描述的因果关系统，不要推测
2. 定量关系优先，定性次之
3. confidence评分: 0.9+有实验验证, 0.7+有仿真验证, 0.5+有理论推导, <0.5仅为推测
4. novelty_score: 1.0=全新发现, 0.5=已有验证, 0=已知常识
"""

    DUPLICATE_CHECK_PROMPT = """以下是文献中提取的新因果关系:
{new_edges}

以下是现有因果图中已有的关系:
{existing_edges}

请判断每条新关系与现有关系的重复度（0.0=完全不同, 1.0=完全相同）,
以及是否需要修正现有关系（如新文献提供了更精确的定量数据）:

{{
  "dedup_results": [
    {{
      "new_edge": "X→Y",
      "most_similar_existing": "A→B",
      "similarity": 0.0-1.0,
      "action": "add_new / merge / replace_existing / discard",
      "reason": "判断理由"
    }}
  ]
}}
"""

    def __init__(self, api_key: str = None, model: str = "claude-sonnet-4-6"):
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        self.model = model
        self._client = None

    def _get_llm(self):
        if self._client is None:
            try:
                from anthropic import Anthropic
                self._client = Anthropic(api_key=self.api_key)
            except ImportError:
                print("[提取器] anthropic SDK未安装，使用规则模式")
                self._client = None
        return self._client

    def extract_from_papers(self, papers: List,
                            existing_causal_graph=None) -> NewKnowledge:
        """
        批量提取：遍历所有文献，提取因果机理、故障数据、仿真参数
        """
        knowledge = NewKnowledge()
        client = self._get_llm()

        for paper in papers:
            print(f"[精读] {paper.title[:60]}...")

            if client:
                result = self._extract_with_llm(client, paper)
            else:
                result = self._extract_with_rules(paper)

            if result:
                knowledge.causal_edges.extend(result.get("causal_relationships", []))
                knowledge.fault_scenarios.extend(result.get("fault_descriptions", []))
                knowledge.experimental_data.extend(result.get("experimental_data", []))
                # 合并仿真参数（后出现的覆盖先出现的）
                knowledge.sim_params.update(result.get("simulation_parameters", {}))

        # 去重：与新知识内部和现有因果图比对
        if knowledge.causal_edges and existing_causal_graph is not None:
            knowledge = self._deduplicate(knowledge, existing_causal_graph, client)

        print(f"[精读] 提取完成: {len(knowledge.causal_edges)}条新因果边, "
              f"{len(knowledge.fault_scenarios)}个新故障场景, "
              f"{len(knowledge.sim_params)}个仿真参数")
        return knowledge

    def _extract_with_llm(self, client, paper) -> Optional[Dict]:
        """使用Claude API进行文献精读"""
        try:
            prompt = self.EXTRACT_CAUSAL_PROMPT.format(
                title=paper.title,
                source=getattr(paper, 'source', '未知'),
                year=getattr(paper, 'year', 0),
                abstract=getattr(paper, 'abstract', '')[:3000],
            )
            resp = client.messages.create(
                model=self.model, max_tokens=4096,
                messages=[{"role": "user", "content": prompt}],
            )
            text = resp.content[0].text
            json_match = re.search(r'\{[\s\S]*\}', text)
            if json_match:
                return json.loads(json_match.group())
        except Exception as e:
            print(f"  LLM提取失败: {e}")
        return None

    def _extract_with_rules(self, paper) -> Optional[Dict]:
        """无LLM时的规则提取（fallback）"""
        abstract = getattr(paper, 'abstract', '').lower()
        result = {
            "causal_relationships": [],
            "fault_descriptions": [],
            "simulation_parameters": {},
            "experimental_data": [],
        }

        # 规则1: 识别钢铁设备关键词 → 因果方向
        causal_patterns = [
            (r'(冷却水|冷却壁|cooling).*(温度|temperature)', {'cause': '冷却水流量', 'effect': '设备温度', 'direction': 'negative', 'confidence': 0.6}),
            (r'(结垢|fouling).*(换热|heat transfer)', {'cause': '换热器结垢', 'effect': '换热效率', 'direction': 'negative', 'confidence': 0.7}),
            (r'(阀门|valve).*(卡滞|stuck|堵塞|block)', {'cause': '阀门状态', 'effect': '流量', 'direction': 'negative', 'confidence': 0.8}),
            (r'(催化剂|catalyst).*(失活|deactivation)', {'cause': '催化剂活性', 'effect': '反应速率', 'direction': 'negative', 'confidence': 0.7}),
            (r'(轴承|bearing).*(磨损|wear).*(振动|vibration)', {'cause': '轴承磨损', 'effect': '振动幅值', 'direction': 'positive', 'confidence': 0.8}),
            (r'(炉况|furnace).*(异常|anomal).*(温度|temp)', {'cause': '炉况异常', 'effect': '炉温波动', 'direction': 'positive', 'confidence': 0.6}),
        ]
        for pattern, causal in causal_patterns:
            if re.search(pattern, abstract, re.IGNORECASE):
                result["causal_relationships"].append(causal)

        # 规则2: 提取数值参数
        param_patterns = [
            (r'换热系数[：:=]?\s*([\d.]+)\s*[Ww]/([m㎡]', '换热系数'),
            (r'导热率[：:=]?\s*([\d.]+)', '导热率'),
            (r'侵蚀速率[：:=]?\s*([\d.]+)', '侵蚀速率'),
        ]
        for pattern, param_name in param_patterns:
            match = re.search(pattern, abstract)
            if match:
                try:
                    result["simulation_parameters"][param_name] = {
                        "value": float(match.group(1)),
                        "source": f"{getattr(paper, 'title', '')[:60]}"
                    }
                except ValueError:
                    pass

        if result["causal_relationships"]:
            return result
        return None

    def _deduplicate(self, knowledge: NewKnowledge, existing_graph, client) -> NewKnowledge:
        """去重：新知识 vs 现有因果图"""
        new_edges_str = json.dumps([{
            "cause": e.get("cause", ""), "effect": e.get("effect", ""),
            "mechanism": e.get("mechanism", ""), "confidence": e.get("confidence", 0)
        } for e in knowledge.causal_edges], ensure_ascii=False)

        existing_edges_str = json.dumps([
            {"cause": u, "effect": v, "mechanism": d.get("mechanism", "")}
            for u, v, d in existing_graph.edges(data=True)
        ], ensure_ascii=False)

        if client and len(new_edges_str) > 100:
            try:
                prompt = self.DUPLICATE_CHECK_PROMPT.format(
                    new_edges=new_edges_str, existing_edges=existing_edges_str,
                )
                resp = client.messages.create(
                    model=self.model, max_tokens=2048,
                    messages=[{"role": "user", "content": prompt}],
                )
                text = resp.content[0].text
                json_match = re.search(r'\{[\s\S]*\}', text)
                if json_match:
                    dedup = json.loads(json_match.group())
                    # 按action筛选
                    kept = [e for e, d in zip(knowledge.causal_edges,
                            dedup.get("dedup_results", []))
                            if d.get("action") != "discard"]
                    knowledge.causal_edges = kept
            except Exception as e:
                print(f"  去重失败: {e}")

        return knowledge

    def extract_sim_params_from_text(self, text: str) -> Dict:
        """从任意文本中提取可用的仿真参数（热力学/换热/侵蚀）"""
        params = {}
        param_specs = {
            "换热系数_水侧": r'(?:水侧|冷却水).*换热系数[：:=]?\s*([\d.]+)\s*[Ww]',
            "换热系数_气侧": r'(?:气侧|烟气).*换热系数[：:=]?\s*([\d.]+)\s*[Ww]',
            "导热率_炉衬": r'炉衬.*导热率[：:=]?\s*([\d.]+)',
            "侵蚀速率_炉壁": r'(?:炉壁|冷却壁).*侵蚀.*速率[：:=]?\s*([\d.]+)',
            "结垢热阻": r'结垢.*热阻[：:=]?\s*([\d.]+)',
            "反应活化能": r'活化能[：:=]?\s*([\d.]+)\s*[kK][Jj]',
            "催化剂失活速率": r'催化剂.*失活.*速率[：:=]?\s*([\d.]+)',
        }
        for param_name, pattern in param_specs.items():
            match = re.search(pattern, text)
            if match:
                try:
                    params[param_name] = float(match.group(1))
                except ValueError:
                    pass
        return params


if __name__ == "__main__":
    # 测试
    from literature_crawler import LiteratureCrawler, Paper
    crawler = LiteratureCrawler()
    papers = crawler.search_semantic_scholar("blast furnace cooling stave fault diagnosis")
    extractor = LiteratureExtractor()
    knowledge = extractor.extract_from_papers(papers)
    print(f"\n提取结果: {len(knowledge.causal_edges)} 因果边, {len(knowledge.sim_params)} 仿真参数")
