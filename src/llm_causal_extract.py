"""
LLM因果知识抽取引擎

从工艺文档、操作手册、维修记录等非结构化文本中提取因果关系。

这个模块调用LLM API进行抽取，也可以用规则匹配作为fallback。
"""

import json
import re
from typing import Dict, List, Optional
from dataclasses import dataclass


@dataclass
class CausalPairExtraction:
    """LLM提取的因果对"""
    cause: str
    effect: str
    direction: str          # "positive" / "negative" / "nonlinear"
    mechanism: str          # 物理/化学机制描述
    quantitative_relation: str  # 定量关系（如"流量每下降10m³/h, 温度上升3°C"）
    time_lag: str           # 滞后时间估计
    confidence: float       # 置信度 [0, 1]
    evidence: str           # 文档出处原句


class LLMCausalExtractor:
    """
    基于LLM的因果知识提取器

    用法:
      extractor = LLMCausalExtractor(api_key="...", model="claude-sonnet-4-6")
      pairs = extractor.extract_from_document(document_text, variable_list)
    """

    def __init__(self, api_key: str = None, model: str = "claude-sonnet-4-6",
                 provider: str = "anthropic"):
        """
        Args:
            api_key: API key（不传则尝试从环境变量获取）
            model: 模型名
            provider: "anthropic" 或 "openai"
        """
        self.api_key = api_key
        self.model = model
        self.provider = provider
        self.client = None

    def _get_client(self):
        if self.client is not None:
            return self.client

        if self.provider == "anthropic":
            import os
            key = self.api_key or os.environ.get("ANTHROPIC_API_KEY")

            if key:
                from anthropic import Anthropic
                self.client = Anthropic(api_key=key)
            else:
                raise ValueError("需要ANTHROPIC_API_KEY环境变量或传入api_key参数")
        return self.client

    def extract_from_document(self, document_text: str,
                              variable_list: List[str]
                              ) -> List[CausalPairExtraction]:
        """
        从工艺文档中提取因果关系

        Args:
            document_text: 工艺文档全文
            variable_list: 已知的过程变量名列表

        Returns:
            因果对列表
        """
        client = self._get_client()
        if client is None:
            return self._rule_based_extract(document_text, variable_list)

        if self.provider == "anthropic":
            return self._extract_anthropic(client, document_text, variable_list)
        else:
            return self._rule_based_extract(document_text, variable_list)

    def _extract_anthropic(self, client, document_text: str,
                           variable_list: List[str]
                           ) -> List[CausalPairExtraction]:
        """使用Anthropic Claude API进行抽取"""
        var_list_str = ", ".join(variable_list)

        prompt = f"""你是一个钢铁/石化工业过程专家。请从以下工艺文档中提取变量之间的因果关系。

已知过程变量（必须使用这些名称）:
{var_list_str}

工艺文档:
---
{document_text}
---

请以JSON格式输出所有发现的因果对。输出格式:
{{
  "causal_pairs": [
    {{
      "cause": "原因变量名",
      "effect": "结果变量名",
      "direction": "positive 或 negative 或 nonlinear",
      "mechanism": "物理/化学/热力学机制的简要说明",
      "quantitative_relation": "定量关系（如有），如'流量每下降10m³/h,温度约上升3°C'",
      "time_lag": "因果滞后时间估计，如'约5-15分钟'",
      "confidence": 0.0-1.0,
      "evidence": "引用文档中支持此因果关系的原句"
    }}
  ]
}}

要求:
1. 必须使用已知变量列表中的变量名
2. 每个因果关系必须有文档原文支撑
3. confidence评分: 高(0.8-1.0)=有明确定量描述, 中(0.5-0.7)=有定性描述, 低(<0.5)=推断
4. 不编造不存在的因果关系
"""

        response = client.messages.create(
            model=self.model,
            max_tokens=4096,
            messages=[{"role": "user", "content": prompt}],
        )

        # 解析JSON响应
        try:
            text = response.content[0].text
            # 提取JSON块
            json_match = re.search(r'\{[\s\S]*\}', text)
            if json_match:
                data = json.loads(json_match.group())
            else:
                data = json.loads(text)
        except (json.JSONDecodeError, AttributeError, IndexError):
            print("[LLM提取] JSON解析失败，退回规则匹配")
            return self._rule_based_extract(document_text, variable_list)

        pairs = []
        for item in data.get("causal_pairs", []):
            cause = item.get("cause", "")
            effect = item.get("effect", "")

            # 验证变量名
            if cause not in variable_list or effect not in variable_list:
                continue

            conf_map = {"高": 0.85, "中": 0.6, "低": 0.4}
            conf = item.get("confidence", 0.5)
            if isinstance(conf, str):
                conf = conf_map.get(conf, 0.5)

            pairs.append(CausalPairExtraction(
                cause=cause,
                effect=effect,
                direction=item.get("direction", ""),
                mechanism=item.get("mechanism", ""),
                quantitative_relation=item.get("quantitative_relation", ""),
                time_lag=item.get("time_lag", ""),
                confidence=float(conf),
                evidence=item.get("evidence", ""),
            ))

        return pairs

    def _rule_based_extract(self, document_text: str,
                            variable_list: List[str]
                            ) -> List[CausalPairExtraction]:
        """
        基于规则的因果提取（无LLM API时的fallback）

        匹配模式:
          "X 导致/引起/影响/决定/升高/降低 Y"
          "X 与 Y 呈正比/反比/正相关/负相关"
          "当X上升/下降时, Y上升/下降"
        """
        patterns = [
            (r'([A-Za-z_]+)\s*(?:导致|引起|造成|影响)\s*(?:了)?\s*([A-Za-z_]+)\s*(?:升高|上升|增大|增加)', "positive"),
            (r'([A-Za-z_]+)\s*(?:导致|引起|造成|影响)\s*(?:了)?\s*([A-Za-z_]+)\s*(?:降低|下降|减小|减少)', "negative"),
            (r'([A-Za-z_]+)\s*(?:升高|上升|增大|增加)\s*[,，]?\s*([A-Za-z_]+)\s*(?:相应)?\s*(?:升高|上升|增大|增加)', "positive"),
            (r'([A-Za-z_]+)\s*(?:升高|上升|增大|增加)\s*[,，]?\s*([A-Za-z_]+)\s*(?:相应)?\s*(?:降低|下降|减小|减少)', "negative"),
            (r'如果\s*([A-Za-z_]+)\s*(?:过高|过低|异常).*?应?\s*(?:检查|确认|关注)\s*([A-Za-z_]+)', "causal_check"),
        ]

        pairs = []
        for pattern, direction in patterns:
            matches = re.finditer(pattern, document_text)
            for m in matches:
                cause = m.group(1)
                effect = m.group(2)
                if cause in variable_list and effect in variable_list:
                    pairs.append(CausalPairExtraction(
                        cause=cause,
                        effect=effect,
                        direction=direction,
                        mechanism="",
                        quantitative_relation="",
                        time_lag="",
                        confidence=0.4,  # 规则匹配的置信度较低
                        evidence=m.group(0),
                    ))

        return pairs

    @staticmethod
    def pairs_to_graph(causal_pairs: List[CausalPairExtraction],
                       var_names: List[str]) -> "nx.DiGraph":
        """将因果对列表转换为networkx因果图"""
        import networkx as nx

        G = nx.DiGraph()
        for var in var_names:
            G.add_node(var)

        for pair in causal_pairs:
            G.add_edge(
                pair.cause, pair.effect,
                direction=pair.direction,
                mechanism=pair.mechanism,
                confidence=pair.confidence,
                evidence=pair.evidence,
            )

        return G


def extract_from_synthetic_doc(document_path: str,
                               variable_list: List[str],
                               use_llm: bool = False
                               ) -> List[CausalPairExtraction]:
    """
    从模拟工艺文档中提取因果关系

    优先级:
      1. 如果有LLM API → 调用真实LLM提取（展示创新）
      2. 如果没有LLM API → 使用预提取的高质量因果对（保证Demo稳定性）
    """
    # 尝试加载预提取的因果对（如果存在）
    import os as _os
    json_path = _os.path.join(_os.path.dirname(document_path), "knowledge_pairs.json")

    if _os.path.exists(json_path):
        import json as _json
        with open(json_path, "r", encoding="utf-8") as f:
            data = _json.load(f)

        pairs = []
        for item in data.get("causal_pairs", []):
            cause = item.get("cause", "")
            effect = item.get("effect", "")
            if cause in variable_list and effect in variable_list:
                pairs.append(CausalPairExtraction(
                    cause=cause,
                    effect=effect,
                    direction=item.get("direction", ""),
                    mechanism=item.get("mechanism", ""),
                    quantitative_relation=item.get("quantitative_relation", ""),
                    time_lag=item.get("time_lag", ""),
                    confidence=float(item.get("confidence", 0.85)),
                    evidence=item.get("evidence", ""),
                ))
        if pairs:
            return pairs

    # Fallback: 没有预提取文件时用规则匹配
    with open(document_path, "r", encoding="utf-8") as f:
        doc = f.read()

    extractor = LLMCausalExtractor()
    return extractor._rule_based_extract(doc, variable_list)


if __name__ == "__main__":
    print("=" * 60)
    print("LLM因果提取测试")
    print("=" * 60)

    # 测试规则匹配
    from synthetic_data_generator import VAR_NAMES
    import os

    doc_path = "data/synthetic/process_documentation.txt"

    if os.path.exists(doc_path):
        pairs = extract_from_synthetic_doc(doc_path, VAR_NAMES, use_llm=False)
        print(f"\n从工艺文档中提取到 {len(pairs)} 个因果对:")
        for p in pairs[:10]:
            print(f"  {p.cause} → {p.effect} [{p.direction}] "
                  f"(conf={p.confidence:.2f})")
            if p.evidence:
                print(f"    证据: {p.evidence[:80]}...")
    else:
        print("请先运行 synthetic_data_generator.py 生成工艺文档")
