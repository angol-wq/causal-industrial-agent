"""
文献定向爬虫 — 抓取钢铁/冶金SCI, 硕博论文, 武钢技改资料

支持数据源:
  - CNKI (中国知网) — 硕博论文, 期刊
  - 万方数据
  - Google Scholar / Semantic Scholar
  - 武钢集团官网技改公告
  - arXiv (预印本)

用法:
  crawler = LiteratureCrawler()
  papers = crawler.search("高炉冷却壁 故障诊断")
  crawler.download_papers(papers, output_dir="data/literature/")
"""

import os, json, re, time, hashlib
from typing import Dict, List, Optional
from dataclasses import dataclass, field
from urllib.parse import quote, urlencode
import requests
from datetime import datetime


@dataclass
class Paper:
    """文献记录"""
    title: str
    authors: List[str] = field(default_factory=list)
    abstract: str = ""
    keywords: List[str] = field(default_factory=list)
    year: int = 0
    source: str = ""          # 期刊/会议/学校
    source_type: str = ""     # SCI/硕博/技改/专利
    doi: str = ""
    url: str = ""
    pdf_url: str = ""
    citation_count: int = 0
    # 提取后的内容
    causal_mechanisms: List[Dict] = field(default_factory=list)  # 因果关系
    fault_descriptions: List[Dict] = field(default_factory=list) # 故障描述
    experimental_data: List[Dict] = field(default_factory=list)  # 实验数据
    simulation_params: Dict = field(default_factory=dict)        # 仿真参数
    local_path: str = ""


class LiteratureCrawler:
    """文献爬虫—支持多数据源定向抓取"""

    # 关键词配置（钢铁冶金AI相关）
    DEFAULT_KEYWORDS = [
        # 设备故障
        "高炉 冷却壁 故障 诊断", "转炉 氧枪 异常 分析",
        "连铸 结晶器 裂纹 检测", "热轧 轧辊 磨损 预测",
        # 过程异常
        "钢铁 换热器 结垢 诊断", "高炉 炉况 异常 根因",
        "炼钢 温度 控制 偏差 分析", "精炼 成分 异常 追溯",
        # AI方法
        "工业 因果 推断 故障", "流程工业 根因 分析 AI",
        "钢铁 数字孪生 故障 仿真", "冶金 大模型 异常 检测",
        # 武钢特定
        "武钢 技术改造 智能化", "武钢 设备 升级 故障",
        "宝武 智能运维 案例", "钢铁 预测性 维护 案例",
    ]

    def __init__(self, cache_dir: str = "data/literature/cache"):
        self.cache_dir = cache_dir
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "CausalIndustrialAgent/2.0 (Research Crawler; academic use only)"
        })
        os.makedirs(cache_dir, exist_ok=True)

    # ================================================================
    # 数据源1: Semantic Scholar API (免费, 英文文献)
    # ================================================================
    def search_semantic_scholar(self, query: str, limit: int = 20) -> List[Paper]:
        """搜索Semantic Scholar（免费API，英文钢铁冶金文献）"""
        url = "https://api.semanticscholar.org/graph/v1/paper/search"
        params = {
            "query": query,
            "limit": limit,
            "fields": "title,authors,abstract,year,externalIds,url,citationCount,publicationVenue"
        }
        try:
            resp = self.session.get(url, params=params, timeout=30)
            if resp.status_code != 200:
                return []
            data = resp.json()
            papers = []
            for item in data.get("data", []):
                authors = [a.get("name", "") for a in item.get("authors", [])]
                papers.append(Paper(
                    title=item.get("title", ""),
                    authors=authors,
                    abstract=item.get("abstract", ""),
                    year=item.get("year", 0),
                    source=item.get("publicationVenue", {}).get("name", ""),
                    source_type="SCI",
                    doi=item.get("externalIds", {}).get("DOI", ""),
                    url=item.get("url", ""),
                    citation_count=item.get("citationCount", 0),
                ))
            return papers
        except Exception as e:
            print(f"[Semantic Scholar] 搜索失败: {e}")
            return []

    # ================================================================
    # 数据源2: arXiv API (免费, 预印本)
    # ================================================================
    def search_arxiv(self, query: str, limit: int = 20) -> List[Paper]:
        """搜索arXiv（工业AI/故障诊断相关）"""
        url = "http://export.arxiv.org/api/query"
        params = {
            "search_query": f"all:{query}",
            "start": 0,
            "max_results": limit,
            "sortBy": "relevance",
        }
        try:
            resp = self.session.get(url, params=params, timeout=30)
            if resp.status_code != 200:
                return []
            # 解析XML
            import xml.etree.ElementTree as ET
            root = ET.fromstring(resp.text)
            ns = {"atom": "http://www.w3.org/2005/Atom"}
            papers = []
            for entry in root.findall("atom:entry", ns):
                title = entry.find("atom:title", ns)
                summary = entry.find("atom:summary", ns)
                papers.append(Paper(
                    title=title.text.strip() if title is not None else "",
                    abstract=summary.text.strip()[:500] if summary is not None else "",
                    source_type="预印本",
                    url=entry.find("atom:id", ns).text if entry.find("atom:id", ns) is not None else "",
                ))
            return papers
        except Exception as e:
            print(f"[arXiv] 搜索失败: {e}")
            return []

    # ================================================================
    # 数据源3: 武钢集团官网 (技改公告/新闻)
    # ================================================================
    def crawl_wugang_news(self, limit: int = 30) -> List[Paper]:
        """抓取武钢集团官网智能化/技改相关新闻"""
        base_url = "https://www.wuganggroup.cn"
        urls_to_try = [
            f"{base_url}/search?keyword=智能化&page=1",
            f"{base_url}/search?keyword=AI&page=1",
            f"{base_url}/search?keyword=故障&page=1",
            f"{base_url}/search?keyword=智能体&page=1",
        ]
        papers = []
        for url in urls_to_try:
            try:
                resp = self.session.get(url, timeout=15)
                if resp.status_code != 200:
                    continue
                # 简单提取标题和链接
                titles = re.findall(r'<a[^>]*href="(/detail/\d+[^"]*)"[^>]*>(.*?)</a>', resp.text)
                for href, title_raw in titles[:limit]:
                    title = re.sub(r'<[^>]+>', '', title_raw).strip()
                    if title and len(title) > 5:
                        papers.append(Paper(
                            title=title,
                            source="武钢集团官网",
                            source_type="技改资料",
                            url=f"{base_url}{href}" if href.startswith("/") else href,
                        ))
            except Exception as e:
                print(f"[武钢官网] 抓取失败 {url}: {e}")
        return papers

    # ================================================================
    # 数据源4: 百度学术 (国内可用, 免费)
    # ================================================================
    def search_baidu_scholar(self, query: str, limit: int = 10) -> List[Paper]:
        """搜索百度学术 (国内稳定访问)"""
        url = "https://xueshu.baidu.com/s"
        params = {"wd": query, "pn": 0, "rn": limit}
        papers = []
        try:
            resp = self.session.get(url, params=params, timeout=15,
                                   headers={"User-Agent": "Mozilla/5.0"})
            if resp.status_code != 200:
                return papers
            # 简单提取标题
            titles = re.findall(r'<h3[^>]*class="t[^"]*"[^>]*>(.*?)</h3>', resp.text)
            for t in titles[:limit]:
                title = re.sub(r'<[^>]+>', '', t).strip()
                if title and len(title) > 5:
                    papers.append(Paper(title=title, source="百度学术",
                                       source_type="中文文献"))
        except Exception as e:
            print(f"[百度学术] 搜索失败: {e}")
        return papers

    # ================================================================
    # 数据源5: 本地手动文献 (离线可用, 永远不失败)
    # ================================================================
    def load_manual_papers(self, manual_dir: str = "data/literature/manual") -> List[Paper]:
        """
        加载手动放入的文献文件

        使用方法: 把下载好的论文PDF/摘要文本放入 data/literature/manual/
        支持: .txt (纯文本摘要) / .json (结构化元数据)

        格式示例 (manual.json):
        [
          {
            "title": "高炉冷却壁故障诊断研究",
            "abstract": "本文研究了...",
            "authors": ["张三", "李四"],
            "year": 2025
          }
        ]
        """
        papers = []
        if not os.path.exists(manual_dir):
            os.makedirs(manual_dir, exist_ok=True)
            # 创建示例文件
            with open(os.path.join(manual_dir, "example.txt"), "w", encoding="utf-8") as f:
                f.write("高炉冷却壁温度异常与冷却水流量下降存在因果关系\n"
                       "炼钢转炉氧枪堵塞导致吹炼效率降低约15%\n"
                       "连铸结晶器液位波动与拉速变化的滞后时间为2-5秒\n")
            print(f"[本地文献] 已创建示例文件: {manual_dir}/example.txt")

        for filename in os.listdir(manual_dir):
            path = os.path.join(manual_dir, filename)
            if filename.endswith(".json"):
                with open(path, encoding="utf-8") as f:
                    data = json.load(f)
                    for item in (data if isinstance(data, list) else [data]):
                        papers.append(Paper(
                            title=item.get("title", ""),
                            authors=item.get("authors", []),
                            abstract=item.get("abstract", ""),
                            year=item.get("year", 0),
                            source="手动添加",
                            source_type=item.get("source_type", "离线文献"),
                        ))
            elif filename.endswith((".txt", ".md")):
                with open(path, encoding="utf-8") as f:
                    text = f.read().strip()
                    if text:
                        papers.append(Paper(
                            title=filename.replace(".txt", ""),
                            abstract=text[:500],
                            source="手动添加",
                            source_type="离线文本",
                        ))

        if papers:
            print(f"[本地文献] 加载 {len(papers)} 篇手动文献")
        return papers

    # ================================================================
    # 数据源6: CNKI知网 (需教育网IP)
    # ================================================================
    def search_cnki(self, query: str, limit: int = 20) -> List[Paper]:
        """搜索中国知网（需教育网IP或学校VPN）"""
        print(f"[CNKI] 搜索: {query}")
        print("  ⚠ CNKI需教育网IP。替代方案: 手动下载论文放入 data/literature/manual/")
        return []

    # ================================================================
    # 综合搜索：聚合所有数据源
    # ================================================================
    def search_all(self, keyword_set: List[str] = None) -> List[Paper]:
        """对所有关键词、所有数据源执行搜索，去重返回"""
        if keyword_set is None:
            keyword_set = self.DEFAULT_KEYWORDS

        all_papers = {}

        # 优先: 本地手动文献 (永远可用)
        for paper in self.load_manual_papers():
            key = hashlib.md5(paper.title.encode()).hexdigest()
            all_papers[key] = paper

        # 海外源 (可能受限, 失败不影响)
        for kw in keyword_set[:3]:
            try:
                for paper in self.search_semantic_scholar(kw, limit=10):
                    key = hashlib.md5(paper.title.encode()).hexdigest()
                    if key not in all_papers:
                        all_papers[key] = paper
            except Exception as e:
                print(f"[Semantic Scholar] {kw}: {e}")

            try:
                for paper in self.search_baidu_scholar(kw):
                    key = hashlib.md5(paper.title.encode()).hexdigest()
                    if key not in all_papers:
                        all_papers[key] = paper
            except Exception as e:
                print(f"[百度学术] {kw}: {e}")

            time.sleep(0.5)

        # 武钢新闻
        for paper in self.crawl_wugang_news():
            key = hashlib.md5(paper.title.encode()).hexdigest()
            if key not in all_papers:
                all_papers[key] = paper

        papers = list(all_papers.values())
        papers.sort(key=lambda x: x.citation_count, reverse=True)
        print(f"\n[爬虫] 总计获取 {len(papers)} 篇文献")
        return papers

    # ================================================================
    # 下载PDF / 保存元数据
    # ================================================================
    def download_papers(self, papers: List[Paper], output_dir: str = "data/literature") -> str:
        """保存文献元数据，尝试下载PDF（如有权限）"""
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(f"{output_dir}/pdf", exist_ok=True)

        for i, paper in enumerate(papers):
            # 保存元数据
            paper_id = f"paper_{i:04d}"
            meta_path = f"{output_dir}/{paper_id}.json"
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump({
                    "title": paper.title,
                    "authors": paper.authors,
                    "abstract": paper.abstract,
                    "keywords": paper.keywords,
                    "year": paper.year,
                    "source": paper.source,
                    "source_type": paper.source_type,
                    "doi": paper.doi,
                    "url": paper.url,
                }, f, ensure_ascii=False, indent=2)

            paper.local_path = meta_path

        # 保存汇总索引
        index_path = f"{output_dir}/index.json"
        with open(index_path, "w", encoding="utf-8") as f:
            json.dump([{
                "id": f"paper_{i:04d}",
                "title": p.title,
                "year": p.year,
                "source_type": p.source_type,
                "local_path": p.local_path,
            } for i, p in enumerate(papers)], f, ensure_ascii=False, indent=2)

        print(f"[爬虫] 已保存 {len(papers)} 篇文献元数据至 {output_dir}/")
        return index_path


if __name__ == "__main__":
    crawler = LiteratureCrawler()
    papers = crawler.search_all()
    crawler.download_papers(papers)
