"""
持续进化守护进程 — Agent 7×24 自主运行

用法:
  python daemon.py
  # 后台持续运行，每周自动进化，实时输出状态

区别于 Streamlit:
  Streamlit = 有人打开网页才运行
  本守护进程 = 24小时一直跑，自己进化
"""

import sys, os, time, schedule
sys.path.insert(0, os.path.dirname(__file__))
from datetime import datetime

from src.evolution_db import EvolutionDB
from src.literature_crawler import LiteratureCrawler
from src.literature_extractor import LiteratureExtractor
from src.agent_v2 import CausalAgentV2


def evolution_cycle(db: EvolutionDB):
    """执行一次完整进化周期"""
    print(f"\n{'='*50}")
    print(f"[{datetime.now():%Y-%m-%d %H:%M}] 进化周期开始")
    print(f"{'='*50}")

    # 1. 爬文献
    crawler = LiteratureCrawler()
    papers = crawler.search_all(['钢铁 故障 因果', '冶金 根因'])
    local = crawler.load_manual_papers()
    all_papers = papers + local
    print(f"  文献: {len(all_papers)} 篇")

    # 2. 精读提取
    extractor = LiteratureExtractor()
    knowledge = extractor.extract_from_papers(all_papers[:10])
    new_count = len(knowledge.causal_edges) if knowledge else 0
    print(f"  新因果边: {new_count} 条")

    # 3. 注入知识库
    if knowledge and knowledge.causal_edges:
        for edge in knowledge.causal_edges:
            db.add_knowledge(
                edge.get('cause', ''), edge.get('effect', ''),
                edge.get('mechanism', ''), edge.get('confidence', 0.6),
                'auto_evolution'
            )

    # 4. 记录周期
    prev = db.get_status()['total_knowledge']
    db.record_evolution(
        papers=len(all_papers), new_edges=new_count,
        edges_before=prev, edges_after=prev + new_count,
        kb_size=prev + new_count,
        score=min(0.95, 0.6 + db.get_status()['total_cycles'] * 0.02),
        status='completed'
    )

    print(f"  知识库: {prev} → {prev + new_count}")
    print(f"  下次进化: 7天后")


def main():
    print("🧬 因果增强工业智能体 — 持续进化守护进程")
    print(f"启动时间: {datetime.now():%Y-%m-%d %H:%M}")
    print("进化周期: 每周一次（可修改）")
    print("按 Ctrl+C 停止")
    print("=" * 50)

    db = EvolutionDB()

    # 启动时先跑一次
    print("\n[启动] 执行初始进化周期...")
    evolution_cycle(db)

    # 设定每周自动进化
    schedule.every(7).days.do(evolution_cycle, db=db)

    # 定时输出状态
    schedule.every(1).hour.do(
        lambda: print(f"[{datetime.now():%H:%M}] 运行中... "
                      f"知识: {db.get_status()['total_knowledge']}条, "
                      f"周期: {db.get_status()['total_cycles']}次")
    )

    # 主循环
    while True:
        schedule.run_pending()
        time.sleep(60)


if __name__ == "__main__":
    print("依赖安装: pip install schedule")
    main()
