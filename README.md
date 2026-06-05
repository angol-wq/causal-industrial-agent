# 因果增强工业智能体

> **Causal Enhanced Industrial Agent** — 基于双通道因果图融合的工业异常根因分析与反事实推理系统

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](你的Streamlit链接)

## 🏆 创智青山AI智能体创新大赛 · 技术挑战赛道

**溯因智工** | 武汉科技大学 | 钢铁石化新材料方向

---

## 💡 核心创新

传统工业AI **只报警不解释** → 本方案实现 **感知→理解→决策** 完整因果闭环。

| 传统方案 | 本方案 |
|---------|--------|
| "Reactor_Temp 偏高" | "Reactor_Temp 偏高，**根因：CW_Valve卡滞**，因果链：Valve→Flow→Temp" |
| 无法建议处置 | "**恢复阀门开度至60%**，预期温度可从192降至165°C" |
| 8个报警同时响，不知哪个是根因 | 因果图自动回溯 + 评分排序 |

## 🔧 技术架构

```
传感器数据 ──→ PCMCI+因果发现 ──┐
                                  ├──→ 双通道融合因果图 ──→ 根因分析
工艺文档  ──→ LLM因果抽取 ────────┘         │
                                            ├──→ 反事实推理
                                            └──→ 处置建议
```

- **通道1 (知识驱动)**：LLM从操作手册抽取因果对 → 知识因果图
- **通道2 (数据驱动)**：PCMCI+从传感器时序数据发现因果结构 → 数据因果图
- **融合层**：冲突消解 + 置信度加权（原创算法）

## 🚀 快速开始

```bash
# 安装依赖
pip install -r requirements.txt

# 运行演示（三种方式）
python run_demo.py         # 合成数据演示
python run_tep_demo.py     # 真实TEP工业数据演示
streamlit run app.py       # Web可视化界面

# 或使用一键启动脚本（Windows）
run.bat
```

## 📊 实验验证

- **合成CSTR数据**：12变量反应器，4种故障模式，已知Ground Truth
- **真实TEP数据**：52变量，21种故障（MIT Braatz Group公开数据集）
  - IDV(6) A进料损失: 因果链准确检出 ✓
  - IDV(14) 阀门卡滞: 根因正确识别 ✓

## 👥 团队

| 角色 | 姓名 | 单位 | 专业 |
|------|------|------|------|
| 负责人 | 郑志浩 | 武汉科技大学 | 能源与动力工程 |

## 📄 许可

MIT License
