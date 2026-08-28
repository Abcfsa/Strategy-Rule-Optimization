# Strategy-Rule-Optimization (SRO)

A first attempt at a novel agent prompt self-evolve approach.

将自进化 prompt 优化中的知识分为三层存储与演化（技术路线报告中的 GEM 方案）：

- **策略层（Strategy）**：带条款化的全局策略，慢速进化，受双证据门控与条款级正则化约束（TextReg 机制）
- **规则层（Rule）**：条件→动作的任务特定规则，中速更新，对比归纳产生，可检索注入
- **支持集（Support set）**：验证过正确的经验轨迹片段，按查询检索少量注入

训练期（`train`）：GEPA 进化骨架 + 规则蒸馏 + 条款晋升门控。
测试期（`adapt`）：零适应迁移，或少量带反馈样本下的快速适应协议。

详细设计见外部技术路线报告：`../可迁移自进化_技术路线/技术路线报告_可迁移自进化Prompt优化.md`

## 目录结构

```
.
├── main.py                     # 唯一入口（train / adapt 两个子命令）
├── requirements.txt
├── configs/
│   └── default.yaml            # 默认超参（晋升阈值、λ、预算等）
├── src/
│   ├── memory/                 # 三层记忆：条款/规则/支持集的数据结构与存储
│   ├── evolve/                 # 训练期进化：反射、Pareto、早停、Judge、数据筛选
│   ├── distill/                # 规则蒸馏：错误分类、对比归纳、条款级 A/B 验证
│   ├── adapt/                  # 测试时快速适应：DG 式规则提取、flip ratio 监控
│   ├── assemble/               # prompt 组装：三层检索与预算控制
│   ├── llm/                    # LLM 客户端（timeout 可调）
│   ├── dataset/                # AIME / MATH 数据加载
│   ├── utils/                  # 通用工具
│   └── prompts/                # jinja2/md 提示模板
│       └── templates/
├── scripts/                    # 运行脚本（占位）
├── tests/                      # 单元测试（占位）
└── docs/                       # 设计文档索引
```

## 使用方式（规划）

```bash
# 训练期：进化 + 蒸馏，产出策略层/规则库/支持集
python main.py train --dataset aime --config configs/default.yaml

# 测试期：零适应迁移或快速适应
python main.py adapt --dataset aime25 --mode zero-shot      # 仅策略层
python main.py adapt --dataset aime25 --mode few-shot -n 10 # 快速适应协议
```

## 依赖

见 `requirements.txt`（openai SDK、datasets、numpy、python-dotenv、jinja2 等）。

## 状态

当前为框架骨架阶段：目录与模块职责已定义，算法实现待填充。
