# LSRG-ByteTrack: 面向多目标跟踪 ID 失效机理与因果物理风险建模研究

本项目基于工业级多目标跟踪框架 [ByteTrack](https://github.com/ifzhang/ByteTrack)（ECCV 2022，MIT License），系统性开展针对 **身份跳变（Identity Switch, IDS）** 失效模式的机理分类、DEN 在线门控探索，以及最新的**严格在线因果物理风险建模、经验分布（ECDF）校准与全局误报雪崩（FPR Avalanche）归因**研究。

---

## 🌟 最新研究成果速览 (Phase 5)

| 交付成果 | 入口位置 | 说明 |
|---|---|---|
| 📖 **万字深度复盘长文** | [`research/专家博客.md`](research/专家博客.md) | 从科研原点复盘：三维因果风险建模、ECDF分布校准与误报雪崩归因 |
| 📊 **学术汇报完整 PPTX** | [`research/reports/LSRG_ByteTrack_组会汇报_v8.pptx`](research/reports/LSRG_ByteTrack_组会汇报_v8.pptx) | 7 页宽屏学术报告（含统一尺度大表、四大模型盲测对比与全屏 ROC） |
| ⚡ **一键科研流水线总控** | [`research/code/run_all_pipeline.py`](research/code/run_all_pipeline.py) | 一键端到端复现特征提取、模型评测、归因分析与图表生成 |
| 📑 **核心研究研报系列** | [`research/reports/`](research/reports/) | 包含 Day 1~3 研报、类别隔离评测报告、离线 HTML 交互研报 |

---

## 📁 目录架构全景

```text
ByteTrack/
├── research/                               # 【核心】LSRG 活跃科研主工作区
│   ├── README.md                           # 科研区完整导航与全景索引
│   ├── 专家博客.md                         # 深度长文：因果物理风险与雪崩机理
│   ├── docs/                               # 活跃台账 (开发进度.md / COE.md / 研究方向.md)
│   ├── reports/                            # 交付研报 (Day1-3研报 / 组会PPTX / HTML报告)
│   ├── code/                               # 科研代码库 (README.md / run_all_pipeline.py)
│   │   ├── risk_features.py                # 因果风险特征抽取与 164.7万负样本 ECDF 校准
│   │   ├── risk_aggregation.py             # 四大聚合模型 5-Fold 盲测与 ROC 评测
│   │   ├── class_risk_breakdown.py         # S_c / S_r / S_h 类别隔离评测与雪崩归因
│   │   ├── comprehensive_class_dataset_breakdown.py # 跨数据集/类别交叉大表生成
│   │   └── generate_*.py                   # 论文级 2x2 ROC / 代价矩阵 / 组会 PPT 生成
│   ├── data/                               # 冻结基准输入 (*_events.csv, *_events_metrics.csv)
│   ├── taxonomy/                           # 统一产物库 (模型二进制/数据表/ROC图表)
│   └── diag_exp/                           # [已归档] Day 3 四特征可观测性诊断实验
│
├── archive/                                # 【归档】历史阶段资产与离线展示包
│   └── phase1_id_switch_report/            # 阶段 1 挖掘报告与 436MB 视频/可视化资产 (index.html)
│
├── yolox/                                  # 上游跟踪器算法与模型库 (包含 DEN 门控模块)
├── exps/                                   # 实验配置文件 (MOT17, MOT20, SportsMOT)
├── tools/                                  # 数据清洗与基准运行工具 (clean_mot_gt.py, track_v001.py 等)
├── docs/                                   # 顶层说明与阶段 1 开发历史 (阶段1开发历史_归档.md)
├── datasets/                               # (Git 忽略) MOT17 / MOT20 / SportsMOT 数据集
├── pretrained/                             # (Git 忽略) YOLOX-X 预训练模型权重
├── YOLOX_outputs/                          # (Git 忽略) 规范化全量运行输出 (*_v001_full, *_den_alert_full)
└── .gitignore                              # 规范化忽略规则 (排除 datasets/npz/大型输出)
```

---

## 🚀 快速开始与复现指南

### 1. 环境准备
```bash
# 激活 Conda 环境
conda activate bytetrack

# 安装开发依赖
python setup.py develop
```

### 2. 运行官方 ByteTrack 基线全量跟踪
```bash
# MOT17 全量 21 序列评估示例
python tools/track_v001.py \
    -f exps/example/mot/yolox_x_mot17_v001.py \
    -c pretrained/bytetrack_x_mot17.pth.tar \
    -expn mot17_v001_full -b 1 -d 1 --fp16 --fuse
```

### 3. 一键执行 LSRG 科研分析流水线
```bash
cd research/code

# 一键端到端运行 (特征提取 -> ECDF校准 -> 聚合模型评测 -> 类别归因 -> 汇报生成)
python run_all_pipeline.py --all

# 或分阶段运行
python run_all_pipeline.py --stage features       # 提取因果风险特征与 ECDF 校准器
python run_all_pipeline.py --stage eval           # 四大聚合模型 5-Fold 评测
python run_all_pipeline.py --stage presentation   # 重新编译生成组会 PPTX
```

---

## 📊 数据集与基准说明

| 数据集 | 序列规模 | 标注状态 | 基准 MOTA / IDF1 / IDS |
|---|---|---|---|
| **MOT17** | 21 序列 (train) | GT 已清洗（仅保留 `conf==1` 行人） | 77.8 / 74.9 / 546 |
| **MOT20** | 4 序列 (train) | GT 已清洗（仅保留 `conf==1` 行人） | 77.8 / 74.6 / 1,600 |
| **SportsMOT** | 90 序列 (含GT子集) | 原始 GT 格式规范 | 97.1 / 74.8 / 2,567 |

---

## 🧭 文档导航

- **科研总入口**：[`research/README.md`](research/README.md)
- **踩坑与协作手册**：[`research/docs/COE.md`](research/docs/COE.md)
- **最新进度台账**：[`research/docs/开发进度.md`](research/docs/开发进度.md)
- **历史阶段 1 报告**：`archive/phase1_id_switch_report/index.html`
