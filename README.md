# ByteTrack 多目标跟踪分析项目

基于 [ByteTrack](https://github.com/ifzhang/ByteTrack)（ECCV 2022，MIT License）的多目标跟踪复现与分析项目。
在 MOT17 / MOT20 / SportsMOT 三个数据集上完成全量评估，并对 **ID 切换（switch）/ ID 复用（reuse）**
事件做了深度分析与可视化；同时内置 BFT（鸟群跟踪）私有数据集，供后续研究使用。

## 目录结构

```
ByteTrack/
├── yolox/                  # YOLOX 框架 + BYTETracker（官方代码）
├── tools/                  # 全部运行脚本（评估 / 事件分析 / 可视化 / 数据集工具）
├── exps/example/mot/       # 实验配置：yolox_x_mix_det 基类 + 3 个数据集 v001 配置
├── datasets/               # MOT17 / MOT20 / SportsMOT / BFT（私有）/ DanceTrack
├── pretrained/             # 官方预训练权重（3 × ~800MB）
├── YOLOX_outputs/          # 全量评估结果与分析产物（track_results、事件表、指标、报告）
├── docs/                   # 项目文档：开发进度 / 经验沉淀 / 复现报告 / 使用指南 / 图表
├── reports/                # 自包含 HTML 分析报告（离线可打开）
├── scripts/                # 报告打包与校验脚本
├── videos/                 # 示例视频（BFT / DanceTrack 样本）
├── requirements.txt        # 依赖清单
├── setup.cfg / setup.py    # 包配置（python setup.py develop 安装 yolox）
└── LICENSE                 # MIT License（官方）
```

## 核心成果

| 项目 | 结果 |
|------|------|
| 三数据集全量评估 | MOT17 **MOTA 49.9%** / MOT20 **78.8%** / SportsMOT **97.1%**（与官方同口径量级一致） |
| 事件全量挖掘 | **7,156 条** switch/reuse 事件，逐序列与 motmetrics `num_switches` 100% 一致 |
| 可视化产物 | 115 个跟踪视频、白布可视化、6 张切换轨迹图、12 个典型例子（372 帧标注图） |
| 分析报告 | 自包含 HTML 报告，`reports/ByteTrack_ID分析报告/index.html` 双击即看 |

> 注：`YOLOX_outputs/` 下的完整跟踪视频与标注图原图已按需精简（可再生），
> 所有指标产物（track_results、事件表 CSV、指标分布、轨迹图、报告）完整保留。

## 快速开始

```bash
# 1. 环境（Python 3.8 + PyTorch + CUDA）
conda activate bytetrack
python setup.py develop

# 2. 全量评估（MOT17 示例，不指定 --sequence 即跑全部序列）
python tools/track_v001.py \
    -f exps/example/mot/yolox_x_mot17_v001.py \
    -c pretrained/bytetrack_x_mot17.pth.tar \
    -expn mot17_v001_full -b 1 -d 1 --fp16 --fuse

# 3. ID 切换/复用事件挖掘
python tools/find_id_events.py -expn mot17_v001_full -ds MOT17

# 4. 跟踪可视化（视频）
python tools/draw_tracks_video.py -expn mot17_v001_full -ds MOT17
```

## 文档索引

| 文档 | 说明 |
|------|------|
| `docs/开发进度.md` | 步骤 1-8 开发全记录（环境 / 数据 / 评估 / 可视化 / 指标） |
| `docs/reproduction_results.md` | 三数据集复现结果报告（逐序列指标） |
| `docs/experience.md` | 经验与踩坑沉淀（8 个专题） |
| `docs/USAGE_GUIDE.md` | 使用指南（数据集、命令、参数调优、算法解析） |
| `reports/ByteTrack_ID分析报告/` | 自包含 HTML 分析报告（6 页，离线可用） |
| `YOLOX_outputs/analysis/*.md` | 各阶段原始报告（事件 / 白布 / 指标） |

## 数据集说明

| 数据集 | 内容 | 状态 |
|--------|------|------|
| MOT17 | 行人跟踪，21 序列 | 全量评估完成，`annotations/eval.json` 就绪 |
| MOT20 | 拥挤场景行人，4 序列 | 全量评估完成，`annotations/eval.json` 就绪 |
| SportsMOT | 体育运动，240 序列（90 有 GT） | 全量评估完成，`annotations/eval.json` 就绪 |
| BFT | 鸟群跟踪（私有），106 序列 | COCO 标注（train/val/test_v1.5.json），可直接使用 |
| DanceTrack | 下载中的基准数据 | `test2.zip` 待处理 |

> 本项目在官方代码基础上未改动 `yolox/` 框架；评估链路适配（v001 配置、
> `tools/track_v001.py`、`mot_evaluator.py` 视频切换逻辑）详见 `docs/开发进度.md` 步骤 4。
