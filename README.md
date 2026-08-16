# ByteTrack 多目标跟踪项目

本仓库包含两个明确分区：

1. **官方 ByteTrack 基线复现区（只读 / 归档）**：基于 [ByteTrack](https://github.com/ifzhang/ByteTrack)（ECCV 2022，MIT License）的复现与评估链路，包含 `yolox/` 上游框架、`exps/` 实验配置、`tools/` 运行工具、`datasets/`（Git 忽略）、`pretrained/`（Git 忽略）与 `YOLOX_outputs/`（Git 忽略）。
2. **LSRG 活跃研究区（`research/`）**：ID 失效机理分类、DEN 门控验证、可观测性诊断与后续 OC-SORT / C-BIoU 关联函数重构工作。**当前研究进展以 [`research/README.md`](research/README.md) 为唯一入口。**

> `docs/` 下的旧版开发文档已归档，仅作历史参考；新研究文档统一维护在 `research/docs/` 与 `research/reports/`。

## 目录结构

```text
ByteTrack/
├── .gitignore                           # 排除 datasets/videos/大型输出
├── requirements.txt                     # 补齐 scipy, matplotlib 等依赖
├── setup.py                             # 配置 exclude 规则，防止污染命名空间
├── README.md                            # 统一入口门户（复现基线 + research 索引）
├── yolox/                               # 上游框架
├── exps/                                # 实验配置
├── tools/                               # 通用运行/转换/评估工具（修复路径依赖与入口保护）
├── research/                            # 活跃研究主区
│   ├── README.md                        # 研究区总索引
│   ├── docs/                            # COE.md / 开发进度.md / 研究方向.md
│   ├── reports/                         # day1_report / day2_report / day3_report
│   ├── code/                            # analysis.py / den_online_eval.py / 备份 diff
│   ├── data/                            # 唯一权威冻结输入 + SHA256SUMS.txt
│   ├── taxonomy/                        # 规范化产物 + 映射说明
│   └── diag_exp/                        # 独立诊断实验
├── docs/                                # 旧版开发文档（标注已归档）
├── reports/                             # 自包含交付报告与打包产物
├── scripts/                             # 打包与校验辅助脚本（消除绝对盘符）
├── datasets/                            # (Git 忽略) 数据集
├── pretrained/                          # (Git 忽略) 权重文件
└── YOLOX_outputs/                       # (Git 忽略) 运行输出
```

## 官方 ByteTrack 基线复现区（只读 / 归档）

本分区保留三数据集（MOT17 / MOT20 / SportsMOT）的 V001 全量评估链路与结果。

```bash
# 1. 环境（Python 3.8 + PyTorch + CUDA）
conda activate bytetrack
python setup.py develop

# 2. 全量评估（MOT17 示例，不指定 --sequence 即跑全部序列）
python tools/track_v001.py \
    -f exps/example/mot/yolox_x_mot17_v001.py \
    -c pretrained/bytetrack_x_mot17.pth.tar \
    -expn mot17_v001_full -b 1 -d 1 --fp16 --fuse
```

历史 ID 事件挖掘/报告生成脚本已下线；事件表唯一权威 = `research/data/`（由
`tools/build_ids_events.py` 从当前 GT + track_results 重生成，命令见
`research/data/README.md`）。

## LSRG 活跃研究区

- 研究总索引：[`research/README.md`](research/README.md)
- 跨会话经验与坑位：[`research/docs/COE.md`](research/docs/COE.md)
- 研究进度台账：[`research/docs/开发进度.md`](research/docs/开发进度.md)
- 冻结数据源：[`research/data/README.md`](research/data/README.md)（含 `SHA256SUMS.txt`）

## 文档索引（历史归档）

| 文档 | 说明 |
|------|------|
| `docs/开发进度.md` | 步骤 1-8 开发全记录（已归档） |
| `docs/experience.md` | 经验与踩坑沉淀（已归档） |
| `docs/USAGE_GUIDE.md` | 使用指南（已归档） |
| `research/reports/day{1,2,3}_report.md` | 研究结论报告（方法学存档，数字基于 2026-08-16 重生成前的旧事件池） |

## 数据集说明

| 数据集 | 内容 | 状态 |
|--------|------|------|
| MOT17 | 行人跟踪，21 序列 | 全量评估完成；GT 已清洗（仅 conf==1 行人） |
| MOT20 | 拥挤场景行人，4 序列 | 全量评估完成；GT 已清洗（仅 conf==1 行人） |
| SportsMOT | 体育运动，240 序列（90 有 GT） | 全量评估完成；GT 原始即干净 |

> 本项目在官方代码基础上未改动 `yolox/` 框架核心算法；评估链路适配（v001 配置、
> `tools/track_v001.py`、`mot_evaluator.py` 视频切换逻辑）详见 `docs/开发进度.md`。
