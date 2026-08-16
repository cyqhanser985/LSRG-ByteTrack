# diag_exp — 小型可观测性诊断实验（Day 3，Go/No-Go）

依据《../docs/研究方向.md》执行的"小型可观测性诊断实验"独立工作区：在冻结 switch 事件池（2026-08-16 重生成后 S_c 1,828 / S_r 1,899 / S_h 986，共 4,713）上验证四个廉价正交门控信号（Margin / Motion Surprise / Occlusion / Swap Instability）的联合识别率（分类别 + 全部），裁决 Gate 路线去留。

**判定：情况 B（No-Go）** — 四信号并集 $TPR@FPR{\le}1\%$（u1234 合并）：S_r 20.7% / 37.6% / 36.2%（MOT17/MOT20/SportsMOT，合并 33.5%）；u2（margin∪swap，成分级预算）合并 36.2%；全部 < 40% 终止线。终止 Gate-centric DEN 路线，转向关联函数重构（OC-SORT 运动补偿 / C-BIoU 搜索空间自适应）。判定与旧池（2026-08-16 前）一致，数字微动（旧：u1234 合并 33.0%，u2 合并 35.7%）。

## 目录

| 文件 | 内容 |
|---|---|
| `../reports/day3_report.md` | 最终报告：协议、四特征诊断、召回率表、并集分解、Go/No-Go 判定（数字已按 2026-08-16 事件池更新） |
| `run_diag.py` | 实验脚本（复用 `code/analysis.py` 工具，SANITY 硬断言 vs 冻结 CSV） |
| `results/` | 产物：逐事件特征（4,713 行）/ ROC 汇总 / 并集汇总 / ROC 图（2026-08-16 重生成） |

## 复现

```bash
PY=E:/anaconda/envs/bytetrack/python.exe   # 或 conda 环境解释器
cd "research/diag_exp"
$PY run_diag.py --datasets mot17 --no-figure   # 冒烟（~4s）
$PY run_diag.py                                 # 全量三数据集（需先重跑 analysis.py 生成 SANITY 锚点）
```

数据源只读：`../data/*_events_metrics.csv`、`YOLOX_outputs/*_v001_full/track_results/`、`../taxonomy/gate_feasibility_events.csv`（SANITY 锚点）。本目录不写入 `data/` 与 `taxonomy/`。

## 关键数字（合并口径，$TPR@FPR{\le}1\%$，2026-08-16 重生成）

**S_r（活跃接管，n=1,899）**：Margin 24.5% | Motion 14.7% | Occlusion 6.1% | **Swap ΔC 31.7%** | **Union u2 36.2%**（u1234 33.5%；@2% u1234 40.6%）

**全部 switch 事件（n=4,713）**：Union(1∪2∪3∪4) = **22.3%**（@2% 26.4%）；分类别并集（u1234a）：S_c 11.9% / S_r 30.6% / S_h 31.1%
