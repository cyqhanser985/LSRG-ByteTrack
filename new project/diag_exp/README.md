# diag_exp — 小型可观测性诊断实验（Day 3，Go/No-Go）

依据《研究方向.md》执行的"小型可观测性诊断实验"独立工作区：在冻结 switch 事件池（S_c 1,841 / S_r 1,921 / S_h 994，共 4,756）上验证四个廉价正交门控信号（Margin / Motion Surprise / Occlusion / Swap Instability）的联合识别率（分类别 + 全部），裁决 Gate 路线去留。

**判定：情况 B（No-Go）** — 四信号并集 $TPR@FPR{\le}1\%$：S_r 24.1% / 37.0% / 36.2%（MOT17/MOT20/SportsMOT，合并 33.6%）；**全部 switch 事件合并仅 22.6%**；全部 < 40% 终止线（S_c 9.5% / S_h 25.7% / S_r 33.6%）。终止 Gate-centric DEN 路线，转向关联函数重构（OC-SORT 运动补偿 / C-BIoU 搜索空间自适应）。汇报用识别率表见 `recognition_summary.md`。

## 目录

| 文件 | 内容 |
|---|---|
| `recognition_summary.md` | **全类别识别率汇报表**（S_c/S_r/S_h/全部 switch，导师汇报用） |
| `day3_report.md` | 最终报告：协议、四特征诊断、召回率表、并集分解、Go/No-Go 判定 |
| `run_diag.py` | 实验脚本（复用 `code/analysis.py` 工具，SANITY 硬断言 vs 冻结 CSV） |
| `results/` | 产物：逐事件特征（4,756 行）/ ROC 汇总 / 并集汇总 / ROC 图 |

## 复现

```bash
PY=E:\anaconda\envs\bytetrack\python.exe
cd "new project/diag_exp"
$PY run_diag.py --datasets mot17 --no-figure   # 冒烟（~4s）
$PY run_diag.py                                 # 全量三数据集（~70s）
```

数据源只读：`../data/*_events_metrics.csv`、`YOLOX_outputs/*_v001_full/track_results/`、`../taxonomy/gate_feasibility_events.csv`（SANITY 锚点）。本目录不写入 `data/` 与 `taxonomy/`。

## 关键数字（合并口径，$TPR@FPR{\le}1\%$）

**S_r（活跃接管，n=1,921）**：Margin 24.5% | Motion 14.7% | Occlusion 6.1% | **Swap ΔC 31.7%** | **Union 33.6%**（@2% 40.6%）

**全部 switch 事件（n=4,756）**：Union(1∪2∪3∪4) = **22.6%**（@2% 26.8%）；分类别并集：S_c 9.5% / S_r 33.6% / S_h 25.7%
