# research — LSRG-ByteTrack 研究方向工作区

物理机理驱动的分类诊断：面向多目标跟踪 ID 失效的机理分析与最小定向纠错（LSRG-ByteTrack）。核心思路：ID 切换不是单一问题，而是三类互斥失效模式——冷启动接管 S_c / 活跃接管 S_r / 历史重新激活 S_h，各有不同物理签名；最终闭环为"分类诊断 → 轻量风险判断（S_r 风险门控）→ S_r 局部纠错 / 保持 ByteTrack"。

## 文档导航

| 文档 | 内容 |
|---|---|
| `docs/开发进度.md` | 进度台账：各分析步骤状态、产物位置、复现命令、下一步建议 |
| `docs/COE.md` | 跨对话经验：环境、数据口径、历史教训 —— **新会话先读这个** |
| `reports/day1_report.md` | 研究报告一：失效分类、设计收缩与几何门控基线 |
| `reports/day2_report.md` | 研究报告二：候选机制离线筛选（V8）与 DEN 在线插桩验证 |
| `reports/day3_report.md` | 研究报告三（独立工作区）：可观测性诊断实验与 Go/No-Go 裁决（情况 B） |
| `docs/研究方向.md` | 研究目标与技术路径（当前版本含 Go/No-Go 判据） |

## 目录布局

```
research/
├── README.md          本索引
├── docs/              进度台账、COE、研究方向
├── reports/           day1/day2/day3 研究报告 + lib/katex
├── code/              活脚本（analysis.py 离线分析 + den_online_eval.py 在线对齐）
├── data/              冻结输入：{MOT17,MOT20,SportsMOT}_events.csv + *_events_metrics.csv + *_events_summary.csv（分类标准见 data/README.md）
├── taxonomy/          产物（分类标准见 taxonomy/README.md）：
│                      event_counts_by_sequence.csv    序列级分类计数（冻结，SANITY 读取源）
│                      gate_feasibility_events.csv     逐事件门控特征（analysis.py 生成）
│                      gate_feasibility_summary.csv    门控触发率长表（analysis.py 生成）
│                      gate_feasibility_roc.png        三机制 ROC 图（analysis.py 生成）
│                      den_online_{ds}_full.csv        在线-离线对齐数据（den_online_eval.py 生成）
│                      event_taxonomy_report.md        静态报告：分类/签名/可挽回性
└── diag_exp/          Day 3 诊断实验独立工作区（自包含，只读 data/ 与 taxonomy/）：
                       run_diag.py                     四特征诊断脚本（复用 code/analysis.py 工具）
                       results/                        diag_features_events.csv / diag_roc_summary.csv /
                                                       diag_union_summary.csv / diag_roc.png
                       README.md                       工作区索引与复现
```

## 数据文件分类标准

**按数据流角色（输入 / 产物）分类，不按文件类型**：

| 目录 | 角色 | 内容 | 规则 |
|---|---|---|---|
| `data/` | 冻结输入 | 三数据集 V1 事件表（`*_events.csv` + `*_events_metrics.csv`） | 只读，脚本不写入；修改 = 改变全部分析基准 |
| `taxonomy/` | 产物 | 分析数据 CSV + 图片 + 静态报告 | 代码生成即写入；活产物可覆写，冻结产物与报告禁止覆写 |

产物侧再按**生成阶段**细分：① 冻结产物（事件表重生成时同步重建，现为 `analysis.py` SANITY 锚点：`event_counts_by_sequence.csv`）；② 离线分析产物（`analysis.py`：`gate_feasibility_*`）；③ 在线插桩产物（`den_online_eval.py`：`den_online_*_full.csv`）。CSV 与 PNG 同属产物时共存于 `taxonomy/`（同源同生命期，如 `gate_feasibility_{events,summary}.csv` 与 `gate_feasibility_roc.png` 一次运行同生）；输入侧无图片。

完整清单与判定规则见 `data/README.md` 与 `taxonomy/README.md`。

## 与主仓库的关系

- **所有代码与产物统一在本目录**：脚本在 `code/` 原位运行（`_repo_root()` 自动探测仓库根），产物直接写入 `taxonomy/`，无双位置
- 主仓库 `E:\科研\ByteTrack` 侧只作**只读数据源**：`datasets/` GT（已清洗）、`YOLOX_outputs/*_v001_full/track_results/`
- `tools/` 保留跟踪运行入口 `track_v001.py`（在线插桩阶段使用）与数据/事件工具：`tools/build_ids_events.py`（事件表重生成）、`tools/build_event_counts.py`（分类计数锚点）、`tools/clean_mot_gt.py`（GT 清洗）、`tools/plot_results.py`（评估图）；V1/V2 事件分析脚本与报告生成脚本已删除（2026-08-08 / 2026-08-16）
- 注意：主仓库现为 Git 仓库，`baseline-v001` 为重构前基线；文件操作仍应谨慎
- **报告与代码解耦**：`taxonomy/event_taxonomy_report.md` 与 `reports/day{1,2,3}_report.md` 是手写静态文档；day1/2/3 数字已于 2026-08-16 按当前事件池更新，`event_taxonomy_report.md` 为旧池方法学存档；改报告直接编辑 md，不需要重跑代码

## 当前状态速览（2026-08-16：GT 清洗后事件全量重生成，数字为当前事件池）

> 2026-08-15 清洗 MOT17/MOT20 GT（conf==1 保留，删 ignore 区域行），2026-08-16 用
> `tools/build_ids_events.py` 以当前 GT + 冻结 track_results 重跑 motmetrics 时序
> 匹配，全量重建事件集与全部下游产物（`data/`、`taxonomy/`、`diag_exp/results/`、
> `den_online_*_full.csv`）。**所有结论方向不变**（数字微动，SportsMOT 因 GT 未变
> 完全复现旧值）；结论报告 `reports/day{1,2,3}` 数字已按当前事件池更新，
> `taxonomy/event_taxonomy_report.md` 为旧事件池方法学存档。

- ✅ 事件分类：**4,713 条 IDS switch**（MOT17 546 / MOT20 1,600 / SportsMOT 2,567）→ S_c 1,828（38.8%）/ S_r 1,899（40.3%）/ S_h 986（20.9%）；reuse 2,320 条；双视角不嵌套，旧 β 公式被否定
- ✅ 物理签名：S_r 强局域（gap=1、IoU_last 中位 0.854）vs S_h 长尾漂移（gap 中位 6）显著分离
- ✅ 冷启动可挽回上限：Recoverability_S_c(L) = 43.1% → 44.4% 饱和
- ✅ 几何拥挤度回归：**不支持**（速度是主导协变量，全局 κ 无解释力）
- ✅ 预警可分离性：**部分支持**（H≥1 PR-AUC 增益 +0.04 ~ +0.10）
- ✅ 触发率代理："90% bypass" 被否定（top2≥0.2 时 76.2% 帧含歧义）
- ✅ 几何门控基线：合并 S_r 覆盖 3.90% → 24.75%（ε=0.05→0.25），C 组误触发 0.07% → 0.75%；margin 是覆盖率瓶颈
- ✅ 三候选机制验证：**KMC 运动一致性 NOT SUPPORTED / KF 自适应裕度机制级 NOT SUPPORTED / DEN 局域密度 SUPPORTED（SportsMOT +6.2pp，TPR 27.3%@FPR≤1%）**——DEN 是唯一值得在线插桩的机制
- ✅ **DEN 在线插桩（检测模块，alert 模式）**：管线成立、零行为变化、N 近似无偏、快照一致率 100%；在线 FPR 预算内点存在（ε0=0.15, γ=1.25 → 0.75-0.81%）；S_r box 级命中数据集依赖（MOT20 29% / MOT17 10%，离线 24%/18%）；修复模块（reject）留待下一轮
- ✅ **可观测性诊断（Go/No-Go）→ 情况 B（No-Go，终止 Gate 路线）**：四特征并集 TPR@FPR≤1% = MOT17 20.7% / MOT20 37.6% / SportsMOT 36.2%（合并 u1234 33.5%；u2 margin∪swap 合并 36.2%）；FPR≤2% 时合并 u1234 40.6% 仍距 50–60% 继续线 ~10pp；**Swap Instability 为新发现的最强信号**（合并 S_r 31.7% @1%，AUC 0.844）；66.4% 事件对全部廉价信号不可见 → 转向关联函数重构（OC-SORT 运动补偿 / C-BIoU 搜索空间自适应）
- 细节见 `taxonomy/event_taxonomy_report.md`（事件侧，旧池存档）、`reports/day1_report.md`（设计收敛与门控基线）、`reports/day2_report.md`（三机制筛选与在线插桩）、`reports/day3_report.md`（可观测性诊断与路线裁决）
