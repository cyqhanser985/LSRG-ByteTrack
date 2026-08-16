# taxonomy/ — 产物（分析输出 + 静态报告）

**定位**：本目录是分析链的**输出侧**（"taxonomy" = 失效分类学）。所有代码生成的数据文件、图片与手写静态报告统一存放于此；脚本（`code/analysis.py` → OUT、`code/den_online_eval.py` → `../taxonomy`）只写入本目录。输入侧规则见 `data/README.md`。

## 分类标准（产物分级）

| 级 | 子类 | 判定规则 | 文件 |
|---|---|---|---|
| ① | **冻结产物** | 事件表重生成时同步重建，仍被活脚本读取（SANITY 锚点） | `event_counts_by_sequence.csv` |
| ② | **离线分析产物** | `analysis.py` 每次运行重新生成（活产物，可覆写） | `gate_feasibility_events.csv`、`gate_feasibility_summary.csv`、`gate_feasibility_roc.png` |
| ③ | **在线插桩产物** | `den_online_eval.py` 生成（活产物，可覆写） | `den_online_{mot17,mot20,sportsmot}_full.csv` |
| ④ | **静态报告** | 手写维护，与代码解耦；改报告直接编辑 md | `event_taxonomy_report.md`（事件侧）；`research/reports/day1_report.md`（设计收敛与门控基线）、`research/reports/day2_report.md`（三机制筛选与在线插桩）、`research/reports/day3_report.md`（可观测性诊断与路线裁决） |

> 2026-08-16：`event_classification.csv` 与 `warning_features.csv`（旧 V1/V5 冻结
> 产物，生成脚本已删、无代码读取、基于旧事件池）已删除；分类逻辑唯一来源为
> `analysis.py` 的 `classify()`，计数锚点唯一来源为 `event_counts_by_sequence.csv`
> （由 `tools/build_event_counts.py` 从事件表重建）。

## 文件清单（6 个数据文件 + 1 篇报告，2026-08-16 整理后）

| 文件 | 生成源 | 级 | 内容 |
|---|---|---|---|
| `event_counts_by_sequence.csv` | `tools/build_event_counts.py`（冻结） | ① | 114 序列级分类计数（S_c/S_r/S_h × switch/reuse）；`analysis.py` SANITY 读取源 |
| `gate_feasibility_events.csv` | `analysis.py` | ② | 4,713 行逐事件：class + top1/top2/margin + cos_theta/r_v/v_obs_norm + sigma_norm + n_neighbor + geom_020 |
| `gate_feasibility_summary.csv` | `analysis.py` | ② | 长表：geom 基线 A/B/C × ε + kmc/kf/den 最优触发点 |
| `gate_feasibility_roc.png` | `analysis.py` | ② | 三机制 ROC 图（全目录唯一图片） |
| `risk_features_events.npy` | `risk_features.py` | ② | 4,713×3 核心风险张量（r_weak/r_comp/r_swap，严格[0,1]，零NaN） |
| `risk_features_events.npz` | `risk_features.py` | ② | 包含风险矩阵、原始矩阵、类别标签与序列/帧/ID元数据的压缩包 |
| `risk_features_events.csv` | `risk_features.py` | ② | 逐事件人类可读 CSV（含原始特征与校准风险分值） |
| `risk_features_negatives.npz` | `test_neg_cache.py` / `risk_features.py` | ② | 164.7万负样本校准风险特征及序列元数据缓存包 |
| `risk_ecdf_calibrator.npz` | `risk_features.py` | ② | 负样本（164.7万框）ECDF经验分布校准器模型与分位数表 |
| `risk_aggregation_summary.csv` | `risk_aggregation.py` | ② | 四大聚合模型在 Oracle 与 Test 口径下的 FPR@TPR 与 pAUC 评测大表 |
| `risk_aggregation_tpr_grid.csv` | `risk_aggregation.py` | ② | TPR 60%..100% 以 5% 为跨度的 FPR 详细对齐大表 |
| `risk_aggregation_roc.png` | `risk_aggregation.py` | ② | 四大聚合模型 TPR 60%..100% ROC 曲线与 5% 步长折线图 |
| `den_online_{mot17,mot20,sportsmot}_full.csv` | `den_online_eval.py` | ③ | 在线-离线对齐：ε0×γ 网格 FPR/TPR + 锚点指标 |
| `event_taxonomy_report.md` | 手写 | 报告 | 事件分类/物理签名/可挽回性（数字基于 2026-08-16 前旧事件池，方法学存档） |

## 使用规则

- ②③ 类为活产物：重跑对应脚本即覆写；① 类与报告禁止被脚本覆写。
- 删除任何文件前先读内容（Git 已启用，`baseline-v001` 为重构前基线；仍应谨慎操作）。

## 旧版命名映射

- `event_counts_by_sequence.csv` 即早期版本中的 `taxonomy_by_sequence.csv`（序列级分类计数）。
- 早期 `taxonomy_partition.csv` / `event_classification.csv`（逐事件分类）与 `warning_features.csv`（V5 预警特征集）已于 2026-08-16 删除。
- 旧版命名已不再使用；后续代码与文档统一引用当前文件名。
