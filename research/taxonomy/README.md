# taxonomy/ — LSRG-ByteTrack 产物库（模型、图表、数据表与报告）

**定位**：本目录是科研分析链的**统一产物库**。所有代码生成的特征张量、校准器模型、统计大表、论文图表与静态分析报告均收纳于此。

---

## 一、 产物分类与清册

### 1. 核心风险张量与模型校准器 (Binaries & Models)

| 文件 | 生成脚本 | 说明 |
|---|---|---|
| `risk_ecdf_calibrator.npz` (22MB) | `risk_features.py` | 基于 164.7 万正常帧负样本的 ECDF 经验分布校准器模型（Git 忽略） |
| `risk_features_negatives.npz` (15.7MB) | `risk_features.py` | 164.7 万负样本校准风险特征及序列元数据包（Git 忽略） |
| `risk_features_events.npy` (113KB) | `risk_features.py` | 4,713 条 ID Switch 事件的 3 维因果风险特征核心张量 $[N, 3]$（Git 忽略） |
| `risk_features_events.npz` (178KB) | `risk_features.py` | 包含风险矩阵、原始物理特征、标签与序列元数据的完整数据包（Git 忽略） |

### 2. 统计评测与跨数据集对比数据表 (Tables: CSV / JSON)

| 文件 | 生成脚本 | 说明 |
|---|---|---|
| `event_counts_by_sequence.csv` | `tools/build_event_counts.py` | 114 个序列的分类事件基准计数（SANITY 硬断言读取源） |
| `risk_features_events.csv` | `risk_features.py` | 逐事件人类可读 CSV（含 $f_{\text{weak}}, f_{\text{comp}}, f_{\text{swap}}$ 原始值与校准风险分） |
| `risk_aggregation_summary.csv` | `risk_aggregation.py` | 四大聚合模型在 Oracle 与 Test 口径下的 FPR@TPR 与 pAUC 评测汇总表 |
| `risk_aggregation_tpr_grid.csv` | `risk_aggregation.py` | TPR 60%~100%（5% 步长）四大模型的全量 FPR 详细对齐表 |
| `class_dataset_breakdown_master_table.csv` | `comprehensive_class_dataset_breakdown.py` | 跨数据集（MOT17/20/SportsMOT）与跨类别（$S_c, S_r, S_h$）综合汇总主表 |
| `class_dataset_intra_benchmark_table.csv` | `comprehensive_class_dataset_breakdown.py` | 数据集内各类别相对占比与风险表现基准表 |
| `class_dataset_percentiles_table.csv` | `comprehensive_class_dataset_breakdown.py` | 风险特征在正负样本中的关键分位数（P50/P90/P99）对比表 |
| `class_dataset_breakdown_full.json` | `comprehensive_class_dataset_breakdown.py` | 全量跨维度嵌套统计明细 JSON（1.7MB） |
| `gate_feasibility_events.csv` | `analysis.py` | 早期几何门控逐事件特征表 |
| `gate_feasibility_summary.csv` | `analysis.py` | 早期门控触发率与候选机制筛选长表 |
| `den_online_{ds}_full.csv` | `den_online_eval.py` | DEN 在线插桩 alert 模式逐数据集网格评估结果 |
| `den_online_{ds}_smoke.csv` | `den_online_eval.py` | DEN 在线插桩冒烟序列评估结果 |

### 3. 学术论文与诊断图表 (Figures: PNG)

| 文件 | 生成脚本 | 说明 |
|---|---|---|
| `fig_dataset_roc_2x2_grid.png` | `generate_paper_tables_and_figures.py` | 论文主图：三数据集独立及整体合并的 2×2 四宫格 ROC 曲线 |
| `fig_risk_score_tail_quantiles.png` | `generate_paper_tables_and_figures.py` | 尾部分位数风险分值断崖分布对比折线图 |
| `fig_roc_{mot17,mot20,sportsmot,overall}.png` | `generate_split_paper_figures.py` | 各数据集独立的高分辨率 ROC 曲线图 |
| `fig_geometry_iou_cost_matrix.png` | `generate_geometric_diagram.py` | 空间重叠与匈牙利分配代价矩阵几何示意图 |
| `fig_single_feature_discriminability.png` | `class_risk_breakdown.py` | 单一因果物理特征的判别力分布图 |
| `fig_sr_cross_dataset_escalation.png` | `class_risk_breakdown.py` | $S_r$ 活跃接管类别跨数据集雪崩激增趋势图 |
| `fig_sr_model_benchmark.png` | `class_risk_breakdown.py` | $S_r$ 针对四大聚合模型的基准性能对比图 |
| `class_risk_tpr_fpr_comparison.png` | `class_risk_breakdown.py` | 三大类别隔离评测 TPR-FPR 对比大图 |
| `risk_aggregation_roc.png` | `risk_aggregation.py` | 四大聚合模型 TPR 60%~100% 对应 FPR 对比曲线 |
| `gate_feasibility_roc.png` | `analysis.py` | 早期三候选机制 ROC 对比图 |

### 4. 静态分析报告 (Static Reports: Markdown)

| 文件 | 维护方式 | 说明 |
|---|---|---|
| `event_taxonomy_report.md` | 手写存档 | 早期事件分类、物理签名与可挽回性理论分析报告（方法学存档） |

---

## 二、 产物生成与覆写规则

- **可自动覆写产物**：运行 `../code/run_all_pipeline.py` 即可按需重新生成所有 CSV、PNG、JSON 与模型二进制。
- **只读保护产物**：`event_counts_by_sequence.csv` 为基准校验锚点，由 `tools/build_event_counts.py` 严格维护。
- **大文件追踪**：`.npz` 与 `.npy` 大型模型包已被 `.gitignore` 规则自动忽略，保障 Git 仓库轻量化。
