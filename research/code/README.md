# research/code/ — LSRG-ByteTrack 科研代码库与执行指南

本目录包含 LSRG-ByteTrack 项目的核心科研脚本。所有脚本均设计为**原位可执行**（通过 `_repo_root()` 自动探测项目根目录），输入与输出遵循统一的数据流规范。

---

## 一、 脚本模块化全景分类

```
research/code/
├── [模块 1: 数据与因果特征管道]
│   ├── risk_features.py                  # 因果物理风险特征抽取与 164.7万负样本 ECDF 校准
│   ├── analysis.py                       # 基础工具函数库、几何门控基线与三机制验证
│   └── den_online_eval.py                # DEN 在线插桩 alert 日志与离线对齐评测
│
├── [模块 2: 聚合模型与雪崩归因]
│   ├── risk_aggregation.py               # 四大聚合模型(Linear, Max, MLP, ECDF) 5-Fold 评测
│   ├── class_risk_breakdown.py           # S_c / S_r / S_h 类别隔离评测与高召回 FPR 雪崩归因
│   └── comprehensive_class_dataset_breakdown.py # 全量跨数据集-跨类别交叉统计
│
├── [模块 3: 成果、图表与材料生成]
│   ├── generate_lab_presentation.py      # 一键生成 7 页学术汇报 PPTX
│   ├── generate_paper_tables_and_figures.py # 生成论文主图表 (ROC, 分位数折线)
│   ├── generate_split_paper_figures.py   # 生成单张/拆分版论文图
│   └── generate_geometric_diagram.py     # 生成 IoU 几何测度与代价矩阵示意图
│
├── [模块 4: 新增实验（混淆矩阵 / 双特征消融 / ID 自愈恢复）]
│   ├── eval_confusion_matrix.py          # 任务 1: 计算并生成各数据集混淆矩阵
│   ├── risk_aggregation_two_features.py  # 任务 2: 双特征消融实验训练与评测
│   ├── id_recovery_dino.py               # 任务 3: DINO 视觉相似度匹配与恢复评估
│   └── id_recovery_vlm.py                # 任务 3: VLM 选择题兜底恢复评估
│
└── run_all_pipeline.py                   # [流水线总控] 一键端到端运行与复现入口
```

---

## 二、 脚本职责与数据流向

### 1. 数据与因果特征管道

| 脚本 | 核心功能 | 输入源 | 核心产物（写入 `../taxonomy/`） |
|---|---|---|---|
| [`risk_features.py`](risk_features.py) | 构建三维严格在线因果特征 $\mathbf{f}=(f_{\text{weak}}, f_{\text{comp}}, f_{\text{swap}})$，利用 164.7 万正常帧检测负样本进行 ECDF 分布校准，输出严格 $[0, 1]$ 且零 NaN 的风险张量 | `../data/*_events_metrics.csv`<br>`YOLOX_outputs/*_v001_full/` | `risk_features_events.{npy,npz,csv}`<br>`risk_ecdf_calibrator.npz`<br>`risk_features_negatives.npz` |
| [`analysis.py`](analysis.py) | 逐序列主循环，几何门控基线（A/B/C 组）与三候选机制（KMC/KF/DEN）离线筛选 | `../data/*_events_metrics.csv`<br>`YOLOX_outputs/*_v001_full/` | `gate_feasibility_events.csv`<br>`gate_feasibility_summary.csv`<br>`gate_feasibility_roc.png` |
| [`den_online_eval.py`](den_online_eval.py) | DEN 在线插桩 alert 运行日志评估 | `YOLOX_outputs/*_den_alert_full/` | `den_online_{ds}_full.csv` |

### 2. 模型评测与机理归因

| 脚本 | 核心功能 | 输入源 | 核心产物 |
|---|---|---|---|
| [`risk_aggregation.py`](risk_aggregation.py) | 5-Fold 跨序列盲测评估 Linear, Max, MLP, ECDF 四大聚合模型在 TPR 60%~100% 下的 FPR 与 pAUC | `risk_features_events.npz`<br>`risk_features_negatives.npz` | `risk_aggregation_summary.csv`<br>`risk_aggregation_tpr_grid.csv`<br>`risk_aggregation_roc.png` |
| [`class_risk_breakdown.py`](class_risk_breakdown.py) | 分离 $S_c$（冷启动）、$S_r$（活跃接管）、$S_h$（历史激活），揭示各类别在不同召回率下的 FPR 贡献 | 同上 | `class_risk_tpr_fpr_comparison.png`<br>`../reports/class_risk_breakdown_report.md` |
| [`comprehensive_class_dataset_breakdown.py`](comprehensive_class_dataset_breakdown.py) | 全量跨数据集（MOT17, MOT20, SportsMOT）与跨类别多维交叉对比大表 | 同上 | `class_dataset_breakdown_master_table.csv`<br>`class_dataset_breakdown_full.json` 等 |

### 3. 成果、图表与材料生成

| 脚本 | 核心功能 | 核心产物 |
|---|---|---|
| [`generate_lab_presentation.py`](generate_lab_presentation.py) | 自动排版生成包含深邃学术封面、IoU代价示意、统一尺度大表、模型对比与全屏ROC大图的完整PPTX | `../reports/LSRG_ByteTrack_组会汇报_v8.pptx` |
| [`generate_paper_tables_and_figures.py`](generate_paper_tables_and_figures.py) | 生成论文发表级 2×2 四宫格 ROC 图与分位数分析图 | `fig_dataset_roc_2x2_grid.png`<br>`fig_risk_score_tail_quantiles.png` |
| [`generate_split_paper_figures.py`](generate_split_paper_figures.py) | 生成各单数据集独立 ROC 图表（MOT17, MOT20, SportsMOT, Overall） | `fig_roc_{ds}.png` |
| [`generate_geometric_diagram.py`](generate_geometric_diagram.py) | 绘制几何 IoU 代价矩阵原理图 | `fig_geometry_iou_cost_matrix.png` |

### 4. 新增实验（混淆矩阵 / 双特征消融 / ID 自愈恢复）

| 脚本 | 核心功能 | 输入源 | 核心产物 |
|---|---|---|---|
| [`eval_confusion_matrix.py`](eval_confusion_matrix.py) | 任务 1: 计算并生成各数据集混淆矩阵 | `risk_features_events.npz`<br>`risk_features_negatives.npz` | `confusion_matrix_report.csv` |
| [`risk_aggregation_two_features.py`](risk_aggregation_two_features.py) | 任务 2: 双特征消融实验训练与评测 | `risk_features_events.npz`<br>`risk_features_negatives.npz` | `two_features_vs_baseline.csv`<br>`fig_two_features_roc.png` |
| [`id_recovery_dino.py`](id_recovery_dino.py) | 任务 3: DINO 视觉相似度匹配与恢复评估 | `datasets/` 图像帧<br>`YOLOX_outputs/` 跟踪框 | `id_recovery_dino_summary.csv`<br>`id_recovery_hard_cases.npz` (Hard Cases) |
| [`id_recovery_vlm.py`](id_recovery_vlm.py) | 任务 3: VLM 选择题兜底恢复评估 | `id_recovery_hard_cases.npz`<br>拼图可视化网格 | `id_recovery_vlm_results.csv` |


---

## 三、 标准执行流水线

可以通过 `run_all_pipeline.py` 进行一键式流水线执行：

```bash
# 激活环境
conda activate bytetrack
cd research/code

# 1. 查看流水线阶段与帮助
python run_all_pipeline.py --help

# 2. 一键执行完整科研流水线（特征抽取 -> 聚合评测 -> 归因分析 -> 材料生成）
python run_all_pipeline.py --all

# 3. 指定分步执行
python run_all_pipeline.py --stage features       # 仅提取因果特征与校准器
python run_all_pipeline.py --stage eval           # 仅运行四大聚合模型评测
python run_all_pipeline.py --stage breakdown      # 仅运行类别雪崩归因
python run_all_pipeline.py --stage presentation   # 仅重新生成组会汇报 PPTX
```
