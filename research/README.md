# research — LSRG-ByteTrack 科研主工作区

面向多目标跟踪 ID 失效的机理分析、在线风险建模与定向纠错研究（LSRG-ByteTrack）。

---

## 📑 核心文档与交付导航

| 文档 / 交付物 | 类型 | 核心内容 |
|---|---|---|
| [`专家博客.md`](专家博客.md) | 深度长文 | **必读复盘**：从因果物理风险建模、ECDF分布校准到全局误报雪崩的机理归因（万字详述） |
| [`reports/LSRG_ByteTrack_组会汇报_v8.pptx`](reports/LSRG_ByteTrack_组会汇报_v8.pptx) | 学术 PPT | 7 页宽屏学术组会汇报（含统一尺度大表、模型盲测对比与全屏 ROC） |
| [`docs/开发进度.md`](docs/开发进度.md) | 进度台账 | 实时台账：各阶段分析状态、产物位置、复现命令与下一步计划 |
| [`docs/COE.md`](docs/COE.md) | 踩坑手册 | 跨会话经验：环境、数据口径、硬件不确定性与历史教训 —— **新会话必读** |
| [`reports/class_risk_breakdown_report.md`](reports/class_risk_breakdown_report.md) | 专题研报 | $S_c, S_r, S_h$ 三大失效模式隔离评测与高召回 FPR 雪崩机理归因报告 |
| [`reports/day3_report.md`](reports/day3_report.md) | 阶段研报 | 小型可观测性诊断实验与 Go/No-Go 裁决报告（判定情况 B） |
| [`reports/day2_report.md`](reports/day2_report.md) | 阶段研报 | 三候选机制离线筛选与 DEN 在线插桩验证报告 |
| [`reports/day1_report.md`](reports/day1_report.md) | 阶段研报 | 失效分类学（$S_c / S_r / S_h$）、设计收敛与几何门控基线 |
| [`docs/研究方向.md`](docs/研究方向.md) | 技术路径 | 研究目标演进与技术路线图 |

---

## 📁 目录布局与模块划分

```
research/
├── README.md                          # 本导航索引
├── 专家博客.md                        # 万字深度学术复盘长文
├── docs/                              # 进度台账、COE、研究方向
├── reports/                           # Day1-3研报、类别分析报告、组会PPTX、HTML报告
├── code/                              # 科研代码库 (详见 code/README.md)
│   ├── run_all_pipeline.py            # [总控] 一键端到端科研复现流水线
│   ├── risk_features.py               # 提取三维因果风险特征与 164.7万负样本 ECDF 校准
│   ├── risk_aggregation.py            # 四大聚合模型(Linear, Max, MLP, ECDF) 5-Fold 盲测
│   ├── class_risk_breakdown.py        # S_c / S_r / S_h 类别隔离评测与雪崩归因
│   ├── comprehensive_class_dataset_breakdown.py # 跨数据集/类别交叉统计
│   ├── generate_lab_presentation.py   # 生成 7 页组会汇报 PPTX
│   └── generate_*.py                  # 论文 2x2 ROC / 代价矩阵 / 折线图生成
├── data/                              # 冻结基准输入：{MOT17,MOT20,SportsMOT}_events*.csv
├── taxonomy/                          # 统一产物库：特征张量、模型包、数据表、ROC图表
└── diag_exp/                          # [已归档] Day 3 诊断实验独立区 (情况 B No-Go)
```

---

## 🔬 科研进展全景（Phase 1 ~ Phase 5）

- ✅ **Phase 1（基准复现与事件挖掘）**：建立 MOT17/20/SportsMOT 统一基准，清洗 GT（去 ignore 区域），精确重构 4,713 条 IDS 事件（$S_c$ 1,828 / $S_r$ 1,899 / $S_h$ 986）。
- ✅ **Phase 2（失效机理与物理签名）**：证明 $S_r$ 强局域（gap=1, IoU_last 中位 0.854）vs $S_h$ 长尾漂移（gap 中位 6）显著分离；否定旧 $\beta$ 公式与拥挤度直接相关假设。
- ✅ **Phase 3（DEN 在线门控插桩）**：完成三候选机制筛选（KMC/KF 否定，DEN 支持）；在线 alert 模式插桩验证零行为变化与特征无偏性。
- ✅ **Phase 4（可观测性诊断与 No-Go 裁决）**：四特征并集 $TPR@FPR{\le}1\% \approx 33.5\%$，未达到 50%~60% 门槛，正式判定为**情况 B（No-Go）**，终止纯外挂 Gate 路线。
- ✅ **Phase 5（因果物理风险建模与 ECDF 经验校准）**：
  - **因果特征闭环**：构建三维正交物理特征 $\mathbf{f}=(f_{\text{weak}}, f_{\text{comp}}, f_{\text{swap}})$，破除前置门控，严格依赖前序时空，零 NaN。
  - **ECDF 经验分布校准**：以 164.7 万正常检测为基底，完成正单调概率映射 $\mathbf{r}=(r_{\text{weak}}, r_{\text{comp}}, r_{\text{swap}}) \in [0, 1]^3$。
  - **四大聚合模型盲测**：5-Fold 跨序列评测 Linear, Max, MLP, ECDF，锁定高召回工作点。
  - **全局误报雪崩归因**：精确定位 TPR $\ge 85\%$ 时 FPR 剧增的机理源于冷启动 $S_c$ 与历史漂移 $S_h$ 的“自信而错”伪装，指出从风控向内部关联函数重构（OC-SORT / C-BIoU）转型的必然性。

---

## 🚀 快速复现命令

```bash
conda activate bytetrack
cd research/code

# 一键端到端复现全部 Phase 5 实验与汇报材料
python run_all_pipeline.py --all
```
