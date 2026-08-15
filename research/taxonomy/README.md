# taxonomy/ — 产物（分析输出 + 静态报告）

**定位**：本目录是分析链的**输出侧**（"taxonomy" = 失效分类学）。所有代码生成的数据文件、图片与手写静态报告统一存放于此；脚本（`code/analysis.py` → OUT、`code/den_online_eval.py` → `../taxonomy`）只写入本目录。输入侧见 `data/README.md`。

**为什么 CSV/PNG 在此处与报告共存、而 data/ 也有 CSV**：分类按**数据流角色**——凡"代码生成、随阶段演化"的数据都是产物，与同阶段产物同生共死（如 `gate_feasibility_{events,summary}.csv` 与 `gate_feasibility_roc.png` 均由 `analysis.py` 同一次运行生成，故同目录）；`data/` 只留冻结输入事件表。

## 分类标准（产物三级分类）

| 级 | 子类 | 判定规则 | 文件 |
|---|---|---|---|
| ① | **冻结产物** | V1/V2 阶段生成、不再变化，仍被活脚本读取（SANITY 锚点） | `event_classification.csv`、`event_counts_by_sequence.csv`、`warning_features.csv` |
| ② | **离线分析产物** | `analysis.py` 每次运行重新生成（活产物，可覆写） | `gate_feasibility_events.csv`、`gate_feasibility_summary.csv`、`gate_feasibility_roc.png` |
| ③ | **在线插桩产物** | `den_online_eval.py` 生成（活产物，可覆写） | `den_online_{mot17,mot20,sportsmot}_full.csv` |
| — | **静态报告** | 手写维护，与代码解耦；改报告直接编辑 md | `event_taxonomy_report.md`（事件侧）；`reports/day1_report.md`（设计收敛与门控基线）、`reports/day2_report.md`（三机制筛选与在线插桩，由原 gate_report + den_online_report 合并，2026-08-10）、`reports/day3_report.md`（可观测性诊断与路线裁决） |

## 文件清单（10 个，2026-08-10 整理后）

| 文件 | 生成源 | 级 | 内容 |
|---|---|---|---|
| `event_classification.csv` | V1（冻结） | ① | 4,756 条 IDS 逐条分类（S_c 1,841 / S_r 1,921 / S_h 994） |
| `event_counts_by_sequence.csv` | V1（冻结） | ① | 114 序列级分类计数；`analysis.py` SANITY 读取源 |
| `warning_features.csv` | V5（冻结） | ① | 预警特征集（H=0/1/3/5 正负样本特征，1.7MB） |
| `gate_feasibility_events.csv` | `analysis.py` | ② | 4,756 行逐事件：class + top1/top2/margin + cos_theta/r_v/v_obs_norm + sigma_norm + n_neighbor + geom_020 |
| `gate_feasibility_summary.csv` | `analysis.py` | ② | 长表：geom 基线 A/B/C × ε + kmc/kf/den 最优触发点 |
| `gate_feasibility_roc.png` | `analysis.py` | ② | 三机制 ROC 图（全目录唯一图片） |
| `den_online_{mot17,mot20,sportsmot}_full.csv` | `den_online_eval.py` | ③ | 在线-离线对齐：ε0×γ 网格 FPR/TPR + 锚点指标 |
| `event_taxonomy_report.md` | 手写 | 报告 | 事件分类/物理签名/可挽回性 |

## 使用规则

- ②/③ 类为活产物：重跑对应脚本即覆写；① 类与报告禁止被脚本覆写。
- 删除任何文件前先读内容（Git 已启用，`baseline-v001` 为重构前基线；仍应谨慎操作）。

## 旧版命名映射

- `event_classification.csv` 即早期版本中的 `taxonomy_partition.csv`（逐事件分类结果）。
- `event_counts_by_sequence.csv` 即早期版本中的 `taxonomy_by_sequence.csv`（序列级分类计数）。
- 旧版命名已不再使用；后续代码与文档统一引用当前文件名。
