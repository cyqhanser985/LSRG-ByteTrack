# data/ — 冻结输入（只读）

**定位**：本目录是整条分析链的**输入侧**。所有文件为 V1/V2 阶段事件分析脚本（已删除）的冻结产物快照，复制自主仓库 `YOLOX_outputs/analysis/`（见 COE.md §2）。脚本（`code/analysis.py`）从这里**只读**数据，**不写入**。

**为什么 CSV 不在此处统一存放**：文件分类按**数据流角色**（输入 / 产物）而非文件类型。凡是由代码生成、随分析阶段演化的数据（CSV/PNG）一律进入 `taxonomy/`；本目录只保留"冻结后不再变化、作为后续一切分析基准"的输入事件表。

## 分类标准（冻结输入判定规则）

1. 文件是**某个已完结分析阶段的直接产物**，且后续阶段仍以它为输入/校验锚点 → 冻结，入 `data/`（或作为冻结产物留在 `taxonomy/`，见下）；
2. 文件被任何活脚本**写入**或**每次运行重新生成** → 产物，必须入 `taxonomy/`；
3. 图片（PNG）只作为产物出现（`taxonomy/gate_feasibility_roc.png`）；输入侧无图片。
   - 特例：`taxonomy/` 中的 `event_classification.csv` / `event_counts_by_sequence.csv` / `warning_features.csv` 同为冻结产物，因生成于分析链中段、后续仍被 `analysis.py` 读取（SANITY 锚点），按"产物留在产物目录、标注冻结"处理，不迁入 `data/`。

## 文件清单（6 个，全部冻结，2026-08-07）

| 文件 | 内容 |
|---|---|
| `{MOT17,MOT20,SportsMOT}_events.csv` | 原始 IDS 事件表：seq, frame, type(switch/reuse), gt_id_old, gt_id_new, track_id, note |
| `{MOT17,MOT20,SportsMOT}_events_metrics.csv` | 事件 + 物理指标：gap, IoU_last, dx/dy/dist_last, IoU_prev/next/swap, dist_swap, area_ratio, **na_flag**（S_c/S_r/S_h 分类键；`analysis.py` 的 `load_events()` 读取源） |

## 使用规则

- **只读**：任何脚本不得写入本目录；如需新输入，走"冻结 → 复制 → 校验"流程。
- 修改事件表 = 修改全部分析基准；无 git，改动前先备份并同步更新 `taxonomy/event_counts_by_sequence.csv`（SANITY 硬断言源）。
- 本目录不存放报告（报告在根目录 `day{1,2}_report.md` 与 `taxonomy/event_taxonomy_report.md`）。
