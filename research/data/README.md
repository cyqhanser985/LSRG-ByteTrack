# data/ — 冻结输入（只读，Canonical）

**定位**：本目录是整条分析链的**唯一权威冻结输入**。三数据集事件表由
`tools/build_ids_events.py` 从**当前 GT**（`datasets/{DS}/V*/gt/gt.txt`，2026-08-15
清洗后 conf==1 版）与冻结跟踪结果（`YOLOX_outputs/*_v001_full/track_results/`）
经 motmetrics 时序匹配重新生成（2026-08-16 全量重生成；语义口径与 V1/V2 冻结版
一致，见 `tools/build_ids_events.py` 头注释与 COE.md §3）。脚本
（`code/analysis.py`、`code/den_online_eval.py`、`diag_exp/run_diag.py`）从这里
**只读**数据，**不写入**。

> **Canonical 声明**：`research/data/` 是后续所有实验与分析代码的唯一读取来源。
> 旧归档镜像（`YOLOX_outputs/analysis/` 与 HTML 报告包的 tables）已于
> 2026-08-16 删除，不再存在任何第二份事件表。

## 为什么 CSV 不在此处统一存放

文件分类按**数据流角色**（输入 / 产物）而非文件类型。凡是由代码生成、随分析阶段演化的数据（CSV/PNG）一律进入 `taxonomy/`；本目录只保留"冻结后不再变化、作为后续一切分析基准"的输入事件表。

## 分类标准（冻结输入判定规则）

1. 文件是**某个已完结分析阶段的直接产物**，且后续阶段仍以它为输入/校验锚点 → 冻结，入 `data/`；
2. 文件被任何活脚本**写入**或**每次运行重新生成** → 产物，必须入 `taxonomy/`；
3. 图片（PNG）只作为产物出现（`taxonomy/gate_feasibility_roc.png`）；输入侧无图片。
   - 特例：`taxonomy/` 中的 `event_counts_by_sequence.csv` 同为冻结产物，因生成于分析链中段、后续仍被 `analysis.py` 读取（SANITY 锚点），按"产物留在产物目录、标注冻结"处理，不迁入 `data/`。

## 文件清单（9 个，全部冻结，2026-08-16 重生成）

| 文件 | 内容 |
|---|---|
| `{MOT17,MOT20,SportsMOT}_events.csv` | 原始 IDS 事件表：seq, frame, type(switch/reuse), gt_id_old, gt_id_new, track_id, note |
| `{MOT17,MOT20,SportsMOT}_events_metrics.csv` | 事件 + 物理指标：gap, IoU_last, dx/dy/dist_last, IoU_prev/next/swap, dist_swap, area_ratio, **na_flag**（S_c/S_r/S_h 分类键；`analysis.py` 的 `load_events()` 读取源） |
| `{MOT17,MOT20,SportsMOT}_events_summary.csv` | 逐序列计数核对表：switch/reuse 计数 vs motmetrics SWITCH/TRANSFER 行数 + status |

重生成命令（修改 GT 或 track_results 后必须执行，并同步更新
`taxonomy/event_counts_by_sequence.csv` 与 `SHA256SUMS.txt`）：

```bash
PY=E:/anaconda/envs/bytetrack/python.exe   # motmetrics 必需
$PY tools/build_ids_events.py              # 全量三数据集
$PY tools/build_event_counts.py            # 重新生成 taxonomy/event_counts_by_sequence.csv
cd research/data && sha256sum *.csv > SHA256SUMS.txt
```

## 校验

- `SHA256SUMS.txt` 记录了本目录全部 9 个 CSV 的 SHA-256 校验值。修改任何文件后必须重新生成并提交该文件。
- 语义自校验：`tools/build_ids_events.py` 输出的 `num_switches_metric` 与 switch 计数逐序列一致（status=OK）即视为与 motmetrics 事件流对齐。

## 使用规则

- **只读**：任何脚本不得写入本目录；如需新输入，走"冻结 → 复制 → 校验"流程。
- 修改事件表 = 修改全部分析基准；Git 已启用（`baseline-v001` 为重构前基线），改动前仍应先备份并同步更新 `taxonomy/event_counts_by_sequence.csv`（SANITY 硬断言源）。
- 本目录不存放报告（结论报告在 `research/reports/day{1,2,3}_report.md` 与 `taxonomy/event_taxonomy_report.md`；注意其数字基于 2026-08-16 重生成前的旧事件池，作为方法学存档）。
