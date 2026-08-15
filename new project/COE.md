# COE — 跨对话经验沉淀

> 目标：任何后续会话的 agent 接手本方向时，先读本文件即可对齐环境、口径与历史教训，避免重复犯错。
> 进度看《开发进度.md》，研究结论看《day1_report.md》《day2_report.md》，本文件只讲"怎么干、别踩什么坑"。

## 1. 环境与运行（最易错，优先读）

| 项 | 值 | 备注 |
|---|---|---|
| conda 环境 | `E:\anaconda\envs\bytetrack\python.exe` | Python 3.8，torch 2.1.2+cu118，RTX 4060 8GB |
| PATH 上的 `python` | ❌ 缺 motmetrics | 跑 motmetrics 相关脚本必须显式用 conda 解释器 |
| **sklearn** | ❌ **不在 bytetrack 环境**（此前"已可用"的说法有误） | 回归用 numpy 手写 IRLS + scipy Wald（历史实现已随旧脚本删除，如需回归需重新实现） |
| **pip 安装** | ❌ 不可用（清华镜像 SSL 握手失败，无外网） | 不要依赖 pip 补包；statsmodels 也装不上，显著性用 scipy 自算 |
| 版本控制 | ❌ **不是 git 仓库** | 任何删除/覆盖不可恢复；操作文件前先看内容，删除先问 |
| 跟踪实验 | `YOLOX_outputs/{mot17,mot20,sportsmot}_v001_full/track_results/*.txt` | 三数据集各一个 expn，名称已确认 |
| **PYTHONPATH** | ⚠️ `python tools/track_v001.py` **必须带 `PYTHONPATH=E:\科研\ByteTrack`** | 本 conda 环境 .pth 处理损坏（`sys.path[0]` 是假前缀 `D:\path\to\python`，easy-install.pth 不生效）；脚本执行时 sys.path[0]=tools/，import yolox 失败；`python -c` 能导入只是因为 cwd 入 path |
| **fp16 非确定性** | ⚠️ 同配置重跑结果逐行不同（base vs base 差异 ~87% 行、行数也变），`--seed 1` **无效**（cudnn.benchmark + fp16） | 任何 A/B 对比必须**同轮次**跑（base 与变体一次启动内对比）；冻结 v001_full 结果只作数量级参照，不可作逐行 diff 基准 |

## 2. 目录地图（**所有代码与产物统一放 `new project/`，单位置**）

**`new project/`（本项目，唯一运行与产物位置）**：
- `开发进度.md`（进度台账）· `COE.md`（本文）· `README.md`（索引）· `day1_report.md` / `day2_report.md`（研究结论）· `diag_exp/`（Day 3 诊断实验独立工作区：`run_diag.py` + `results/` + `day3_report.md` + `README.md`，自包含，只读 data/ 与 taxonomy/）
- `code/` — **活脚本（2 个）**：`analysis.py`（几何门控基线 + 三候选机制验证）、`den_online_eval.py`（在线-离线对齐，自包含只读 CSV；均用 `_repo_root()` 自动向上探测仓库根）。**2026-08-08 整理：V1–V8 历史脚本全部删除**（结论冻结在根目录 `day1_report.md`/`day2_report.md` 与 `taxonomy/` 报告及 CSV；report 与代码解耦——改报告直接编辑 md，不重跑代码）。**2026-08-10：`diag_exp/run_diag.py` 通过 `import analysis` 复用其全部工具函数**（main 有守卫，import 无副作用），新特征（occlusion IoF / swap ΔC）为纯 numpy 向量化，全量三数据集 ~70s（无需 KF 重放）
- `data/` — 冻结输入：`{ds}_events.csv` + `{ds}_events_metrics.csv`（主仓库 analysis 快照；只读，分类标准见 `data/README.md`）
- `taxonomy/` — **产物（10 个，见名知义）**：`event_classification.csv` / `event_counts_by_sequence.csv`（冻结）、`warning_features.csv`（冻结 V5 特征集）、`gate_feasibility_{events,summary}.csv` + `gate_feasibility_roc.png`（analysis.py 生成）、`den_online_{ds}_full.csv`（den_online_eval.py 生成）、`event_taxonomy_report.md`（手写静态报告；门控/在线报告已合并至根目录 `day2_report.md`）

**主仓库 `E:\科研\ByteTrack`（只读数据源）**：
- `datasets/` — GT + `YOLOX_outputs/{mot17,mot20,sportsmot}_v001_full/track_results/` — 跟踪输出（analysis.py 输入）
- `YOLOX_outputs/analysis/` — 遗留评估产物与**冻结输入**：`{ds}_events.csv`、`switch_metrics/{ds}_events_metrics.csv`、`taxonomy/`（只读，脚本从这里读输入、不写入）
- `exps/example/mot/yolox_x_{mot17,mot20,sportsmot}_v001.py` — 跟踪配置（继承 `yolox_x_mix_det.py` / `yolox_x_mix_mot20_ch.py`，后者 test_size=(896,1600)）
- `yolox/` — 框架代码（**2026-08-08 起 DEN 在线插桩，alert 模式，`--den-gate` 默认关**）：`yolox/tracker/den_gate.py`（新，纯函数）+ `byte_tracker.py`/`mot_evaluator.py`/`tools/track_v001.py` 薄改动；**改前备份在 `new project/code/backup/`（含 SHA256SUMS.txt）**；analysis.py 按文件路径加载 `yolox/tracker/kalman_filter.py` 做 KF 重放，不 import 包
- `tools/` — 保留 `track_v001.py`（跟踪运行入口，在线插桩阶段使用）+ 数据集转换/可视化/诊断工具；**V1/V2 事件分析脚本已删除**（2026-08-08，产物冻结）

**同步规则**：无 git，但**不再有双位置复制**——新脚本直接写在 `new project/code/` 运行，产物直接写入 `new project/taxonomy/`；主仓库侧只有只读数据源。

## 3. 数据口径（易错点）

- **事件定义**：一次 IDS 事件 = (seq, frame, track_id, gt_id_old → gt_id_new)，"目标换了 tracker"。`events.csv` 中 type=switch（GT 视角）与 type=reuse（tracker 视角，仅扫 MATCH 行）是同一批物理现象的**两个观察视角**。
- **三类分类逻辑**（`analysis.py` 的 `classify()`，基于 `na_flag` 三态）：
  - `no_last_seen` → **S_c**（接管方此前从未输出，冷启动）
  - 无 `no_prev` → **S_r**（紧邻帧 F-1 有输出，活跃接管）
  - 有 `no_prev` → **S_h**（更早帧有输出，历史重新激活）
  - 分类只看**输出历史状态**，不依赖 ByteTrack 内部 active/lost/removed（需在线插桩后才做口径对齐）
- **tracker 视角事件**（历史口径，脚本已删）：只取 MATCH/SWITCH/TRANSFER 行（RESOLVED 集合）；**RAW 行必须排除**（是前一帧候选对回放，会污染 per-tracker 历史）
- **方法学事实（勿重复推导）**：|S∩Reuse| = 0 —— reuse 挖掘只扫 MATCH 行，对 SWITCH 行上的顶替天然盲视；这否定了旧经验公式 β = R/((1−α)S)（MOT17 反解 β=1.33 > 1，不可能是概率）
- **特征纪律（防数据泄露）**：gap / IoU_last / dist_last 等是**离线诊断指标**（源自 GT 事后分析），只用于机理画像，**不得作为在线门控输入**；在线只能用因果特征（卡尔曼 innovation、Top1-Top2 margin、neighbor geometry、lost age）
- **CSV schema 速查**：
  - `events`：seq, frame, type, gt_id_old, gt_id_new, track_id, note
  - `events_metrics`：seq, frame, type, gt_id_old, gt_id_new, track_id, old_hid, last_frame, gap, IoU_last, dx_last, dy_last, dist_last, IoU_prev, IoU_next, IoU_swap, dist_swap, area_ratio, **na_flag**（三态分类键）
  - `tracker_events`：seq, frame, track_id, gt_id_new
  - `taxonomy_partition`：…, old_hid, class, in_tracker_join, in_reuse_join
  - `taxonomy_by_sequence`：dataset, seq, n_S_c, n_S_r, n_S_h, n_switch, n_reuse
- **Day 3 诊断实验口径**（2026-08-10，详见 `diag_exp/day3_report.md` 与 `diag_exp/recognition_summary.md`）：
  - 正样本 = 全部冻结 switch 事件（S_c 1,841 / S_r 1,921 / S_h 994 = 4,756；分类别 + overall 分别分析）；负样本 = C 组全部检测（**含无 F−1 输出的非事件帧**——analysis.py 的 `prev is None` 分支 `c_n += n_det` 计入，该类检测 top1=top2=0 永不触发，仅稀释 FPR 分母；diag 首版漏计 396/371/878 条，与冻结 summary 对不上即此因）
  - **无 F−1 输出的"事件帧"也会被 `prev is None` 分支跳过**——SportsMOT V082 帧 407 的 S_c 事件（tid 有输出但 F−1 整帧无输出）漏计 1 条；analysis.py 语义 = 无活跃轨迹 → 门控特征未定义 → no-box 行。扩展类别分析时该分支必须写 no-box 事件行
  - **S_c/S_h 的 Motion Surprise 结构上不可计算**（接管方 F−1 无输出，定义数 0/1,841、0/994）——四特征对非 S_r 类别实际只有 3 个可用；S_c 合格事件（top1≥0.2∧top2≥0.2）仅 12.0%（SportsMOT 43/1,446），"冷启动结构性盲视"量化确认
  - **Swap 变体在类别间反转**：S_r 用变体 B（D2=T2 自有检测，31.7%）更强；S_c/S_h 用变体 A（D2=T1 自有检测，S_h 29.7% vs 8.1%、MOT20 S_c 58.8% vs 7.0%）更强；全部 switch 并集 22.6% → 24.0%（变体 A 重建）
  - 共享资格 E = top1≥0.2 ∧ top2≥0.2（所有特征触发的前置，保证并集口径自洽）；TPR 分母 = 全部有框 S_r 事件（未定义特征按不触发计），与 V8 的"defined-only 分母"不同——报告需注明
  - **Swap Instability 定义**：ΔC = (c12+c21) − (c11+c22)，c11=top1、c12=top2，D2 = T2 自有检测（变体 B，文档"当前分配 D1→T1, D2→T2"读法；变体 A 取 T1 自有检测，性能相近 AUC 0.846/0.844）；M<2 或 D2 不存在时 NaN；触发方向 high（ΔC>thr，thr 实测约 −0.4~−0.55）
  - **并集搜索必须给每特征加"关闭"哨兵**（低方向 −1e9 / 高方向 +1e9）：候选阈值全为负样本分位数时并集无法复现子集最优点，曾致 MOT20 u1234(19.1%) < u12(29.0%) 的反常结果
  - **冻结 CSV 逐值比对容差 5.0001e-5**：gate_feasibility_events.csv 存 "%.4f" 四舍五入值，同一代码路径的浮点噪声被放大到 5e-5（实测最大 4.97e-5、均值 ~0）；1e-6 容差会误报
  - Occlusion IoF 用同帧检测框（在线可观测）；SportsMOT 上无判别力（AUC 0.50，中位 0.55 vs 0.52 重叠）——"被遮挡→定位失真"机理不适用于高速稀疏场景
- **新增口径**（2026-08-07/08，详见 `taxonomy/event_taxonomy_report.md` 与根目录 `day1_report.md` / `day2_report.md`）：
  - S_c 事件中 `old_hid` = 持有目标 g 的旧 tracker A；"在场未认领"= A 在事件帧 F 仍有输出（380 条）；可达性 = 恒速外推 A 最后框到 F 与冷启动新检测框的 IoU≥0.3（主）或中心距≤0.5·对角线（敏感性，并集为上限口径）
  - track_results 是**关联后输出**（非原始检测），歧义比例是下界代理；margin<0.05 档含 top1=top2=0 的孤立检测（全数据仅 0.12%，可忽略）
  - κ 定义（已否定但口径复用）：k1 邻域数 = 帧内中心距<0.5·(diag_i+diag_j)；k2 成对重叠 = 帧内 IoU>0 对的平均 IoU；k3 归一化密度 = n·均值框面积/帧面积；**κ 与速度组内强共线（|ρ| 0.43–1.0），勿同时解释两系数**
  - V5 预警特征（因果，仅过去观测）：innovation（恒速外推残差/框对角线）、max_overlap、margin、n_competitor、lost_age；正样本锚点 = 旧 tracker A 在 [F−H, F) 持续持有 g；负样本取 T′≤F−H−2 的同序列同密度（±5 活跃数）帧
  - **门控分析锚点 = 接管方 tid**（与 V5 的 old_hid 区分）；S_r 事件 gap 恒=1 且 (seq,frame,tid) 无重复
  - **帧数/检测数地标只统计"有活跃轨迹（F−1 有输出）的帧"**——`prev is None` 分支里计数不加；no-active 事件帧的检测**不计入 C 组**
  - **KF 协方差演化与测量值无关**（只依赖 predict/update 调度与 h 序列）→ 离线重放 σ ≈ 在线 σ；σ 归一化必须用 **mean[3]（状态 h）**而非测量 h（检测框抖动会污染 σ̂）；`updates + init = 输出行数`（新 track 首帧 activate 不 update）
  - r_v 在任一分量速度 <1e-3·框对角线时视为**未定义（NaN）**——静止目标"由静到动"是检测噪声不是运动突变；曾误用 inf 导致 20% 假阳性
  - **判定规则（双口径）**：绝对口径 = FPR≤1% 下 TPR（SUPPORTED≥25% / PARTIAL 10–25% / NOT SUPPORTED<10%）；机制口径 = 相对固定基线的增量增益（KF/DEN 的覆盖率可能高而机制本身增益≈0，判定必须分开报告）
  - **DEN 在线插桩口径**（2026-08-08 起，见根目录 `day2_report.md` §二/§三）：
    - 快照 = update() 帧末 output_stracks 复刻 mot_evaluator 过滤（vertical w/h>1.6 + min_box_area）+ 0.1 舍入，与冻结 track_results 人口同构（同运行逐帧一致率 100%）
    - **帧号资格**：仅 `cur_frame == snap_frame + 1` 才计算门控（mot_evaluator 只在检测非空时调 update，帧号与 frame_id 脱钩——跳帧用旧快照即口径错位/泄露）
    - **跨运行 track_id 不可比**（STrack._count 全局计数 + 运行非确定性 → ID 漂移；同运行内 ID 才有意义）→ 事件-日志对齐用 **box 级 IoU>0.5**（事件 track 的冻结 F 输出框 vs 日志检测框），不要用 tid 匹配
    - **在线 FPR 代理系统性高于离线 1% 预算**（V002 3.0% / V043 5.7% @0.20/1.25）——候选人口不同（原始检测框 vs KF 滤波输出框、top1-track N vs 接管方自身 N）；未分解前不得据此判定"门控过松"
    - 候选行日志存原始特征（top1/top2/margin/N + 检测框）→ **ε0×γ 网格可后验重算，无需重跑**（`den_online_eval.py`）

## 4. 已知结论速查（验证过的数字，勿重复推导）

- 4,756 条 IDS：S_c 1,841（38.7%）/ S_r 1,921（40.4%）/ S_h 994（20.9%）；序列级 114 条（MOT17 21 / MOT20 4 / SportsMOT 89）
- 场景参数：MOT17 α=22.4%、π_r=58.5%；MOT20 α=16.6%、π_r=70.3%；SportsMOT α=56.3%、π_r=63.3%
- S_r 局域：IoU_last 中位 0.854、dist_last P90 14.9 px；S_h 长尾：gap 中位 6、gap>5 占 56.3%、dist_last P90 97.3 px、IoU_last 中位 0.383
- S_c：380 样本（20.6%）旧 tracker 事件帧仍在场，IoU_swap 中位 0.235、中心距中位 28.8 px
- π_r 场景排序（MOT20 > SportsMOT > MOT17）与拥挤度排序**不一致**（SportsMOT 最稀疏却高于 MOT17），"越拥挤 π_r 越高"不成立
- **拥挤度回归结论**：π_r 不由全局 κ_geo 决定（k_neighbor/k_overlap 不显著、k_density 显著为负，方向不一致 → 判定不支持）；**速度是主导协变量**（b≈−0.64~−0.69，聚类稳健 p<0.001），κ 经速度通道呈现假性正相关；McFadden R²≈0.03–0.04
- **可挽回性结论**：Recoverability_S_c(L) = 43.1%（L=30）→ 44.4%（L≥120）饱和；在场未认领 380 条贡献 20.6 个百分点；瓶颈是可达性而非间隔
- **触发率代理结论**："90% bypass" 被否定：top2≥0.2 时 76.2% 帧含歧义检测（bypass 23.8%）；margin<0.1 仅 0.4% 检测——触发语义选择决定触发率量级
- **预警结论**：PR-AUC 全 H 高于先验基线：H=0 +0.126 / H=1 +0.044 / H=3 +0.101 / H=5 +0.095（部分支持）；max_overlap、margin 为主导特征；H=1 最弱
- **几何门控结论**：S_r 覆盖 3.85%→24.47%（ε=0.05→0.25），C 组误触发 0.07%→0.75%（选择性 33–55×）；S_r 的 top1 中位 0.875 但 margin 中位 0.406——margin 是覆盖率瓶颈；S_c 结构性盲视
- **三机制验证结论**：① KMC（cosθ/r_v 运动一致性）三数据集 NOT SUPPORTED——S_r 接管方运动正常（cos 中位 0.72–0.90），"confident-but-wrong 伴随运动突变"被否定；② KF 自适应裕度机制级 NOT SUPPORTED（增益 +0.00/+0.62/+0.99pp）——S_r 的 σ̂ 与正常帧重叠（中位 0.0575–0.0584）、margin 与 σ 无相关（ρ≈0.03–0.06）；③ **DEN 局域密度 PARTIAL→SUPPORTED（增益 +3.6/+6.6/+6.2pp，SportsMOT TPR 27.3%@FPR≤1%）——全局拥挤度在 ego-centric 局部化后复活（SportsMOT：S_r N=0 仅 11% vs 正常 50%），唯一值得在线插桩的机制**
- **Day 3 诊断实验结论（Go/No-Go = 情况 B，终止 Gate 路线）**：四特征并集 TPR@FPR≤1% = MOT17 24.1% / MOT20 37.0% / SportsMOT 36.2%（合并 33.6%）——全部 < 40% 终止线；FPR≤2% 合并 40.6%（MOT20/SportsMOT 43–45%）仍距 50–60% 继续线 ~10pp。单特征合并 TPR@1%：margin 24.5% / motion(v_obs_norm) 14.7% / cos 2.2%（再次否定）/ occlusion 6.1%（SportsMOT AUC 0.50 无效）/ **swap ΔC 31.7%（AUC 0.844，最强信号，事件中位 −0.66~−0.89 vs 正常 −1.16~−1.22）**；margin∧swap 互补（并集 +9.1pp），其余信号边际 ≈0；66.4% 事件对全部信号不可见（"自信而错"结构性盲区，与 day1 瓶颈同源）。**全类别扩展（S_c/S_r/S_h/全部 switch）**：并集识别率 S_c 9.5%（SportsMOT 2.6%，结构性盲视）/ S_h 25.7% / S_r 33.6% / **全部 switch 22.6%**（@2% 26.8%）；MOT20 拥挤场景例外（S_c 44.9%、S_h 42.9%、swap 变体 A 下 58.8%/56.2%）。**裁决动作：不再堆叠门控，转向关联函数重构（OC-SORT 观测中心运动补偿 / C-BIoU 搜索空间自适应）**；swap ΔC 保留为关联函数改进的离线诊断指标（AUC 0.84）

## 5. 历史教训（血泪清单）

1. **2026-08-03 清理时误删 `exps/example/mot/yolox_x_mix_mot20_ch.py`**，后按官方样式 + 本地验证参数重建 —— 教训：无 git，删除前先读目标内容、与描述不符先提出；重建配置需用 Gflops 日志验证 test_size
2. **115 个视频和例子 PNG 已删**（可再生，报告包内有 JPEG 版）—— 别再找不存在的文件
3. `yolox_x_bft.py`（BFT 私有数据集配置）**尚未创建**；USAGE_GUIDE 中"已创建"是过时说法
4. 旧脚本的 `ROOT = dirname(dirname(abspath(__file__)))` 硬编码仓库布局（V1/V2 遗留，仍在 `tools/` 运行）；**V3–V6 脚本已改为 `_repo_root()` 自动探测**（从 `new project/code/` 或 `tools/` 运行均可）
5. 脚本读数据用 `encoding="utf-8-sig"`（容忍 BOM）、写出用 `utf-8`；脚本为纯 stdlib + ASCII 注释（历史脚本曾用 `statistics.stdev` 而非 numpy）
6. 序列目录已重组为 `V001/V002/...` 命名（如 `datasets/MOT17/V001/gt/gt.txt`），与官方 MOT 命名（MOT17-01 等）不同
7. 数据集说明：BFT = 用户私有鸟群跟踪数据集（106 序列，COCO 标注 train/val/test_v1.5.json，MOT 格式图像）；公开集有 MOT17 / MOT20 / SportsMOT / DanceTrack（已重组）
8. motmetrics `acc.events` 结构排查曾需专门调试脚本（已删）——若再排查，注意其事件行语义与 RAW/MATCH 区分
9. **报告 md 的 `%%` 字面量陷阱**：非 `%` 格式化的字符串里写 `%%` 会原样输出为 `%%`——只有被 `%` 操作符格式化的字符串才需要 `%%` 转义
10. **手写 IRLS 的伪零系数陷阱**：设计矩阵含全零列（如剔除数据集后哑变量列）时 Hessian 奇异，IRLS 停在 β=0——表现为系数恰好 0.0000；剔除子样本时要同步删对应列
11. **逐对 IoU 必须 numpy 向量化**（历史教训：纯 Python 逐对 IoU 全量约 10+ 分钟；`analysis.py` 的 `iou_matrix` 是向量化实现，可复用）
12. **`boxes_array()` 返回顺序是 (arr, tids)**——V8 曾解包反（`prev_ids, prev_arr =`）导致 searchsorted "object too deep"；解包必须是 `prev_arr, prev_ids =`（V7 里 `pred_arr, _ =` 只取第一个所以没暴露）
13. **KF 重放"σ 稳态 sanity"的三层修正**：① 序列只记输出帧（lost 帧只累积 Q 显得不稳定）；② 归一化用状态 h（mean[3]）而非测量 h（检测框抖动 → 假不稳定）；③ 长 track（≥100 输出帧）任一 10 帧窗口稳态即过（initiate 收敛段会污染整段检查）；σ̂ 稳态值 ~0.1（按 h 归一）
14. **KF 批量 predict（multi_predict）只对 ≥50 track 的帧划算**（数组构造 overhead），少 track 用逐 track predict；全量三数据集从 10+ 分钟降到 3 分钟（MOT20 大序列曾 30ms/帧）
15. **`np.where(cond, x, y)` 会完整计算 x 与 y 再选取**——除零处产生 inf 后再被 mask 为 NaN 可以，但不要在 x/y 里依赖"惰性求值"
16. **冻结 CSV 只存 4 位小数**——与冻结文件做逐值 SANITY 时容差取 5e-5（"%.4f" 舍入噪声上限 0.5e-4），别用 1e-6（曾误报 1,191 处"不一致"，实为同一代码路径的舍入差异，均值 ~0 无系统偏差）
17. **并集网格搜索要含"关闭"档**（每特征追加 ±1e9 哨兵）——否则并集无法复现子集最优点，出现 u1234 < u12 的非单调反常结果（2026-08-10 MOT20 实例）
18. **C 组（正常帧）人口含"无 F−1 输出的非事件帧"检测**——analysis.py 在 `prev is None` 分支仍 `c_n += n_det`；只算"有 F−1 的帧"会使 C 计数对不上冻结 summary（MOT17 差 396 条）
19. **`import analysis` 复用工具可行**——analysis.py 的 `main()` 有 `if __name__ == "__main__"` 守卫，import 无副作用；新脚本 `sys.path.insert(0, "../code")` 后可直接复用 load_frames/boxes_array/iou_matrix/extrapolate_box/kmc_arrays/_top1_top2_margin 等（diag_exp/run_diag.py 全量 70s，无 KF 重放时更快）

## 6. 文档与编码惯例

- 研究文档（开发进度 / COE / README / day1_day2 报告）：中文、UTF-8、GitHub Flavored Markdown
- 脚本注释：英文、纯 ASCII（历史约定，避免编码问题）；报告产物用英文
- 术语统一：冷启动接管 S_c / 活跃接管 S_r / 历史重新激活 S_h；LSRG-ByteTrack 已收缩为"分类诊断 → 轻量风险判断 → S_r 局部纠错 / 保持 ByteTrack"最小闭环（2026-08-07，原 Diagnose → Gate → Route → Resolve 四阶段与三头 Gate 因 V3–V6 裁决移除，见《day1_report.md》第二、三节）

## 7. 常用命令速查

```bash
# 唯一的分析入口（new project/code/analysis.py 原位运行，产物写入 new project/taxonomy/；
# 输入为只读：data/ 冻结事件表 + YOLOX_outputs/*_v001_full/track_results/）
E:\anaconda\envs\bytetrack\python.exe "new project/code/analysis.py"                                  # 全量三数据集（约 3 分钟）
E:\anaconda\envs\bytetrack\python.exe "new project/code/analysis.py" --datasets mot17 --ideas kmc     # 冒烟（~15s）
# 跟踪运行（在线插桩阶段入口，tools/ 保留）
E:\anaconda\envs\bytetrack\python.exe tools/track_v001.py -f exps/example/mot/yolox_x_mot17_v001.py \
    -c pretrained/bytetrack_x_mot17.pth.tar -expn mot17_v001_full -b 1 -d 1 --fp16 --fuse
# 报告是手写静态文档（根目录 day1_report.md / day2_report.md + taxonomy/event_taxonomy_report.md）：改报告直接编辑 md，不重跑代码
```
