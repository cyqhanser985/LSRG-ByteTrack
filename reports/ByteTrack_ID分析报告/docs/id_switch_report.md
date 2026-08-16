# ByteTrack ID 切换 / ID 复用事件分析报告

> 日期:2026-08-01(初版);2026-08-16(按清洗后 GT 全量重算事件表)｜ 环境:Windows + Python 3.8 + motmetrics 1.4.0 ｜ 数据:三数据集全量跟踪结果(V001 标准格式)+ 清洗后 GT
> 分析脚本:`tools/find_id_events.py`(事件挖掘)、`tools/gen_examples.py`(典型例子)、`tools/draw_tracks_video.py`(可视化视频)
> 前置结果:`YOLOX_outputs/{mot17|mot20|sportsmot}_v001_full/track_results/V*.txt`(见 `reproduction_results.md`)

---

## 1. 事件定义(机器定义 = 文档定义)

### 1.1 输入与匹配参数

| 项 | 取值 |
|----|------|
| GT | `datasets/{ds}/V*/gt/gt.txt`,`mm.io.loadtxt(fmt='mot15-2D', min_confidence=1)`(仅保留 conf≥1 的标注行) |
| Track | `YOLOX_outputs/{expn}/track_results/V*.txt`,`min_confidence=-1`(不过滤任何检测) |
| 匹配 | `mm.utils.compare_to_groundtruth(gt, ts, 'iou', distth=0.5)`,IoU≥0.5 视为关联 |
| 事件流 | `acc.events`,列 `Type/OId/HId/D`,`FrameId` 为索引第 0 级(版本 1.4.0 实测确认) |
| 分析范围 | 只有 gt 的序列:MOT17 21 个、MOT20 4 个、SportsMOT 90 个(240 个 V 序列中仅 90 个有 gt) |

### 1.2 事件一:ID 切换(switch)

**机器判定:** `acc.events` 中每一行 `Type == 'SWITCH'` 产生一条事件。

**语义**(motmetrics 源码 `mot.py` 第 263 行):GT 对象 `OId` 之前由 tracker A 跟踪,本帧被匹配给 tracker B(B≠A,且距该对象上次出现 ≤ max_switch_time)。即 **"同一个被跟踪目标换了一个 tracker ID"**。

**CSV 行:** `seq, frame=事件帧, type=switch, gt_id_old=OId, gt_id_new=OId(目标未变), track_id=B(新 tracker)`;note 中给出 **旧 tracker A** = 该 OId 在 switch 帧之前最近一次 MATCH 事件的 HId(与 motmetrics 内部 `self.m[OId]` 旧值一致,逐条抽查核验通过)。

### 1.3 事件二:ID 复用(reuse)

**机器判定:** 对每个 tracker ID,将其全部 **MATCH** 事件按帧序排列;每当关联的 GT 对象发生变化(A→B,A≠B),在 B 首次 MATCH 帧产生一条事件。即 **"同一个 tracker ID 先分配给目标 A,之后又被分配给目标 B(新物体顶替旧 ID)"**。

**CSV 行:** `seq, frame=B 首次 MATCH 帧, type=reuse, gt_id_old=A, gt_id_new=B, track_id=tracker ID`;note 中给出 A 的最后 MATCH 帧。

**边界:** 严格按任务定义仅扫描 MATCH 行;tracker 在 SWITCH 行上发生的目标切换不计入 reuse(§2.3 讨论)。

### 1.4 与 motmetrics 原生事件的关系

| 本报告事件 | motmetrics 事件 | 关系 |
|-----------|----------------|------|
| switch | `SWITCH` 行 | 定义相同,数量逐序列完全相等(§2.2) |
| reuse | `TRANSFER` 行(近似) | 定义同源("tracker 换了目标"),但实现边界不同,数量有差异(§2.3) |

---

## 2. 统计概览

### 2.1 三数据集总数

| 数据集 | 序列数 | 帧数 | **switch** | **reuse** | num_switches(指标) | TRANSFER 行 |
|--------|-------|------|-----------|-----------|-------------------|-------------|
| MOT17 | 21 | 7,977 | **546** | **537** | 546 | 489 |
| MOT20 | 4 | 8,931 | **1,600** | **1,021** | 1,600 | 1,034 |
| SportsMOT | 90 | 55,544 | **2,567** | **762** | 2,567 | 974 |
| **合计** | **115** | **72,452** | **4,713** | **2,320** | 4,713 | 2,497 |

### 2.2 交叉验证(必做项)

- **逐序列 switch 数 == motmetrics `num_switches` 指标:115/115 全部一致**(`find_id_events.py` 对每个序列用 `mh.compute(acc, metrics=['num_switches'])` 复算比对,不一致即 exit 1,本次运行全部 OK)
- switch 总数与 `reproduction_results.md` 三数据集 **IDs 指标(546 / 1,600 / 2,567)完全一致** —— MOTA 表中的 IDs 即 num_switches,这是与独立评估流程的二次印证

### 2.3 reuse 与 TRANSFER 行数的差异说明

两者定义同源但实现边界不同,且两个方向的偏差同时存在(净效应随数据集而异):

- **reuse 偏多的情况(MOT17:537 vs 489)**:motmetrics 的 TRANSFER 只在匈牙利分配阶段产生,**不覆盖 carry-forward 复连**(步骤 1 的延续匹配即使换过对象也只记 MATCH);而按 MATCH 行扫描会捕获这类切换
- **reuse 偏少的情况(MOT20/SportsMOT:1,021 vs 1,034、762 vs 974)**:tracker 的目标切换若发生在 **SWITCH 行** 上,MATCH 行扫描不会捕获,而 TRANSFER 会
- 两者都是合法的"ID 复用"计数口径;**本报告以任务定义(MATCH 行视角)为准**,差异范围 ≤9%(MOT17)与 ≤22%(SportsMOT)

### 2.4 逐序列明细

见 `research/data/{MOT17|MOT20|SportsMOT}_events_summary.csv`(列:`seq, switch_count, reuse_count, num_switches_metric, transfer_rows, status`),全部 115 行 status=OK。

---

## 3. 典型例子(每数据集 4 个,共 12 个)

每个例子的图片为 **事件帧前后各 15 帧** 的标注图(共 31 张 JPEG,画全部 track 框 + 数字 ID,颜色与视频一致;涉及 tracker 加粗),位于本报告 `assets/examples/{数据集}_{类型}_{序列}_f{事件帧}/`(原始 PNG 已随 2026-08-16 归档清理删除)。表格中 IoU 为事件帧数值。

### 3.1 MOT17(val_half 段,帧 301-600)

**switch 例 1 — V013(MOT17-10-DPM)第 345 帧:gt31 由 tracker 735 切换至 737**

- 事件帧上 737 的框与 gt31 框 IoU=0.64,准确覆盖目标;旧 tracker 735 的框仍在画面中(中心距目标约 30 px,IoU=0.04),正跟踪着附近另一名行人。
- **发生了什么:** 目标 31 身上的框从 735 的颜色变成了 737 的颜色;735 的颜色并未消失,还在画面其他位置移动 —— 一眼可见目标"换色"。
- 图片:`assets/examples/MOT17_switch_V013_f000345_f000330..360.png`

**switch 例 2 — V013 第 542 帧:gt46 由 tracker 751 切换至 753**

- 753 框与 gt46 框 IoU=0.81;旧 751 在目标左上方约 45 px(IoU=0.00)。
- **发生了什么:** 目标 46 换到 753,旧 751 继续跟踪它自己的目标,两个 ID 在画面中共存。
- 图片:`assets/examples/MOT17_switch_V013_f000542_f000527..557.png`

**reuse 例 1 — V019(MOT17-13-DPM)第 431 帧:tracker 996 从 gt70 复用到 gt142**

- 996 最后匹配 gt70 于第 423 帧(旧目标消失约 8 帧),第 431 帧起匹配 gt142(IoU=0.62)。
- **发生了什么:** 旧目标离开/丢失后,同一个 ID 996 被配置给了新出现的目标 142 —— 新物体顶替旧 ID 的典型形态。
- 图片:`assets/examples/MOT17_reuse_V019_f000431_f000416..446.png`

**reuse 例 2 — V013 第 564 帧:tracker 754 从 gt72 复用到 gt68**

- 754 最后匹配 gt72 于第 561 帧,仅 3 帧后(564)即匹配 gt68(IoU=0.77)。
- **发生了什么:** 极短间隔的 ID 顶替:目标 72 短暂丢失后,同一 ID 立即出现在紧邻的目标 68 上。
- 图片:`assets/examples/MOT17_reuse_V013_f000564_f000549..579.png`

### 3.2 MOT20(全帧)

**switch 例 1 — V001(MOT20-01)第 314 帧:gt77 由 tracker 71 切换至 121**(2026-08-16 重生成后该帧事件更新)

- 121 框与 gt77 框 IoU=0.87;旧 tracker 71 上一帧(313)还有输出,事件帧已无框(新初始化 tracker 直接接管)。
- **发生了什么:** 目标 77 在事件帧被新出现的 121 接管(71 同时在 313 帧丢失),拥挤人流中的冷启动接管代表。
- 图片:`assets/examples/MOT20_switch_V001_f000314_f000299..329.png`

**switch 例 2 — V002(MOT20-02)第 1495 帧:gt38 由 tracker 416 切换至 421**

- 421 框与 gt38 框 IoU=0.65;旧 416 在目标左下方(IoU=0.03)。
- **发生了什么:** 目标 38 换到 421,旧 416 继续跟踪其他目标;长序列(2,782 帧)中后期的一次典型切换。
- 图片:`assets/examples/MOT20_switch_V002_f001495_f001480..1510.png`

**reuse 例 1 — V004(MOT20-05)第 2052 帧:tracker 3445 从 gt939 复用到 gt467**

- 3445 最后匹配 gt939 于第 2051 帧,紧接着第 2052 帧匹配 gt467(IoU=0.73,目标在画面顶部)。
- **发生了什么:** 相隔仅 1 帧的 ID 复用 —— 上一帧还在目标 939 上的 ID,下一帧就顶替到了目标 467 上。
- 图片:`assets/examples/MOT20_reuse_V004_f002052_f002037..2067.png`

**reuse 例 2 — V004 第 2172 帧:tracker 3782 从 gt532 复用到 gt528**

- 3782 最后匹配 gt532 于第 2157 帧,第 2172 帧起匹配 gt528(IoU=0.62,目标在画面最左侧)。
- **发生了什么:** 旧目标消失 15 帧后,ID 3782 复用于左侧新目标 528。
- 图片:`assets/examples/MOT20_reuse_V004_f002172_f002157..2187.png`

### 3.3 SportsMOT(全帧)

**switch 例 1 — V019(篮球 v_2j7kLB-vEEk_c009)第 281 帧:gt9 由 tracker 951 切换至 949**

- 949 框与 gt9 框 IoU=0.86(极高);旧 951 在目标左侧约 95 px(IoU=0.10)。
- **发生了什么:** 快速跑动中目标 9 被 949 接管,旧 951 仍在左侧跟踪另一名运动员 —— 同色球衣运动员之间的典型 ID 交换。
- 图片:`assets/examples/SportsMOT_switch_V019_f000281_f000266..296.png`

**switch 例 2 — V040(篮球 v_HdiyOtliFiw_c003)第 709 帧:gt4 由 tracker 1746 切换至 1776**

- 1776 框与 gt4 框 IoU=0.73;旧 1746 在目标右下方(IoU=0.01)。
- **发生了什么:** 目标 4 换到 1776,旧 1746 跟踪着紧邻的另一名运动员。
- 图片:`assets/examples/SportsMOT_switch_V040_f000709_f000694..724.png`

**reuse 例 1 — V087(足球 v_ITo3sCnpw_k_c007)第 332 帧:tracker 4870 从 gt6 复用到 gt10**

- 4870 最后匹配 gt6 于第 323 帧,第 332 帧起匹配 gt10(IoU=0.53)。
- **发生了什么:** 旧目标 6 消失约 9 帧后,ID 4870 复用于中场的另一名球员 10。
- 图片:`assets/examples/SportsMOT_reuse_V087_f000332_f000317..347.png`

**reuse 例 2 — V052(足球 val/v_0kUtTtmLaJA_c004)第 17 帧:tracker 2411 从 gt3 复用到 gt4**

- 2411 最后匹配 gt3 于第 15 帧,第 17 帧起匹配 gt4(IoU=0.71)。
- **发生了什么:** 序列极早期(第 17 帧)的 ID 复用 —— tracker 刚初始化、尚未稳定时的典型现象,新目标 4 顶替了初始化目标 3。
- 图片:`assets/examples/SportsMOT_reuse_V052_f000017_f000002..032.png`

---

## 4. 视频可视化说明(含帧率假设)

**产物:** 115 个 mp4(21 + 4 + 90),路径 `YOLOX_outputs/{expn}/track_videos/Vxxx.mp4`(原始文件已于 2026-08-03 精简删除;报告包内保留 7 个典型序列于 `assets/videos/`,完整清单见 index.html 附录)。

| 项 | 说明 |
|----|------|
| 帧率 | **30 fps(假设)**——序列无 fps 元数据,统一 30;如需原速播放,按各序列真实帧率重新编码即可 |
| 帧范围 | MOT17:301-600(val_half 评测段);MOT20/SportsMOT:1..N(全帧) |
| 分辨率 | 原图分辨率不变(MOT17/MOT20 约 1920px 宽,MOT20-05 本地为 1654px 宽,SportsMOT 约 1280px 宽) |
| 颜色 | 同序列内 `hue(id) = (id × 137.5) mod 180`(黄金角),S=V=255;**同 ID 恒定同色,相邻 ID 色差 ≥ 42.5°** |
| 标注 | 框 + 黑底白字数字 ID(putText 不支持中文,只标数字) |
| 编码 | OpenCV `mp4v`,30 fps,无音频 |

**验收核对:**
- 帧数:视频帧数 = track txt 最小..最大帧跨度(含无检测的间隙帧,以空帧补齐保持连续);抽样 7 个序列(每数据集 2-3 个)视频帧数 == 跨度 == txt 唯一帧数,全部一致;V082 等个别序列存在 ≤3 帧无检测间隙(无目标,属正常)
- 框数:逐帧框数由 txt 对应行直接绘制,天然一致(抽样帧逐行核对通过)
- 画质抽查:框边像素与理论 HSV 颜色比对,80% 精确匹配(其余为拥挤场景重叠框覆盖边框所致),标签黑底条存在

---

## 5. 交付物清单

| 产物 | 路径 | 数量 |
|------|------|------|
| 跟踪可视化视频 | `assets/videos/`(7 个打包;完整 115 个原始文件已精简删除) | 7(打包) |
| 事件表 | `research/data/{MOT17|MOT20|SportsMOT}_events.csv`(列 `seq,frame,type,gt_id_old,gt_id_new,track_id,note`) | 3 份,1,083 / 2,621 / 3,329 行 |
| 逐序列统计 | `research/data/{数据集}_events_summary.csv` | 3 份 |
| 典型例子标注图 | `assets/examples/`(JPEG) | 372 张(12 例 × 31 帧) |
| 本报告 | `docs/id_switch_report.md`(本报告包) | 1 份 |
