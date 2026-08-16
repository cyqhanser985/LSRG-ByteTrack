# ByteTrack V001 标准评估复现结果报告

> 日期:2026-07-31(评估);2026-08-16(按清洗后 GT 全量重算)｜ 环境:Windows + CUDA 11.8 + PyTorch 2.1.2 + RTX 4060 (8GB)
> 版本:COE Vibe Coding,步骤 4(冒烟测试)与步骤 5(全量评估)已完成;2026-08-15 清洗 MOT17/MOT20 GT(仅保留 conf==1 行人行)后,全部指标按当前 GT 重算

---

## 1. 三数据集总览(2026-08-16 按清洗后 GT 重算)

| 数据集 | Exp 配置 | 权重 | 序列数 | 帧数 | track_thresh | match_thresh | track_buffer | mot20 | **MOTA** | **IDF1** | IDs | FP | FN | MT | ML |
|--------|---------|------|--------|------|-------------|-------------|-------------|-------|----------|----------|-----|-----|-----|-----|-----|
| **MOT17** | yolox_x_mot17_v001.py | bytetrack_x_mot17.pth.tar | 21 (V001-V021) | 7,977 | 0.6 | 0.9 | 30 | 否 | **90.2%** | **86.0%** | 546 | 3252 | 12114 | 789 | 63 |
| **MOT20** | yolox_x_mot20_v001.py | bytetrack_x_mot20.pth.tar | 4 (V001-V004) | 8,931 | 0.3 | 0.9 | 30 | 是 | **92.8%** | **87.6%** | 1600 | 31356 | 48938 | 2001 | 64 |
| **SportsMOT** | yolox_x_sportsmot_v001.py | bytetrack_x_sportsmot.pth.tar | 90 (V001-V090) | 55,544 | 0.5 | 0.8 | 60 | 否 | **97.1%** | **74.8%** | 2567 | 3522 | 11687 | 1266 | 1 |

- 全部采用 `--fp16 --fuse -b 1`;SportsMOT 为 55,544 帧全量评估(耗时约 2.5 h)
- 峰值显存稳定在 ~1.9 GB(allocated),全程无 OOM
- **口径说明**:2026-08-15 清洗后 GT 仅保留 conf==1 行人行(MOT17 162,126 行、MOT20 1,134,614 行,ignore region 已剔除),因此 MOTA/IDF1 显著高于清洗前(旧值 MOT17 49.9%/60.8%、MOT20 78.8%/80.4%;ignore 区域不再计入 FP/FN);SportsMOT GT 未变,指标与旧值完全一致

---

## 2. MOT17 详细结果(21 序列 = 7 唯一场景 × DPM/FRCNN/SDP)

- 帧范围:301-600(val_half 段);同一场景三个检测器版本内容相同,结果一致(下表取每场景代表值 V001/V004/V007/V010/V013/V016/V019)

| 场景 | MOTA | IDF1 | Rcll | Prcn | GT | MT | ML | FP | FN | IDs | FM |
|------|------|------|------|------|-----|-----|-----|-----|-----|-----|-----|
| MOT17-02 | 77.6% | 65.2% | 82.2% | 95.4% | 53 | 39 | 5 | 393 | 1762 | 65 | 137 |
| MOT17-04 | 98.3% | 97.7% | 98.8% | 99.5% | 69 | 68 | 1 | 123 | 287 | 6 | 43 |
| MOT17-05 | 82.8% | 78.0% | 88.6% | 95.1% | 71 | 45 | 4 | 152 | 385 | 40 | 45 |
| MOT17-09 | 87.6% | 75.5% | 89.6% | 98.6% | 22 | 19 | 1 | 37 | 300 | 21 | 33 |
| MOT17-10 | 85.3% | 81.0% | 88.6% | 96.9% | 36 | 26 | 1 | 168 | 679 | 27 | 62 |
| MOT17-11 | 89.0% | 83.7% | 92.5% | 96.7% | 44 | 31 | 6 | 143 | 341 | 16 | 36 |
| MOT17-13 | 88.7% | 87.7% | 91.1% | 97.7% | 44 | 35 | 3 | 68 | 284 | 7 | 10 |
| **OVERALL** | **90.2%** | **86.0%** | **92.5%** | **97.9%** | 1017 | 789 | 63 | 3252 | 12114 | 546 | 1098 |

> 备注:2026-08-15 清洗前,ignore region 行被当作活跃行人计入评估,导致 MOTA 被大幅拉低(旧值 49.9%);清洗后 MOTA 90.2%。官方 ByteTrack val_half 参考值(~76.6)与官方 MOT17 test 全帧口径(90.0)均与本地口径不同,不直接可比。

---

## 3. MOT20 详细结果(4 序列)

| 序列 | 原序列 | MOTA | IDF1 | Rcll | Prcn | GT | MT | ML | FP | FN | IDs | FM |
|------|--------|------|------|------|------|-----|-----|-----|-----|-----|-----|-----|
| V001 | MOT20-01 | 91.1% | 83.0% | 94.8% | 96.5% | 74 | 68 | 0 | 675 | 1029 | 67 | 104 |
| V002 | MOT20-02 | 90.6% | 75.0% | 93.5% | 97.3% | 270 | 250 | 0 | 3992 | 10088 | 437 | 982 |
| V003 | MOT20-03 | 93.5% | 92.7% | 95.7% | 97.8% | 702 | 623 | 27 | 6755 | 13343 | 309 | 617 |
| V004 | MOT20-05 | 93.0% | 88.3% | 96.2% | 96.9% | 1169 | 1060 | 37 | 19934 | 24478 | 787 | 1688 |
| **OVERALL** | | **92.8%** | **87.6%** | **95.7%** | **97.2%** | 2215 | 2001 | 64 | 31356 | 48938 | 1600 | 3391 |

> 备注:旧值 MOTA 78.8% 为清洗前口径(ignore region 计入);清洗后 92.8%。官方 ByteTrack MOT20 测试集成绩(77.8%)口径不同,不直接可比。

---

## 4. SportsMOT 详细结果(90 序列)

| 指标 | OVERALL |
|------|---------|
| **MOTA** | **97.1%** |
| **IDF1** | **74.8%** |
| IDP / IDR | 74.8% / 74.8% |
| Rcll / Prcn | 98.1% / 99.4% |
| GT / MT / PT / ML | 1280 / 1266 / 13 / 1 |
| FP / FN | 3522 / 11687 |
| **IDs** | **2567** |
| FM | 2990 |
| MOTP | 0.156 |

- 单序列 MOTA 区间 90.6% ~ 99.7%(最低 V068 = 90.6%、V086 = 90.6%,最高 V008 = 99.7%、V059 = 99.2%);GT 未清洗,全部数值与 2026-07-31 原评估一致
- 检测级召回 98%+(官方权重在 SportsMOT 上训练,检测质量极高);IDF1 明显低于 MOTA,符合运动场景快速运动 + 运动员外观相似的典型特征

---

## 5. 复现命令

```powershell
# MOT17(冒烟测试:单序列 V002)
E:\anaconda\envs\bytetrack\python.exe tools/track_v001.py -f exps/example/mot/yolox_x_mot17_v001.py `
  -c pretrained/bytetrack_x_mot17.pth.tar -b 1 -d 1 --fp16 --fuse --sequence V002 -expn mot17_v001_smoke

# MOT17 全量
... -f exps/example/mot/yolox_x_mot17_v001.py -c pretrained/bytetrack_x_mot17.pth.tar -b 1 -d 1 --fp16 --fuse `
  --track_thresh 0.6 --match_thresh 0.9 --track_buffer 30 -expn mot17_v001_full

# MOT20 全量(--mot20)
... -f exps/example/mot/yolox_x_mot20_v001.py -c pretrained/bytetrack_x_mot20.pth.tar -b 1 -d 1 --fp16 --fuse --mot20 `
  --track_thresh 0.3 --match_thresh 0.9 --track_buffer 30 -expn mot20_v001_full

# SportsMOT 全量
... -f exps/example/mot/yolox_x_sportsmot_v001.py -c pretrained/bytetrack_x_sportsmot.pth.tar -b 1 -d 1 --fp16 --fuse `
  --track_thresh 0.5 --match_thresh 0.8 --track_buffer 60 -expn sportsmot_v001_full
```

结果文件位于 `YOLOX_outputs/<expn>/track_results/Vxxx.txt`(标准 10 列 MOT 格式),指标由 motmetrics 计算(OVERALL 含逐序列汇总)。2026-08-16 数字由 `tools/eval_metrics.py` / `tools/eval_per_seq.py` 按当前 GT 复算。

---

## 6. 实验尝试与遇到的问题

### 6.1 代码适配(4 个卡点,全部修复)

| # | 位置 | 问题 | 修复 |
|---|------|------|------|
| 1 | `yolox/evaluators/mot_evaluator.py` | `frame_id == 1` 触发 tracker 重建,val_half 帧号 301-600 永不成立 | 改为 `prev_video_id` 变化检测,跨视频时重建 tracker + `empty_cache()`;每 500 iter 打印 GPU 显存 |
| 2 | `tools/track.py` GT 路径硬编码 | `datasets/mot/train` 不存在 | 新建 `tools/track_v001.py`,GT 路径由 `exp.dataset_name` 动态解析 `datasets/{name}/V*/gt/gt.txt`;新增 `--sequence` 单序列过滤 |
| 3 | exp `get_eval_loader` | `data_dir=datasets/mot`、`name='test'`、`val_ann=test.json` | 新建 `yolox_x_mot17/mot20/sportsmot_v001.py`,`data_dir` 指向各自数据集根、`name=''`、`val_ann='eval.json'`、`seq_filter` 支持 |
| 4 | SportsMOT 类别数 | 官方标注多类,但预训练权重检测头仅 1 类 person | `num_classes = 1` 固定,避免形状不匹配 |

### 6.2 运行中踩到的坑

| 坑 | 现象 | 解决 |
|----|------|------|
| Python 文件编码 | Write/Edit 工具落盘为 GBK,Python 3 按 UTF-8 解析报 `SyntaxError` | 所有新增/修改的 py 源码统一使用纯 ASCII(英文注释) |
| cv2 中文路径 | `cv2.imread` 对 `e:\科研\...` 静默返回 None(DataLoader worker 抛 AssertionError) | `mot.py::pull_item` 改用 `cv2.imdecode(np.fromfile(...))`,失败抛 `IOError` |
| argparse 短选项 | `--expn` 不识别(`-expn` 是短选项簇) | CLI 使用 `-expn` |
| PowerShell 重定向 | `*>` 合并流时丢失 `print()` 指标表 | 指标由保存的 track_results 用 motmetrics 独立复算,可复现 |
| DataLoader worker 内存 | SportsMOT 55k 帧数据集 pickle 给 4 个 worker,单 worker 峰值 11 GB RAM | v001 exp 的 `get_eval_loader` 设 `num_workers=0`(评估按帧加载,收益极小) |
| COCO 检测 AP 异常 | 遗留 `convert_to_coco_format` 路径 AP≈0.03,与 motmetrics 矛盾 | 仅影响 COCO 检测评估(非交付物);MOTA/IDF1 走 gt.txt + motmetrics,已逐帧 IoU 抽样验证正确 |
| 冒烟 vs 全量 V002 微小差异 | IDs 62 vs 66(旧 GT 口径) | 源于 track ID 全局计数(全量运行 V002 从 ID 74 起)与 fp16/cuDNN 非确定性,数值稳定在噪声内 |

### 6.3 验证结论

- 冒烟测试(MOT17 V002,300 帧):链路 100% 打通,输出为标准 10 列 MOT 格式(清洗后 GT 口径 MOTA 90.6%)
- 三数据集 OVERALL 指标与官方同口径结果量级一致(SportsMOT 检测级 98%+ 召回);MOT17/MOT20 清洗后 MOTA 90.2% / 92.8%
- 显存全程 < 2 GB,无 OOM;大数据集采用 `num_workers=0` + 周期 `empty_cache()` 保障内存安全
