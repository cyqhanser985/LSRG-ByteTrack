# ByteTrack 项目使用指南

> 本文档为历史归档版本，部分历史事件分析与验证脚本已下线；当前活跃研究请以 `research/README.md` 为准。

## 目录
1. [项目概述](#1-项目概述)
2. [环境配置](#2-环境配置)
3. [数据集准备](#3-数据集准备)
4. [模型下载](#4-模型下载)
5. [快速上手：视频/图像跟踪](#5-快速上手视频图像跟踪)
6. [训练自己的检测器](#6-训练自己的检测器)
7. [评估跟踪性能](#7-评估跟踪性能)
8. [核心参数调优](#8-核心参数调优)
9. [核心算法解析](#9-核心算法解析)
10. [常见问题](#10-常见问题)

---

## 1. 项目概述

ByteTrack 是基于 YOLOX 的多目标跟踪(MOT)算法。核心创新是 **BYTE 策略**：同时利用高分和低分检测框进行双阶段匹配，有效解决遮挡场景下的轨迹断裂问题。

```
整体流程: 图像 → YOLOX检测 → BYTETracker跟踪 → MOT格式输出
```

**目录结构：**

```
ByteTrack/
├── yolox/                    # YOLOX核心框架
│   ├── tracker/              # 跟踪核心模块
│   │   ├── byte_tracker.py   # BYTETracker主算法
│   │   ├── kalman_filter.py  # 卡尔曼滤波器
│   │   ├── matching.py       # 匈牙利匹配算法
│   │   └── basetrack.py      # 轨迹基类
│   ├── models/               # YOLOX模型定义
│   ├── evaluators/           # 评估器(MOTEvaluator)
│   └── data/datasets/        # MOTDataset数据加载
├── tools/                    # 工具脚本
│   ├── track_v001.py         # V001批量评估跟踪（标准入口）
│   ├── clean_mot_gt.py       # MOT17/MOT20 gt.txt 清洗（过滤 ignore region）
│   ├── build_ids_events.py   # IDS 事件表重生成（motmetrics 时序匹配）
│   ├── build_event_counts.py # 序列级分类计数锚点重建
│   └── plot_results.py       # 结果绘图
├── exps/example/mot/         # 实验配置（v001系列）
├── docs/                     # 项目文档（开发进度/经验/指南）
└── datasets/                 # 数据集目录（MOT17/MOT20/SportsMOT）
```

---

## 2. 环境配置

### 2.1 安装依赖

```bash
# 创建环境 (Python 3.8+)
conda create -n bytetrack python=3.8
conda activate bytetrack

# 安装PyTorch (根据CUDA版本选择)
pip install torch==1.10.0+cu111 torchvision==0.11.0 -f https://download.pytorch.org/whl/torch_stable.html

# 安装项目依赖
cd ByteTrack
pip install -r requirements.txt
pip install cython
pip install 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'
pip install cython_bbox
```

### 2.2 编译安装

```bash
python setup.py develop
```

---

## 3. 数据集准备

### 3.1 数据集概览

| 数据集 | 内容 | 类别 | 当前状态 |
|--------|------|------|----------|
| MOT17 | 行人跟踪 | person | 已清洗（gt.txt 仅保留 conf==1 行人） |
| MOT20 | 极度拥挤场景行人 | person | 已清洗（gt.txt 仅保留 conf==1 行人） |
| SportsMOT | 体育运动跟踪 | person/ball等 | 原始标注即干净，直接可用 |

**验证过的图像数量：**
- MOT17 train: 21序列，600~1500张/序列
- MOT17 test: 21序列，450~1500张/序列
- MOT20 train: 4序列(MOT20-01/02/03/05)，429~3315张/序列
- MOT20 test: 4序列(MOT20-04/06/07/08)，585~2080张/序列

### 3.2 MOT17/MOT20 GT 清洗（ignore region 过滤）

```bash
# 从 annotations/eval.json 重建各序列 gt.txt，仅保留 conf==1 的活跃行人
# （移除 MOT17 约 44.7% / MOT20 约 15.1% 的 conf==0 ignore region）:
python tools/clean_mot_gt.py                  # MOT17 + MOT20
python tools/clean_mot_gt.py --datasets mot20 # 仅 MOT20
```

### 3.3 SportsMOT 数据集 (已就绪)

```bash
# 标注: datasets/SportsMOT/annotations/
# 图像: datasets/SportsMOT/V{xxx}/img1/
```

### 3.4 数据集完整性

GT 加载器（`research/diag_exp/bad_cases/badcase_common.py::load_gt_frames`）
内置防御过滤（conf<1 / 无效框 / 越界框 / 超大框 / 静态目标），
加载时自动剔除异常标注；`tools/track_v001.py` 为评估标准入口。

---

## 4. 模型下载

下载预训练模型到 `pretrained/` 目录：

| 模型 | 用途 | 下载链接 |
|------|------|----------|
| bytetrack_x_mot17 | MOT17行人跟踪 | [Google Drive](https://drive.google.com/file/d/1P4mY0Yyd3PPTybgZkjMYhFri88nTmJX5) |
| bytetrack_x_mot20 | MOT20拥挤场景 | [Google Drive](https://drive.google.com/file/d/1HX2_JpMOjO5e05L1LL2eKCs4CLkf7mwT) |
| bytetrack_s_mot17 | 轻量版(速度快) | 见官方仓库 |

```bash
mkdir pretrained
# 将下载的 .pth.tar 文件放入 pretrained/ 目录
```

---

## 5. 快速上手：视频/图像跟踪

### 5.1 单序列跟踪评估

```bash
python tools/track_v001.py \
    -f exps/example/mot/yolox_x_mot17_v001.py \
    -c pretrained/bytetrack_x_mot17.pth.tar \
    -expn mot17_v001_test \
    --sequence V002 \
    --device gpu
```

### 5.2 全量评估（全部序列）

```bash
python tools/track_v001.py \
    -f exps/example/mot/yolox_x_mot17_v001.py \
    -c pretrained/bytetrack_x_mot17.pth.tar \
    -expn mot17_v001_full -b 1 -d 1 --fp16 --fuse
```

### 5.3 关键参数说明

```bash
python tools/track_v001.py \
    -f exps/example/mot/yolox_x_mot17_v001.py \  # 实验配置文件
    -c pretrained/bytetrack_x_mot17.pth.tar \    # 模型权重路径
    -expn mot17_v001_test \          # 输出目录名(YOLOX_outputs/<expn>)
    --sequence V002 \                # 只跑指定序列(不填=全量)
    --device gpu \                   # gpu 或 cpu
    --track_thresh 0.5 \             # 跟踪置信度阈值(0.3~0.7)
    --match_thresh 0.8 \             # 匹配IoU阈值(0.7~0.95)
    --track_buffer 30 \              # 丢失轨迹保留帧数(14~60)
    --mot20 \                        # MOT20模式(使用纯IoU匹配)
    --fp16 --fuse                    # 半精度+模型融合
```

### 5.4 MOT20 拥挤场景跟踪

```bash
python tools/track_v001.py \
    -f exps/example/mot/yolox_x_mot20_v001.py \
    -c pretrained/bytetrack_x_mot20.pth.tar \
    -expn mot20_v001_test \
    --mot20 \
    --track_thresh 0.3 \
    --device gpu
```

### 5.5 输出格式

跟踪结果保存为MOT Challenge标准格式：
```
帧号, 目标ID, x, y, w, h, 置信度, -1, -1, -1
```

---

## 6. 训练自己的检测器

> ⚠️ 本项目已移除训练脚本（`tools/train.py` 等），聚焦评估与分析；
> 如需训练请从 [官方仓库](https://github.com/ifzhang/ByteTrack) 恢复 `tools/train.py`，以下示例仅供参考。

### 6.1 创建实验配置

参考 `exps/example/mot/yolox_x_mix_det.py` 创建自己的配置文件：

```python
# 示例（num_classes 等按实际任务修改）
class Exp(MyExp):
    def __init__(self):
        super().__init__()
        self.num_classes = 1          # 类别数
        self.train_ann = "train.json" # 训练集标注文件
        self.val_ann = "val.json"     # 验证集标注文件
        self.input_size = (896, 1600) # 输入尺寸
        self.test_size = (896, 1600)
        self.max_epoch = 80           # 最大训练轮数
        self.test_conf = 0.001        # 测试置信度阈值
        self.nmsthre = 0.7            # NMS阈值
```

### 6.2 SportsMOT 训练示例

先创建配置文件，修改 `data_dir` 指向SportsMOT：

```python
# exps/example/mot/yolox_x_sportsmot.py
self.num_classes = 4  # person, ball, goalkeeper, referee

# get_data_loader中:
data_dir=os.path.join(get_yolox_datadir(), "SportsMOT"),
json_file=self.train_ann,
name='train',   # 图像在 train/ 子目录下
```

训练命令：
```bash
python tools/train.py -f exps/example/mot/yolox_x_sportsmot.py -d 1 -b 16 --fp16
```

### 6.4 混合数据集训练

将多数据集合并训练：
1. 将所有COCO JSON合并为一个
2. 统一符号链接图像目录
3. 修改配置文件中的 `data_dir` 和 `train_ann`

---

## 7. 评估跟踪性能

### 7.1 运行评估

```bash
# MOT17评估
python tools/track_v001.py \
    -f exps/example/mot/yolox_x_mot17_v001.py \
    -c pretrained/bytetrack_x_mot17.pth.tar \
    -expn mot17_v001_full -b 1 -d 1 --fp16 --fuse

# MOT20评估
python tools/track_v001.py \
    -f exps/example/mot/yolox_x_mot20_v001.py \
    -c pretrained/bytetrack_x_mot20.pth.tar \
    -expn mot20_v001_full -b 1 -d 1 --fp16 --fuse --mot20

# SportsMOT评估
python tools/track_v001.py \
    -f exps/example/mot/yolox_x_sportsmot_v001.py \
    -c pretrained/bytetrack_x_sportsmot.pth.tar \
    -expn sportsmot_v001_full -b 1 -d 1 --fp16 --fuse
```

### 7.2 评估指标说明

| 指标 | 全称 | 含义 | 越高越好 |
|------|------|------|----------|
| MOTA | Multi-Object Tracking Accuracy | 综合跟踪精度 | ✓ |
| MOTP | Multi-Object Tracking Precision | 定位精度 | ✓ |
| IDF1 | ID F1 Score | ID保持能力 | ✓ |
| MT | Mostly Tracked | 大部分被跟踪的比例 | ✓ |
| ML | Mostly Lost | 大部分丢失的比例 | ✗ |
| FP | False Positives | 误检数 | ✗ |
| FN | False Negatives | 漏检数 | ✗ |
| IDSw | ID Switches | ID切换次数 | ✗ |
| Frag | Fragmentations | 轨迹断裂次数 | ✗ |

### 7.3 对比其他跟踪器

SORT / DeepSORT / MOTDT 对比脚本已随官方冗余移除，如需对比请从
[官方仓库](https://github.com/ifzhang/ByteTrack) 获取 `tools/track_sort.py` / `tools/track_deepsort.py` / `tools/track_motdt.py`。

---

## 8. 核心参数调优

### 8.1 BYTETracker 参数

| 参数 | 默认值 | 范围 | 调优策略 |
|------|--------|------|----------|
| `track_thresh` | 0.6 | 0.3-0.7 | 拥挤场景降低(保留更多检测)；简单场景提高(减少误检) |
| `match_thresh` | 0.9 | 0.7-0.95 | 低帧率场景降低；高质量检测可提高 |
| `track_buffer` | 30 | 14-60 | 快速移动/严重遮挡减小；稳定场景增大 |
| `min_box_area` | 100 | 10-200 | 小目标场景减小；大目标场景增大 |

### 8.2 场景化推荐

```
┌─────────────────┬──────────┬──────────┬───────────┐
│ 场景             │ track_thresh │ match_thresh│ track_buffer│
├─────────────────┼──────────┼──────────┼───────────┤
│ 行人跟踪(MOT17)  │ 0.6      │ 0.9      │ 30        │
│ 拥挤场景(MOT20)  │ 0.3      │ 0.9      │ 30        │
│ 体育运动         │ 0.5      │ 0.8      │ 60        │
│ 高帧率视频       │ 0.6      │ 0.9      │ 30        │
│ 低帧率视频       │ 0.4      │ 0.7      │ 60        │
└─────────────────┴──────────┴──────────┴───────────┘
```

### 8.3 视频级自适应参数

MOTEvaluator 内置了对特定视频的参数微调（参考 [mot_evaluator.py 第138-157行](file:///e:/科研/ByteTrack/yolox/evaluators/mot_evaluator.py#L138-L157)）：

```python
# 快速移动序列: 减少track_buffer
if video_name in ['MOT17-05-FRCNN', 'MOT17-06-FRCNN']:
    track_buffer = 14

# 极度拥挤序列: 降低track_thresh
if video_name in ['MOT20-06', 'MOT20-08']:
    track_thresh = 0.3
```

---

## 9. 核心算法解析

### 9.1 BYTE策略：ByteTrack的核心创新

**SORT问题:** 只保留高分检测框，低分框直接丢弃。遮挡场景下检测分数下降 → 轨迹断裂。

**BYTE方案:**

```
一帧检测框:
├── 高分框 (score > track_thresh)      → Step 2: 优先与现有轨迹匹配
└── 低分框 (0.1 < score < track_thresh) → Step 3: 恢复被遮挡/模糊的目标

Step 1: 将检测框分为高分组和低分组
Step 2: 高分框 + 融合Score的IoU + 匈牙利匹配 → 主要关联
Step 3: 未匹配轨迹 + 纯IoU + 低分框 → 恢复关联
Step 4: 未确认轨迹与剩余高分框匹配
Step 5: 初始化新轨迹 / 清理过期轨迹
```

### 9.2 卡尔曼滤波器 (8维状态)

```
状态向量: [x, y, a, h, vx, vy, va, vh]
         (中心x, 中心y, 宽高比, 高度, 以及各自速度)

观测向量: [x, y, a, h]  (直接从检测框测量)

噪声参数:
  std_position = 1/20 × height   (位置噪声与目标高度成正比)
  std_velocity = 1/160 × height
```

### 9.3 匈牙利匹配算法

使用 `lapjv` (Jonker-Volgenant算法) 求解线性分配问题：

```python
cost_matrix = 1 - IoU(轨迹预测框, 检测框)

# MOT17模式: Score融合
cost_matrix = 1 - (1 - cost) * detection_scores

# MOT20模式: 纯IoU(不融合score，避免拥挤场景中score不可靠)
# 不做融合

matches, unmatched_tracks, unmatched_detections = linear_assignment(cost_matrix, threshold)
```

### 9.4 四大关键模块调用关系

```
BYTETracker.update(outputs, img_info, img_size)
├── 1. 检测框坐标缩放 → scores = obj_score * cls_score
├── 2. STrack.multi_predict(tracked + lost)
│        └── KalmanFilter.multi_predict()  ← 批量卡尔曼预测
├── 3. matching.iou_distance() → matching.linear_assignment()
│        └── lap.lapjv()  ← 匈牙利算法
├── 4. track.update(det) / track.re_activate(det)
│        └── KalmanFilter.update()  ← 卡尔曼更新
├── 5. track.activate(kf, frame_id)
│        └── KalmanFilter.initiate()  ← 初始化新轨迹
└── 6. 状态清理: mark_lost() / mark_removed()
```
---

## 附录: 数据集状态速查

| 数据集 | 标注格式 | 图像状态 | 可用性 |
|--------|----------|----------|--------|
| MOT17 | MOT格式(已清洗 conf==1) | 完整(21序列) | 直接可用 |
| MOT20 | MOT格式(已清洗 conf==1) | 完整(4序列) | 直接可用 |
| SportsMOT | MOT格式(原始即干净) | train/val/test已解压 | 直接可用 |

---

> **相关文件路径:**
> - 核心跟踪器: [yolox/tracker/byte_tracker.py](file:///e:/科研/ByteTrack/yolox/tracker/byte_tracker.py)
> - 卡尔曼滤波: [yolox/tracker/kalman_filter.py](file:///e:/科研/ByteTrack/yolox/tracker/kalman_filter.py)
> - 匹配算法: [yolox/tracker/matching.py](file:///e:/科研/ByteTrack/yolox/tracker/matching.py)
> - 评估器: [yolox/evaluators/mot_evaluator.py](file:///e:/科研/ByteTrack/yolox/evaluators/mot_evaluator.py)
> - 数据加载: [yolox/data/datasets/mot.py](file:///e:/科研/ByteTrack/yolox/data/datasets/mot.py)
> - GT清洗: [tools/clean_mot_gt.py](file:///e:/科研/ByteTrack/tools/clean_mot_gt.py)
> - 评估入口: [tools/track_v001.py](file:///e:/科研/ByteTrack/tools/track_v001.py)
> - 实验配置: [exps/example/mot/yolox_x_mot17_v001.py](file:///e:/科研/ByteTrack/exps/example/mot/yolox_x_mot17_v001.py)
