# ByteTrack 项目经验沉淀

> COE 知识库 — 只保留会导致静默失败或数小时 debug 的核心经验

---

## 1. 数据格式标准

### 1.1 V001 目录结构（唯一约定）

```
datasets/{数据集名}/
  V001/
    img1/000001.jpg ...
    gt/gt.txt              # MOT 10列格式
  V002/ ...
  annotations/             # COCO JSON
  mapping.txt              # V001 → 原始序列名
```

- file_name 统一 `Vxxx/img1/000xxx.jpg`，COCO JSON 必须同步
- gt.txt 后四列统一 `1,-1,-1,-1`（无论原始值是多少）
- 序列按字母序编号，`mapping.txt` 记录映射

### 1.2 COCO JSON 必需字段

```json
{
  "images": [{"id", "file_name", "frame_id", "video_id", "width", "height"}],
  "annotations": [{"id", "image_id", "category_id", "track_id", "bbox": [x,y,w,h], "area"}],
  "videos": [{"id", "file_name"}],
  "categories": [{"id": 1, "name": "person"}]
}
```

**缺一不可：** `frame_id`（帧号 1-based）、`video_id`（序列编号）、`track_id`、`bbox` 必须是 `[x,y,w,h]`（非 xyxy）。

---

## 2. 致命陷阱

### 2.1 中文路径 = 静默失败

**Python 脚本中硬编码中文路径**（即使加了 `# -*- coding: utf-8 -*-`）→ 如果文件被 IDE 存为 GBK 编码，Python 解析器直接 `SyntaxError`。**只用 `os.getcwd()` 派生路径。**

**cv2.imread 中文路径** → 静默返回 `None`，后续 `assert img is not None` 才暴露。**必须用字节流：**

```python
img = cv2.imdecode(np.fromfile(str(img_path), dtype=np.uint8), cv2.IMREAD_COLOR)
```

### 2.2 Conda + PowerShell：activate 是假的

`conda activate bytetrack` 在 PowerShell 中执行后不持久——后续 `pip install` 实际跑在 base 环境。**直接使用 Python 全路径：**

```powershell
E:\anaconda\envs\bytetrack\python.exe -m pip install ...
E:\anaconda\envs\bytetrack\python.exe tools/track_v001.py ...
```

**版本冲突链（Windows, CUDA 11.8, Python 3.8）：**

| 坑 | 表现 | 解决 |
|----|------|------|
| 清华镜像无旧 torch GPU wheel | pip 只列 2.6.0+ | 官方 URL 直接装 `torch==2.1.2+cu118` |
| onnx 1.8.1 + protobuf ≥ 5 | `Descriptors cannot be created` | `protobuf==3.20.3` |
| cocoapi git 源码编译 | MSVC 不支持 `/Wno-cpp` | `pycocotools==2.0.7` 预编译 wheel |
| numpy ≥ 1.24 | `np.float` 被移除 | `numpy==1.23.5` |

### 2.3 模型权重验证

仅 `torch.load` 不足以确认权重可用。必须 **strict load + 前向推理**：

```python
model = exp.get_model()                         # 从 exp 构建正确结构
ckpt = torch.load(ckpt_path, map_location='cpu')
model.load_state_dict(ckpt["model"], strict=True)  # 形状不匹配立即报错
model.eval(); model(torch.randn(1,3,640,640))      # 前向验证
```

- MOT17/MOT20/SportsMOT 三个权重均为 YOLOX-X（depth=1.33, width=1.25），**全是 1 类 person**（SportsMOT 预训练权重的检测头仅 1 类，即使官方标注含 4 类）
- 三个文件体积相同（~793MB）但 SHA256 不同 → 非同一文件拷贝，不可混用

### 2.4 评估代码与 V001 结构不兼容（四处硬编码）

官方 `tools/track.py`（已随官方冗余移除）不能直接在 V001 结构上运行，必须修复以下四点（最终方案：新建 `tools/track_v001.py` 整合全部修复）：

**① exp 的 `get_eval_loader`：** `data_dir` 指向 `datasets/mot`，`name='test'`，`val_ann='test.json'`。改为：
```python
data_dir=os.path.join(get_yolox_datadir(), "MOT17")  # 数据集根
name=''           # file_name 已是 V001/img1/xxx.jpg 完整路径
val_ann="eval.json"
```

**② track.py GT glob 硬编码：** `datasets/mot/train/*/gt/gt*.txt` 不存在。改为 `datasets/MOT17/V*/gt/gt.txt`（MOT20/SportsMOT 同理）。

**③ mot_evaluator.py 视频切换：** 用 `frame_id == 1` 判断新视频——val_half 帧号是 301-600，永不为 1。**必须改用 `video_id` 变化检测。**

**④ gt_type 后缀：** 原代码按 `val_ann == 'val_half.json'` 给 gt 文件名加 `_val_half` 后缀，V001 下不适用。

### 2.5 Python 源码编码：非 ASCII 注释 = SyntaxError

Write/Edit 工具在 Windows 落盘可能为 GBK；Python 3 默认按 UTF-8 解析，含中文注释的 py 文件直接 `SyntaxError`（非静默但同样烧时间）。**所有新增/修改的 py 源码统一纯 ASCII（英文注释），中文只写进 .md 文档。**

### 2.6 DataLoader worker 内存暴涨

大数据集（SportsMOT 55k 帧）pickle 给多个 worker → 单 worker 峰值可达 11 GB RAM。评估模式按帧加载，多 worker 收益极小：**eval loader 固定 `num_workers=0`**。

### 2.7 两个 Windows/argparse 小坑

| 坑 | 表现 | 解决 |
|----|------|------|
| argparse 短选项簇 | `--expn` 不存在，报 unrecognized | CLI 一律写 `-expn` |
| PowerShell `*>` 合并流 | `print()` 指标表丢失 | 指标从保存的 track_results 用 motmetrics 独立复算，不依赖 stdout |

---

## 3. 算法核心

### 3.1 BYTETracker.update() 输入契约

```python
output_results: np.ndarray   # (N,5) [x1,y1,x2,y2,score] 或 (N,7) [...,obj,cls,cls_id]
# 像素绝对坐标，score 0~1
```

### 3.2 场景参数速查

| 场景 | track_thresh | match_thresh | track_buffer | mot20 |
|------|-------------|-------------|-------------|-------|
| MOT17 | 0.6 | 0.9 | 30 | 否 |
| MOT20 | 0.3 | 0.9 | 30 | 是 |
| SportsMOT | 0.5 | 0.8 | 60 | 否 |

`mot20=True` 时匹配用纯 IoU（不融合 score，拥挤场景 score 不可靠）。

---

## 4. 操作规范

### 4.1 eval.json 构建

本地帧切片后原始 JSON 不可用，必须重建 `annotations/eval.json`：

1. file_name 重映射：原始序列名 → `Vxxx/img1/xxx.jpg`
2. video_id 重映射：`int(v_name[1:])`
3. 帧切片：只保留 `frame_id ≤ max(gt.txt 帧号)`
4. id 重编号：image_id / annotation id 从 1 连续
5. categories 统一 `[{"id":1,"name":"person"}]`
6. **必须逐视频验证帧范围与 gt.txt 一致，不一致立即 `sys.exit(1)`**

**坑：** 原始 JSON 可能跨 train.json / val.json 两个文件 → 按 `mapping.txt` 在两者中搜索。SportsMOT mapping 值为 `train/v_xxx` 格式，需 `split("/")[-1]` 去前缀。

### 4.2 权重命名

| 原始名 | 标准化名 |
|--------|---------|
| `yolox_x_MOT17_test.pt` | `pretrained/bytetrack_x_mot17.pth.tar` |
| `yolox_x_MOT20_test.pt` | `pretrained/bytetrack_x_mot20.pth.tar` |
| `yolox_x_sportsmot.pt` | `pretrained/bytetrack_x_sportsmot.pth.tar` |

后缀不影响 `torch.load`，统一命名便于管理。

### 4.3 COCO → gt.txt 转换

```python
seq = img['file_name'].split('/')[0]  # V001
line = f"{frame_id},{track_id},{x:.1f},{y:.1f},{w:.1f},{h:.1f},1,-1,-1,-1\n"
```

- gt 的 conf 必须强制为 1（无论 COCO 中原始值）
- 同序列内所有帧尺寸相同，只读首帧取 `(h,w)`

---

## 5. 结果解读与口径（防误判）

### 5.1 三数据集全量基线（2026-07-31 实测，RTX 4060）

| 数据集 | 序列/帧 | MOTA | IDF1 | IDs | 与官方对比 |
|--------|---------|------|------|-----|-----------|
| MOT17（test 权重, val_half 段） | 21 / 7,977 | 49.9% | 60.8% | 549 | 官方 76.6% 是 train 全帧口径，**不可直接比** |
| MOT20 | 4 / 8,931 | 78.8% | 80.4% | 1,640 | ≈ 官方 77.8%，健康 |
| SportsMOT | 90 / 55,544 | 97.1% | 74.8% | 2,567 | 检测级召回 98%+，官方权重在本数据集训练 |

**指标与论文/官方对不上 ≠ 链路 Bug。** 先核对四类口径差异再排查：
1. **权重口径**：MOT17 官方 90.0/76.6 用 train 模型全帧评测；`bytetrack_x_mot17.pth.tar` 是 **test 模型 + val_half 段（帧 301-600）**，检测召回仅 ~51%，MOTA 49.9% 属正常
2. **评测段**：val_half vs 全帧，帧号起点 301 而非 1（还连带触发 2.4③ 的 tracker 重建 bug）
3. **检测器冗余**：MOT17 每场景 DPM/FRCNN/SDP 三份序列内容相同，OVERALL 被拖低
4. **序列集**：SportsMOT 本地 240 个 V 序列仅 90 个有 gt，评估只跑 90

验证手段：motmetrics 从 `track_results/Vxxx.txt` 独立复算 + 逐帧 IoU 抽样检测召回，二者一致才可信。

### 5.2 其他口径事实

- **冒烟 vs 全量同序列差异属噪声**：track ID 全局计数不同（全量时 V002 从 ID 74 起）+ fp16/cuDNN 非确定性，MOTA 差 0.2pp、IDs 差几条均在噪声内
- **COCO 检测 AP 是遗留死路**：`convert_to_coco_format` 路径 AP≈0.03 与 MOTA 矛盾，仅影响 COCO 检测评估（非交付物）；MOT 指标一律以 gt.txt + motmetrics 为准
- SportsMOT 的 MOTA 极高而 IDF1 偏低，是"快速运动 + 运动员外观相似"的典型形态，不是跟踪异常

---

## 6. 可视化与 ID 事件分析（步骤 6 新坑）

### 6.1 motmetrics events 结构（1.4.0 实测）

- `acc.events` 的索引是 MultiIndex（FrameId, Event），**FrameId 不是列**，`reset_index()` 后才可当列用；实际列只有 `Type/OId/HId/D`
- 事件类型 8 种：RAW/FP/MISS/SWITCH/MATCH/TRANSFER/ASCEND/MIGRATE
- **SWITCH 与 TRANSFER 语义（mot.py:263 附近）**：SWITCH = GT 对象换了 tracker（对象视角，不要求旧 tracker 还活着）；TRANSFER = tracker 换了匹配对象（tracker 视角，且**只在匈牙利分配阶段产生，不覆盖 carry-forward 复连**）
- 按 MATCH 行扫描"tracker 换目标"与 TRANSFER 行数有差异（两个方向都存在：carry-forward 使 MATCH 视角偏多、SWITCH 行上的切换使 MATCH 视角偏少），净效应随数据集而定，报告里必须写清口径

### 6.2 CSV 字段内禁止逗号

note 字段写进 CSV 时若含逗号（如 `"(last frame 306), now gt 76"`）会破坏列结构，下游按 `split(',')` 解析直接错位。**字段内容一律不含逗号**（用空格分隔）。

### 6.3 Git Bash 调用 python 全路径必须正斜杠

`E:\anaconda\envs\bytetrack\python.exe` 在 bash 里反斜杠被吞成 `E:anacondaenvsbytetrackpython.exe`（command not found）。**用 `E:/anaconda/envs/bytetrack/python.exe`**。

### 6.4 视频 ID 着色：排名均匀色相在 ID 多时不可区分

按 ID 排名均匀分配色相，56 个 ID 时相邻色差仅 3.2°，肉眼难分。**改用黄金角 `hue = (id × 137.5) mod 180`**：整圈均匀分布 + 相邻数值 ID 色差 ≥42.5°，同 ID 恒定同色。

### 6.5 视频帧范围口径

视频帧数 = track txt 的 min..max 帧（**含无检测的间隙帧**，空帧补齐保持连续）；MOT17 为 301-600（val_half 段），MOT20/SportsMOT 全帧。帧率序列无元数据，统一假设 30fps。

### 6.6 CPU 争抢拖慢后台任务

OpenCV 渲染（多线程）与 motmetrics LAP 计算同时跑，事件脚本被拖慢 3-5 倍（MOT17 单数据集 10+ 分钟）。**后台任务错峰运行**。

---

## 7. 白布可视化（步骤 7 新坑）

### 7.1 高饱和颜色在纯白底上对比度低 → 黑描边

**现象：** 黄金角着色（HSV S=V=255）的纯蓝/蓝紫色框（V013 id732、V019 id948）亮度 ~226，与纯白底亮度差仅 29.1，白布上几乎不可见；两序列实测均存在。

**原因：** 高饱和 ≠ 与白底可分。白底对比看**亮度差**（Rec.601 亮度 Y = 0.114B + 0.587G + 0.299R，BGR 顺序）；纯蓝/黄/青等单双通道满值颜色在 255 底上只剩一个通道的差异。

**解决：** **黑描边方案**——先画 line_w+2 黑色粗线再画 line_w 彩线，任何颜色在白底上保持可读，ID 数字仍为彩色（黑底条内不受影响）。白底画面 mp4v 压缩率高（327+1825 帧两视频共 ~35MB）。**决策依据必须程序化**：验证脚本输出每 ID 亮度差，取"最亮（最接近白底）"为最差，而非直觉上的"最鲜艳"。

### 7.2 验证脚本自身两个 bug（已修复）

**① 对比度"最差 ID"排序两错。** 第一版按 id 排序输出（误当按 gap 排）；第二版按亮度升序（拿到最暗而非最亮）。"最差" = 与白底最接近 = 最亮，排序键必须明确。

**② 线宽 1 时边框采样点落空。** 采样偏移 `max(1, line_w//2)` 在 1280×720（line_w=1）序列上落在框内空白，误报整帧所有边框"NOT found"；1920×1080（line_w=2）序列未暴露。修复：`off = line_w // 2`（线带中心）。**教训：验证几何必须对参数化线宽健壮，低分辨率序列也要跑。**

### 7.3 MOT17 val_half 帧范围因序列而异

id_switch_report 概括的"301-600"只对 V001 等成立：**V013 实际为 328-654**（track txt 与 gt.txt 双印证）。涉及 MOT17 序列的帧范围一律以 track txt min..max 为准，报告口径要写实际值。

---

## 8. 事件指标统计与轨迹图（步骤 8 新坑）

### 8.1 parse_track 缓存键必须含数据集名

按 seq 缓存 `parse_track` 结果时，**MOT17/MOT20/SportsMOT 都有 V001~V0xx**,只用 seq 作键会让后处理的数据集命中前一个数据集的缓存（实测:MOT20 的选例打分全部误用了 MOT17 V001 的数据,before/after 帧数完全错误）。**缓存键一律 `(ds, seq)`**;选例打分打印时必带 seq（MOT17 有 DPM/FRCNN/SDP 三副本序列,V004/V005/V006 内容相同,不带 seq 的候选表会被副本污染）。

### 8.2 密集轨迹图:圆点/描边互相覆盖把彩线盖没

目标位移小(人群/静止)时,523 帧轨迹中心点挤在 ~25px 范围内:半径 3+1 的圆点黑描边互相重叠成大片黑色,把彩线盖得只剩几十像素,图不可读。**解决:① 画法——圆点 r=2 描边 1、彩线 4px/黑底 6px;② 选例——打分必须加"路径位移"(中心点轨迹总长),位移量级是轨迹图可读性的第一要素**。仅按帧数选例会选到视觉不可读的例子。

### 8.3 帧号标签白底框会盖住轨迹点

帧号标注若只避开标题条,白底框可能落在轨迹上(实测 f2052 标签框盖住 2043 帧的轨迹点,中心变白,验证暴露)。**候选偏移加大(±36px)且检查候选框与全部轨迹中心(±8px margin)不重叠**——"放切换点旁空白处"要程序化落实。

### 8.4 起点方块(黑外白内)盖住起点附近密集帧

方块画在轨迹之上时,白色内芯(8×8)盖住起点附近多帧的圆点(实测第二帧中心变白)。**标记方块先画、轨迹后画**(标记在轨迹下层);终点方块白外黑内被轨迹线帽覆盖后中心仍为黑,不受影响。

### 8.5 标题条/边界矩形盖四角 → 验证"四角=255"失败

标题条与边界灰矩形若从 (0,0) 画起,会盖住四角像素,程序化验证"四角 == 纯白"必然失败。**标题条与边界框内缩 2px**((2,2)-(w-3,h-3)),视觉无损、四角保持纯白。

### 8.6 同一 (seq, frame) 可有多条同型事件

事件 CSV 中同一序列同一帧可能存在多条 switch(不同 tracker 分别接管不同目标)。**按 (seq, frame, track_id) 三键精确定位事件**,只按 seq+frame 匹配会拿到第一条(实测选例 tid=30 被匹配成 tid=25)。

---
> **版本:** 2026-08-02 | **范围:** 数据标准 + 致命陷阱 + 算法核心 + 操作规范 + 结果解读 + 可视化与 ID 事件分析（步骤 6）+ 白布可视化（步骤 7）+ 事件指标统计与轨迹图（步骤 8 完成）
