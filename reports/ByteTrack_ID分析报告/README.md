# ByteTrack 多目标跟踪 ID 切换分析报告(HTML 版)

本文件夹为 ByteTrack 在 MOT17 / MOT20 / SportsMOT 三个数据集上的 **ID 切换(switch)与 ID 复用(reuse)分析**的自包含展示报告,
由项目全量实验产物自动打包生成,用于汇报 / 展示。

## 打开方式

- 双击 `index.html`,用任意现代浏览器打开即可(Chrome / Edge 推荐)。
- **完全离线可用**,无任何外部依赖、无需联网、无需服务器。

## 报告结构(左侧导航 6 页)

| 页 | 内容 |
|----|------|
| 01 总览 | 任务完成情况、实验设置(MOTA/IDF1/IDs)、两种事件定义、事件总数 |
| 02 任务1 | 可视化视频规格、12 个典型例子(事件帧 ±15 帧动画播放器 + 视频片段 + 完整视频链接) |
| 03 任务2 | 白布可视化口径、2 个例子(全序列视频 + 事件窗口动画) |
| 04 任务3 | 指标口径、核心结论、NA 统计、gap 分布、全部指标分布表(自动生成) |
| 05 任务4 | 6 张切换轨迹图(switch + reuse 每数据集各 1)+ 中文说明 |
| 06 附录 | 打包清单、7 个完整序列视频、115 视频总清单、全部 7,156 条事件表(可检索)、指标 CSV、原始报告文档 |

## 目录结构

```
index.html                主报告(单文件多页)
assets/
  examples/               12 个典型例子 × 31 帧标注图(JPEG 压缩版)
  clips/                  14 个事件窗口视频片段(30 fps)
  videos/                 7 个典型序列完整跟踪视频
  canvas_videos/          2 个白布全序列视频(V013 / V019)
  canvas/                 2 个白布事件窗口 PNG(62 张)
  trajectory/             6 张切换轨迹图
  tables/                 事件表 / 逐序列统计 / 逐事件指标 CSV / 分布表 markdown
  data/                   页面数据(frame_data.js / events.js / video_inventory.js)
docs/                     原始报告(id_switch_report、canvas_report、switch_metrics_report、reproduction_results)
```

## 注意事项

1. **视频帧率 30 fps 为假设** —— 数据集序列无 fps 元数据,统一按 30 fps 编码;如需原速播放可按真实帧率重新编码。
2. `assets/examples/` 中为 **JPEG 压缩版**(原始 PNG 约 630 MB,压缩后约 100 MB),用于页面内动画播放;
   原始无损 PNG 与全部 115 个完整视频(约 5.5 GB)已从 `YOLOX_outputs/` 精简删除(均为脚本可再生产物),
   如需重新生成:`tools/draw_tracks_video.py -expn {mot17|mot20|sportsmot}_v001_full -ds {MOT17|MOT20|SportsMOT}`
   与 `tools/gen_examples.py`;产物清单见报告附录。
3. 所有指标均由项目脚本计算并经独立验证脚本复核(事件数逐序列与 motmetrics `num_switches` 100% 一致)。
4. 如需更小体积发送,可删除 `assets/videos/`(约 254 MB),页面其余功能不受影响。

## 数据来源

- 跟踪结果:`YOLOX_outputs/{mot17|mot20|sportsmot}_v001_full/track_results/V*.txt`
- 事件挖掘:`tools/find_id_events.py`(motmetrics 1.4.0)
- 可视化:`tools/draw_tracks_video.py` / `tools/draw_boxes_canvas.py`
- 指标统计:`tools/id_switch_metrics.py`;轨迹图:`tools/draw_switch_trajectory.py`

生成日期:2026-08-02
