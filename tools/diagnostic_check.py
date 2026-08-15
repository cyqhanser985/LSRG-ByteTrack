# -*- coding: utf-8 -*-
"""
数据集与路径鲁棒性校验脚本 (COE 规范) — 步骤3 防错机制
- 使用 pathlib.Path + __file__ 推导仓库根，禁止硬编码盘符/绝对路径
- 检测 datasets/{MOT17,MOT20,SportsMOT} 目录结构是否符合 V001 标准化格式
- 读取 mapping.txt 验证 V 编号与原始序列名映射
- 抽样读取每个数据集 1 张图像（规避中文路径编码问题）
- 校验 annotations JSON 的 file_name 与磁盘目录一致性（找出未标准化的 JSON）
- 校验 gt.txt 帧范围与 JSON 中序列帧范围一致性
"""
import json
import os
import sys
import traceback
from pathlib import Path

import cv2
import numpy as np

BASE = Path(__file__).resolve().parents[1]
DATASETS = ["MOT17", "MOT20", "SportsMOT"]

# 每个数据集评估用 JSON（prepare_eval_json.py 生成的标准化文件）
EVAL_JSON = {"MOT17": "eval.json", "MOT20": "eval.json", "SportsMOT": "eval.json"}

FAILS = 0
WARNS = 0


def report(tag, msg):
    global FAILS, WARNS
    if tag == "FAIL":
        FAILS += 1
    elif tag == "WARN":
        WARNS += 1
    print(f"  [{tag}] {msg}")


def parse_mapping(mapping_path):
    """解析 mapping.txt -> {V编号: 原始序列名}"""
    mapping = {}
    if not mapping_path.exists():
        report("WARN", f"mapping.txt 不存在: {mapping_path}")
        return mapping
    for line in mapping_path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) >= 2:
            mapping[parts[0]] = parts[1]
    return mapping


def check_dataset_structure(name):
    """检查 V001 标准化目录结构"""
    print(f"\n===== {name} 目录结构 =====")
    root = BASE / "datasets" / name
    if not root.is_dir():
        report("FAIL", f"{root} 不存在")
        return None

    v_dirs = sorted([d for d in root.iterdir() if d.is_dir() and d.name.startswith("V")])
    report("OK", f"共 {len(v_dirs)} 个 V 序列目录")

    # 期望编号连续: V001..V{n}
    expected = [f"V{i:03d}" for i in range(1, len(v_dirs) + 1)]
    actual = [d.name for d in v_dirs]
    if actual == expected:
        report("OK", "V 编号连续且按序排列 V001..V%03d" % len(v_dirs))
    else:
        report("FAIL", f"V 编号不连续! 期望前3={expected[:3]} 实际前3={actual[:3]}")

    # 每个序列检查 img1/ 与 gt/gt.txt
    img_counts = {}
    gt_ranges = {}
    n_img_ok = 0
    n_gt_ok = 0
    for d in v_dirs:
        img1 = d / "img1"
        imgs = sorted(img1.glob("*.jpg")) if img1.is_dir() else []
        img_counts[d.name] = len(imgs)
        if imgs:
            n_img_ok += 1
        gt_file = d / "gt" / "gt.txt"
        if gt_file.exists():
            frames = set()
            with open(gt_file, "r", encoding="utf-8", errors="replace") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        frames.add(int(line.split(",")[0]))
            if frames:
                gt_ranges[d.name] = (min(frames), max(frames), len(frames))
                n_gt_ok += 1
    report("OK", f"{n_img_ok}/{len(v_dirs)} 序列 img1/ 含图像")
    report("OK", f"{n_gt_ok}/{len(v_dirs)} 序列 gt/gt.txt 存在")
    if img_counts:
        sample = list(img_counts.items())[:3]
        report("OK", f"图像数抽样(前3): {sample}")

    return {"root": root, "v_dirs": v_dirs, "img_counts": img_counts, "gt_ranges": gt_ranges}


def check_mapping(name, v_dirs):
    """校验 mapping.txt 与 V 序列一致性"""
    print(f"\n===== {name} mapping.txt =====")
    mapping = parse_mapping(BASE / "datasets" / name / "mapping.txt")
    if not mapping:
        return
    map_v = sorted(mapping.keys())
    actual = sorted([d.name for d in v_dirs]) if v_dirs else []
    if map_v == actual:
        report("OK", f"mapping 覆盖全部 {len(map_v)} 个序列")
    else:
        report("FAIL", f"mapping 的 V 编号与目录不一致: mapping={map_v[:5]}... dirs={actual[:5]}...")
    # 抽样展示映射关系
    sample = list(mapping.items())[:3]
    report("OK", f"映射抽样(前3): {sample}")


def check_json_consistency(name, info):
    """校验 JSON file_name 与磁盘目录一致性（核心防错点）
    规则: eval.json 为评估唯一依据，严格校验(Vxxx + 无缺失);
          其余为原始下载 JSON(引用原始序列名), 仅提示不阻断。
    """
    print(f"\n===== {name} JSON 一致性 =====")
    root = BASE / "datasets" / name
    ann_dir = root / "annotations"
    if not ann_dir.is_dir():
        report("FAIL", f"annotations 目录不存在: {ann_dir}")
        return

    jsons = sorted(ann_dir.glob("*.json"))
    report("OK", f"annotations 下 {len(jsons)} 个 JSON: {[j.name for j in jsons]}")

    for jf in jsons:
        is_eval = jf.name == "eval.json"
        try:
            with open(jf, "r", encoding="utf-8") as f:
                data = json.load(f)
            imgs = data.get("images", [])
            if not imgs:
                report("WARN", f"{jf.name}: images 为空")
                continue
            # 序列前缀集合
            seqs = sorted(set(i["file_name"].split("/")[0] for i in imgs))
            # 检查 file_name 指向的目录是否存在
            miss = 0
            for i in imgs[:50]:
                if not (root / i["file_name"]).exists():
                    miss += 1
            n_missing_50 = f"{miss}/50"
            # 判断是否使用 Vxxx 前缀
            v_ok = all(s.startswith("V") for s in seqs)
            cat_names = [c.get("name") for c in data.get("categories", [])]
            if is_eval:
                status = "OK" if v_ok and miss == 0 else "FAIL"
                tag = "FAIL" if status == "FAIL" else "OK"
            else:
                # 原始下载 JSON: 非评估依据, 缺失/非Vxxx 仅提示
                tag = "OK" if v_ok and miss == 0 else "WARN"
            report(
                tag,
                f"{jf.name}: images={len(imgs)}, 序列数={len(seqs)}, 前缀Vxxx={v_ok}, "
                f"前50张缺失={n_missing_50}, categories={cat_names}"
                + ("" if is_eval else " [原始JSON, 评估以 eval.json 为准]"),
            )
            if tag == "FAIL":
                report("INFO", f"  -> 序列前缀抽样: {seqs[:5]}")
        except Exception as e:
            report("FAIL", f"{jf.name} 解析异常: {e}")
            traceback.print_exc()


def check_sample_image(name, info):
    """抽样读取每个数据集 1 张图像，验证中文路径下 cv2 可解码"""
    print(f"\n===== {name} 图像抽样读取 =====")
    root = BASE / "datasets" / name
    v_dirs = info["v_dirs"] if info else []
    if not v_dirs:
        report("FAIL", "无 V 序列可抽样")
        return
    img_dir = v_dirs[0] / "img1"
    imgs = sorted(img_dir.glob("*.jpg")) if img_dir.is_dir() else []
    if not imgs:
        report("FAIL", f"{img_dir} 无图像")
        return
    # 读首张 + 中间一张（中文路径需用 np.fromfile + cv2.imdecode 字节流解码）
    for img_path in [imgs[0], imgs[len(imgs) // 2]]:
        img = cv2.imdecode(np.fromfile(str(img_path), dtype=np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            report("FAIL", f"cv2.imdecode(np.fromfile) 失败: {img_path}")
        else:
            h, w = img.shape[:2]
            report("OK", f"{img_path.name} 读取成功 ({w}x{h})")


def check_gt_json_range(name, info):
    """校验 gt.txt 帧范围与评估用 JSON 的帧范围一致"""
    print(f"\n===== {name} gt.txt 与 JSON 帧范围 =====")
    jf_name = EVAL_JSON.get(name)
    if not jf_name:
        return
    jf = BASE / "datasets" / name / "annotations" / jf_name
    if not jf.exists():
        report("FAIL", f"评估 JSON 不存在: {jf}")
        return
    with open(jf, "r", encoding="utf-8") as f:
        data = json.load(f)
    # 按视频分组帧范围
    json_ranges = {}
    for i in data["images"]:
        vid = i["video_id"]
        fr = i["frame_id"]
        if vid not in json_ranges:
            json_ranges[vid] = [fr, fr]
        else:
            json_ranges[vid][0] = min(json_ranges[vid][0], fr)
            json_ranges[vid][1] = max(json_ranges[vid][1], fr)

    gt_ranges = info["gt_ranges"] if info else {}
    checked = 0
    for v_name, (lo, hi, n) in sorted(gt_ranges.items()):
        # 通过 V 编号找 video_id (V001 -> video_id 1)
        try:
            vid = int(v_name[1:])
        except ValueError:
            continue
        if vid in json_ranges:
            jlo, jhi = json_ranges[vid]
            if lo == jlo and hi == jhi:
                report("OK", f"{v_name}: gt[{lo}-{hi}] 与 JSON 帧范围一致")
            else:
                report("WARN", f"{v_name}: gt[{lo}-{hi}] vs JSON[{jlo}-{jhi}] 不一致")
            checked += 1
        else:
            report("WARN", f"{v_name}: JSON 中无 video_id={vid}")
    report("OK", f"共核对 {checked} 个序列的帧范围")


def main():
    print(f"项目根目录: {BASE}")
    print(f"YOLOX_DATADIR 环境变量: {os.getenv('YOLOX_DATADIR', '(未设置, 默认使用 {BASE}/datasets)')}")

    infos = {}
    for name in DATASETS:
        info = check_dataset_structure(name)
        infos[name] = info
        if info:
            check_mapping(name, info["v_dirs"])
            check_json_consistency(name, info)
            check_sample_image(name, info)
            check_gt_json_range(name, info)

    print("\n" + "=" * 60)
    print(f"[SUMMARY] FAIL={FAILS}, WARN={WARNS}")
    if FAILS:
        print("[RESULT] 存在 FAIL 项，请先修复后再进行跟踪评估")
        sys.exit(1)
    print("[RESULT] 数据集校验通过")
    sys.exit(0)


if __name__ == "__main__":
    main()
