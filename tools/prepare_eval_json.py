# -*- coding: utf-8 -*-
"""
构建标准化评估 JSON (COE 规范) — 步骤3 修复数据层
- 产物: datasets/{name}/annotations/eval.json (file_name 统一为 Vxxx/img1/000xxx.jpg)
- MOT17:    从 val_half.json 复制(已标准化)，gt 覆盖 301-600 后半段
- MOT20:    从 train.json 切片，file_name 重映射 MOT20-XX -> Vxxx (gt 为全序列)
- SportsMOT: 对有 gt 的 Vxxx 序列，从 train/val.json 切片到本地帧范围 + 重映射
- 全程只使用 pathlib.Path 与 __file__ 派生的仓库根路径
"""
import json
import os
import sys
import traceback
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]


def load_mapping(name):
    """mapping.txt -> {V编号: 原始序列名}"""
    mapping = {}
    mp = BASE / "datasets" / name / "mapping.txt"
    if not mp.exists():
        return mapping
    for line in mp.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) >= 2:
            mapping[parts[0]] = parts[1]
    return mapping


def gt_frames(name, v_name):
    """读取 gt.txt 的帧号集合"""
    gt_file = BASE / "datasets" / name / v_name / "gt" / "gt.txt"
    frames = set()
    if gt_file.exists():
        for line in gt_file.read_text(encoding="utf-8", errors="replace").splitlines():
            line = line.strip()
            if line:
                frames.add(int(line.split(",")[0]))
    return frames


def img_count(name, v_name):
    d = BASE / "datasets" / name / v_name / "img1"
    return len(list(d.glob("*.jpg"))) if d.is_dir() else 0


def save_json(name, data):
    out = BASE / "datasets" / name / "annotations" / "eval.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=True)
    print(f"[OK] {name}: 写入 {out} (images={len(data['images'])}, annotations={len(data['annotations'])})")


def prepare_mot17():
    print("===== MOT17 =====")
    src = BASE / "datasets" / "MOT17" / "annotations" / "val_half.json"
    with open(src, encoding="utf-8") as f:
        data = json.load(f)
    save_json("MOT17", data)


def prepare_mot20():
    print("===== MOT20 =====")
    mapping = load_mapping("MOT20")  # MOT20-XX -> Vxxx
    src = BASE / "datasets" / "MOT20" / "annotations" / "train.json"
    with open(src, encoding="utf-8") as f:
        data = json.load(f)

    v_of_seq = {seq: v for v, seq in mapping.items()}  # V001 -> MOT20-01
    new_images = []
    new_anns = []
    # 保持 id 唯一性: 全量重编号
    img_id_map = {}
    ann_id = 1
    for im in sorted(data["images"], key=lambda x: (x["video_id"], x["frame_id"])):
        seq = im["file_name"].split("/")[0]
        v_name = v_of_seq.get(seq)
        if v_name is None:
            print(f"[WARN] {seq} 不在 mapping 中，跳过")
            continue
        new_id = len(new_images) + 1
        img_id_map[im["id"]] = new_id
        new_images.append(
            {
                "id": new_id,
                "file_name": f"{v_name}/img1/{im['file_name'].split('/')[-1]}",
                "frame_id": im["frame_id"],
                "video_id": int(v_name[1:]),
                "width": im["width"],
                "height": im["height"],
                "prev_image_id": im.get("prev_image_id", -1),
                "next_image_id": im.get("next_image_id", -1),
            }
        )
    for a in data["annotations"]:
        if a["image_id"] in img_id_map:
            a = dict(a)
            a["id"] = ann_id
            ann_id += 1
            a["image_id"] = img_id_map[a["image_id"]]
            new_anns.append(a)

    data["images"] = new_images
    data["annotations"] = new_anns
    data["videos"] = [{"id": i, "file_name": f"V{i:03d}"} for i in range(1, 5)]
    save_json("MOT20", data)


def prepare_sportsmot():
    print("===== SportsMOT =====")
    mapping = load_mapping("SportsMOT")  # Vxxx -> train/v_xxx
    # 加载两个候选源 JSON
    sources = {}
    for jf in ["train.json", "val.json"]:
        with open(BASE / "datasets" / "SportsMOT" / "annotations" / jf, encoding="utf-8") as f:
            d = json.load(f)
        seq_imgs = {}
        for im in d["images"]:
            seq = im["file_name"].split("/")[0]
            seq_imgs.setdefault(seq, []).append(im)
        sources[jf] = (d, seq_imgs)

    new_images = []
    new_anns = []
    img_id_map = {}
    ann_id = 1

    v_dirs = sorted([d for d in (BASE / "datasets" / "SportsMOT").iterdir()
                     if d.is_dir() and d.name.startswith("V")])
    for v_dir in v_dirs:
        v_name = v_dir.name
        local_imgs = img_count("SportsMOT", v_name)
        gt = gt_frames("SportsMOT", v_name)
        if not gt:
            continue  # 无 gt 的序列不纳入评估
        max_frame = max(gt)
        orig = mapping.get(v_name, "")
        seq = orig.split("/")[-1]  # 去 split 前缀

        # 在 train.json / val.json 中定位该序列
        src_data, seq_imgs = None, None
        for jf, (d, si) in sources.items():
            if seq in si:
                src_data, seq_imgs = d, si[seq]
                break
        if src_data is None:
            print(f"[FAIL] {v_name} ({orig}) 在 JSON 中找不到对应序列")
            sys.exit(1)

        # 仅保留本地帧范围内的图像
        kept = [im for im in seq_imgs if im["frame_id"] <= max_frame]
        if len(kept) != local_imgs:
            print(f"[FAIL] {v_name}: 本地 {local_imgs} 张 vs JSON 前 {max_frame} 帧 {len(kept)} 张，不一致")
            sys.exit(1)
        for im in kept:
            new_id = len(new_images) + 1
            img_id_map[im["id"]] = new_id
            new_images.append(
                {
                    "id": new_id,
                    "file_name": f"{v_name}/img1/{im['file_name'].split('/')[-1]}",
                    "frame_id": im["frame_id"],
                    "video_id": int(v_name[1:]),
                    "width": im["width"],
                    "height": im["height"],
                    "prev_image_id": im.get("prev_image_id", -1),
                    "next_image_id": im.get("next_image_id", -1),
                }
            )
        for a in src_data["annotations"]:
            if a["image_id"] in img_id_map:
                a = dict(a)
                a["id"] = ann_id
                ann_id += 1
                a["image_id"] = img_id_map[a["image_id"]]
                new_anns.append(a)

    n_videos = len(set(i["video_id"] for i in new_images))
    data = {
        "images": new_images,
        "annotations": new_anns,
        "videos": [{"id": i, "file_name": f"V{i:03d}"} for i in range(1, n_videos + 1)],
        "categories": [{"id": 1, "name": "person"}],
    }
    save_json("SportsMOT", data)


def verify(name):
    """校验 eval.json 帧范围与 gt.txt 一致"""
    print(f"----- 校验 {name} -----")
    jf = BASE / "datasets" / name / "annotations" / "eval.json"
    with open(jf, encoding="utf-8") as f:
        data = json.load(f)
    json_ranges = {}
    for im in data["images"]:
        vid, fr = im["video_id"], im["frame_id"]
        if vid not in json_ranges:
            json_ranges[vid] = [fr, fr]
        else:
            json_ranges[vid][0] = min(json_ranges[vid][0], fr)
            json_ranges[vid][1] = max(json_ranges[vid][1], fr)

    v_dirs = sorted([d for d in (BASE / "datasets" / name).iterdir()
                     if d.is_dir() and d.name.startswith("V")])
    checked = 0
    for v_dir in v_dirs:
        vid = int(v_dir.name[1:])
        gt = gt_frames(name, v_dir.name)
        if not gt:
            continue
        if vid not in json_ranges:
            print(f"[FAIL] {v_dir.name}: JSON 中无 video_id={vid}")
            sys.exit(1)
        jlo, jhi = json_ranges[vid]
        glo, ghi = min(gt), max(gt)
        if (jlo, jhi) != (glo, ghi):
            print(f"[FAIL] {v_dir.name}: JSON[{jlo}-{jhi}] vs gt[{glo}-{ghi}]")
            sys.exit(1)
        checked += 1
    print(f"[OK] {name}: {checked} 个序列 JSON/gt 帧范围全部一致, 共 {len(data['images'])} 张图像")


def main():
    for name, fn in [("MOT17", prepare_mot17), ("MOT20", prepare_mot20), ("SportsMOT", prepare_sportsmot)]:
        try:
            fn()
            verify(name)
        except Exception as e:
            print(f"[FAIL] {name} 处理异常: {e}")
            traceback.print_exc()
            sys.exit(1)
    print("\n[RESULT] 全部数据集 eval.json 构建完成")


if __name__ == "__main__":
    main()
