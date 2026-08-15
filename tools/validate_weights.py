# -*- coding: utf-8 -*-
"""
权重文件完整性验证脚本 (COE 规范)
- 使用 pathlib.Path + __file__ 推导仓库根，禁止硬编码绝对路径
- 逐一 torch.load(map_location='cpu') 验证可加载性
- 用与 exps/example/mot/yolox_x_mix_det.py 一致的网络结构(YOLOX-X, 1类)构建模型并加载权重
"""
import os
import sys
import traceback
from pathlib import Path

import torch

BASE = Path(__file__).resolve().parents[1]
PRETRAINED = BASE / "pretrained"


def build_model(num_classes=1, depth=1.33, width=1.25):
    import torch.nn as nn
    from yolox.models import YOLOPAFPN, YOLOX, YOLOXHead

    in_channels = [256, 512, 1024]
    backbone = YOLOPAFPN(depth, width, in_channels=in_channels)
    head = YOLOXHead(num_classes, width, in_channels=in_channels)
    model = YOLOX(backbone, head)

    def init_yolo(M):
        for m in M.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eps = 1e-3
                m.momentum = 0.03

    model.apply(init_yolo)
    return model


def main():
    if not PRETRAINED.is_dir():
        print(f"[ERROR] pretrained 目录不存在: {PRETRAINED}")
        sys.exit(1)

    weight_files = sorted(PRETRAINED.glob("*.pth.tar")) or sorted(PRETRAINED.glob("*.pt"))
    if not weight_files:
        print(f"[ERROR] {PRETRAINED} 下未找到任何权重文件 (*.pth.tar / *.pt)")
        sys.exit(1)

    print(f"[INFO] 扫描到 {len(weight_files)} 个权重文件:")
    ok = True
    for wf in weight_files:
        try:
            print(f"\n{'='*70}")
            print(f"[FILE] {wf.name}  ({wf.stat().st_size / 1e6:.1f} MB)")
            ckpt = torch.load(str(wf), map_location="cpu")
            print(f"  top-level keys: {list(ckpt.keys())}")

            model_state = ckpt.get("model", ckpt.get("state_dict", None))
            if not isinstance(model_state, dict):
                print("  [FAIL] checkpoint 中无 'model'/'state_dict' 字段")
                ok = False
                continue

            n_classes = None
            for k in model_state:
                if "head.cls_preds" in k and k.endswith(".weight"):
                    n_classes = model_state[k].shape[0]
                    break
            print(f"  推断类别数 num_classes = {n_classes}")

            model = build_model(num_classes=n_classes or 1)
            model.load_state_dict(model_state, strict=True)
            n_params = sum(p.numel() for p in model.parameters()) / 1e6
            print(f"  [OK]   strict 权重加载成功，模型参数量: {n_params:.1f}M")

            # 前向推理冒烟测试（CPU，1x3x640x640）
            dummy = torch.randn(1, 3, 640, 640)
            model.eval()
            with torch.no_grad():
                outs = model(dummy)
            print(f"  [OK]   前向推理通过，输出 {len(outs)} 个特征层")
        except Exception as e:
            print(f"[FAIL] {wf.name} 验证失败: {e}")
            traceback.print_exc()
            ok = False

    print(f"\n{'='*70}")
    if ok:
        print("[RESULT] 全部权重文件验证通过")
    else:
        print("[RESULT] 存在验证失败的权重文件，请人工检查")
        sys.exit(1)


if __name__ == "__main__":
    main()
