# -*- coding: utf-8 -*-
"""id_recovery_vlm.py — VLM 多模态选择题兜底恢复评测

Part of LSRG-ByteTrack Research Workspace.

This script loads the Hard Cases saved by id_recovery_dino.py, stitches the
query image and candidates into a unified choice grid, and invokes Qwen2.5-VL
locally via Transformers to match the correct ID.
"""

import os
import csv
import sys
import numpy as np
import cv2
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
TAXONOMY_DIR = os.path.join(ROOT, "research", "taxonomy")
SCRATCH_DIR = os.path.join(ROOT, "research", "scratch")
VLM_TEMP_DIR = os.path.join(SCRATCH_DIR, "vlm_temp")

os.makedirs(VLM_TEMP_DIR, exist_ok=True)

def log(msg):
    print(f"[id_recovery_vlm] {msg}")

def create_visual_choice_grid(query_path, candidates, output_path):
    """Stitch Query and candidates into a visual choice grid.
    Layout:
    +---------------+-----------------+-----------------+
    |               |  Option A       |  Option B       |
    |  TARGET       |  [Candidate 0]  |  [Candidate 1]  |
    |  (Query)      |                 |                 |
    +---------------+-----------------+-----------------+
    """
    q_img = cv2.imread(query_path)
    if q_img is None:
        return None
        
    # Resize query to standard height
    h_std = 250
    w_q = int(q_img.shape[1] * (h_std / q_img.shape[0]))
    q_img = cv2.resize(q_img, (w_q, h_std))
    
    # Add label "TARGET"
    cv2.putText(q_img, "TARGET", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    cv2.rectangle(q_img, (0, 0), (w_q-1, h_std-1), (0, 0, 255), 3)

    cand_imgs = []
    # Map index to options (A, B, C, D...)
    options_map = {}
    
    for idx, cand in enumerate(candidates):
        c_path = cand["crop_path"]
        c_img = cv2.imread(c_path)
        if c_img is None:
            continue
        
        w_c = int(c_img.shape[1] * (h_std / c_img.shape[0]))
        c_img = cv2.resize(c_img, (w_c, h_std))
        
        opt_char = chr(65 + idx)  # A, B, C...
        options_map[opt_char] = cand["gt_id"]
        
        cv2.putText(c_img, f"Option {opt_char}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        cv2.rectangle(c_img, (0, 0), (w_c-1, h_std-1), (255, 0, 0), 2)
        cand_imgs.append(c_img)

    if not cand_imgs:
        return None

    # Stitch all images horizontally
    canvas = [q_img] + cand_imgs
    stitched = np.hstack(canvas)
    cv2.imwrite(output_path, stitched)
    
    return options_map

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Task 3: VLM Choice-Based Re-ID Recovery")
    parser.add_argument("--model", default="Qwen/Qwen2.5-VL-3B-Instruct", 
                        help="Hugging Face VLM model path/ID")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    hard_cases_npz = os.path.join(SCRATCH_DIR, "id_recovery_hard_cases.npz")
    if not os.path.exists(hard_cases_npz):
        log(f"Hard cases file not found: {hard_cases_npz}. Run id_recovery_dino.py first.")
        return

    # 1. Load Hard Cases
    data = np.load(hard_cases_npz, allow_pickle=True)
    hard_cases = data["hard_cases"]
    log(f"Loaded {len(hard_cases)} hard cases to evaluate via VLM.")

    if len(hard_cases) == 0:
        log("No hard cases found. Re-ID evaluation finished.")
        return

    # 2. Load VLM local model
    log(f"Loading local VLM: {args.model} on {args.device}...")
    try:
        from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
        from qwen_vl_utils import process_vision_info
        
        processor = AutoProcessor.from_pretrained(args.model)
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            args.model, torch_dtype="auto", device_map=args.device
        )
        log("VLM successfully loaded.")
    except Exception as e:
        log(f"Error loading VLM: {e}.")
        log("Make sure 'transformers', 'accelerate', 'qwen-vl-utils' are installed.")
        log("VLM execution skipped (planned script is ready).")
        return

    vlm_recovered = 0
    total_eval = 0

    results = []

    for idx, case in enumerate(hard_cases):
        ds = case["dataset"]
        seq = case["seq"]
        frame = case["frame"]
        gt_correct = case["gt_correct_id"]
        q_path = case["query_crop_path"]
        candidates = case["candidates"]
        dino_correct = case["dino_correct"]

        # 3. Stitch Query and Candidates into choice grid image
        grid_path = os.path.join(VLM_TEMP_DIR, f"grid_{ds}_{seq}_F{frame}.jpg")
        options_map = create_visual_choice_grid(q_path, candidates, grid_path)
        
        if not options_map:
            continue

        # Look up correct option char
        correct_option = "None"
        for opt_char, gt_id in options_map.items():
            if gt_id == gt_correct:
                correct_option = opt_char
                break

        if correct_option == "None":
            # Correct ID is not in candidates window (occlusion/departure case)
            continue

        total_eval += 1

        # 4. Construct prompt and call VLM
        prompt = (
            "Task: Identify which Option on the right is the same object as the Target on the left.\n"
            "Options are labeled A, B, C, etc.\n"
            "Compare clothing, colors, patterns, and shape carefully.\n"
            "Respond ONLY with the option letter (e.g., A, B, C) and nothing else."
        )

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": grid_path},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(args.device)

        with torch.no_grad():
            generated_ids = model.generate(**inputs, max_new_tokens=10)
            generated_ids_trimmed = [
                out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0].strip()

        # Parse VLM letter choice
        predicted_char = output_text.upper().replace(".", "").strip()
        vlm_pred_gt_id = options_map.get(predicted_char, -1)

        is_correct = (vlm_pred_gt_id == gt_correct)
        if is_correct:
            vlm_recovered += 1

        log(f"[{idx+1}/{len(hard_cases)}] Seq={seq} Frame={frame} -> GT={correct_option} | VLM Pred={predicted_char} | Correct={is_correct}")

        results.append({
            "Dataset": ds,
            "Sequence": seq,
            "Frame": frame,
            "GT_Correct_Option": correct_option,
            "VLM_Choice": predicted_char,
            "DINO_Correct": dino_correct,
            "VLM_Correct": int(is_correct)
        })

    # 5. Output comparison results to CSV
    vlm_summary_path = os.path.join(TAXONOMY_DIR, "id_recovery_vlm_results.csv")
    with open(vlm_summary_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["Dataset", "Sequence", "Frame", "GT_Correct_Option", "VLM_Choice", "DINO_Correct", "VLM_Correct"])
        writer.writeheader()
        writer.writerows(results)

    vlm_acc = vlm_recovered / total_eval if total_eval > 0 else 0.0
    log(f"VLM Accuracy on Hard Cases = {vlm_acc*100:.2f}% ({vlm_recovered}/{total_eval})")
    log(f"VLM detailed results saved to {vlm_summary_path}")

if __name__ == "__main__":
    main()
