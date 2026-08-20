# -*- coding: utf-8 -*-
"""
Generate High-Resolution Geometric Primer Diagram: IoU, Cost Matrix, and 2x2 Swap
Output: research/taxonomy/fig_geometry_iou_cost_matrix.png (300 DPI)
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import os

# Set font and style
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 4.5), dpi=300)
fig.patch.set_facecolor('#f8fafc')

# ── Palette ─────────────────────────────────────────────────────────────────
C_DET = '#1f77b4'       # Blue: Detection Box
C_PRED = '#2ca02c'      # Green: Kalman Prediction Box
C_IOU = '#ff7f0e'       # Orange/Amber: Intersection
C_TEXT = '#1c2430'
C_MUTED = '#55627a'
C_BORDER = '#dfe5ef'

# =============================================================================
# PANEL 1: IoU (Intersection over Union)
# =============================================================================
ax1.set_facecolor('#ffffff')
ax1.set_xlim(0, 10)
ax1.set_ylim(0, 10)
ax1.set_aspect('equal')
ax1.axis('off')

# Title
ax1.text(5, 9.4, "(a) IoU 空间重叠度 (交并比)", fontsize=13, weight='bold', ha='center', color='#0b1c33')

# Detection Box D (Blue)
rect_d = patches.Rectangle((1.5, 2.2), 4.8, 5.0, linewidth=2.5, edgecolor=C_DET, facecolor=C_DET, alpha=0.15)
ax1.add_patch(rect_d)
ax1.text(2.2, 7.4, "当前帧检测框 D", fontsize=10.5, color=C_DET, weight='bold')

# Prediction Box P (Green)
rect_p = patches.Rectangle((3.8, 3.6), 4.8, 4.8, linewidth=2.5, edgecolor=C_PRED, facecolor=C_PRED, alpha=0.15, linestyle='--')
ax1.add_patch(rect_p)
ax1.text(6.0, 8.6, "轨迹外推预测 P", fontsize=10.5, color=C_PRED, weight='bold')

# Intersection Area (Orange Hatch)
rect_inter = patches.Rectangle((3.8, 3.6), 2.5, 3.6, linewidth=1.5, edgecolor=C_IOU, facecolor=C_IOU, alpha=0.45, hatch='//')
ax1.add_patch(rect_inter)
ax1.text(5.05, 5.2, "重叠交集\nA ∩ B", fontsize=10, color='#8a4b00', weight='bold', ha='center', va='center')

# Formula banner below
ax1.text(5, 1.0, r"$\mathrm{IoU}(D, P) = \frac{\mathrm{Area}(D \cap P)}{\mathrm{Area}(D \cup P)} \in [0, 1]$", 
         fontsize=12, ha='center', color='#0f243e', weight='bold',
         bbox=dict(boxstyle='round,pad=0.5', facecolor='#eef4fc', edgecolor='#c4d7ed'))

# =============================================================================
# PANEL 2: Bipartite Matching Cost Matrix
# =============================================================================
ax2.set_facecolor('#ffffff')
ax2.set_xlim(0, 10)
ax2.set_ylim(0, 10)
ax2.set_aspect('equal')
ax2.axis('off')

ax2.text(5, 9.4, "(b) 二部图关联与代价矩阵", fontsize=13, weight='bold', ha='center', color='#0b1c33')

# Draw a 2x2 / 3x2 Matrix Grid
# Matrix data: Cost = 1 - IoU
iou_matrix = np.array([
    [0.82, 0.35],
    [0.15, 0.78]
])
cost_matrix = 1.0 - iou_matrix

# Draw cells
cell_w, cell_h = 2.4, 2.0
ox, oy = 3.0, 3.0

# Col Headers (Tracks T1, T2)
ax2.text(ox + 0.5 * cell_w, oy + 2 * cell_h + 0.3, "轨迹 T1", fontsize=11, weight='bold', ha='center', color=C_PRED)
ax2.text(ox + 1.5 * cell_w, oy + 2 * cell_h + 0.3, "轨迹 T2", fontsize=11, weight='bold', ha='center', color=C_PRED)

# Row Headers (Detections D1, D2)
ax2.text(ox - 0.3, oy + 1.5 * cell_h, "检测 D1", fontsize=11, weight='bold', ha='right', va='center', color=C_DET)
ax2.text(ox - 0.3, oy + 0.5 * cell_h, "检测 D2", fontsize=11, weight='bold', ha='right', va='center', color=C_DET)

for r in range(2):
    for c in range(2):
        cx = ox + c * cell_w
        cy = oy + (1 - r) * cell_h
        val_iou = iou_matrix[r, c]
        val_cost = cost_matrix[r, c]
        
        # Color intensity: match on diagonal is best (green tint)
        is_match = (r == c)
        fc = '#eaf5ee' if is_match else '#fdf6ee'
        ec = '#2ca02c' if is_match else '#d5dbe6'
        lw = 2.0 if is_match else 1.0
        
        rect = patches.Rectangle((cx, cy), cell_w, cell_h, facecolor=fc, edgecolor=ec, linewidth=lw)
        ax2.add_patch(rect)
        
        # Text inside cell
        ax2.text(cx + cell_w/2, cy + cell_h*0.65, f"IoU = {val_iou:.2f}", fontsize=10.5, weight='bold', ha='center', 
                 color='#1f6b41' if is_match else '#55627a')
        ax2.text(cx + cell_w/2, cy + cell_h*0.3, f"Cost = {val_cost:.2f}", fontsize=9.5, ha='center', color='#8a94a6')

# Tag Top1 / Top2
ax2.text(ox + 0.5 * cell_w, oy + 1.5 * cell_h - 0.6, "★ top1", fontsize=9, color='#1f6b41', weight='bold', ha='center')
ax2.text(ox + 1.5 * cell_w, oy + 1.5 * cell_h - 0.6, "top2", fontsize=9, color='#c2572a', ha='center')

# Formula banner below
ax2.text(5, 1.0, r"$\mathrm{Cost}(D_i, T_j) = 1.0 - \mathrm{IoU}(D_i, T_j)$", 
         fontsize=12, ha='center', color='#0f243e', weight='bold',
         bbox=dict(boxstyle='round,pad=0.5', facecolor='#eef4fc', edgecolor='#c4d7ed'))

# =============================================================================
# PANEL 3: 2x2 Swap Cost Perturbation
# =============================================================================
ax3.set_facecolor('#ffffff')
ax3.set_xlim(0, 10)
ax3.set_ylim(0, 10)
ax3.set_aspect('equal')
ax3.axis('off')

ax3.text(5, 9.4, r"(c) 局部 2×2 交换扰动 ($\Delta C_{\mathrm{swap}}$)", fontsize=13, weight='bold', ha='center', color='#0b1c33')

# Diagram of D1, D2 and T1, T2
# Left: D1, D2
ax3.scatter([2.5, 2.5], [6.5, 3.5], s=260, color=C_DET, zorder=5)
ax3.text(2.5, 6.5, "D1", color='white', weight='bold', ha='center', va='center', fontsize=10)
ax3.text(2.5, 3.5, "D2", color='white', weight='bold', ha='center', va='center', fontsize=10)

# Right: T1, T2
ax3.scatter([7.5, 7.5], [6.5, 3.5], s=260, color=C_PRED, zorder=5)
ax3.text(7.5, 6.5, "T1", color='white', weight='bold', ha='center', va='center', fontsize=10)
ax3.text(7.5, 3.5, "T2", color='white', weight='bold', ha='center', va='center', fontsize=10)

# Direct lines (Original Matching: D1-T1, D2-T2) -> Solid Green
ax3.annotate("", xy=(7.0, 6.5), xytext=(3.0, 6.5), arrowprops=dict(arrowstyle="->", color="#1f7d4b", lw=2.5))
ax3.annotate("", xy=(7.0, 3.5), xytext=(3.0, 3.5), arrowprops=dict(arrowstyle="->", color="#1f7d4b", lw=2.5))
ax3.text(5.0, 6.8, "原始分配 c11+c22", fontsize=9.5, color="#1f7d4b", weight='bold', ha='center')

# Swap cross lines (D1-T2, D2-T1) -> Dashed Red
ax3.annotate("", xy=(7.0, 3.8), xytext=(3.0, 6.2), arrowprops=dict(arrowstyle="->", color="#ba3428", lw=2.0, linestyle="--"))
ax3.annotate("", xy=(7.0, 6.2), xytext=(3.0, 3.8), arrowprops=dict(arrowstyle="->", color="#ba3428", lw=2.0, linestyle="--"))
ax3.text(5.0, 4.8, "对调分配 c12+c21", fontsize=9.5, color="#ba3428", weight='bold', ha='center')

# Formula banner below
ax3.text(5, 1.0, r"$\Delta C_{\mathrm{swap}} = (c_{12}+c_{21}) - (c_{11}+c_{22})$", 
         fontsize=12, ha='center', color='#0f243e', weight='bold',
         bbox=dict(boxstyle='round,pad=0.5', facecolor='#fef3ee', edgecolor='#f5cbbd'))

# Adjust layout
plt.tight_layout()

# Save
out_img_path = r"e:\科研\ByteTrack\research\taxonomy\fig_geometry_iou_cost_matrix.png"
plt.savefig(out_img_path, dpi=300, bbox_inches='tight', facecolor=fig.get_facecolor())
plt.close()
print(f"[SUCCESS] Geometric primer diagram generated: {out_img_path}")
