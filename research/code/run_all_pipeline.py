# -*- coding: utf-8 -*-
"""run_all_pipeline.py — LSRG-ByteTrack 一键端到端科研复现流水线

Part of LSRG-ByteTrack Research Workspace.

Usage:
    python run_all_pipeline.py --help
    python run_all_pipeline.py --all
    python run_all_pipeline.py --stage features
    python run_all_pipeline.py --stage eval
    python run_all_pipeline.py --stage breakdown
    python run_all_pipeline.py --stage figures
    python run_all_pipeline.py --stage presentation
"""

import argparse
import os
import subprocess
import sys
import time

# Ensure UTF-8 output on Windows consoles if possible
if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

HERE = os.path.dirname(os.path.abspath(__file__))
PY_EXE = sys.executable

STAGES = {
    "features": [
        ("提取三维因果风险特征与ECDF经验分布校准", "risk_features.py"),
    ],
    "eval": [
        ("四大聚合模型 5-Fold 盲测与 ROC/FPR 评测", "risk_aggregation.py"),
    ],
    "breakdown": [
        ("三大失效类别隔离评测与雪崩机理归因", "class_risk_breakdown.py"),
        ("全量跨数据集/跨类别细分统计大表生成", "comprehensive_class_dataset_breakdown.py"),
    ],
    "figures": [
        ("生成几何 IoU 代价矩阵图", "generate_geometric_diagram.py"),
        ("生成论文主图表 (2x2 ROC 与分位数折线)", "generate_paper_tables_and_figures.py"),
        ("生成单数据集拆分 ROC 图", "generate_split_paper_figures.py"),
    ],
    "presentation": [
        ("生成 7 页学术组会汇报 PPTX", "generate_lab_presentation.py"),
    ],
}


def run_script(description, script_name):
    script_path = os.path.join(HERE, script_name)
    if not os.path.exists(script_path):
        print(f"[ERROR] 找不到脚本: {script_path}")
        return False

    print("\n" + "=" * 70)
    print(f">> [RUN] {description} ({script_name})")
    print("=" * 70)
    t0 = time.time()
    ret = subprocess.run([PY_EXE, script_path], cwd=HERE)
    elapsed = time.time() - t0

    if ret.returncode == 0:
        print(f"[OK] 成功完成 [{elapsed:.1f}s]: {script_name}")
        return True
    else:
        print(f"[FAIL] 执行失败 (退出码 {ret.returncode}): {script_name}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="LSRG-ByteTrack 端到端科研流水线总控脚本",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""示例:
  python run_all_pipeline.py --all                # 顺序执行全部 5 个阶段
  python run_all_pipeline.py --stage features     # 仅提取因果特征与校准器
  python run_all_pipeline.py --stage eval         # 仅运行聚合模型评测
  python run_all_pipeline.py --stage presentation # 仅重新生成组会 PPTX
"""
    )
    parser.add_argument(
        "--stage",
        choices=["features", "eval", "breakdown", "figures", "presentation"],
        help="指定运行特定阶段"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="顺序执行全部阶段"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="仅打印将要执行的脚本清单，不实际运行"
    )

    args = parser.parse_args()

    if not args.stage and not args.all:
        parser.print_help()
        sys.exit(0)

    stages_to_run = list(STAGES.keys()) if args.all else [args.stage]

    total_tasks = sum(len(STAGES[s]) for s in stages_to_run)
    print("*" * 70)
    print(f"  LSRG-ByteTrack 科研流水线启动: 共 {len(stages_to_run)} 个阶段, {total_tasks} 个任务")
    print("*" * 70)

    if args.dry_run:
        print("[DRY-RUN] 将按序执行以下脚本:")
        for s in stages_to_run:
            print(f"  阶段 [{s}]:")
            for desc, script in STAGES[s]:
                print(f"    - {script} ({desc})")
        return

    pipeline_t0 = time.time()
    for s in stages_to_run:
        print(f"\n>>> 阶段: {s.upper()} <<<")
        for desc, script in STAGES[s]:
            success = run_script(desc, script)
            if not success:
                print(f"\n[ABORT] 流水线在 {script} 中断，请检查上方报错！")
                sys.exit(1)

    total_elapsed = time.time() - pipeline_t0
    print("\n" + "*" * 70)
    print(f"[DONE] 全部任务执行完毕！总耗时: {total_elapsed:.1f}s")
    print("*" * 70)


if __name__ == "__main__":
    main()
