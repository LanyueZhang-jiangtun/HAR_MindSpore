# -*- coding: utf-8 -*-
"""【乙】基线对比：4 个 sklearn 模型 LOSO + 对比表 + 最佳模型混淆矩阵 + 特征 t-SNE
运行: E:/python_ziliao/.venv-ms311/Scripts/python.exe scripts/02_baseline.py
      [--feature-set full|time|freq|extended]
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from config import RESULT_DIR
from src import visualize
from src.baseline import BASELINE_MODELS, loso_evaluate
from src.dataset import load_raw_windows
from src.evaluate import compare_table
from src.features import extract_features


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature-set", default="full",
                        choices=["full", "time", "freq", "extended"],
                        help="特征消融：full=12维(默认) / time=9维 / freq=3维 / extended=26维")
    args = parser.parse_args()

    X, y, people, names = load_raw_windows()
    F = extract_features(X, subset=args.feature_set)
    tag = "baseline" if args.feature_set == "full" else f"baseline_{args.feature_set}"
    print(f"窗口数: {len(y)}，特征集: {args.feature_set}，特征维度: {F.shape[1]}\n")

    results = {}
    for name, factory in BASELINE_MODELS.items():
        r = loso_evaluate(factory, F, y, people, names)
        r["model"] = name
        results[name] = r
        print(f"{name:<20} acc={r['accuracy']:.4f}  macroF1={r['macro_f1']:.4f}  耗时={r['seconds']:.1f}s")

    compare_table(results, tag=tag)

    (RESULT_DIR / f"baseline_per_subject_{args.feature_set}.json").write_text(
        json.dumps({n: r["per_subject"] for n, r in results.items()}, indent=2, ensure_ascii=False),
        encoding="utf-8")
    (RESULT_DIR / f"baseline_results_{args.feature_set}.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")

    best = max(results, key=lambda n: results[n]["accuracy"])
    conf_tag = f"confusion_{best}" if args.feature_set == "full" else f"confusion_{best}_{args.feature_set}"
    visualize.plot_confusion(np.array(results[best]["confusion"]), names,
                             tag=conf_tag, title=f"{best} confusion matrix ({args.feature_set})")
    if args.feature_set == "full":
        visualize.plot_tsne(F, y, names)
    print(f"\n最佳基线: {best}，混淆矩阵/特征投影已保存 outputs/figures/")


if __name__ == "__main__":
    main()
