# -*- coding: utf-8 -*-
"""【丁】一键重绘全部报告图（依赖 02/03/03b 的输出，改了参数重跑后再跑本脚本即可）
总对比表覆盖所有已有 LOSO 结果：1D-CNN / BiLSTM / CNN+增强 / MLP / sklearn 基线(12维、26维)。
运行: E:/python_ziliao/.venv-ms311/Scripts/python.exe scripts/04_report_figures.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from config import RESULT_DIR
from src import visualize
from src.evaluate import compare_table, load_loso


def main():
    missing = [t for t in ("cnn", "mlp") if load_loso(t) is None]
    if missing:
        print(f"缺少 {missing} 的 LOSO 结果，请先运行 scripts/03_train_cnn.py")
        sys.exit(1)

    # 1) 总对比表：所有 MindSpore 模型 + sklearn 基线（12 维 / 26 维特征集）
    rows = {}
    for tag, label in [("cnn", "1D-CNN(原始信号)"),
                       ("bilstm", "BiLSTM(原始信号)"),
                       ("cnn_aug", "1D-CNN+数据增强"),
                       ("mlp", "MLP(手工特征)")]:
        r = load_loso(tag)
        if r is not None:
            rows[label] = r
    for fn, subset in [("baseline_results_full.json", "12维特征"),
                       ("baseline_results_extended.json", "26维特征")]:
        p = RESULT_DIR / fn
        if not p.exists():
            continue
        for m, r in json.loads(p.read_text(encoding="utf-8")).items():
            rows[f"{m}({subset})"] = r
    compare_table(rows, tag="final")

    # 2) 重绘混淆矩阵与学习曲线
    for tag, title in [("cnn", "1D-CNN confusion matrix"), ("mlp", "MLP confusion matrix")]:
        r = load_loso(tag)
        visualize.plot_confusion(np.array(r["confusion"]), r["labels"],
                                 tag=f"confusion_{tag}", title=title)
        curve = json.loads((RESULT_DIR / f"curve_{tag}.json").read_text(encoding="utf-8"))
        visualize.plot_curves(curve, tag=f"curve_{tag}", title=f"{tag.upper()} learning curve (LOSO avg)")

    # 3) 数据图（波形/谱图/t-SNE 已由 01/02 生成，这里不重复）
    print("报告图已更新: outputs/figures/；总对比表: outputs/results/final_comparison.md")


if __name__ == "__main__":
    main()
