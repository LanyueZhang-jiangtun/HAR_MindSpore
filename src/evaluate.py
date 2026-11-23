# -*- coding: utf-8 -*-
"""
【丁】评估与对比
====================================================
  compare_table : 多个 LOSO 结果（sklearn 基线 + MindSpore 模型）-> 对比表（md + csv）
  load_loso     : 读取 outputs/results/loso_<tag>.json
TODO(丁)：
  1. 逐类 recall / 混淆对分析（哪两类最易混，写进报告）
  2. 学习曲线对比图（基线无曲线，只画 MindSpore 两组）
  3. 显著性：LOSO 10 折的配对检验（McNemar / Wilcoxon）
"""
import csv
import json

import numpy as np

from config import RESULT_DIR


def load_loso(tag):
    """读取单个 LOSO 结果 JSON，不存在返回 None"""
    p = RESULT_DIR / f"loso_{tag}.json"
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def compare_table(results: dict, tag: str = "comparison"):
    """
    results: {模型名: loso结果dict}（sklearn 与 MindSpore 结果同构，可直接合并）。
    输出 outputs/results/<tag>_comparison.md 与 .csv，并打印。
    """
    names = list(results)
    rows = ["| 指标 | " + " | ".join(names) + " |",
            "|" + "---|" * (len(names) + 1)]
    for key, label in [("accuracy", "Accuracy"), ("macro_f1", "Macro-F1"), ("seconds", "耗时(s)")]:
        fmt = "{:.1f}" if key == "seconds" else "{:.4f}"
        rows.append("| " + label + " | " + " | ".join(
            fmt.format(results[n][key]) for n in names) + " |")
    md = "\n".join(rows)
    (RESULT_DIR / f"{tag}_comparison.md").write_text(md + "\n", encoding="utf-8")
    with open(RESULT_DIR / f"{tag}_comparison.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["metric"] + names)
        w.writerow(["accuracy"] + [f"{results[n]['accuracy']:.4f}" for n in names])
        w.writerow(["macro_f1"] + [f"{results[n]['macro_f1']:.4f}" for n in names])
        w.writerow(["seconds"] + [f"{results[n]['seconds']:.1f}" for n in names])
    print(md)
    return md
