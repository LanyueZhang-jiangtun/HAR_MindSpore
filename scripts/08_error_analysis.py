# -*- coding: utf-8 -*-
"""
【乙】逐被试误差分析：全模型逐被试精度表 + 各模型最易混类别对 + 手机验证汇总
=====================================================================
读取 outputs/results/ 下的 loso_*.json / baseline_per_subject_full.json / phone_test.json，
写 outputs/results/error_analysis.md（报告可直接引用）。
运行: E:/python_ziliao/.venv-ms311/Scripts/python.exe scripts/08_error_analysis.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import RESULT_DIR

MODEL_FILES = [
    ("1D-CNN", "loso_cnn.json"),
    ("MLP", "loso_mlp.json"),
    ("BiLSTM", "loso_bilstm.json"),
    ("CNN+增强", "loso_cnn_aug.json"),
]


def main():
    lines = ["# 逐被试误差分析", ""]

    # 1) 逐被试精度表（行=被试，列=模型）
    per_subj = {}
    for label, fn in MODEL_FILES:
        p = RESULT_DIR / fn
        if p.exists():
            per_subj[label] = json.loads(p.read_text(encoding="utf-8"))["per_subject"]
    bp = RESULT_DIR / "baseline_per_subject_full.json"
    if bp.exists():
        for m, ps in json.loads(bp.read_text(encoding="utf-8")).items():
            per_subj[m] = ps
    # sklearn 模型也计入易混类别对（第 3 节用）；用 extend 避免函数内全局变量被当局部
    MODEL_FILES.extend([("LogisticRegression", "baseline_results_full.json"),
                        ("SVM", "baseline_results_full.json"),
                        ("RandomForest", "baseline_results_full.json"),
                        ("GradientBoosting", "baseline_results_full.json")])
    subjects = sorted(set().union(*[set(ps) for ps in per_subj.values()]))

    lines.append("## 1. 逐被试精度（LOSO，每格=该被试做测试折时的正确率）")
    lines.append("")
    lines.append("| 被试 | " + " | ".join(per_subj) + " |")
    lines.append("|---" * (len(per_subj) + 1) + "|")
    for s in subjects:
        cells = []
        for ps in per_subj.values():
            if s in ps:
                c, t = ps[s]["correct"], ps[s]["total"]
                cells.append(f"{c}/{t} ({c / t:.0%})")
            else:
                cells.append("—")
        lines.append(f"| {s.replace('mHealth_subject', 'S')} | " + " | ".join(cells) + " |")
    lines.append("")

    # 2) 各模型最差/最好被试
    lines.append("## 2. 各模型最好/最差被试")
    lines.append("")
    for label, ps in per_subj.items():
        accs = {s: v["correct"] / v["total"] for s, v in ps.items()}
        best = max(accs, key=accs.get)
        worst = min(accs, key=accs.get)
        lines.append(f"- {label}: 最好 {best.replace('mHealth_subject', 'S')} "
                     f"({accs[best]:.0%})，最差 {worst.replace('mHealth_subject', 'S')} ({accs[worst]:.0%})")
    lines.append("")

    # 3) 各模型最易混类别对（混淆矩阵 top-3 非对角）
    lines.append("## 3. 各模型最易混类别对（LOSO 汇总混淆矩阵 top-3）")
    lines.append("")
    for label, fn in MODEL_FILES:
        p = RESULT_DIR / fn
        if not p.exists():
            continue
        r = json.loads(p.read_text(encoding="utf-8"))
        r = r.get(label, r) if label in r else r  # baseline_results_*.json 是 {模型: 结果} 嵌套
        cm, names = r["confusion"], r["labels"]
        pairs = sorted(((cm[i][j], names[i], names[j])
                        for i in range(len(names)) for j in range(len(names)) if i != j),
                       reverse=True)[:3]
        lines.append(f"- {label}: " + ", ".join(f"{a}→{b}×{n}" for n, a, b in pairs))
    lines.append("")

    # 4) 手机验证汇总
    phone = RESULT_DIR / "phone_test.json"
    if phone.exists():
        r = json.loads(phone.read_text(encoding="utf-8"))
        lines.append("## 4. 手机真机验证（P01 三段，真实标签见 data/phone_labels.json）")
        lines.append("")
        lines.append(f"模型类别: {', '.join(r['model_classes'])}，轴对齐: {r.get('axis_aligned', True)}")
        lines.append("")
        lines.append("| 文件 | 真实 | 窗口 | 正确 | 预测分布 |")
        lines.append("|---|---|---|---|---|")
        for f in r["files"]:
            dist = ", ".join(f"{k}×{v}" for k, v in f["pred_distribution"].items())
            lines.append(f"| {f['file'][8:20]}… | {f['true_label']} | {f['windows']} "
                         f"| {f['correct']} | {dist} |")
        lines.append("")

    out = "\n".join(lines)
    (RESULT_DIR / "error_analysis.md").write_text(out, encoding="utf-8")
    print(out)
    print(f"已保存 {RESULT_DIR / 'error_analysis.md'}")


if __name__ == "__main__":
    main()
