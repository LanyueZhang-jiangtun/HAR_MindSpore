# -*- coding: utf-8 -*-
"""
【丁】手机真机数据验证：重采样 -> 轴对齐 -> 3s 切窗 -> 加载 checkpoint 推理 -> 预测 vs 真实标签
====================================================
运行: E:/python_ziliao/.venv-ms311/Scripts/python.exe scripts/05_phone_test.py
      可选: --data-dir 数据目录  --pattern 文件名匹配（默认 ../a机器学习1122 下 P*.csv，
      覆盖 P01~P04 全部被试）

前置条件：
  1. 已跑过 scripts/03_train_cnn.py（产出 cnn_full.ckpt + cnn_full_preprocess.npz，
     类别 = config.USE_LABELS，当前 7 类）
  2. 手机 CSV 列名: time_s, ax_m_s2, ay_m_s2, az_m_s2, label, participant
     （每文件单一活动、单一被试）

处理链（与训练保持一致的关键点）：
  - 手机原始采样率（实测 ≈60 Hz）重采样到 50 Hz（np.interp）
  - 轴对齐：mHealth 胸戴传感器重力在 x 轴（均值 ≈ -9.8），手机重力轴不定
    （P01 实测在 z 轴）。取 |直流分量| 最大的轴映射为 x 并把符号翻成负值。
    注意：y/z 两轴的相对方向无法从数据恢复，这是已知局限（可写报告）。
  - 不重叠 3s 窗（150 采样点），用训练集的 mean/std 标准化后推理
TODO(丁)：
  1. 补齐其余 6 类的手机录制后重跑本脚本，得完整混淆矩阵
  2. 轴对齐开/关消融对比（报告"跨设备"章节素材）
  3. 多人录制后按 participant 分组分析跨人一致性
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import mindspore as ms

from config import DATA_DIR, RATE, RESULT_DIR, USE_LABELS, WINDOW
from src.models import MotionCNN

NAMES = list(USE_LABELS.values())


def load_phone_csv(path):
    """读手机 CSV -> (t, xyz (M,3) float64, label, participant)"""
    with open(path) as f:
        rows = list(csv.DictReader(f))
    t = np.array([float(r["time_s"]) for r in rows])
    xyz = np.array([[float(r["ax_m_s2"]), float(r["ay_m_s2"]), float(r["az_m_s2"])] for r in rows])
    labels = sorted(set(r["label"] for r in rows))
    parts = sorted(set(r["participant"] for r in rows))
    assert len(labels) == 1, f"{path.name} 含多个标签 {labels}，请按每文件一种活动录制"
    assert len(parts) == 1, f"{path.name} 含多个被试 {parts}"
    return t, xyz, labels[0], parts[0]


def resample_to_50hz(t, xyz):
    """任意采样率 -> 50 Hz（与训练一致）"""
    t_new = np.arange(0.0, t[-1] + 1e-9, 1.0 / RATE)
    xyz_new = np.stack([np.interp(t_new, t, xyz[:, c]) for c in range(3)], axis=1)
    return xyz_new.astype(np.float32)


def align_axes(xyz):
    """重力轴对齐到 x（训练约定：x 轴均值 ≈ -9.8）
    返回 (对齐后 xyz, 原重力轴下标)"""
    dc = xyz.mean(axis=0)
    g = int(np.argmax(np.abs(dc)))
    order = [g] + [i for i in range(3) if i != g]
    aligned = xyz[:, order]
    if aligned[:, 0].mean() > 0:
        aligned[:, 0] = -aligned[:, 0]  # 符号翻负，对齐训练分布
    return aligned, g, dc


def main():
    parser = argparse.ArgumentParser()
    default_dir = Path(__file__).resolve().parent.parent.parent / "a机器学习1122"
    parser.add_argument("--data-dir", default=str(default_dir))
    parser.add_argument("--pattern", default="P*.csv")
    parser.add_argument("--labels", default=None,
                        help="文件名->真实标签 的 JSON（覆盖 CSV 的 label 列）；"
                             "默认 data/phone_labels.json（存在时生效）")
    parser.add_argument("--no-align", action="store_true",
                        help="轴对齐消融：不重排/翻转轴向，按原始轴直接推理")
    args = parser.parse_args()

    labels_file = Path(args.labels) if args.labels else DATA_DIR / "phone_labels.json"
    overrides = json.loads(labels_file.read_text(encoding="utf-8")) if labels_file.exists() else {}

    ms.set_context(mode=ms.PYNATIVE_MODE)

    # 加载 03 产出的全量模型与预处理参数
    pp = np.load(RESULT_DIR / "cnn_full_preprocess.npz")
    mean, std = pp["mean"], pp["std"]
    net = MotionCNN(len(NAMES))
    ms.load_checkpoint(str(RESULT_DIR / "cnn_full.ckpt"), net=net, strict_load=True)
    net.set_train(False)

    results = {"model_classes": NAMES,
               "labels_file": str(labels_file) if overrides else None,
               "axis_aligned": not args.no_align,
               "files": []}
    all_true, all_pred = [], []
    for path in sorted(Path(args.data_dir).glob(args.pattern)):
        t, xyz, csv_label, participant = load_phone_csv(path)
        label = overrides.get(path.name, csv_label)
        if label != csv_label:
            print(f"{path.name}: 真实标签 {label}（CSV 标为 {csv_label}，已被 {labels_file.name} 覆盖）")
        xyz50 = resample_to_50hz(t, xyz)
        if args.no_align:
            aligned, g = xyz50, None
        else:
            aligned, g, _ = align_axes(xyz50)
        n_win = len(aligned) // WINDOW
        X = np.stack([aligned[i * WINDOW:(i + 1) * WINDOW] for i in range(n_win)]).transpose(0, 2, 1)
        Xs = ((X - mean) / std).astype(np.float32)
        pred = net(ms.Tensor(Xs)).asnumpy().argmax(axis=1)
        pred_names = [NAMES[p] for p in pred]
        ok = int(np.sum(np.array(pred_names) == label))
        uniq, cnt = np.unique(pred_names, return_counts=True)
        row = {
            "file": path.name, "participant": participant, "true_label": label,
            "windows": n_win, "correct": ok, "acc": ok / max(n_win, 1),
            "predictions": pred_names,
            "gravity_axis": g, "dc_offset": [round(float(v), 2) for v in aligned.mean(axis=0)],
            "pred_distribution": {k: int(v) for k, v in zip(uniq, cnt)},
        }
        results["files"].append(row)
        all_true += [label] * n_win
        all_pred += pred_names
        gtxt = "无(--no-align)" if g is None else chr(97 + g)
        print(f"{path.name}: 真实={label}  acc={row['acc']:.0%} ({ok}/{n_win})  "
              f"重力轴={gtxt}  预测分布={row['pred_distribution']}")

    if all_true:
        known = [p for t_, p in zip(all_true, all_pred) if t_ in NAMES]
        known_true = [t_ for t_ in all_true if t_ in NAMES]
        results["overall"] = {
            "windows": len(all_true),
            "known_class_windows": len(known_true),
            "acc_on_known_classes": float(np.mean(np.array(known) == np.array(known_true))) if known else None,
        }
    out_name = "phone_test_no_align.json" if args.no_align else "phone_test.json"
    (RESULT_DIR / out_name).write_text(
        json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n总体: {results.get('overall')}")
    print(f"已保存 outputs/results/{out_name}")
    print("说明: 真实标签不在模型类别集内的窗口不计入准确率——先补录缺失类别、重跑 03 再跑本脚本")


if __name__ == "__main__":
    main()
