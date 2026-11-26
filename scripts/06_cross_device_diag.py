# -*- coding: utf-8 -*-
"""
【丁/可选】跨设备差异定量诊断
====================================================
数据：默认目录扫描 P*.csv（仓库内 data/phone/，当前 23 段：P02×11 / P04×12，
空时回退旧目录 ../a机器学习1122；标错的文件仍可用 data/phone_labels.json 覆盖真实标签）

每个手机文件输出：
1. 高频 RMS（0.5~8 Hz 带通模值，方位不变量）vs 胸戴同类别中位数
   （walk 对胸戴 walk，jog 对胸戴 jog；类外活动无参照）
2. |标准化输入|（用训练集 mean/std）vs 胸戴整体值：CNN 输入分布漂移程度
3. RF（12 维模值特征，胸戴全量训练）跨设备预测：模值特征通路是否抗跨设备
运行: E:/python_ziliao/.venv-ms311/Scripts/python.exe scripts/06_cross_device_diag.py
"""
import csv
import json
import sys
from pathlib import Path

import numpy as np
from scipy import signal as sp
from sklearn.ensemble import RandomForestClassifier

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import DATA_DIR, RATE, RESULT_DIR, USE_LABELS, WINDOW
from src.dataset import load_raw_windows
from src.features import extract_features

NAMES = list(USE_LABELS.values())

REPO_PHONE_DIR = Path(__file__).resolve().parent.parent / "data" / "phone"
LEGACY_PHONE_DIR = Path(__file__).resolve().parent.parent.parent / "a机器学习1122"


def phone_data_dir():
    """仓库内 data/phone/ 有 P*.csv 时优先，否则回退旧目录"""
    return REPO_PHONE_DIR if any(REPO_PHONE_DIR.glob("P*.csv")) else LEGACY_PHONE_DIR


def resample_to_50hz(t, xyz):
    t_new = np.arange(0.0, t[-1] + 1e-9, 1.0 / RATE)
    return np.stack([np.interp(t_new, t, xyz[:, c]) for c in range(3)], axis=1).astype(np.float32)


def hp_rms(xyz_t3):
    """0.5~8 Hz 带通模值的 RMS（m/s²），方位不变，反映运动强度。输入 (T, 3)"""
    mag = np.linalg.norm(xyz_t3, axis=1)
    b, a = sp.butter(2, [0.5 / 25.0, 8.0 / 25.0], btype="band")
    return float(np.sqrt(np.mean(sp.filtfilt(b, a, mag) ** 2)))


def _windows(xyz_t3):
    """(T, 3) -> (N, 3, WINDOW) 不重叠 3s 窗"""
    n = len(xyz_t3) // WINDOW
    return np.stack([xyz_t3[i * WINDOW:(i + 1) * WINDOW] for i in range(n)]).transpose(0, 2, 1)


def align_axes(xyz_t3):
    """|直流| 最大的轴映射为 x 并翻负号（胸戴训练约定：重力在 x、均值≈-9.8）"""
    dc = xyz_t3.mean(axis=0)
    g = int(np.argmax(np.abs(dc)))
    order = [g] + [i for i in range(3) if i != g]
    aligned = xyz_t3[:, order]
    if aligned[:, 0].mean() > 0:
        aligned[:, 0] = -aligned[:, 0]
    return aligned, g


def main():
    default_dir = phone_data_dir()
    labels_file = DATA_DIR / "phone_labels.json"
    overrides = json.loads(labels_file.read_text(encoding="utf-8")) if labels_file.exists() else {}

    X, y, people, _ = load_raw_windows()

    # 1) 胸戴参照：逐类高频 RMS
    chest_rms = {}
    print("== 1) 胸戴参照 高频RMS（运动强度，方位不变量，m/s²）==")
    for c in ("walk", "jog", "bike"):
        arr = np.array([hp_rms(w.T) for w in X[y == NAMES.index(c)]])
        chest_rms[c] = arr
        print(f"  {c}: 中位数 {np.median(arr):.3f}  [{np.quantile(arr, .1):.3f}, {np.quantile(arr, .9):.3f}]")

    # 2) CNN 标准化输入参照
    pp = np.load(RESULT_DIR / "cnn_full_preprocess.npz")
    mean, std = pp["mean"], pp["std"]
    chest_std_in = float(np.abs((X - mean) / std).mean())

    # 3) RF 特征通路（胸戴全量训练）
    F = extract_features(X)
    rf = RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1).fit(F, y)
    print(f"\n  （RF 特征通路在胸戴数据上的训练集精度: {rf.score(F, y):.4f}）")

    print("\n== 2)+3) 逐手机文件（真实标签来自 phone_labels.json）==")
    for path in sorted(default_dir.glob("P*.csv")):
        rows = list(csv.DictReader(open(path)))
        t = np.array([float(r["time_s"]) for r in rows])
        xyz = np.array([[float(r["ax_m_s2"]), float(r["ay_m_s2"]), float(r["az_m_s2"])] for r in rows])
        xyz50 = resample_to_50hz(t, xyz)
        truth = overrides.get(path.name, sorted(set(r["label"] for r in rows))[0])

        phone_rms = hp_rms(xyz50)
        aligned, _ = align_axes(xyz50)
        phone_std_in = float(np.abs((_windows(aligned) - mean) / std).mean())
        Fp = extract_features(_windows(xyz50))  # 模值特征方位不变，用原始数据提取
        pred = rf.predict(Fp)
        n = len(pred)
        uniq, cnt = np.unique(pred, return_counts=True)
        dist = ", ".join(f"{NAMES[i]}×{int(c)}" for i, c in zip(uniq, cnt))

        pos = path.name.split("_")[2]
        print(f"\n{path.name}  真实={truth}  位置={pos}")
        if truth in chest_rms:
            ref = chest_rms[truth]
            n_true = int((pred == NAMES.index(truth)).sum())
            print(f"  高频RMS {phone_rms:.3f} = 胸戴{truth}中位数({np.median(ref):.3f})的 "
                  f"{phone_rms / np.median(ref):.2f}x")
            print(f"  RF(模值特征): {n_true}/{n} 判为 {truth}  分布: {dist}")
        else:
            print(f"  高频RMS {phone_rms:.3f}（类外活动，无胸戴参照）")
            print(f"  RF(模值特征): 类外不计对错  分布: {dist}")
        print(f"  |标准化输入| {phone_std_in:.3f}（胸戴 {chest_std_in:.3f} 的 {phone_std_in / chest_std_in:.2f}x）")


if __name__ == "__main__":
    main()
