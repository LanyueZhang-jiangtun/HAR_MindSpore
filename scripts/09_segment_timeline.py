# -*- coding: utf-8 -*-
"""
【丁】长片段活动时间线识别：给一段长录音，输出"第 X 秒到第 Y 秒是什么活动"
====================================================
方案（滑动窗口 + 时序平滑 + 时间段合并）：
  1. 输入 CSV（~60 Hz）重采样到 50 Hz（np.interp，支持任意输入采样率）
  2. 滑动窗口：3 秒窗（150 点）、1 秒步长（50% 重叠）→ 1 秒时间分辨率
  3. 每窗提取 26 维模值特征（方位不变量）-> RF 分类器预测标签 + 最大概率（置信度）
  4. 时序平滑：±1 窗多数投票（消除边界单窗抖动）
  5. 连续同标签窗合并为时间段；短于 --min-seg 的段并入置信度更高的邻段

训练数据：mHealth 胸戴 4 类部署集（walk/stairs/bike/jog，798 窗）；
  --add-phone P04 把手机被试数据加入训练（run 自动映射为 jog，类外标签跳过）。

用法:
  python scripts/09_segment_timeline.py --file 长片段.csv
  python scripts/09_segment_timeline.py --file 长片段.csv --add-phone P04
  # 演示：把多个 30 秒片段拼接成长片段（时间自动连续）
  python scripts/09_segment_timeline.py --files a.csv b.csv c.csv --name demo

输出:
  控制台时间线表；
  outputs/results/timeline_<name>.json（时间段 + 置信度 + 逐窗明细）；
  outputs/figures/timeline_<name>.png（色块时间线图）
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import USE_LABELS, WINDOW, RATE, RESULT_DIR, FIG_DIR
from src.dataset import load_raw_windows
from src.features import extract_features

CLS4 = ["walk", "stairs", "bike", "jog"]
LABEL_MAP = {"run": "jog"}  # P02 label 写法 run -> 模型类别名 jog
REPO_PHONE_DIR = Path(__file__).resolve().parent.parent / "data" / "phone"
LEGACY_PHONE_DIR = Path(__file__).resolve().parent.parent.parent / "a机器学习1122"
COLORS = {"walk": "#4c9be8", "stairs": "#f5a623", "bike": "#7ed321", "jog": "#d0021b"}


def load_csv_signal(path):
    """CSV -> (t (M,), xyz (M,3))，t 单位秒"""
    rows = list(csv.DictReader(open(path, encoding="utf-8-sig")))
    t = np.array([float(r["time_s"]) for r in rows])
    xyz = np.array([[float(r["ax_m_s2"]), float(r["ay_m_s2"]), float(r["az_m_s2"])]
                    for r in rows])
    return t, xyz, rows[0].get("label", "")


def resample50(t, xyz):
    t_new = np.arange(0.0, t[-1] + 1e-9, 1.0 / RATE)
    out = np.stack([np.interp(t_new, t, xyz[:, c]) for c in range(3)], axis=1)
    return out.astype(np.float32)


def load_long(paths):
    """一个或多个 CSV 拼成连续 50Hz 信号；返回 (xyz (T,3), 总时长 s)"""
    chunks, total = [], 0.0
    for p in paths:
        t, xyz, _ = load_csv_signal(p)
        chunks.append(resample50(t, xyz))
        total += float(t[-1])
    return np.concatenate(chunks, axis=0), total


def sliding_windows(xyz, win=WINDOW, stride=None):
    """(T,3) -> (N,3,win) 滑动窗口，默认 1 秒步长"""
    stride = stride or RATE
    n = (len(xyz) - win) // stride + 1
    idx = np.arange(n)[:, None] * stride + np.arange(win)[None, :]
    return xyz[idx].transpose(0, 2, 1)


def phone_data_dir():
    """仓库内 data/phone/ 有 P*.csv 时优先，否则回退旧目录"""
    return REPO_PHONE_DIR if any(REPO_PHONE_DIR.glob("P*.csv")) else LEGACY_PHONE_DIR


def load_phone_windows(subj):
    """手机被试数据 -> (X (N,3,W), labels idx)，仅 4 类内窗口"""
    xs, ys = [], []
    for p in sorted(phone_data_dir().glob(f"{subj}_*.csv")):
        t, xyz, raw = load_csv_signal(p)
        label = LABEL_MAP.get(raw, raw)
        if label not in CLS4:
            continue
        x50 = resample50(t, xyz)
        n = len(x50) // WINDOW
        for i in range(n):
            xs.append(x50[i * WINDOW:(i + 1) * WINDOW])
            ys.append(CLS4.index(label))
    if not xs:
        return None, None
    return np.stack(xs).transpose(0, 2, 1), np.array(ys)


def train_model(add_phone=None):
    """RF 26 维，胸戴 4 类（798 窗）+ 可选手机数据"""
    names7 = list(USE_LABELS.values())
    idx4 = [names7.index(c) for c in CLS4]
    X, y, _, _ = load_raw_windows()
    m = np.isin(y, idx4)
    Xtr = X[m]
    ytr = np.array([idx4.index(int(v)) for v in y[m]])
    n_phone = 0
    if add_phone:
        Xp, yp = load_phone_windows(add_phone)
        if Xp is not None:
            Xtr = np.concatenate([Xtr, Xp])
            ytr = np.concatenate([ytr, yp])
            n_phone = len(yp)
    F = extract_features(Xtr, subset="extended")
    rf = RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1).fit(F, ytr)
    return rf, n_phone


def smooth_labels(lab, k=3):
    """±k//2 窗多数投票"""
    out = lab.copy()
    for i in range(len(lab)):
        lo, hi = max(0, i - k // 2), min(len(lab), i + k // 2 + (k % 2))
        vals, cnts = np.unique(lab[lo:hi], return_counts=True)
        out[i] = vals[int(cnts.argmax())]
    return out


def merge_short(segs, min_seg):
    """最短段反复并入平均置信度更高的邻段，直到所有段 >= min_seg"""
    segs = [dict(s) for s in segs]
    while len(segs) > 1:
        durs = [s["end_s"] - s["start_s"] for s in segs]
        i = int(np.argmin(durs))
        if durs[i] >= min_seg:
            break
        prev, nxt = (segs[i - 1] if i > 0 else None), (segs[i + 1] if i + 1 < len(segs) else None)
        if prev is None:
            j = i + 1
        elif nxt is None:
            j = i - 1
        else:
            j = i - 1 if prev["confidence"] >= nxt["confidence"] else i + 1
        a, b = (segs[i], segs[j]) if i < j else (segs[j], segs[i])
        a["start_s"], a["end_s"] = min(a["start_s"], b["start_s"]), max(a["end_s"], b["end_s"])
        a["n_windows"] += b["n_windows"]
        segs.pop(i if i < j else j)
    return segs


def make_plot(name, total, segs, conf, t_start):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(max(8.0, total / 12.0), 4.2),
                                   sharex=True, gridspec_kw={"height_ratios": [1, 1.2]})
    for s in segs:
        ax1.axvspan(s["start_s"], s["end_s"], color=COLORS[s["label"]], alpha=0.85)
        mid = (s["start_s"] + s["end_s"]) / 2
        if s["end_s"] - s["start_s"] > 4:
            ax1.text(mid, 0.5, f'{s["label"]}\n{s["start_s"]:.0f}-{s["end_s"]:.0f}s',
                     ha="center", va="center", fontsize=9, color="white")
    ax1.set_yticks([0.5])
    ax1.set_yticklabels(["timeline"])
    ax1.set_ylim(0, 1)
    ax1.set_title(f"Activity timeline: {name} ({total:.0f}s)")
    ax2.step(t_start, conf, where="post", color="#555555", lw=1.2)
    ax2.set_ylabel("confidence")
    ax2.set_xlabel("time (s)")
    ax2.set_xlim(0, total)
    ax2.grid(alpha=0.3)
    fig.tight_layout()
    out = FIG_DIR / f"timeline_{name}.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", default=None, help="长片段 CSV")
    ap.add_argument("--files", nargs="+", default=None, help="多个 CSV 拼接成长片段（演示用）")
    ap.add_argument("--add-phone", default=None, help="把手机被试数据加入训练（如 P04）")
    ap.add_argument("--stride", type=float, default=1.0, help="滑动步长（秒，默认 1）")
    ap.add_argument("--win", type=float, default=3.0, help="窗口长度（秒，默认 3）")
    ap.add_argument("--min-seg", type=float, default=2.0, help="最短时间段（秒，默认 2）")
    ap.add_argument("--name", default=None, help="输出文件前缀")
    args = ap.parse_args()

    if args.file:
        paths, base = [args.file], Path(args.file).stem
    elif args.files:
        paths, base = args.files, "concat"
    else:
        ap.error("需要 --file 或 --files")
    name = args.name or base

    rf, n_phone = train_model(args.add_phone)
    xyz, total = load_long(paths)
    if len(xyz) < WINDOW:
        sys.exit(f"信号太短（{total:.1f}s < {args.win}s），无法切窗")

    stride = int(args.stride * RATE)
    win = int(args.win * RATE)
    W = sliding_windows(xyz, win, stride)
    F = extract_features(W, subset="extended")
    pred = rf.predict(F)
    conf = rf.predict_proba(F).max(axis=1)
    lab = smooth_labels(pred, k=3)
    t_start = np.arange(len(lab)) * args.stride
    t_end = t_start + args.win

    segs, i = [], 0
    while i < len(lab):
        j = i
        while j + 1 < len(lab) and lab[j + 1] == lab[i]:
            j += 1
        segs.append({"start_s": round(float(t_start[i]), 1), "end_s": round(float(t_end[j]), 1),
                     "label": CLS4[lab[i]], "confidence": round(float(conf[i:j + 1].mean()), 3),
                     "n_windows": int(j - i + 1)})
        i = j + 1
    segs = merge_short(segs, args.min_seg)

    print(f"== 时间线 {name}（总长 {total:.1f}s，{args.win:.0f}s 窗 / {args.stride:.0f}s 步长，"
          f"RF 26 维 胸戴{798}窗" + (f" + 手机{n_phone}窗" if n_phone else "") + "）==")
    for s in segs:
        bar = "█" * max(1, int((s["end_s"] - s["start_s"]) // args.stride))
        print(f'  {s["start_s"]:6.1f} - {s["end_s"]:6.1f}s  {s["label"]:<7} '
              f'置信度 {s["confidence"]:.2f}  {bar}')

    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    out_json = RESULT_DIR / f"timeline_{name}.json"
    out_json.write_text(json.dumps({
        "name": name, "total_duration_s": round(total, 1),
        "window_s": args.win, "stride_s": args.stride,
        "model": f"RF 26-dim extended, chest 798 windows" + (f" + phone {n_phone} windows" if n_phone else ""),
        "segments": segs,
        "per_window": [{"t_s": round(float(t), 1), "label": CLS4[l], "conf": round(float(c), 3)}
                       for t, l, c in zip(t_start, lab, conf)],
    }, ensure_ascii=False, indent=1), encoding="utf-8")
    out_png = make_plot(name, total, segs, conf, t_start)
    print(f"已保存: {out_json.relative_to(Path(__file__).resolve().parent.parent)}"
          f"  {out_png.relative_to(Path(__file__).resolve().parent.parent)}")


if __name__ == "__main__":
    main()
