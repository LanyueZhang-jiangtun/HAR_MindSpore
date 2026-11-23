# -*- coding: utf-8 -*-
"""
【甲】数据加载与窗口化
====================================================
UCI mHealth：10 名被试、50 Hz 胸戴三轴加速度计、13 个活动码（0~12）。
提供：
  load_raw_windows()  -> X (N, 3, WINDOW) 原始波形, y (N,) 标签索引, people, names
  data_summary()      -> 逐被试逐类窗口数（写报告用）
TODO(甲)：
  1. 确认 0 号活动码含义（休息/过渡），决定 USE_LABELS 的最终集合
  2. 扩充 USE_LABELS 到 6~8 类，重跑 01_eda 检查每被试每类窗口数
     （LOSO 要求每个被试都覆盖全部类别，否则该被试无法做测试折）
  3. 可选：读取 ankle / arm 传感器通道（log 中另有列），做多传感器融合
"""
import zipfile
from pathlib import Path

import numpy as np

from config import DATASET_ZIP, USE_LABELS, WINDOW


def _iter_logs(zip_path):
    """逐被试读取，yield (被试名, xyz (M,3) float32, activity (M,) int)"""
    with zipfile.ZipFile(zip_path) as archive:
        for name in sorted(n for n in archive.namelist() if n.endswith(".log")):
            with archive.open(name) as handle:
                data = np.loadtxt(handle, usecols=(0, 1, 2, 23))
            yield Path(name).stem, data[:, :3].astype(np.float32), data[:, -1].astype(int)


def load_raw_windows(use_labels=None):
    """
    按 WINDOW(3s) 不重叠切窗，跳过长度不足一个窗口的活动段。

    Returns
    -------
    X : (N, 3, WINDOW) float32  原始 xyz 波形（丙的 CNN 直接吃这个）
    y : (N,) int64              标签索引（0..len(names)-1）
    people : (N,) object        每个窗口所属被试
    names : list[str]           类别名（y 的索引对应 names）
    """
    use_labels = use_labels or USE_LABELS
    names = list(use_labels.values())
    xs, ys, ps = [], [], []
    for person, xyz, act in _iter_logs(DATASET_ZIP):
        boundaries = np.r_[0, np.flatnonzero(np.diff(act)) + 1, len(xyz)]
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            label = int(act[start])
            if label not in use_labels:
                continue
            for i in range(start, end - WINDOW + 1, WINDOW):
                xs.append(xyz[i:i + WINDOW])
                ys.append(use_labels[label])
                ps.append(person)
    X = np.stack(xs).transpose(0, 2, 1)  # (N, 3, W)
    y = np.array([names.index(v) for v in ys], dtype=np.int64)
    return X, y, np.array(ps, dtype=object), names


def data_summary(use_labels=None):
    """逐被试逐类窗口数 + 总量，返回 (table, n_windows, names)"""
    X, y, people, names = load_raw_windows(use_labels)
    table = {}
    for person in sorted(set(people.tolist())):
        mask = np.array([p == person for p in people])
        table[person] = {n: int(((y == i) & mask).sum()) for i, n in enumerate(names)}
    return table, len(X), names
