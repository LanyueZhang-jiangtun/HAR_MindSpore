# -*- coding: utf-8 -*-
"""
【甲】特征工程
====================================================
X (N, 3, W) 原始信号 -> (N, D) 特征矩阵。全部基于 xyz 模值（方位不变量），
对传感器佩戴位置/手机朝向天然鲁棒（跨设备验证已证实这一点）。

特征集（02_baseline.py --feature-set）：
  full      12 维：时域 9 + 频域 3（项目默认，现有对比表基于此）
  time      9 维：仅时域（消融）
  freq      3 维：仅频域（消融）
  extended  26 维：full + 14（逐轴 std×3、逐轴过零率×3、
             偏度/峰度/中位数×3、低/中/高频带能量占比×3、
             谱熵、样本熵）
"""
import numpy as np

from config import RATE, WINDOW

FEATURE_NAMES = [
    "mag_mean", "mag_std", "mag_min", "mag_max", "mag_p10", "mag_p90",
    "mag_rms", "mag_mean_abs_diff", "mag_max_abs_diff",
    "dominant_freq", "spectral_centroid", "band_energy_ratio",
]

EXTENDED_FEATURE_NAMES = [
    "ax_std", "ay_std", "az_std",
    "ax_zcr", "ay_zcr", "az_zcr",
    "mag_skew", "mag_kurt", "mag_median",
    "band_low_ratio", "band_mid_ratio", "band_high_ratio",
    "spectral_entropy", "sample_entropy",
]


def _time_features(mag: np.ndarray) -> list:
    d = np.abs(np.diff(mag))
    return [
        float(mag.mean()), float(mag.std()), float(mag.min()), float(mag.max()),
        float(np.quantile(mag, 0.1)), float(np.quantile(mag, 0.9)),
        float(np.sqrt(np.mean(mag ** 2))), float(d.mean()), float(d.max()),
    ]


def _power_freq(mag: np.ndarray):
    centered = mag - mag.mean()
    power = np.abs(np.fft.rfft(centered * np.hanning(WINDOW))) ** 2
    freq = np.fft.rfftfreq(WINDOW, 1 / RATE)
    return power, freq


def _freq_features(mag: np.ndarray) -> list:
    """频域：主导步频 / 谱质心 / 步频带(0.7~6Hz)能量占比"""
    power, freq = _power_freq(mag)
    band = (freq >= 0.7) & (freq <= 6.0)
    dominant = float(freq[band][np.argmax(power[band])]) if band.any() else 0.0
    centroid = float(np.sum(freq * power) / (np.sum(power) + 1e-12))
    hf_ratio = float(power[band].sum() / (power.sum() + 1e-12))
    return [dominant, centroid, hf_ratio]


def _sample_entropy(x: np.ndarray, m: int = 2, r: float = 0.25) -> float:
    """样本熵（m=2, r=0.25*std）；近似常值信号返回 0"""
    x = x - x.mean()
    std = float(x.std())
    if std < 1e-9:
        return 0.0
    r *= std
    X = np.lib.stride_tricks.sliding_window_view(x, m)
    X2 = np.lib.stride_tricks.sliding_window_view(x, m + 1)

    def _count(Xv):
        c = 0
        for i in range(len(Xv)):
            d = np.abs(Xv - Xv[i]).max(axis=1)
            c += int((d < r).sum()) - 1
        return c

    B, A = _count(X), _count(X2)
    return float(np.log(B / A)) if A > 0 and B > 0 else 0.0


def _extended_features(X: np.ndarray, mag: np.ndarray) -> np.ndarray:
    """扩充 14 维。X (N,3,W)，mag (N,W)"""
    power, freq = _power_freq(mag)
    # 逐轴 std / 过零率（去均值后符号变化比例）
    ax_std = X.std(axis=2)
    xc = X - X.mean(axis=2, keepdims=True)
    s = np.sign(xc)
    s[s == 0] = 1.0
    ax_zcr = np.mean(np.abs(np.diff(s, axis=2)) > 0, axis=2)
    # 分布形态：偏度 / 超额峰度 / 中位数
    z = (mag - mag.mean(axis=1, keepdims=True)) / (mag.std(axis=1, keepdims=True) + 1e-12)
    skew = (z ** 3).mean(axis=1)
    kurt = (z ** 4).mean(axis=1) - 3.0
    median = np.median(mag, axis=1)
    # 低(0.5~2) / 中(2~6) / 高(6~12) 频带能量占比
    total = power.sum(axis=1) + 1e-12
    low = (freq >= 0.5) & (freq < 2.0)
    mid = (freq >= 2.0) & (freq <= 6.0)
    high = (freq > 6.0) & (freq <= 12.0)
    band_ratios = np.stack(
        [power[:, low].sum(1), power[:, mid].sum(1), power[:, high].sum(1)], axis=1
    ) / total[:, None]
    # 谱熵
    p = power / total[:, None]
    sp_ent = -(p * np.log2(p + 1e-12)).sum(axis=1)
    # 样本熵
    se = np.array([_sample_entropy(m) for m in mag])
    return np.column_stack([ax_std, ax_zcr, skew, kurt, median[:, None],
                            band_ratios, sp_ent[:, None], se[:, None]])


def extract_features(X: np.ndarray, subset: str = "full") -> np.ndarray:
    """X (N, 3, W) -> (N, D)。subset: full(12)/time(9)/freq(3)/extended(26)"""
    mag = np.linalg.norm(X, axis=1)  # (N, W) 模值
    time_f = np.stack([np.array(_time_features(m)) for m in mag])
    freq_f = np.stack([np.array(_freq_features(m)) for m in mag])
    if subset == "time":
        return time_f
    if subset == "freq":
        return freq_f
    base = np.concatenate([time_f, freq_f], axis=1)
    if subset == "full":
        return base
    if subset == "extended":
        return np.concatenate([base, _extended_features(X, mag)], axis=1)
    raise ValueError(f"未知特征集: {subset}")
