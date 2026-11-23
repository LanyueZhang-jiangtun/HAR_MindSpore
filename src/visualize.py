# -*- coding: utf-8 -*-
"""
【丁】图表（统一输出到 outputs/figures/）
====================================================
图内标签一律用英文，避免 Windows 下中文字体缺失显示方块；
中文标题放进报告正文。
TODO(丁)：
  1. 统一风格（字体大小/配色/图例位置），直接嵌报告
  2. 增加特征分布图（各特征 boxplot，按类别着色）
  3. 部署章：用 P01_walk_*.csv 实测数据推理的效果图
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from config import FIG_DIR, RATE

plt.rcParams["axes.unicode_minus"] = False


def plot_waveform(X, y, names, tag="waveform"):
    """每类取一个窗口的三轴波形"""
    fig, axes = plt.subplots(len(names), 1, figsize=(8, 2.0 * len(names)), sharex=True)
    t = np.arange(X.shape[-1]) / RATE
    for ax, (i, n) in zip(np.atleast_1d(axes), enumerate(names)):
        idx = int(np.where(y == i)[0][0])
        for c in range(X.shape[1]):
            ax.plot(t, X[idx, c], lw=0.8, label=f"axis {c}")
        ax.set_ylabel(names[i], fontsize=9)
        ax.legend(loc="upper right", fontsize=8, ncol=3)
    axes[-1].set_xlabel("time (s)")
    fig.suptitle("Example raw signal window per class")
    fig.tight_layout()
    fig.savefig(FIG_DIR / f"{tag}.png", dpi=150)
    plt.close(fig)


def plot_spectrogram(X, y, names, tag="spectrogram"):
    """每类一个窗口的幅度谱图（50 点帧长，50% 重叠）"""
    fig, axes = plt.subplots(len(names), 1, figsize=(8, 2.2 * len(names)))
    for ax, (i, n) in zip(np.atleast_1d(axes), enumerate(names)):
        idx = int(np.where(y == i)[0][0])
        mag = np.linalg.norm(X[idx], axis=0)
        frame, hop = 50, 25
        frames = np.stack([mag[i:i + frame] for i in range(0, len(mag) - frame + 1, hop)])
        spec = np.abs(np.fft.rfft(frames * np.hanning(frame), axis=1))
        im = ax.imshow(10 * np.log10(spec + 1e-6), origin="lower", aspect="auto",
                       extent=[0, len(mag) / RATE, 0, RATE / 2])
        ax.set_ylabel(names[i], fontsize=9)
        fig.colorbar(im, ax=ax, shrink=0.7)
    axes[-1].set_xlabel("time (s)")
    fig.suptitle("Magnitude spectrogram per class (dB)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / f"{tag}.png", dpi=150)
    plt.close(fig)


def plot_confusion(cm, names, tag="confusion", title=None):
    """混淆矩阵热力图（行=真实，列=预测）"""
    cm = np.asarray(cm)
    cm_n = cm / cm.sum(axis=1, keepdims=True).clip(min=1)
    fig, ax = plt.subplots(figsize=(4.5, 3.8))
    im = ax.imshow(cm_n, cmap="Blues")
    ax.set_xticks(range(len(names)), names)
    ax.set_yticks(range(len(names)), names)
    for a in range(cm.shape[0]):
        for b in range(cm.shape[1]):
            ax.text(b, a, str(cm[a, b]), ha="center", va="center",
                    color="white" if cm_n[a, b] > 0.5 else "black", fontsize=10)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title(title or "Confusion matrix (normalized)")
    fig.colorbar(im)
    fig.tight_layout()
    fig.savefig(FIG_DIR / f"{tag}.png", dpi=150)
    plt.close(fig)


def plot_tsne(F, y, names, tag="tsne"):
    """手工特征的 t-SNE 二维投影（报告"特征可分性"用）"""
    from sklearn.manifold import TSNE
    emb = TSNE(n_components=2, random_state=42, perplexity=20).fit_transform(F)
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    for i, n in enumerate(names):
        ax.scatter(emb[y == i, 0], emb[y == i, 1], s=12, label=n, alpha=0.8)
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.set_title("t-SNE of handcrafted features")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG_DIR / f"{tag}.png", dpi=150)
    plt.close(fig)


def plot_curves(curve, tag="curve", title=None):
    """学习曲线：LOSO 各折平均的训练 loss 与验证 acc"""
    loss, val = curve["loss"], curve["val_acc"]
    fig, ax1 = plt.subplots(figsize=(6, 4))
    ax1.plot(range(1, len(loss) + 1), loss, "b-", lw=1.5, label="train loss")
    ax1.set_xlabel("epoch")
    ax1.set_ylabel("train loss", color="b")
    ax2 = ax1.twinx()
    ax2.plot(np.arange(5, 5 * len(val) + 5, 5), val, "r-o", ms=4, lw=1.5, label="val acc (LOSO avg)")
    ax2.set_ylabel("val accuracy", color="r")
    ax2.set_ylim(0.9, 1.001)
    ax1.set_title(title or "Learning curve")
    fig.tight_layout()
    fig.savefig(FIG_DIR / f"{tag}.png", dpi=150)
    plt.close(fig)
