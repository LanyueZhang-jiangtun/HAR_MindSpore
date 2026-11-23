# -*- coding: utf-8 -*-
"""【丙】深度模型：1D-CNN（原始信号）vs MLP（手工特征）LOSO 对比 + 全量训练导出
运行: E:/python_ziliao/.venv-ms311/Scripts/python.exe scripts/03_train_cnn.py [--epochs 40] [--lr 1e-3]
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from config import EPOCHS, LEARNING_RATE, RESULT_DIR
from src import visualize
from src.dataset import load_raw_windows
from src.evaluate import compare_table
from src.features import extract_features
from src.models import MotionCNN, MotionMLP
from src.train_mindspore import loso_train, train_all


def _augment(X, y, seed):
    """训练折数据增强：原样 + 时移(±15 采样点) + 高斯噪声(σ=0.05 m/s²)，3 倍数据量"""
    rng = np.random.default_rng(seed)
    n = len(X)
    shifted = np.stack([np.roll(X[i], int(rng.integers(-15, 15)), axis=1) for i in range(n)])
    noisy = X + rng.normal(0.0, 0.05, X.shape)
    return np.concatenate([X, shifted, noisy]), np.concatenate([y, y, y])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--lr", type=float, default=LEARNING_RATE)
    parser.add_argument("--augment", action="store_true",
                        help="数据增强消融：训练折加 时移±15采样点/高斯噪声σ=0.05（3倍数据），"
                             "结果存 cnn_aug，不覆盖 cnn 主结果")
    args = parser.parse_args()

    X, y, people, names = load_raw_windows()
    print(f"窗口数: {len(y)}，类别: {names}\n")
    F = extract_features(X)

    if args.augment:
        print("=== 1D-CNN（原始波形 + 时移/噪声增强）LOSO ===")
        r_cnn = loso_train(lambda nc: MotionCNN(nc), "cnn_aug", X, y, people, names,
                           args.epochs, args.lr, augment=_augment)
        print(f"CNN_aug: acc={r_cnn['accuracy']:.4f} macroF1={r_cnn['macro_f1']:.4f} "
              f"耗时={r_cnn['seconds']:.1f}s")
        compare_table({"CNN(原始信号+增强)": r_cnn}, tag="cnn_aug")
        visualize.plot_confusion(np.array(r_cnn["confusion"]), names, tag="confusion_cnn_aug",
                                 title="1D-CNN (augmented) confusion matrix")
        curve = json.loads((RESULT_DIR / "curve_cnn_aug.json").read_text(encoding="utf-8"))
        visualize.plot_curves(curve, tag="curve_cnn_aug",
                              title="CNN (augmented) learning curve (LOSO avg)")
        print("完成（增强消融）。对照：无增强结果见 loso_cnn.json。")
        return

    print("=== 1D-CNN（原始波形，端到端）LOSO ===")
    r_cnn = loso_train(lambda nc: MotionCNN(nc), "cnn", X, y, people, names,
                       args.epochs, args.lr)
    print(f"CNN: acc={r_cnn['accuracy']:.4f} macroF1={r_cnn['macro_f1']:.4f} "
          f"耗时={r_cnn['seconds']:.1f}s\n")

    print("=== MLP（手工特征，对照组）LOSO ===")
    r_mlp = loso_train(lambda nc: MotionMLP(F.shape[1], nc), "mlp", F, y, people, names,
                       args.epochs, args.lr)
    print(f"MLP: acc={r_mlp['accuracy']:.4f} macroF1={r_mlp['macro_f1']:.4f} "
          f"耗时={r_mlp['seconds']:.1f}s\n")

    compare_table({"CNN(原始信号)": r_cnn, "MLP(手工特征)": r_mlp}, tag="mindspore")
    visualize.plot_confusion(np.array(r_cnn["confusion"]), names, tag="confusion_cnn",
                             title="1D-CNN confusion matrix")
    visualize.plot_confusion(np.array(r_mlp["confusion"]), names, tag="confusion_mlp",
                             title="MLP confusion matrix")
    for tag in ("cnn", "mlp"):
        curve = json.loads((RESULT_DIR / f"curve_{tag}.json").read_text(encoding="utf-8"))
        visualize.plot_curves(curve, tag=f"curve_{tag}",
                              title=f"{tag.upper()} learning curve (LOSO avg)")

    print("=== 全量训练 + checkpoint（给丁做部署） ===")
    train_all(lambda nc: MotionCNN(nc), "cnn_full", X, y, len(names), args.epochs, args.lr)
    print("\n完成。结果: outputs/results/，图: outputs/figures/")


if __name__ == "__main__":
    main()
