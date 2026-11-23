# -*- coding: utf-8 -*-
"""【丙】BiLSTM 第二深度基线：LOSO 与 1D-CNN 对比（结果存 bilstm，不覆盖 cnn）
运行: E:/python_ziliao/.venv-ms311/Scripts/python.exe scripts/03b_train_bilstm.py [--epochs 80]
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
from src.models import MotionBiLSTM
from src.train_mindspore import loso_train


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--lr", type=float, default=LEARNING_RATE)
    args = parser.parse_args()

    X, y, people, names = load_raw_windows()
    print(f"窗口数: {len(y)}，类别: {names}\n=== BiLSTM（原始波形，端到端）LOSO ===")
    r = loso_train(lambda nc: MotionBiLSTM(nc), "bilstm", X, y, people, names,
                   args.epochs, args.lr)
    print(f"BiLSTM: acc={r['accuracy']:.4f} macroF1={r['macro_f1']:.4f} 耗时={r['seconds']:.1f}s")

    compare_table({"BiLSTM(原始信号)": r}, tag="bilstm")
    visualize.plot_confusion(np.array(r["confusion"]), names, tag="confusion_bilstm",
                             title="BiLSTM confusion matrix")
    curve = json.loads((RESULT_DIR / "curve_bilstm.json").read_text(encoding="utf-8"))
    visualize.plot_curves(curve, tag="curve_bilstm",
                          title="BiLSTM learning curve (LOSO avg)")
    print("\n完成。对照：1D-CNN 结果见 loso_cnn.json。")


if __name__ == "__main__":
    main()
