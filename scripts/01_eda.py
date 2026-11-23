# -*- coding: utf-8 -*-
"""【甲】数据 EDA：逐被试逐类窗口数 + 波形/谱图
运行: E:/python_ziliao/.venv-ms311/Scripts/python.exe scripts/01_eda.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import FIG_DIR
from src import visualize
from src.dataset import data_summary, load_raw_windows


def main():
    table, n_win, names = data_summary()
    print(f"总窗口数: {n_win}，类别: {names}\n")
    header = f"{'被试':<20}" + "".join(f"{n:>10}" for n in names) + f"{'合计':>8}"
    print(header)
    print("-" * len(header))
    for p, row in table.items():
        print(f"{p:<20}" + "".join(f"{row[n]:>10}" for n in names) + f"{sum(row.values()):>8}")
    print("\n注意：LOSO 要求每个被试都覆盖全部类别，某被试缺类会导致其无法作为测试折。")

    X, y, people, names = load_raw_windows()
    print(f"\n原始波形张量: {X.shape} (窗口, 3轴, {X.shape[2]}采样点)")
    visualize.plot_waveform(X, y, names)
    visualize.plot_spectrogram(X, y, names)
    print(f"图已保存: {FIG_DIR}\\waveform.png, {FIG_DIR}\\spectrogram.png")


if __name__ == "__main__":
    main()
