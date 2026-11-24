# -*- coding: utf-8 -*-
"""Global paths and experiment settings shared by the project scripts."""
import os
from pathlib import Path

# ---------- 路径 ----------
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "outputs"
FIG_DIR = OUTPUT_DIR / "figures"
RESULT_DIR = OUTPUT_DIR / "results"

# 数据集：UCI mHealth（doi:10.24432/C5TW22）。可用 HAR_DATASET_ZIP 指定任意路径。
_ZIP_CANDIDATES = [
    DATA_DIR / "mhealth-uci.zip",
    ROOT.parent / "a机器学习1122" / "mhealth-uci.zip",
]
_DATASET_ENV = os.environ.get("HAR_DATASET_ZIP")
if _DATASET_ENV:
    _dataset_path = Path(_DATASET_ENV).expanduser()
    DATASET_ZIP = (_dataset_path if _dataset_path.is_absolute() else ROOT / _dataset_path).resolve()
else:
    DATASET_ZIP = next((p for p in _ZIP_CANDIDATES if p.exists()), _ZIP_CANDIDATES[0])

# ---------- 信号参数 ----------
RATE = 50                      # 采样率 Hz（胸戴三轴加速度计）
WINDOW_SECONDS = 3             # 窗口长度（秒）
WINDOW = RATE * WINDOW_SECONDS  # 窗口长度（采样点）= 150

# ---------- 活动映射（README 中 L1~L12 的编号即标签码） ----------
ACTIVITY_MAP = {
    0: "rest",        # README 未列 0 号段，推测为休息/过渡段（甲：看波形确认后可改名或剔除）
    1: "stand",       # L1 站立
    2: "sit",         # L2 坐姿放松
    3: "lie",         # L3 平躺
    4: "walk",        # L4 步行
    5: "stairs",      # L5 爬楼梯
    6: "waist_bend",  # L6 弯腰
    7: "arm_elev",    # L7 手臂前抬
    8: "knee_bend",   # L8 屈膝下蹲
    9: "bike",        # L9 骑车
    10: "jog",        # L10 慢跑
    11: "run",        # L11 快跑
    12: "jump",       # L12 前后跳
}

# 当前阶段使用的类别子集（7 类：3 静态 + 4 运动，含 stairs；
# 已验证 10 个被试全部覆盖这 7 类（subject6 楼梯 18 窗），LOSO 可行，共 1398 窗口）
USE_LABELS = {1: "stand", 2: "sit", 3: "lie", 4: "walk", 5: "stairs", 9: "bike", 10: "jog"}

# ---------- 训练默认值 ----------
# 实测（2026-11）：lr=1e-3 全批量下 40 epoch 不够（MLP val acc 仅 0.67），80 epoch 达 0.99+
SEED = 42
EPOCHS = 80
LEARNING_RATE = 1e-3

# ---------- 输出目录 ----------
for _d in (OUTPUT_DIR, FIG_DIR, RESULT_DIR):
    _d.mkdir(parents=True, exist_ok=True)
