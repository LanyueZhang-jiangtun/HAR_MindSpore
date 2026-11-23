# -*- coding: utf-8 -*-
"""
【丁】部署模型导出 MindIR（.mindir）
====================================================
预处理已烤进模型：Wrapper 内部做标准化（训练集 mean/std），
导出的模型直接吃原始信号（需先重采样到 50 Hz、切 3s=150 点窗）。
产物: outputs/results/cnn_full.mindir
运行: E:/python_ziliao/.venv-ms311/Scripts/python.exe scripts/07_export_mindir.py

坑（2026-11-23 实测，MindSpore 2.11.0）：
  ms.load_checkpoint(ckpt, net=wrapper.net)（加载到子 Cell）会静默失败——
  参数名按顶层 Cell 解析（加 "net." 前缀），全部权重不加载但仍报 strict_load 通过，
  导出的是随机权重模型（样本内 23.7% vs 正确 94.4%）。
  正确做法：先加载到顶层裸 net，再把裸 net 塞进 wrapper（见 main）。
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import mindspore as ms
from mindspore import nn

from config import RESULT_DIR, USE_LABELS
from src.models import MotionCNN

NAMES = list(USE_LABELS.values())


class DeployWrapper(nn.Cell):
    """原始信号 (1, 3, 150) -> 标准化 -> CNN -> logits (1, C)"""

    def __init__(self, mean, std, net):
        """net: 已加载权重的 MotionCNN（不能在这里新建再 load 到子 Cell，见文件头）"""
        super().__init__()
        self.mean = ms.Tensor(mean)
        self.std = ms.Tensor(std)
        self.net = net

    def construct(self, x):
        return self.net((x - self.mean) / self.std)


def main():
    pp = np.load(RESULT_DIR / "cnn_full_preprocess.npz")
    mean, std = pp["mean"].astype(np.float32), pp["std"].astype(np.float32)

    bare = MotionCNN(len(NAMES))
    ms.load_checkpoint(str(RESULT_DIR / "cnn_full.ckpt"), net=bare, strict_load=True)
    bare.set_train(False)

    wrapper = DeployWrapper(mean, std, bare)
    wrapper.set_train(False)

    ms.set_context(mode=ms.GRAPH_MODE)
    x = ms.Tensor(np.zeros((1, 3, 150), dtype=np.float32))
    out_path = str(RESULT_DIR / "cnn_full")
    ms.export(wrapper, x, file_name=out_path, file_format="MINDIR")

    # 冒烟测试：wrapper 输出必须与裸模型一致（确认权重真的进了导出图）
    ref = bare(ms.Tensor(((x.asnumpy() - mean) / std).astype(np.float32))).asnumpy()
    got = wrapper(x).asnumpy()
    same = bool(np.allclose(ref, got, rtol=1e-3, atol=1e-3))
    print(f"MindIR 已导出: {out_path}.mindir")
    print(f"冒烟测试 wrapper vs 裸模型 logits 一致: {same}（logits 形状 {got.shape}）")
    if not same:
        raise SystemExit("wrapper 权重未正确加载，导出的 MindIR 无效")
    print("部署时输入要求: 50 Hz 采样、3 通道、每通道 150 点，单位 m/s²")


if __name__ == "__main__":
    main()
