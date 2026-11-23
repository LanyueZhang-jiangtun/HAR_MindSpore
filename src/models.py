# -*- coding: utf-8 -*-
"""
【丙】MindSpore 模型定义
====================================================
  MotionMLP : 手工特征 (N, D) -> 对照组（复现已有 99.7% 结果）
  MotionCNN : 原始波形 (N, 3, 150) -> 主模型（1D-CNN，端到端）

环境坑（2026-11 实测，MindSpore 2.11.0 CPU / PYNATIVE）：
  numpy>=2.x 下 MindSpore 内部 Squeeze/Reshape 在梯度计算中报 SystemError
  （Conv1d/MaxPool1d 一进 TrainOneStepCell 就崩），根因是 numpy 1.x/2.x
  ABI 不匹配（stderr 提示 "_ARRAY_API not found"）。
  解决：requirements.txt 固定 numpy<2（本环境已装 1.26.4）。
  这个坑可以写进报告"平台适配"部分。
TODO(丙)：
  1. 增加 BiLSTM / ConvLSTM 作为第二个深度基线
  2. 超参搜索：通道数 / 核大小 / lr / dropout，记录每组结果
  3. 数据增强（时移 ±5 采样点 / 加高斯噪声）并做消融
  4. GRAPH 模式 + AMP 对比 PYNATIVE 的训练耗时（报告"华为平台"部分）
"""
from mindspore import nn

from config import WINDOW


class MotionMLP(nn.Cell):
    """对照：两层 MLP（与 a机器学习1122/train_mindspore.py 同结构）"""

    def __init__(self, n_features: int, n_classes: int):
        super().__init__()
        self.net = nn.SequentialCell(
            nn.Dense(n_features, 16), nn.ReLU(), nn.Dense(16, n_classes))

    def construct(self, x):
        return self.net(x)


class MotionCNN(nn.Cell):
    """
    1D-CNN 主模型。
    输入 (N, 3, WINDOW) 原始 xyz 波形（按通道标准化后），输出 (N, n_classes) logits。
    结构：Conv1D->BN->ReLU -> Conv1D->BN->ReLU -> MaxPool×2 -> Dropout -> FC
    池化后长度：150 -> 75 -> 37
    """

    def __init__(self, n_classes: int, n_channels: int = 3,
                 k1: int = 7, k2: int = 5, c1: int = 16, c2: int = 32,
                 dropout: float = 0.5):
        super().__init__()
        self.conv1 = nn.Conv1d(n_channels, c1, kernel_size=k1, pad_mode="same")
        self.bn1 = nn.BatchNorm1d(c1)
        self.conv2 = nn.Conv1d(c1, c2, kernel_size=k2, pad_mode="same")
        self.bn2 = nn.BatchNorm1d(c2)
        self.act = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.drop = nn.Dropout(p=dropout)
        self.flat = nn.Flatten(start_dim=1, end_dim=-1)
        self.fc = nn.Dense(c2 * (WINDOW // 4), n_classes)

    def construct(self, x):
        h = self.act(self.bn1(self.conv1(x)))
        h = self.act(self.bn2(self.conv2(h)))
        h = self.drop(self.pool(self.pool(h)))
        return self.fc(self.flat(h))


class MotionBiLSTM(nn.Cell):
    """
    BiLSTM 第二深度基线。
    输入 (N, 3, WINDOW) 原始 xyz 波形（按通道标准化后）
    -> 转置 (N, W, 3) -> 2 层双向 LSTM(64) -> 时间维平均池化 -> FC
    与 CNN 同为端到端（原始信号 -> logits），用于"深度模型结构"对比。
    """

    def __init__(self, n_classes: int, hidden: int = 64, layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_size=3, hidden_size=hidden,
                            num_layers=layers, bidirectional=True)
        self.drop = nn.Dropout(p=dropout)
        self.fc = nn.Dense(hidden * 2, n_classes)

    def construct(self, x):
        out = self.lstm(x.transpose((0, 2, 1)))  # 返回 (output, hidden...)，取 output (N, W, hidden*2)
        return self.fc(self.drop(out[0].mean(axis=1)))
