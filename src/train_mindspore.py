# -*- coding: utf-8 -*-
"""
【丙】MindSpore 训练框架（LOSO + 学习曲线 + 全量训练导出）
====================================================
  loso_train : 留一被试交叉验证，输出与 sklearn 基线同构的结果 JSON + 学习曲线
  train_all  : 全量数据训练，保存 .ckpt + 预处理参数，并做恢复一致性验证（丁的部署用）
TODO(丙)：
  1. early stopping / 学习率调度（nn.cosine_decay 等）
  2. 数据增强消融
  3. 记录参数量（net.trainable_params() 求和）与单折耗时
"""
import json
import time

import numpy as np
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.model_selection import LeaveOneGroupOut

import mindspore as ms
from mindspore import nn

from config import EPOCHS, LEARNING_RATE, RESULT_DIR, SEED


def _standardize_fit(X_train):
    """按通道拟合均值/方差（只在训练折上拟合，避免数据泄漏）"""
    shape = X_train.shape
    flat = X_train.reshape(shape[0], -1)
    mean = flat.mean(axis=0, keepdims=True)
    std = flat.std(axis=0, keepdims=True) + 1e-8
    return mean.reshape((1,) + shape[1:]), std.reshape((1,) + shape[1:])


def _predict(net, X_std):
    """X_std 必须已用训练折的均值/方差标准化（与训练时一致）"""
    net.set_train(False)
    out = net(ms.Tensor(X_std.astype(np.float32, copy=False))).asnumpy()
    net.set_train(True)
    return out.argmax(axis=1)


def fit_model(factory, X_tr, y_tr, X_te, y_te, n_classes,
              epochs=EPOCHS, lr=LEARNING_RATE, eval_every=5):
    """
    训练一折。factory: n_classes -> nn.Cell。
    Returns (net, (mean, std), history{"loss": [...], "val_acc": [...]})
    """
    ms.set_seed(SEED)
    mean, std = _standardize_fit(X_tr)
    inputs = ms.Tensor(((X_tr - mean) / std).astype(np.float32))
    targets = ms.Tensor(y_tr.astype(np.int32))
    te_in = ms.Tensor(((X_te - mean) / std).astype(np.float32))

    net = factory(n_classes)
    loss_fn = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction="mean")
    opt = nn.Adam(net.trainable_params(), learning_rate=lr)
    step = nn.TrainOneStepCell(nn.WithLossCell(net, loss_fn), opt)
    step.set_train(True)

    history = {"loss": [], "val_acc": []}
    for epoch in range(epochs):
        step(inputs, targets)
        history["loss"].append(float(loss_fn(net(inputs), targets)))
        if (epoch + 1) % eval_every == 0:
            net.set_train(False)
            acc = float((net(te_in).asnumpy().argmax(axis=1) == y_te).mean())
            net.set_train(True)
            history["val_acc"].append(acc)
    net.set_train(False)
    return net, (mean, std), history


def loso_train(factory, tag, X, y, people, names, epochs=EPOCHS, lr=LEARNING_RATE,
               augment=None, mode=None):
    """留一被试 LOSO 训练，写 outputs/results/loso_<tag>.json 与 curve_<tag>.json
    augment: 可选 callable(X_tr, y_tr, seed) -> (X_aug, y_aug)，仅作用于训练折
    mode:    ms.PYNATIVE_MODE（默认）或 ms.GRAPH_MODE（耗时对比实验）
    """
    ms.set_context(mode=mode or ms.PYNATIVE_MODE)
    n_classes = len(names)
    t0 = time.time()
    predictions = np.full(len(y), -1, dtype=np.int64)
    per_subject, curves = {}, {}
    for train, test in LeaveOneGroupOut().split(X, y, people):
        X_tr, y_tr = X[train], y[train]
        if augment is not None:
            X_tr, y_tr = augment(X_tr, y_tr, seed=SEED + 1000 + len(train))
        net, (mean, std), hist = fit_model(factory, X_tr, y_tr, X[test], y[test],
                                           n_classes, epochs, lr)
        # 测试集必须用训练折的 mean/std 标准化（防泄漏 + 与训练一致），否则预测全错
        X_test_std = (X[test] - mean) / std
        predictions[test] = _predict(net, X_test_std)
        p = str(people[test[0]])
        per_subject[p] = {"correct": int((predictions[test] == y[test]).sum()), "total": int(len(test))}
        curves[p] = hist
        print(f"  [{p}] acc={(predictions[test] == y[test]).mean():.3f}", flush=True)

    loss_curve = np.nanmean(np.stack([np.asarray(c["loss"], dtype=float) for c in curves.values()]), axis=0)
    val_curve = np.nanmean(np.stack([np.asarray(c["val_acc"], dtype=float) for c in curves.values()]), axis=0)
    result = {
        "model": tag,
        "mindspore": ms.__version__,
        "device": ms.get_context("device_target"),
        "epochs": epochs, "lr": lr, "seed": SEED, "augment": bool(augment),
        "labels": names, "per_subject": per_subject,
        "windows": len(y), "correct": int((predictions == y).sum()),
        "accuracy": float((predictions == y).mean()),
        "macro_f1": float(f1_score(y, predictions, average="macro")),
        "confusion": confusion_matrix(y, predictions, labels=list(range(len(names)))).tolist(),
        "seconds": time.time() - t0,
    }
    (RESULT_DIR / f"loso_{tag}.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    (RESULT_DIR / f"curve_{tag}.json").write_text(
        json.dumps({"loss": loss_curve.tolist(), "val_acc": val_curve.tolist()}, indent=2),
        encoding="utf-8")
    return result


def train_all(factory, tag, X, y, n_classes, epochs=EPOCHS, lr=LEARNING_RATE):
    """全量训练 -> checkpoint + 预处理参数，恢复一致性验证（丁的部署用）"""
    ms.set_context(mode=ms.PYNATIVE_MODE)
    net, (mean, std), _ = fit_model(factory, X, y, X, y, n_classes, epochs, lr)
    ckpt = str(RESULT_DIR / f"{tag}.ckpt")
    ms.save_checkpoint(net, ckpt)
    np.savez(RESULT_DIR / f"{tag}_preprocess.npz", mean=mean, std=std)
    X_std = (X - mean) / std
    restored = factory(n_classes)
    ms.load_checkpoint(ckpt, net=restored, strict_load=True)
    same = bool(np.array_equal(_predict(restored, X_std), _predict(net, X_std)))
    print(f"  checkpoint 恢复一致性: {same}")
    return net
