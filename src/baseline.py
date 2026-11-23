# -*- coding: utf-8 -*-
"""
【乙】sklearn 基线（多模型 + 留一被试 LOSO）
====================================================
评估协议：每次用 9 个被试训练、剩下的 1 个做测试（共 10 折），
保证"没见过的被试"也能分对，衡量跨人泛化能力（报告重点指标）。
TODO(乙)：
  1. 调参：SVM 的 C/gamma、RF 的棵数、GBDT 的学习率/深度，记录调参过程
  2. 特征消融：全量 12 维 vs 仅时域 9 维 vs 仅频域 3 维
  3. 逐被试误差分析：哪个被试最差、哪两类最易混（写进报告）
"""
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.model_selection import LeaveOneGroupOut, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from config import SEED

BASELINE_MODELS = {
    "LogisticRegression": lambda: make_pipeline(
        StandardScaler(), LogisticRegression(max_iter=2000, C=1.0, random_state=SEED)),
    "SVM": lambda: make_pipeline(
        StandardScaler(), SVC(kernel="rbf", C=10.0, gamma="scale")),
    "RandomForest": lambda: RandomForestClassifier(
        n_estimators=300, random_state=SEED, n_jobs=-1),
    "GradientBoosting": lambda: GradientBoostingClassifier(
        n_estimators=100, learning_rate=0.1, random_state=SEED),
}


def loso_evaluate(factory, X, y, people, names):
    """
    留一被试评估单个模型。

    Returns
    -------
    dict: model / per_subject / windows / correct / accuracy / macro_f1 /
          confusion / labels / seconds（与 src/train_mindspore.py 的 LOSO 结果同构，
          可直接进 src/evaluate.py 的对比表）
    """
    import time
    t0 = time.time()
    model = factory()
    pred = cross_val_predict(model, X, y, groups=people, cv=LeaveOneGroupOut())
    per_subject = {}
    for p in sorted(set(people.tolist())):
        mask = np.array([q == p for q in people])
        per_subject[p] = {"correct": int((pred[mask] == y[mask]).sum()), "total": int(mask.sum())}
    return {
        "model": "sklearn",
        "per_subject": per_subject,
        "windows": len(y),
        "correct": int((pred == y).sum()),
        "accuracy": float((pred == y).mean()),
        "macro_f1": float(f1_score(y, pred, average="macro")),
        "confusion": confusion_matrix(y, pred, labels=list(range(len(names)))).tolist(),
        "labels": names,
        "seconds": time.time() - t0,
    }
