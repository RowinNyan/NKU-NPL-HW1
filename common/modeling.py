"""Logistic Regression 的训练与评价工具（Task 1/2 共用）。

把"训练 + 在验证/测试集上评价 + 分类别指标 + 混淆矩阵"这套流程集中在这里，
使不同文本表示方法之间的评价口径完全一致。
"""

from __future__ import annotations

import time
from collections.abc import Sequence

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)


def train_logistic_regression(
    X_train, y_train, C: float = 1.0, max_iter: int = 1000
) -> tuple[LogisticRegression, float]:
    """训练 Logistic Regression，返回 (模型, 训练秒数)。"""
    clf = LogisticRegression(C=C, max_iter=max_iter)
    started = time.perf_counter()
    clf.fit(X_train, y_train)
    return clf, time.perf_counter() - started


def evaluate_predictions(
    y_true: np.ndarray, y_pred: np.ndarray, class_names: Sequence[str]
) -> dict[str, object]:
    """计算 Accuracy、Macro-F1、分类别 P/R/F1 与混淆矩阵。"""
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=list(class_names), zero_division=0
    )
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "per_class": {
            str(name): {
                "precision": float(p),
                "recall": float(r),
                "f1": float(f),
                "support": int(s),
            }
            for name, p, r, f, s in zip(class_names, precision, recall, f1, support)
        },
        "confusion_matrix": {
            "labels": [str(c) for c in class_names],
            "matrix": confusion_matrix(
                y_true, y_pred, labels=list(class_names)
            ).tolist(),
        },
    }


def run_experiment(
    representation: str,
    X_train,
    y_train: np.ndarray,
    X_val,
    y_val: np.ndarray,
    X_test,
    y_test: np.ndarray,
    class_names: Sequence[str],
    C: float = 1.0,
    max_iter: int = 1000,
    extra: dict[str, object] | None = None,
) -> dict[str, object]:
    """跑完一个完整实验：训练 + 验证/测试集评价。"""
    clf, train_seconds = train_logistic_regression(
        X_train, y_train, C=C, max_iter=max_iter
    )
    n_iter = int(np.max(clf.n_iter_))

    result: dict[str, object] = {
        "representation": representation,
        "n_features": int(X_train.shape[1]),
        "n_train": int(X_train.shape[0]),
        "train_seconds": round(train_seconds, 2),
        "n_iter": n_iter,
        "converged": n_iter < max_iter,
    }
    if extra:
        result.update(extra)

    for part, X, y in (("val", X_val, y_val), ("test", X_test, y_test)):
        metrics = evaluate_predictions(y, clf.predict(X), class_names)
        result[f"{part}_accuracy"] = metrics["accuracy"]
        result[f"{part}_macro_f1"] = metrics["macro_f1"]
        if part == "test":
            result["test_per_class"] = metrics["per_class"]
            result["test_confusion_matrix"] = metrics["confusion_matrix"]

    return result


def describe_result(result: dict[str, object]) -> str:
    """把单个实验的结果排版成可读文本（含分类别指标与混淆矩阵）。"""
    name = result["representation"]
    lines = [
        f"[{name}] 完成: val Acc={result['val_accuracy']:.4f} "
        f"Macro-F1={result['val_macro_f1']:.4f} | "
        f"test Acc={result['test_accuracy']:.4f} "
        f"Macro-F1={result['test_macro_f1']:.4f} "
        f"({result['train_seconds']}s, iter={result['n_iter']}"
        f"{'' if result['converged'] else ', 未收敛!'})"
    ]

    lines.append(f"[{name}] 测试集分类别指标:")
    lines.append(f"    {'类别':<12}{'precision':>11}{'recall':>10}{'f1':>10}{'support':>10}")
    for label, metrics in result["test_per_class"].items():
        lines.append(
            f"    {label:<12}{metrics['precision']:>11.4f}{metrics['recall']:>10.4f}"
            f"{metrics['f1']:>10.4f}{metrics['support']:>10}"
        )

    cm = result["test_confusion_matrix"]
    lines.append(f"[{name}] 测试集混淆矩阵 (行=真实, 列=预测):")
    lines.append("    " + " " * 12 + "".join(f"{lbl:>12}" for lbl in cm["labels"]))
    for label, row in zip(cm["labels"], cm["matrix"]):
        lines.append(f"    {label:<12}" + "".join(f"{value:>12}" for value in row))

    return "\n".join(lines)


def print_summary_table(results: Sequence[dict[str, object]], name_width: int = 34) -> None:
    """打印所有实验的汇总表。"""
    header = (
        f"{'表示方法':<{name_width}}{'Acc(val)':>10}{'F1(val)':>10}"
        f"{'Acc(test)':>11}{'F1(test)':>10}{'用时(s)':>10}"
    )
    print("=" * len(header))
    print(header)
    print("-" * len(header))
    for r in results:
        print(
            f"{r['representation']:<{name_width}}{r['val_accuracy']:>10.4f}"
            f"{r['val_macro_f1']:>10.4f}{r['test_accuracy']:>11.4f}"
            f"{r['test_macro_f1']:>10.4f}{r['train_seconds']:>10.2f}"
        )
    print("=" * len(header))
