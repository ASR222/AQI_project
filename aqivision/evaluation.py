"""Leak-free evaluation helpers.

Image AQI datasets are made of bursts: many frames of the same place at the same time share one
label (TRAQID: dashcam frames 5 s apart on 17 dates; India/Nepal: 12,240 images from 115
location+date+hour moments). A random image-level split puts near-duplicates of the same moment
in train and test, so the model can score well by recognising the moment instead of the haze.
Split by group (date, location, moment) instead, and always report the majority-class baseline.

    groups = df["location"] + "_" + df["date"] + "_" + df["hour"].astype(str)
    train, test = group_split(groups, test_frac=0.2, seed=0)
    print(leakage(groups, train, test))   # must be 0 shared groups
    print(metrics(y_true, y_pred, y_train=labels[train]))
"""

from __future__ import annotations

import numpy as np


def group_split(groups, test_frac: float = 0.2, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Index arrays (train, test) such that no group appears on both sides."""
    groups = np.asarray(groups)
    uniq = np.unique(groups)
    rng = np.random.default_rng(seed)
    rng.shuffle(uniq)
    n_test = max(1, int(round(len(uniq) * test_frac)))
    test_groups = set(uniq[:n_test].tolist())
    is_test = np.array([g in test_groups for g in groups])
    return np.flatnonzero(~is_test), np.flatnonzero(is_test)


def random_split(n: int, test_frac: float = 0.2, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Image-level split (for comparison only: leaks when frames come in groups)."""
    idx = np.random.default_rng(seed).permutation(n)
    n_test = int(round(n * test_frac))
    return np.sort(idx[n_test:]), np.sort(idx[:n_test])


def leave_one_group_out(groups):
    """Yield (group, train_idx, test_idx) for every distinct group."""
    groups = np.asarray(groups)
    for g in np.unique(groups):
        test = groups == g
        yield g, np.flatnonzero(~test), np.flatnonzero(test)


def leakage(groups, train_idx, test_idx) -> dict:
    groups = np.asarray(groups)
    tr, te = set(groups[train_idx].tolist()), set(groups[test_idx].tolist())
    shared = tr & te
    return {
        "train_groups": len(tr),
        "test_groups": len(te),
        "shared_groups": len(shared),
        "test_rows_in_shared_groups": int(np.isin(groups[test_idx], list(shared)).sum()),
    }


def macro_f1(y_true, y_pred, labels=None) -> float:
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    labels = np.unique(np.concatenate([y_true, y_pred])) if labels is None else np.asarray(labels)
    f1s = []
    for c in labels:
        tp = np.sum((y_pred == c) & (y_true == c))
        fp = np.sum((y_pred == c) & (y_true != c))
        fn = np.sum((y_pred != c) & (y_true == c))
        f1s.append(0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn))
    return float(np.mean(f1s))


def metrics(y_true, y_pred, y_train=None, groups=None) -> dict:
    """Accuracy, macro-F1, majority baseline (from the training labels if given), optional per-group accuracy."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    ref = np.asarray(y_train) if y_train is not None else y_true
    vals, counts = np.unique(ref, return_counts=True)
    majority = vals[counts.argmax()]
    out = {
        "n": int(len(y_true)),
        "accuracy": float((y_true == y_pred).mean()),
        "macro_f1": macro_f1(y_true, y_pred),
        "majority_class": majority.item() if hasattr(majority, "item") else majority,
        "majority_baseline": float((y_true == majority).mean()),
    }
    if groups is not None:
        groups = np.asarray(groups)
        out["per_group_accuracy"] = {
            str(g): float((y_true[groups == g] == y_pred[groups == g]).mean()) for g in np.unique(groups)
        }
    return out
