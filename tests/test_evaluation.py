import numpy as np

from aqivision import evaluation as ev


def test_group_split_has_no_shared_groups():
    groups = np.repeat([f"g{i}" for i in range(20)], 30)
    tr, te = ev.group_split(groups, test_frac=0.25, seed=1)
    rep = ev.leakage(groups, tr, te)
    assert rep["shared_groups"] == 0 and rep["test_groups"] == 5
    assert len(tr) + len(te) == len(groups)


def test_random_split_leaks_on_grouped_data():
    groups = np.repeat([f"g{i}" for i in range(20)], 30)
    tr, te = ev.random_split(len(groups), seed=0)
    assert ev.leakage(groups, tr, te)["shared_groups"] > 15


def test_memorising_groups_looks_perfect_on_random_split_only():
    # every group has one label; a "model" that memorises group -> label
    rng = np.random.default_rng(0)
    groups = np.repeat(np.arange(30), 20)
    label_of = rng.integers(0, 3, 30)
    y = label_of[groups]

    def memoriser(train_idx, test_idx):
        lookup = {g: y[i] for g, i in zip(groups[train_idx], train_idx)}
        return np.array([lookup.get(g, 0) for g in groups[test_idx]])

    tr, te = ev.random_split(len(y))
    assert ev.metrics(y[te], memoriser(tr, te))["accuracy"] == 1.0
    tr, te = ev.group_split(groups)
    assert ev.metrics(y[te], memoriser(tr, te))["accuracy"] < 1.0


def test_metrics_baseline_and_f1():
    m = ev.metrics([0, 0, 1, 1], [0, 0, 1, 0], y_train=[0, 0, 0, 1])
    assert m["accuracy"] == 0.75 and m["majority_class"] == 0 and m["majority_baseline"] == 0.5
    assert abs(m["macro_f1"] - (0.8 + 2 / 3) / 2) < 1e-9


def test_leave_one_group_out():
    folds = list(ev.leave_one_group_out(["a", "a", "b", "c"]))
    assert [f[0] for f in folds] == ["a", "b", "c"]
    assert list(folds[0][2]) == [0, 1]
