"""Tests for the hierarchical forecasting pipeline."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from hier_energy import (
    DataError,
    Hierarchy,
    HierarchyError,
    NodeModel,
    reconcile,
    run_pipeline,
)
from hier_energy.demo import TREE, WEIGHTS, synthetic_meters
from hier_energy.features import extract_node_features
from hier_energy.windows import aggregate_windows, cut_windows, pivot_leaves

FAST_FC = {
    "mean": None,
    "sum_values": None,
    "maximum": None,
    "minimum": None,
    "abs_energy": None,
    "standard_deviation": None,
    "linear_trend": [{"attr": "slope"}],
}


def fast_rf():
    from sklearn.ensemble import RandomForestRegressor

    return RandomForestRegressor(n_estimators=50, random_state=7, n_jobs=1)


def run_fast(raw, out_dir, **kw):
    kw.setdefault("regressor_factory", fast_rf)
    return run_pipeline(
        raw,
        Hierarchy(TREE),
        WEIGHTS,
        window_size=12,
        out_dir=out_dir,
        fc_parameters=FAST_FC,
        **kw,
    )


# ---------------------------------------------------------------------- #
# hierarchy
# ---------------------------------------------------------------------- #
def test_hierarchy_structure():
    h = Hierarchy(TREE)
    assert h.root == "total"
    assert h.leaves == ["floor_a1", "floor_a2", "floor_b1"]
    assert h.leaves_under("building_a") == ["floor_a1", "floor_a2"]
    assert h.leaves_under("floor_b1") == ["floor_b1"]
    S = h.summing_matrix()
    assert list(S.index) == h.nodes and list(S.columns) == h.leaves
    assert S.loc["total"].sum() == 3.0
    assert S.loc["building_a"].tolist() == [1.0, 1.0, 0.0]


@pytest.mark.parametrize(
    "tree",
    [
        {"a": ["b"], "c": ["d"]},                 # two roots
        {"a": ["b"], "b": ["a"]},                 # detached cycle, no root
        {"a": ["b", "c"], "d": ["c"]},            # child with two parents
        {},                                        # empty
    ],
)
def test_hierarchy_rejects_bad_trees(tree):
    with pytest.raises(HierarchyError):
        Hierarchy(tree)


# ---------------------------------------------------------------------- #
# aggregation-first features
# ---------------------------------------------------------------------- #
def test_features_are_extracted_from_aggregated_series_not_summed():
    h = Hierarchy(TREE)
    raw = synthetic_meters(n_windows=3, window_size=12)
    wide = pivot_leaves(raw, h.leaves)
    frames, _ = cut_windows(wide, 12)
    agg = aggregate_windows(h, frames)

    feats = extract_node_features("total", agg["total"], FAST_FC)

    from tsfresh.feature_extraction.feature_calculators import abs_energy

    wid = "w0000"
    aggregated = agg["total"][wid].to_numpy()
    # feature equals the non-linear statistic of the aggregated series ...
    assert feats.loc[wid, "abs_energy"] == pytest.approx(abs_energy(aggregated))
    # ... which is NOT the sum of the leaf statistics (non-linearity)
    leaf_sum = sum(abs_energy(frames[wid][l].to_numpy()) for l in h.leaves)
    assert abs(feats.loc[wid, "abs_energy"] - leaf_sum) > 1.0
    # sum_values is linear, so there aggregation and summation coincide
    assert feats.loc[wid, "sum_values"] == pytest.approx(aggregated.sum())


# ---------------------------------------------------------------------- #
# preprocessing reuse
# ---------------------------------------------------------------------- #
def test_preprocessing_is_fitted_on_train_and_reused():
    class Spy:
        def fit(self, X, y):
            return self

        def predict(self, X):
            self.seen = np.asarray(X).copy()
            return np.zeros(len(X))

    spy = Spy()
    model = NodeModel(regressor=spy)
    X_train = pd.DataFrame(
        {
            "a": [1.0, 2.0, np.nan, 4.0],      # median 2.0
            "b": [5.0, 5.0, 5.0, 5.0],          # zero variance -> dropped
            "c": [10.0, 20.0, 30.0, 50.0],      # median 25 != mean 27.5
        }
    )
    model.fit(X_train, pd.Series([1.0, 2.0, 3.0, 4.0]))
    assert model.columns_ == ["a", "c"]
    assert model.fill_["a"] == pytest.approx(2.0)

    X_new = pd.DataFrame({"a": [np.nan], "z": [999.0]})  # 'c' missing, 'z' unseen
    model.predict(X_new)
    a_std = np.std([1.0, 2.0, 2.0, 4.0])  # train median imputed before scaling
    c_std = np.std([10.0, 20.0, 30.0, 50.0])
    expected_a = (2.0 - np.mean([1.0, 2.0, 2.0, 4.0])) / a_std
    expected_c = (25.0 - np.mean([10.0, 20.0, 30.0, 50.0])) / c_std
    assert spy.seen.shape == (1, 2)
    assert spy.seen[0, 0] == pytest.approx(expected_a)
    assert spy.seen[0, 1] == pytest.approx(expected_c)


# ---------------------------------------------------------------------- #
# reconciliation QP
# ---------------------------------------------------------------------- #
def _toy_problem():
    nodes = ["total", "a", "b", "l1", "l2", "l3"]
    leaves = ["l1", "l2", "l3"]
    S = pd.DataFrame(
        [
            [1, 1, 1],
            [1, 1, 0],
            [0, 0, 1],
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
        ],
        index=nodes,
        columns=leaves,
        dtype=float,
    )
    base = pd.Series([10.0, 8.0, 1.0, 5.0, -3.0, 2.0], index=nodes)  # l2 < 0
    weights = pd.Series([3.0, 2.0, 2.0, 1.0, 1.0, 1.0], index=nodes)
    return base, weights, S


def test_reconcile_global_optimum_not_clipping():
    base, weights, S = _toy_problem()
    rec = reconcile(base, weights, S)

    x = np.array([rec["leaves"][l] for l in S.columns])
    assert (x >= -1e-12).all()  # non-negative demand

    # KKT: gradient of the weighted objective vanishes on active coords,
    # is >= 0 on the boundary
    Sv = S.to_numpy()
    b = base.to_numpy()
    w = weights.to_numpy()
    g = Sv.T @ (w * (Sv @ x - b))
    active = x > 1e-10
    assert np.abs(g[active]).max() < 1e-6
    assert g[~active].min() > -1e-6

    # better than clip-at-zero ...
    assert rec["total_cost"] <= rec["clip_baseline"]["total_cost"] + 1e-9
    assert rec["clip_baseline"]["total_cost"] - rec["total_cost"] > 1e-6
    # ... and better than any random feasible perturbation
    rng = np.random.default_rng(0)
    for _ in range(200):
        xp = np.maximum(x + rng.normal(0, 0.5, size=x.shape), 0.0)
        cost_p = np.sum(w * (Sv @ xp - b) ** 2)
        assert cost_p >= rec["total_cost"] - 1e-9

    # node values are the aggregation of the leaves
    for i, n in enumerate(S.index):
        assert rec["nodes"][n] == pytest.approx(Sv[i] @ x)


def test_reconcile_rejects_bad_input():
    base, weights, S = _toy_problem()
    from hier_energy.reconcile import ReconciliationError

    with pytest.raises(ReconciliationError):
        reconcile(base, weights * 0.0, S)  # non-positive weights
    with pytest.raises(ReconciliationError):
        reconcile(base * np.nan, weights, S)  # non-finite base


# ---------------------------------------------------------------------- #
# end-to-end
# ---------------------------------------------------------------------- #
def test_end_to_end_success(tmp_path):
    raw = synthetic_meters(n_windows=14, window_size=12)
    result = run_fast(raw, tmp_path)
    assert result["success"], result["failures"]

    h = Hierarchy(TREE)
    rec = result["reconciliation"]
    # coherence: every node equals the sum of its leaves, root == all leaves
    for parent, children in h.tree.items():
        assert rec["nodes"][parent] == pytest.approx(
            sum(rec["nodes"][c] for c in children), abs=1e-9
        )
    assert rec["nodes"]["total"] == pytest.approx(
        sum(rec["leaves"].values()), abs=1e-9
    )
    # non-negativity
    assert all(v >= 0 for v in rec["leaves"].values())
    # global optimum beats (or ties) the clip baseline
    assert rec["total_cost"] <= rec["clip_baseline"]["total_cost"] + 1e-9
    # base predictions exist for every node and are finite
    assert set(result["base_predictions"]) == set(h.nodes)
    assert all(np.isfinite(list(result["base_predictions"].values())))

    # artifacts
    for name in ["report.md", "status.json", "reconciliation.json",
                 "base_predictions.json", "raw_aggregation.csv",
                 "labels_train.csv", "features/total.csv"]:
        assert (tmp_path / name).exists(), name
    status = json.loads((tmp_path / "status.json").read_text())
    assert status["success"] is True and status["failures"] == []
    report = (tmp_path / "report.md").read_text(encoding="utf-8")
    assert "✅ 成功" in report and "失败明细" not in report

    # raw aggregation artifact is consistent with the declared tree
    agg_csv = pd.read_csv(tmp_path / "raw_aggregation.csv")
    w0 = agg_csv[agg_csv["window"] == "w0000"].set_index("time")
    per_leaf = (
        w0[w0["node"].isin(h.leaves)].pivot(columns="node", values="value")
    )
    total = w0[w0["node"] == "total"]["value"]
    assert np.allclose(per_leaf.sum(axis=1).to_numpy(), total.to_numpy())


def test_missing_leaf_is_reported_not_hidden(tmp_path):
    raw = synthetic_meters(n_windows=14, window_size=12)
    raw = raw[raw["meter"] != "floor_b1"]
    result = run_fast(raw, tmp_path)
    assert result["success"] is False
    assert result["failures"][0]["code"] == "missing_leaf"
    assert result["reconciliation"] is None
    status = json.loads((tmp_path / "status.json").read_text())
    assert status["success"] is False
    report = (tmp_path / "report.md").read_text(encoding="utf-8")
    assert "❌ 失败" in report and "missing_leaf" in report


def test_population_misalignment_is_reported(tmp_path):
    raw = synthetic_meters(n_windows=14, window_size=12)
    # knock out some timestamps of one leaf -> ragged population
    drop = (raw["meter"] == "floor_a2") & (raw.index % 37 == 0)
    result = run_fast(raw[~drop], tmp_path)
    assert result["success"] is False
    assert result["failures"][0]["code"] == "population_misalignment"


def test_unexpected_meter_is_reported(tmp_path):
    raw = synthetic_meters(n_windows=14, window_size=12)
    rogue = raw[raw["meter"] == "floor_a1"].copy()
    rogue["meter"] = "rogue_meter"
    result = run_fast(pd.concat([raw, rogue]), tmp_path)
    assert result["success"] is False
    assert result["failures"][0]["code"] == "unexpected_meter"


def test_fit_failure_is_reported_and_not_reconciled(tmp_path):
    class Exploding:
        def fit(self, X, y):
            raise RuntimeError("boom")

    raw = synthetic_meters(n_windows=14, window_size=12)
    result = run_fast(raw, tmp_path, regressor_factory=Exploding)
    assert result["success"] is False
    codes = {f["code"] for f in result["failures"]}
    assert "fit_failure" in codes
    assert "reconciliation_skipped" in codes
    assert result["reconciliation"] is None
    status = json.loads((tmp_path / "status.json").read_text())
    assert status["success"] is False


def test_solve_failure_is_reported(tmp_path, monkeypatch):
    import importlib

    rec_mod = importlib.import_module("hier_energy.reconcile")

    def boom(A, b):
        raise RuntimeError("solver exploded")

    monkeypatch.setattr(rec_mod, "nnls", boom)
    raw = synthetic_meters(n_windows=14, window_size=12)
    result = run_fast(raw, tmp_path)
    assert result["success"] is False
    assert any(f["code"] == "solve_failure" for f in result["failures"])


def test_insufficient_windows_is_reported(tmp_path):
    raw = synthetic_meters(n_windows=3, window_size=12)
    result = run_fast(raw, tmp_path, min_train_windows=4)
    assert result["success"] is False
    assert result["failures"][0]["code"] == "insufficient_windows"


def test_invalid_weights_are_reported(tmp_path):
    raw = synthetic_meters(n_windows=14, window_size=12)
    bad = dict(WEIGHTS, total="not-a-number")
    result = run_pipeline(
        raw, Hierarchy(TREE), bad, window_size=12, out_dir=tmp_path,
        fc_parameters=FAST_FC, regressor_factory=fast_rf,
    )
    assert result["success"] is False
    assert result["failures"][0]["code"] == "invalid_weights"
    missing = {k: v for k, v in WEIGHTS.items() if k != "total"}
    result = run_pipeline(
        raw, Hierarchy(TREE), missing, window_size=12, out_dir=tmp_path,
        fc_parameters=FAST_FC, regressor_factory=fast_rf,
    )
    assert result["success"] is False
    assert result["failures"][0]["code"] == "invalid_weights"


def test_trailing_rows_are_dropped_and_reported(tmp_path):
    raw = synthetic_meters(n_windows=14, window_size=12)
    extra = raw[raw["meter"] == "floor_a1"].head(5).copy()
    extra["time"] = raw["time"].max() + pd.to_timedelta(
        np.arange(1, 6), unit="h"
    )
    all_leaves = []
    for meter in ["floor_a1", "floor_a2", "floor_b1"]:
        part = extra.copy()
        part["meter"] = meter
        all_leaves.append(part)
    raw2 = pd.concat([raw] + all_leaves)
    result = run_fast(raw2, tmp_path)
    assert result["success"], result["failures"]
    assert result["windowing"]["dropped_trailing_rows"] == 5
