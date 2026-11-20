"""End-to-end orchestration: window -> aggregate -> extract -> fit -> predict
-> reconcile -> report.  Failures are collected, never hidden: any failure
makes the overall status non-success."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from .features import extract_node_features
from .hierarchy import Hierarchy
from .model import NodeModel
from .reconcile import ReconciliationError, reconcile
from .report import write_report
from .windows import (
    DataError,
    aggregate_windows,
    cut_windows,
    next_period_labels,
    pivot_leaves,
)


def _fail(failures, stage, code, message, node=None):
    entry = {"stage": stage, "code": code, "message": str(message)}
    if node is not None:
        entry["node"] = node
    failures.append(entry)


def _json_default(o):
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (pd.Series, pd.DataFrame)):
        return o.to_dict()
    if isinstance(o, (pd.Timestamp, datetime)):
        return o.isoformat()
    return str(o)


def _write_json(path: Path, obj):
    path.write_text(
        json.dumps(obj, indent=2, ensure_ascii=False, default=_json_default),
        encoding="utf-8",
    )


def run_pipeline(
    raw: pd.DataFrame,
    hierarchy: Hierarchy,
    weights: dict,
    window_size: int,
    out_dir,
    *,
    fc_parameters=None,
    regressor_factory=None,
    min_train_windows: int = 4,
) -> dict:
    """Run the full hierarchical forecasting pipeline.

    Parameters
    ----------
    raw : long-format DataFrame with columns ``time``, ``meter``, ``value``.
    hierarchy : declared :class:`Hierarchy` tree.
    weights : positive weight per node (all nodes must be covered).
    window_size : samples per window (>= 8).
    out_dir : directory for the report and machine-readable artifacts.
    fc_parameters : optional tsfresh settings override.
    regressor_factory : optional ``() -> regressor`` override (tests).
    min_train_windows : minimum number of labelled training windows.

    Returns a result dict; ``result["success"]`` is True only if no failure
    occurred at any stage.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    failures: list[dict] = []
    try:
        weights = {str(k): float(v) for k, v in dict(weights).items()}
    except (TypeError, ValueError) as exc:
        weights = {}
        failures.append(
            {"stage": "config", "code": "invalid_weights",
             "message": f"weights must be numeric: {exc}"}
        )
    result = {
        "success": False,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "failures": failures,
        "out_dir": str(out_dir),
        "config": {
            "window_size": window_size,
            "min_train_windows": min_train_windows,
            "weights": weights,
            "tree": hierarchy.tree,
        },
        "windowing": None,
        "raw_aggregation": None,
        "features": {},
        "labels": {},
        "base_predictions": {},
        "train_r2": {},
        "reconciliation": None,
        "artifacts": [],
    }

    def finalize() -> dict:
        result["success"] = not failures
        _write_artifacts(result, hierarchy, out_dir)
        report_path = write_report(out_dir, result, hierarchy)
        result["report_path"] = str(report_path)
        status = {
            "success": result["success"],
            "generated_at": result["generated_at"],
            "failures": failures,
            "windowing": result["windowing"],
            "nodes_with_model": sorted(result["base_predictions"]),
            "reconciled": result["reconciliation"] is not None,
            "artifacts": result["artifacts"] + [str(report_path)],
        }
        status_path = out_dir / "status.json"
        _write_json(status_path, status)
        result["status_path"] = str(status_path)
        return result

    # ---- config checks ------------------------------------------------- #
    if window_size < 8:
        _fail(failures, "config", "invalid_config",
              f"window_size must be >= 8, got {window_size}")
    w = pd.Series(weights, dtype=float)
    missing_w = [n for n in hierarchy.nodes if n not in w.index]
    bad_w = [
        n for n in hierarchy.nodes
        if n in w.index and (not np.isfinite(w[n]) or w[n] <= 0)
    ]
    if missing_w or bad_w:
        _fail(failures, "config", "invalid_weights",
              f"weights must be given and positive for every node; "
              f"missing={missing_w}, non_positive={bad_w}")
    if failures:
        return finalize()
    w = w.reindex(hierarchy.nodes)

    # ---- load: align leaf populations ---------------------------------- #
    try:
        wide = pivot_leaves(raw, hierarchy.leaves)
    except DataError as exc:
        _fail(failures, "load", exc.code, str(exc))
        return finalize()

    # ---- windowing ------------------------------------------------------ #
    window_frames, dropped = cut_windows(wide, window_size)
    wids = list(window_frames)
    if len(wids) < min_train_windows + 1:
        _fail(failures, "window", "insufficient_windows",
              f"need at least {min_train_windows + 1} complete windows, "
              f"got {len(wids)}")
        return finalize()
    train_ids, predict_id = wids[:-1], wids[-1]
    result["windowing"] = {
        "window_size": window_size,
        "n_windows": len(wids),
        "train_windows": train_ids,
        "predict_window": predict_id,
        "dropped_trailing_rows": dropped,
        "n_leaves": len(hierarchy.leaves),
        "leaves": hierarchy.leaves,
    }

    # ---- raw bottom-up aggregation, then labels from leaves ------------ #
    agg = aggregate_windows(hierarchy, window_frames)
    result["raw_aggregation"] = agg
    labels = next_period_labels(hierarchy, window_frames)
    result["labels"] = labels

    # ---- per node: extract features, fit, predict ---------------------- #
    for node in hierarchy.nodes:
        try:
            feats = extract_node_features(node, agg[node], fc_parameters)
        except Exception as exc:  # noqa: BLE001 - recorded, not hidden
            _fail(failures, "features", "feature_failure",
                  f"{type(exc).__name__}: {exc}", node=node)
            continue
        result["features"][node] = feats
        model = NodeModel(regressor_factory() if regressor_factory else None)
        try:
            model.fit(feats.loc[train_ids], labels[node])
            pred = float(model.predict(feats.loc[[predict_id]]).iloc[0])
        except Exception as exc:  # noqa: BLE001
            _fail(failures, "fit", "fit_failure",
                  f"{type(exc).__name__}: {exc}", node=node)
            continue
        if not np.isfinite(pred):
            _fail(failures, "predict", "non_finite_prediction",
                  f"model returned {pred}", node=node)
            continue
        result["base_predictions"][node] = pred
        result["train_r2"][node] = model.train_r2_

    # ---- reconcile: non-negative leaves, nodes aggregated from them ----- #
    if len(result["base_predictions"]) == len(hierarchy.nodes):
        base = pd.Series(result["base_predictions"]).reindex(hierarchy.nodes)
        try:
            rec = reconcile(base, w, hierarchy.summing_matrix())
            rec["coherence"] = _coherence(rec, hierarchy)
            result["reconciliation"] = rec
        except ReconciliationError as exc:
            _fail(failures, "reconcile", "solve_failure", str(exc))
    else:
        missing = [n for n in hierarchy.nodes if n not in result["base_predictions"]]
        _fail(failures, "reconcile", "reconciliation_skipped",
              f"missing base predictions for nodes: {missing}")

    return finalize()


def _coherence(rec: dict, hierarchy: Hierarchy) -> dict:
    """Verify reconciled nodes equal the sum of their children (they must,
    by construction -- this surfaces any float-level violation)."""
    nodes = rec["nodes"]
    worst = 0.0
    for parent, children in hierarchy.tree.items():
        viol = abs(nodes[parent] - sum(nodes[c] for c in children))
        worst = max(worst, viol)
    leaf_sum = sum(rec["leaves"].values())
    return {
        "root": hierarchy.root,
        "root_value": nodes[hierarchy.root],
        "leaf_sum": leaf_sum,
        "root_minus_leaf_sum": abs(nodes[hierarchy.root] - leaf_sum),
        "max_parent_child_violation": worst,
    }


def _write_artifacts(result: dict, hierarchy: Hierarchy, out_dir: Path):
    artifacts = []

    agg = result.get("raw_aggregation")
    if agg:
        rows = []
        for node in hierarchy.nodes:
            if node not in agg:
                continue
            for wid, series in agg[node].items():
                rows.append(
                    pd.DataFrame(
                        {
                            "window": wid,
                            "node": node,
                            "time": series.index,
                            "value": np.asarray(series, dtype=float),
                        }
                    )
                )
        if rows:
            path = out_dir / "raw_aggregation.csv"
            pd.concat(rows, ignore_index=True).to_csv(path, index=False)
            artifacts.append(str(path))

    if result.get("features"):
        feat_dir = out_dir / "features"
        feat_dir.mkdir(exist_ok=True)
        for node, feats in result["features"].items():
            path = feat_dir / f"{node}.csv"
            feats.to_csv(path)
            artifacts.append(str(path))

    if result.get("labels"):
        path = out_dir / "labels_train.csv"
        pd.DataFrame(result["labels"]).to_csv(path)
        artifacts.append(str(path))

    if result.get("base_predictions"):
        path = out_dir / "base_predictions.json"
        _write_json(
            path,
            {
                "base_predictions": result["base_predictions"],
                "train_r2_in_sample": result["train_r2"],
                "predict_window": (result["windowing"] or {}).get("predict_window"),
            },
        )
        artifacts.append(str(path))

    if result.get("reconciliation"):
        path = out_dir / "reconciliation.json"
        _write_json(path, result["reconciliation"])
        artifacts.append(str(path))

    result["artifacts"] = artifacts
