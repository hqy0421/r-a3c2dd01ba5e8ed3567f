"""Windowing of aligned leaf signals and raw bottom-up aggregation."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .hierarchy import Hierarchy


class DataError(RuntimeError):
    """A data-population problem that must be surfaced, never hidden.

    The ``code`` is a stable machine-readable tag (e.g. ``missing_leaf``,
    ``population_misalignment``) so callers can report it faithfully.
    """

    def __init__(self, code: str, message: str, details: dict | None = None):
        super().__init__(message)
        self.code = code
        self.details = details or {}


def pivot_leaves(raw: pd.DataFrame, leaves) -> pd.DataFrame:
    """Align raw long-format meter data into a time x leaf wide frame.

    ``raw`` must have columns ``time``, ``meter``, ``value``.  Every declared
    leaf must be present, no undeclared meter may appear, every meter must
    share the same timestamp grid (no ragged populations) and values must be
    finite numbers.  Any violation raises :class:`DataError`.
    """
    required = {"time", "meter", "value"}
    missing_cols = required - set(raw.columns)
    if missing_cols:
        raise DataError(
            "bad_schema",
            f"raw data must have columns {sorted(required)}; missing {sorted(missing_cols)}",
        )
    declared = [str(l) for l in leaves]
    df = raw.loc[:, ["time", "meter", "value"]].copy()
    df["meter"] = df["meter"].astype(str)

    present = set(df["meter"].unique())
    missing = [l for l in declared if l not in present]
    if missing:
        raise DataError(
            "missing_leaf",
            f"declared leaf meters absent from data: {missing}",
            {"missing": missing},
        )
    extra = sorted(present - set(declared))
    if extra:
        raise DataError(
            "unexpected_meter",
            f"data contains meters not declared in the tree: {extra}",
            {"extra": extra},
        )

    if not (
        np.issubdtype(df["time"].dtype, np.datetime64)
        or np.issubdtype(df["time"].dtype, np.number)
    ):
        try:
            df["time"] = pd.to_datetime(df["time"])
        except Exception as exc:  # noqa: BLE001 - re-raised as DataError
            raise DataError("bad_timestamps", f"unparseable timestamps: {exc}") from exc
    try:
        df["value"] = pd.to_numeric(df["value"])
    except Exception as exc:  # noqa: BLE001
        raise DataError("non_numeric_values", f"non-numeric meter values: {exc}") from exc

    dup = df.duplicated(subset=["time", "meter"])
    if dup.any():
        examples = df.loc[dup, ["time", "meter"]].head(5).to_dict("records")
        raise DataError(
            "duplicate_samples",
            f"{int(dup.sum())} duplicated (time, meter) samples",
            {"examples": [str(e) for e in examples]},
        )

    wide = df.pivot(index="time", columns="meter", values="value").sort_index()
    wide = wide.reindex(columns=declared)

    nan_cells = int(wide.isna().sum().sum())
    if nan_cells:
        bad_times = [str(t) for t in wide.index[wide.isna().any(axis=1)][:5]]
        raise DataError(
            "population_misalignment",
            f"{nan_cells} missing leaf samples after alignment: meters do not share "
            f"the same timestamp grid",
            {"nan_cells": nan_cells, "example_times": bad_times},
        )
    if not np.isfinite(wide.to_numpy(dtype=float)).all():
        raise DataError("non_finite_values", "meter values contain inf")
    return wide


def cut_windows(wide: pd.DataFrame, window_size: int):
    """Cut the aligned frame into consecutive, equal-length windows.

    Returns ``(frames, dropped_rows)`` where ``frames`` maps window id ->
    wide slice.  Trailing rows that do not form a complete window are dropped
    and reported (this is a windowing artefact, not a population defect).
    """
    n_windows = len(wide) // window_size
    frames = {
        f"w{i:04d}": wide.iloc[i * window_size : (i + 1) * window_size]
        for i in range(n_windows)
    }
    dropped = len(wide) - n_windows * window_size
    return frames, dropped


def aggregate_windows(hierarchy: Hierarchy, window_frames: dict) -> dict:
    """Raw bottom-up aggregation: for every window and every node, sum the
    leaf signals of the node's subtree *at each timestamp of the window*.

    Returns ``{node: {window_id: pd.Series(index=time)}}``.  Feature
    extraction runs on these aggregated series afterwards -- leaf features
    are never summed to build a node's features.
    """
    agg = {}
    for node in hierarchy.nodes:
        leaves = hierarchy.leaves_under(node)
        agg[node] = {
            wid: frame.loc[:, leaves].sum(axis=1) for wid, frame in window_frames.items()
        }
    return agg


def next_period_labels(hierarchy: Hierarchy, window_frames: dict) -> dict:
    """Next-period labels composed from leaves: for window ``w_i`` the label
    of node ``n`` is the sum, over the leaves under ``n``, of their total
    demand in window ``w_{i+1}``.

    Returns ``{node: pd.Series(index=train_window_ids)}``; the last window has
    no label (it is the window to predict).
    """
    wids = list(window_frames)
    leaf_demand = {wid: frame.sum(axis=0) for wid, frame in window_frames.items()}
    labels = {}
    for node in hierarchy.nodes:
        leaves = hierarchy.leaves_under(node)
        labels[node] = pd.Series(
            {
                wids[i]: float(sum(leaf_demand[wids[i + 1]][l] for l in leaves))
                for i in range(len(wids) - 1)
            },
            name=node,
        )
        labels[node].index.name = "window"
    return labels
