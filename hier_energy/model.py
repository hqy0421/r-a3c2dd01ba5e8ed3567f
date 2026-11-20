"""Per-node independent regressor with train-fitted preprocessing."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score


class NodeModel:
    """One node's demand model: preprocessing fitted on the training windows
    only, then frozen and re-applied verbatim to any new window.

    Preprocessing pipeline (all statistics come from the training features):
      1. +-inf -> NaN, NaN -> training median (all-NaN training column -> 0.0)
      2. drop columns with zero variance on the training windows
      3. standardise with training mean / std
    At predict time the input is reindexed to the kept training columns, so
    unseen feature columns are ignored and missing ones are imputed with the
    training medians.
    """

    def __init__(self, regressor=None):
        self.regressor = (
            regressor
            if regressor is not None
            else RandomForestRegressor(
                n_estimators=300, random_state=20261120, n_jobs=1
            )
        )
        self.columns_ = None
        self.fill_ = None
        self.mean_ = None
        self.scale_ = None
        self.train_r2_ = None

    # ------------------------------------------------------------------ #
    def fit(self, X: pd.DataFrame, y: pd.Series) -> "NodeModel":
        if len(X) != len(y):
            raise ValueError(f"X and y disagree: {len(X)} rows vs {len(y)} labels")
        if len(X) < 2:
            raise ValueError("need at least 2 training windows to fit")
        X = X.replace([np.inf, -np.inf], np.nan)
        fill = X.median(axis=0).fillna(0.0)
        Xf = X.fillna(fill)
        keep = Xf.columns[Xf.std(axis=0, ddof=0) > 0.0]
        if len(keep) == 0:
            raise ValueError("all features are constant on the training windows")
        Xk = Xf.loc[:, keep]
        mean = Xk.mean(axis=0)
        scale = Xk.std(axis=0, ddof=0).replace(0.0, 1.0)
        Z = (Xk - mean) / scale

        yv = np.asarray(y, dtype=float)
        if not np.isfinite(yv).all():
            raise ValueError("labels contain NaN/inf")
        self.regressor.fit(Z.to_numpy(), yv)

        self.columns_ = list(keep)
        self.fill_ = fill
        self.mean_ = mean
        self.scale_ = scale
        self.train_r2_ = float(r2_score(yv, self.regressor.predict(Z.to_numpy())))
        return self

    def _transform(self, X: pd.DataFrame) -> np.ndarray:
        if self.columns_ is None:
            raise RuntimeError("model is not fitted")
        X = X.replace([np.inf, -np.inf], np.nan)
        X = X.reindex(columns=self.columns_)
        X = X.fillna(self.fill_.reindex(self.columns_))
        Z = (X - self.mean_) / self.scale_
        return Z.to_numpy()

    def predict(self, X: pd.DataFrame) -> pd.Series:
        return pd.Series(self.regressor.predict(self._transform(X)), index=X.index)
