"""Plain, interpretable model-performance metrics (PD discrimination,
calibration, and population stability). No heavy ML -- just the numbers a
model-validation reviewer expects to see."""

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


def auc(y_true, y_score):
    """Area under the ROC curve: chance of ranking a random bad above a random good."""
    return roc_auc_score(y_true, y_score)


def gini(y_true, y_score):
    """Gini = 2*AUC - 1. A common scorecard discrimination summary."""
    return 2 * auc(y_true, y_score) - 1


def ks(y_true, y_score):
    """Kolmogorov-Smirnov: the biggest gap between the cumulative good and bad
    score distributions. Higher = better separation."""
    df = pd.DataFrame({"y": np.asarray(y_true), "s": np.asarray(y_score)}).sort_values("s")
    bads = df["y"].sum()
    goods = len(df) - bads
    cum_bad = df["y"].cumsum() / max(bads, 1)
    cum_good = (1 - df["y"]).cumsum() / max(goods, 1)
    return float((cum_bad - cum_good).abs().max())


def calibration_table(y_true, y_score, n_bins=10):
    """Bucket predictions into deciles and compare predicted vs observed default
    rate. The closer the two columns track, the better-calibrated the PD."""
    df = pd.DataFrame({"y": np.asarray(y_true), "p": np.asarray(y_score)})
    df["bucket"] = pd.qcut(df["p"], q=n_bins, duplicates="drop")
    tbl = df.groupby("bucket", observed=True).agg(
        n=("y", "size"),
        predicted_pd=("p", "mean"),
        observed_default_rate=("y", "mean"),
    ).reset_index()
    tbl["bucket"] = tbl["bucket"].astype(str)
    return tbl


def psi(expected, actual, n_bins=10):
    """Population Stability Index: how far a score has drifted between two
    samples. <0.10 stable, 0.10-0.25 watch, >0.25 material shift."""
    expected = pd.Series(expected).dropna()
    actual = pd.Series(actual).dropna()
    quantiles = np.linspace(0, 1, n_bins + 1)
    edges = np.unique(np.quantile(expected, quantiles))
    edges[0], edges[-1] = -np.inf, np.inf
    e = pd.cut(expected, edges).value_counts(normalize=True).sort_index()
    a = pd.cut(actual, edges).value_counts(normalize=True).sort_index()
    e = e.replace(0, 1e-6)
    a = a.replace(0, 1e-6)
    return float(((a - e) * np.log(a / e)).sum())
