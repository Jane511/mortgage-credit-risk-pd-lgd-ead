"""Plain, interpretable model-performance metrics (PD discrimination,
calibration, and population stability). No heavy ML -- just the numbers a
model-validation reviewer expects to see."""

import numpy as np
import pandas as pd
from scipy import stats
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


def binomial_pd_test(predicted_pd, observed_defaults, n):
    """One-sided binomial calibration test for PD UNDER-estimation (PD-4).

    Under H0 the grade's defaults are Binomial(n, predicted_pd). The returned p-value
    is P(X >= observed_defaults): a SMALL p means materially more defaults occurred
    than the assigned PD predicts (the PD is under-estimated -> review).

    NOTE: this assumes defaults are INDEPENDENT, which understates Type-I error when
    defaults are correlated (WP14) -- so read amber/red as a prompt, not a hard fail.
    """
    p = min(max(float(predicted_pd), 1e-9), 1 - 1e-9)
    return float(stats.binom.sf(int(observed_defaults) - 1, int(n), p))


def hosmer_lemeshow(y_true, y_score, n_bins=10):
    """Hosmer-Lemeshow chi-square calibration test (PD-4). Bucket by predicted score
    into deciles, compare observed vs expected defaults, sum (O-E)^2 / (E(1-E/n))
    over the buckets -> chi-square with (n_bins - 2) dof. Returns (statistic, p_value);
    a small p-value means observed and predicted diverge across the deciles."""
    df = pd.DataFrame({"y": np.asarray(y_true, dtype=float), "p": np.asarray(y_score, dtype=float)})
    df["bin"] = pd.qcut(df["p"], n_bins, duplicates="drop")
    g = df.groupby("bin", observed=True).agg(n=("y", "size"), obs=("y", "sum"), exp=("p", "sum"))
    g = g[g["n"] > 0]
    denom = (g["exp"] * (1 - g["exp"] / g["n"])).replace(0, np.nan)
    stat = float((((g["obs"] - g["exp"]) ** 2) / denom).sum())
    dof = max(len(g) - 2, 1)
    return stat, float(stats.chi2.sf(stat, dof))


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
