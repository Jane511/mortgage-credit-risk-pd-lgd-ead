"""Weight-of-Evidence (WOE) binning and Information Value (IV).

Ported from my consumer-credit scorecard project so the mortgage scorecard uses
the same, proven, interpretable binning logic. WOE replaces each predictor band
with a single number measuring how good/bad that band is; IV summarises how
predictive the whole feature is.
"""

import numpy as np
import pandas as pd


def safe_qcut(series, max_bins=5):
    """Quantile-bin a numeric series into up to max_bins, backing off the bin
    count until the cuts are valid (handles spiky/duplicated values)."""
    clean = series.dropna()
    if clean.nunique() <= 1:
        return None
    if clean.nunique() <= max_bins:
        try:
            return pd.cut(clean, bins=clean.nunique(), duplicates="drop")
        except Exception:
            return None
    for q in range(max_bins, 1, -1):
        try:
            cats = pd.qcut(clean, q=q, duplicates="drop")
            if len(cats.cat.categories) >= 2:
                return cats
        except Exception:
            continue
    return None


def fit_woe(train_series, target, feature_name, max_bins=5, smoothing=0.5):
    """Bin one feature and compute WOE + IV per bin.

    Returns (table, spec, mapping): the per-bin table, the bin definition used to
    re-apply the bins to new data, and a {bin -> woe} mapping. `target` is 1 for
    a bad loan (default)."""
    df = pd.DataFrame({"x": train_series, "target": target})

    if pd.api.types.is_numeric_dtype(df["x"]):
        binned = safe_qcut(df["x"], max_bins)
        if binned is None:
            df["bin"] = df["x"].astype(str).fillna("MISSING")
            spec = {"type": "categorical"}
        else:
            df.loc[binned.index, "bin"] = binned.astype(str)
            df["bin"] = df["bin"].fillna("MISSING")
            intervals = pd.IntervalIndex(binned.cat.categories)
            edges = [intervals[0].left] + [iv.right for iv in intervals]
            spec = {"type": "numeric", "edges": edges}
    else:
        df["bin"] = df["x"].astype(str).fillna("MISSING")
        spec = {"type": "categorical"}

    grp = df.groupby("bin")["target"].agg(total="count", bad="sum").reset_index()
    grp["good"] = grp["total"] - grp["bad"]

    total_good = grp["good"].sum()
    total_bad = grp["bad"].sum()

    # Laplace smoothing keeps WOE finite when a bin has zero goods or bads.
    grp["dist_good"] = (grp["good"] + smoothing) / (total_good + smoothing * len(grp))
    grp["dist_bad"] = (grp["bad"] + smoothing) / (total_bad + smoothing * len(grp))

    grp["woe"] = np.log(grp["dist_good"] / grp["dist_bad"])
    grp["iv_component"] = (grp["dist_good"] - grp["dist_bad"]) * grp["woe"]
    grp["iv"] = grp["iv_component"].sum()
    grp["feature"] = feature_name

    mapping = dict(zip(grp["bin"], grp["woe"]))
    return grp, spec, mapping


def fit_binning(df, features, target, max_bins=5):
    """Fit WOE for several features at once.

    Returns (binning_store, iv_summary, woe_tables): a {feature -> {spec, mapping,
    iv}} store for transforming new data, a per-feature IV summary, and the long
    per-bin WOE table for inspection."""
    store = {}
    iv_rows = []
    tables = []
    for f in features:
        tbl, spec, mapping = fit_woe(df[f], target, f, max_bins=max_bins)
        iv = float(tbl["iv"].iloc[0])
        store[f] = {"spec": spec, "mapping": mapping, "iv": iv}
        iv_rows.append({"feature": f, "information_value": round(iv, 4),
                        "strength": _iv_strength(iv)})
        tables.append(tbl)
    iv_summary = pd.DataFrame(iv_rows).sort_values("information_value", ascending=False)
    woe_tables = pd.concat(tables, ignore_index=True)
    return store, iv_summary, woe_tables


def _iv_strength(iv):
    """Standard IV rule of thumb for how predictive a feature is."""
    if iv < 0.02:
        return "not predictive"
    if iv < 0.10:
        return "weak"
    if iv < 0.30:
        return "medium"
    if iv < 0.50:
        return "strong"
    return "very strong (check)"
