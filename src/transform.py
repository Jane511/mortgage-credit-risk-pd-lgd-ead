"""Apply fitted WOE bins to data (ported from the consumer-credit project)."""

import pandas as pd


def apply_bins(series, spec):
    """Re-bin a series using a bin definition produced by woe.fit_woe."""
    if spec["type"] == "numeric":
        return pd.cut(series, bins=spec["edges"], include_lowest=True).astype(str).fillna("MISSING")
    return series.astype(str).fillna("MISSING")


def transform_to_woe(df, binning_store, default_woe=0.0):
    """Turn raw predictors into their WOE values, one column per feature.
    Bands unseen in training fall back to a neutral WOE of 0."""
    out = pd.DataFrame(index=df.index)
    for feature, meta in binning_store.items():
        bins = apply_bins(df[feature], meta["spec"])
        out[feature] = bins.map(meta["mapping"]).fillna(default_woe)
    return out
