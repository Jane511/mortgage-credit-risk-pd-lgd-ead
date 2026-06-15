"""Turn a WOE logistic PD into a points-based scorecard and a rating master scale.

Standard scorecard scaling (Siddiqi):
    Factor = PDO / ln(2)
    Offset = base_points - Factor * ln(base_odds)
    Score  = Offset + Factor * ln(odds_good:bad) = Offset - Factor * logit_default
A higher score means a safer loan; every `PDO` points the good:bad odds double.
"""

import numpy as np
import pandas as pd

GRADE_LETTERS = list("ABCDEFGHIJ")  # A = best (safest)


def scaling_params(base_points, base_odds, pdo):
    """Return (Factor, Offset) for the chosen anchor: `base_odds`:1 good:bad at
    `base_points`, with `pdo` points to double the odds."""
    factor = pdo / np.log(2)
    offset = base_points - factor * np.log(base_odds)
    return factor, offset


def score_from_logit(default_logit, factor, offset):
    """Score from the model's default log-odds. ln(odds good:bad) = -logit."""
    return offset - factor * default_logit


def assign_grades(scores, n_grades=8):
    """Cut scores into rating grades A (best) .. worst by equal-population bands.
    Returns an ordered categorical so A always sorts first."""
    bands = pd.qcut(scores, q=n_grades, duplicates="drop")
    cats = list(bands.cat.categories)            # ascending score order
    letters = GRADE_LETTERS[:len(cats)][::-1]    # highest score band -> 'A'
    label_map = {cat: letters[i] for i, cat in enumerate(cats)}
    ordered = sorted(letters)
    return bands.map(label_map).astype(pd.CategoricalDtype(ordered, ordered=True))


def master_scale(df, grade_col, pd_col, target_col, exposure_col, score_col=None):
    """The rating master scale: per grade, predicted PD vs observed default rate
    (a calibration check), with loan count and exposure share. Pass `score_col`
    to also show each grade's score band (its min and max score)."""
    total_exp = df[exposure_col].sum()
    aggs = dict(
        loans=(target_col, "size"),
        predicted_pd=(pd_col, "mean"),
        observed_default_rate=(target_col, "mean"),
        exposure=(exposure_col, "sum"),
    )
    if score_col is not None:
        aggs["score_min"] = (score_col, "min")
        aggs["score_max"] = (score_col, "max")
    g = df.groupby(grade_col, observed=True).agg(**aggs).reset_index()
    if score_col is not None:
        g["score_min"] = g["score_min"].round(0)
        g["score_max"] = g["score_max"].round(0)
    g["exposure_share"] = g["exposure"] / total_exp
    g["predicted_pd"] = g["predicted_pd"].round(4)
    g["observed_default_rate"] = g["observed_default_rate"].round(4)
    g["exposure_share"] = g["exposure_share"].round(4)
    g["exposure"] = g["exposure"].round(0)
    return g.sort_values(grade_col).reset_index(drop=True)


def long_run_grade_pd(df, grade_col, target_col, year_col, exposure_col=None):
    """Long-run PD per grade (PD-3): the one-year default rate computed PER YEAR and
    then SIMPLE-averaged across the years -- count-weighted *within* each year, equal
    weight *across* years. This is the framework's calibration basis (APG 113 paras
    110-114; APS 113 Att D PD para 3 -- count-weighted, NOT exposure-weighted).

    If `exposure_col` is given, also returns an EXPOSURE-weighted PD per grade for
    sensitivity review only (APG 113 para 114) -- clearly not the calibration figure.
    """
    per_year = df.groupby([grade_col, year_col], observed=True)[target_col].mean()
    out = per_year.groupby(level=0).mean().rename("long_run_pd").reset_index()
    if exposure_col is not None:
        ew = (df.groupby(grade_col, observed=True)
                .apply(lambda g: np.average(g[target_col], weights=g[exposure_col]))
                .rename("exposure_weighted_pd").reset_index())
        out = out.merge(ew, on=grade_col)
    return out


def scorecard_points(binning_store, woe_tables, coefs, intercept, factor, offset):
    """Per-bin points table: each feature band's contribution to the score.
    Points for a loan sum to its total score, so the card is fully transparent."""
    n = len(coefs)
    base_per_feature = (offset - factor * intercept) / n
    rows = []
    for feature, beta in coefs.items():
        tbl = woe_tables[woe_tables["feature"] == feature]
        for _, r in tbl.iterrows():
            pts = base_per_feature - factor * beta * r["woe"]
            rows.append({
                "feature": feature, "bin": r["bin"], "woe": round(r["woe"], 4),
                "coefficient": round(beta, 4), "points": int(round(pts)),
            })
    return pd.DataFrame(rows)
