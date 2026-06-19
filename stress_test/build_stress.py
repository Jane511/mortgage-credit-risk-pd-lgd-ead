"""
Statistical macro-credit ("satellite") stress test for the Freddie Mac mortgage book.

WHY THIS EXISTS
---------------
The parent project (notebook 07) stresses the book with *observed multipliers* -- the
simple "Method B" of the compliance guidance. This module implements the bank-grade
*statistical* approach ("Method C" / the ECB-style satellite model):

    macro scenario  ->  satellite model  ->  stressed PD  ->  stressed Expected Loss

It is feasible here only because the panel is rich: ~850k loans observed monthly from
2006 to 2025, spanning the GFC, the recovery and COVID -- enough adverse history to
*estimate* (not assume) how default risk responds to the economy.

PIPELINE
--------
1. Build a calendar-quarter POINT-IN-TIME default-rate series from the parent project's
   cached loan-level table (../data/processed/loan_level.parquet). For each quarter:
   default rate = (loans first defaulting that quarter) / (loans at risk that quarter).
2. Merge external US macro variables (macro/macro_annual.csv, interpolated to quarterly).
3. Fit the satellite regression  logit(default_rate_t) ~ macro_t (+ a seasoning control),
   and validate it: coefficient signs, fit, and an out-of-time test on the 2018-2025 tail
   (which includes the COVID shock the model never trained on).
4. Define baseline / mild-recession (Basel CRE36.51) / severe macro paths, run them through
   the satellite model to get a stressed PD, then stressed EL = PD x LGD x EAD with the
   no-diversification rule (APG 113 para 92) and a monotonicity check.

COMPLIANCE: see README.md for the full mapping (CRE36.50-53, APS 113 Att D, APG 113 para
92/93, APS 220 paras 70-76). All macro values and scenarios are illustrative.

Run:  python build_stress.py     (from the stress_test/ folder)
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

HERE = os.path.dirname(os.path.abspath(__file__))
PARENT_PANEL = os.path.join(HERE, "..", "data", "processed", "loan_level.parquet")
MACRO_CSV = os.path.join(HERE, "macro", "macro_annual.csv")
TAB = os.path.join(HERE, "outputs", "tables")
FIG = os.path.join(HERE, "outputs", "charts")
os.makedirs(TAB, exist_ok=True)
os.makedirs(FIG, exist_ok=True)

# mortgage_rate is in the macro file but DELIBERATELY EXCLUDED from the model: in this
# sample it carries a perverse (negative) coefficient because the Fed CUT rates during the
# GFC/COVID, so low rates coincide with high defaults. Failing the economic-sign check, it
# is dropped under the "economic sign restriction" principle (ECB satellite practice; the
# guidance's "check whether coefficient signs make economic sense").
MACRO_VARS = ["unemployment", "hpi_yoy", "gdp_growth"]
# Expected economic sign of each driver on default risk (used as a validation check).
EXPECTED_SIGN = {"unemployment": +1, "hpi_yoy": -1, "gdp_growth": -1, "avg_age_q": +1}

# Portfolio anchors carried over from the parent project's Expected-Loss build so the
# stressed numbers reconcile with notebook 06 (see README).
BASE_PORTFOLIO_PD = 0.0040    # portfolio avg calibrated 1-yr PD (06_el_summary_by_grade)
LGD_CALM = 0.34               # non-GFC realised LGD (04_lgd_model)
LGD_DOWNTURN = 0.565          # GFC realised LGD (04_lgd_model)


def yyyymm_to_qord(s):
    """YYYYMM -> a quarter ordinal (year*4 + quarter-1), so quarters sort and subtract."""
    s = pd.to_numeric(s, errors="coerce")
    y = np.floor(s / 100.0)
    m = s - y * 100.0
    q = np.floor((m - 1) / 3.0)
    return (y * 4 + q)


def qord_to_label(q):
    y = int(q // 4)
    return f"{y}Q{int(q % 4) + 1}"


# ---------------------------------------------------------------------------
# 1. Point-in-time default-rate panel from the parent loan-level table
# ---------------------------------------------------------------------------
def build_default_panel(min_at_risk=2000):
    df = pd.read_parquet(PARENT_PANEL)
    orig_q = yyyymm_to_qord(df["first_payment_date"]).to_numpy()
    last_q = yyyymm_to_qord(df["last_period"]).to_numpy()
    def_q = yyyymm_to_qord(df["default_period"]).to_numpy()  # NaN for non-defaults
    ever = df["ever_default"].to_numpy()
    # Exposure proxy: balance at default for defaulters, else original loan amount.
    ead = np.where(ever, df["ead"].fillna(df["original_upb"]).to_numpy(),
                   df["original_upb"].to_numpy())
    # A loan is "at risk" from origination until it exits (defaults, or is last observed).
    exit_q = np.where(ever, def_q, last_q)
    def_q_filled = np.where(np.isnan(def_q), -1, def_q)

    qmin = int(np.nanmin(orig_q))
    qmax = int(np.nanmax(exit_q))
    rows = []
    for Q in range(qmin, qmax + 1):
        at_risk = (orig_q <= Q) & (exit_q >= Q)
        n = int(at_risk.sum())
        if n < min_at_risk:
            continue
        n_def = int((def_q_filled == Q).sum())
        rows.append({
            "qord": Q,
            "quarter": qord_to_label(Q),
            "year": Q // 4,
            "n_at_risk": n,
            "n_default": n_def,
            "default_rate": n_def / n,
            # average seasoning (quarters since origination) of the at-risk pool --
            # the control that separates "macro is bad" from "the book is at peak-default age".
            "avg_age_q": float((Q - orig_q[at_risk]).mean()),
            "ead_at_risk": float(ead[at_risk].sum()),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 2. Merge macro (annual -> quarterly by linear interpolation)
# ---------------------------------------------------------------------------
def load_quarterly_macro(qords):
    macro = pd.read_csv(MACRO_CSV)
    # Place each annual value at mid-year (Q3 ordinal) then interpolate across quarters.
    idx = pd.Index(range(int(min(qords)), int(max(qords)) + 1), name="qord")
    frame = pd.DataFrame(index=idx)
    for v in MACRO_VARS:
        anchor = {int(y * 4 + 2): val for y, val in zip(macro["year"], macro[v])}
        s = pd.Series({q: anchor.get(q, np.nan) for q in idx}, index=idx)
        frame[v] = s.interpolate(limit_direction="both")
    return frame.reset_index()


# ---------------------------------------------------------------------------
# 3. Fit + validate the satellite model
# ---------------------------------------------------------------------------
def fit_satellite(panel):
    feats = MACRO_VARS + ["avg_age_q"]
    eps = 1e-4
    dr = panel["default_rate"].clip(eps, 1 - eps)
    y = np.log(dr / (1 - dr))                      # logit of the quarterly default rate
    X = panel[feats].astype(float)
    mu, sd = X.mean(), X.std(ddof=0)
    Xs = (X - mu) / sd                             # standardise -> comparable coefficients
    model = LinearRegression().fit(Xs, y)
    pred = model.predict(Xs)
    ss_res = float(((y - pred) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1 - ss_res / ss_tot
    # Observed support of each driver -- scenarios are clipped to this so the model is never
    # used outside the range it was estimated on (no unsupported extrapolation).
    support = {f: (float(X[f].min()), float(X[f].max())) for f in feats}

    coef = pd.DataFrame({"variable": feats, "coefficient_std": model.coef_.round(4)})
    coef["expected_sign"] = [EXPECTED_SIGN[f] for f in feats]
    coef["sign_ok"] = np.sign(coef["coefficient_std"]) == coef["expected_sign"]
    coef = pd.concat([
        pd.DataFrame([{"variable": "intercept", "coefficient_std": round(float(model.intercept_), 4),
                       "expected_sign": np.nan, "sign_ok": np.nan}]),
        coef,
    ], ignore_index=True)

    info = {"model": model, "feats": feats, "mu": mu, "sd": sd, "r2": r2, "coef": coef,
            "support": support}

    # Out-of-time: fit on <2018, predict the 2018-2025 tail (incl. COVID), report fit.
    tr = panel[panel["year"] < 2018]
    te = panel[panel["year"] >= 2018]
    if len(te) >= 4 and len(tr) >= 12:
        ytr = np.log(tr["default_rate"].clip(eps, 1 - eps) / (1 - tr["default_rate"].clip(eps, 1 - eps)))
        m2 = LinearRegression().fit((tr[feats] - mu) / sd, ytr)
        pte = m2.predict((te[feats] - mu) / sd)
        obs = np.log(te["default_rate"].clip(eps, 1 - eps) / (1 - te["default_rate"].clip(eps, 1 - eps)))
        oos_corr = float(np.corrcoef(pte, obs)[0, 1])
        oos_rmse = float(np.sqrt(((pte - obs) ** 2).mean()))
    else:
        oos_corr, oos_rmse = np.nan, np.nan
    info["oos_corr"], info["oos_rmse"] = oos_corr, oos_rmse
    panel = panel.copy()
    panel["fitted_logit"] = pred
    panel["fitted_default_rate"] = 1 / (1 + np.exp(-pred))
    return info, panel


def predict_dr(info, macro_row, clip=True):
    """Predicted quarterly default rate for a macro vector (dict of feats). By default the
    inputs are clipped to the estimation sample's observed range -- the model is never asked
    to extrapolate beyond the data it was fit on."""
    row = dict(macro_row)
    if clip:
        for f in info["feats"]:
            lo, hi = info["support"][f]
            row[f] = min(max(row[f], lo), hi)
    x = pd.DataFrame([row])[info["feats"]].astype(float)
    xs = (x - info["mu"]) / info["sd"]
    logit = float(info["model"].predict(xs)[0])
    return 1 / (1 + np.exp(-logit))


# ---------------------------------------------------------------------------
# 4. Scenarios -> stressed PD -> stressed EL
# ---------------------------------------------------------------------------
def scenario_paths(panel):
    """3-year macro paths. 'baseline' = recent calm average; mild = Basel CRE36.51 example;
    severe = a GFC-like adverse path (kept within the estimation sample's observed envelope
    so the satellite is not used to extrapolate). avg_age held at the current book's level."""
    recent = panel[panel["year"].between(2015, 2019)]
    base = {v: float(recent[v].mean()) for v in MACRO_VARS}
    avg_age = float(panel["avg_age_q"].iloc[-8:].mean())   # current-ish book seasoning

    def yr(unemp, hpi, gdp):
        return {"unemployment": unemp, "hpi_yoy": hpi, "gdp_growth": gdp, "avg_age_q": avg_age}

    return base, {
        "baseline": [yr(base["unemployment"], base["hpi_yoy"], base["gdp_growth"])] * 3,
        # Mild recession (CRE36.51: ~2 quarters of zero growth) -- modest, short.
        "mild recession": [yr(6.5, -3.0, 0.0), yr(7.0, -4.0, 0.5), yr(6.0, 0.0, 1.5)],
        # Severe (GFC-like) -- deep, near the worst the panel actually observed (~unemp 9.6,
        # HPI -9.5%), so it sits at the edge of support rather than beyond it.
        "severe (GFC-like)": [yr(9.0, -9.0, -2.5), yr(9.6, -7.0, -1.0), yr(8.5, -3.0, 1.0)],
    }


def lgd_for_scenario(name, hpi_worst):
    """Scenario LGD anchored to the parent's observed regimes, scaled by the property
    shock (collateral-driven). Baseline ~ calm LGD; severe ~ downturn LGD."""
    if name == "baseline":
        return LGD_CALM
    # linear in the property-price fall between calm (hpi>=0) and downturn (hpi<=-25)
    frac = min(max(-hpi_worst / 25.0, 0.0), 1.0)
    return round(LGD_CALM + frac * (LGD_DOWNTURN - LGD_CALM), 4)


def run_scenarios(info, panel, total_ead):
    base_macro, paths = scenario_paths(panel)
    dr_base = predict_dr(info, paths["baseline"][0])
    rows = []
    for name, path in paths.items():
        # Worst (year-1) macro drives the one-year stressed PD/LGD for the headline number.
        worst = max(path, key=lambda r: predict_dr(info, r))
        dr_s = predict_dr(info, worst)
        pd_mult = dr_s / dr_base
        stressed_pd = min(BASE_PORTFOLIO_PD * pd_mult, 1.0)
        stressed_lgd = lgd_for_scenario(name, worst["hpi_yoy"])
        # No-diversification (APG 113 para 92): PD and LGD shocks stack, no offset.
        el = stressed_pd * stressed_lgd * total_ead
        rows.append({
            "scenario": name,
            "worst_unemployment": worst["unemployment"],
            "worst_hpi_yoy": worst["hpi_yoy"],
            "satellite_default_rate": round(dr_s, 5),
            "pd_multiplier_vs_base": round(pd_mult, 2),
            "stressed_pd": round(stressed_pd, 5),
            "stressed_lgd": stressed_lgd,
            "stressed_EL": round(el, 0),
        })
    out = pd.DataFrame(rows)
    base_el = out.loc[out["scenario"] == "baseline", "stressed_EL"].iloc[0]
    out["EL_uplift_x"] = (out["stressed_EL"] / base_el).round(2)
    return out


# ---------------------------------------------------------------------------
# Charts
# ---------------------------------------------------------------------------
def make_charts(panel, scen):
    # Default rate vs unemployment over time (the satellite relationship, visually).
    fig, ax1 = plt.subplots(figsize=(10, 4.6))
    ax1.bar(panel["qord"], panel["default_rate"] * 100, width=0.8, color="#b2182b",
            alpha=0.55, label="quarterly default rate (%)")
    ax1.plot(panel["qord"], panel["fitted_default_rate"] * 100, color="#222", lw=1.6,
             label="satellite fitted")
    ax1.set_ylabel("quarterly default rate (%)")
    ax2 = ax1.twinx()
    ax2.plot(panel["qord"], panel["unemployment"], color="#2166ac", lw=1.6, label="unemployment (%)")
    ax2.set_ylabel("unemployment (%)", color="#2166ac")
    ticks = panel["qord"][::8]
    ax1.set_xticks(ticks); ax1.set_xticklabels([qord_to_label(q) for q in ticks], rotation=45)
    ax1.set_title("Satellite model: quarterly default rate vs the macro cycle")
    ax1.legend(loc="upper right", frameon=False, fontsize=9)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "satellite_fit.png"), dpi=120); plt.close(fig)

    # Stressed EL by scenario.
    fig, ax = plt.subplots(figsize=(6.6, 4.4))
    colors = {"baseline": "#2166ac", "mild recession": "#f0a500", "severe (GFC-like)": "#b2182b"}
    bars = ax.bar(scen["scenario"], scen["stressed_EL"] / 1e6,
                  color=[colors.get(s, "#888") for s in scen["scenario"]], width=0.6)
    for b, v, x in zip(bars, scen["stressed_EL"] / 1e6, scen["EL_uplift_x"]):
        ax.text(b.get_x() + b.get_width() / 2, v, f"${v:,.0f}m\n{x:.1f}x",
                ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.set_ylabel("stressed 1-yr Expected Loss ($m)")
    ax.set_title("Satellite-model stressed Expected Loss by scenario")
    ax.set_ylim(0, (scen["stressed_EL"] / 1e6).max() * 1.25)
    ax.tick_params(axis="x", rotation=15)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "scenario_expected_loss.png"), dpi=120); plt.close(fig)


def main():
    print("1) building point-in-time default panel from", os.path.relpath(PARENT_PANEL, HERE), "...")
    panel = build_default_panel()
    macro = load_quarterly_macro(panel["qord"])
    panel = panel.merge(macro, on="qord", how="left")
    panel.to_csv(os.path.join(TAB, "satellite_panel.csv"), index=False)
    print(f"   {len(panel)} quarters, {panel['quarter'].iloc[0]}..{panel['quarter'].iloc[-1]}; "
          f"peak default rate {panel['default_rate'].max()*100:.2f}%")

    print("2) fitting satellite model ...")
    info, panel = fit_satellite(panel)
    info["coef"].to_csv(os.path.join(TAB, "satellite_coefficients.csv"), index=False)
    fit_tbl = pd.DataFrame([
        {"metric": "in-sample R2 (logit dr)", "value": round(info["r2"], 4)},
        {"metric": "out-of-time corr (>=2018)", "value": round(info["oos_corr"], 4)},
        {"metric": "out-of-time RMSE (logit)", "value": round(info["oos_rmse"], 4)},
        {"metric": "all coefficient signs economic?", "value": bool(info["coef"]["sign_ok"].dropna().all())},
    ])
    fit_tbl.to_csv(os.path.join(TAB, "satellite_fit.csv"), index=False)
    panel.to_csv(os.path.join(TAB, "satellite_panel.csv"), index=False)
    print("   R2={:.3f}  OOS corr={:.3f}  signs_ok={}".format(
        info["r2"], info["oos_corr"], bool(info["coef"]["sign_ok"].dropna().all())))
    print(info["coef"].to_string(index=False))

    print("3) running scenarios ...")
    total_ead = float(pd.read_parquet(PARENT_PANEL)["original_upb"].sum())
    scen = run_scenarios(info, panel, total_ead)
    scen.to_csv(os.path.join(TAB, "scenario_stressed_el.csv"), index=False)
    print(scen.to_string(index=False))

    # Monotonicity control (more severe should not lose less).
    order = ["baseline", "mild recession", "severe (GFC-like)"]
    els = [scen.loc[scen.scenario == s, "stressed_EL"].iloc[0] for s in order]
    print("   monotonic (baseline <= mild <= severe):", els[0] <= els[1] <= els[2])

    # Triangulation (model-risk control): cross-check the satellite tail against (a) the worst
    # quarterly default rate actually observed and (b) the parent project's simple multiplier
    # approach (notebook 07). A satellite that runs far hotter than both is over-extrapolating.
    base_macro, paths = scenario_paths(panel)
    base_dr = predict_dr(info, paths["baseline"][0])
    obs_peak_mult = float(panel["default_rate"].max() / base_dr)
    tri = [{"method": "satellite severe (this module)",
            "severe_PD_mult_x": float(scen.loc[scen.scenario == "severe (GFC-like)", "pd_multiplier_vs_base"].iloc[0]),
            "note": "logit-linear macro model, inputs clipped to support"},
           {"method": "observed worst quarter (data)", "severe_PD_mult_x": round(obs_peak_mult, 1),
            "note": "peak quarterly default rate / baseline -- the realised ceiling"}]
    try:
        p07 = pd.read_csv(os.path.join(HERE, "..", "outputs", "tables", "07_stress_test.csv"))
        sev = float(p07.loc[p07["scenario"].str.contains("sever", case=False), "pd_mult"].iloc[0])
        tri.append({"method": "parent multiplier approach (nb07)", "severe_PD_mult_x": round(sev, 1),
                    "note": "observed GFC-vs-calm one-year PD ratio"})
    except Exception:
        pass
    pd.DataFrame(tri).to_csv(os.path.join(TAB, "triangulation.csv"), index=False)
    print(f"   triangulation: satellite severe PD x{tri[0]['severe_PD_mult_x']:.1f} | "
          f"observed-peak ceiling x{obs_peak_mult:.1f}")

    make_charts(panel, scen)
    print("done. tables -> outputs/tables/  charts -> outputs/charts/")


if __name__ == "__main__":
    main()
