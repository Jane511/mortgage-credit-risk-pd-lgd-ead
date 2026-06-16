"""
tools/make_figures.py — regenerate the README charts for this repo.

Every figure is built from the committed pipeline outputs in outputs/tables/ (aggregated
results only — rates, totals, distributions; never raw loan-level records), so the
charts regenerate reproducibly with:

    python tools/make_figures.py

Outputs PNGs into outputs/charts/.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "tables"
FIG = ROOT / "outputs" / "charts"
FIG.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "figure.dpi": 130, "savefig.dpi": 130,
    "font.size": 13, "axes.titlesize": 15, "axes.titleweight": "bold",
    "axes.labelsize": 13, "axes.grid": True, "grid.alpha": 0.25,
    "axes.spines.top": False, "axes.spines.right": False,
})
CRISIS, CALM, ACCENT = "#b2182b", "#2166ac", "#4d4d4d"


def save(fig, name):
    fig.tight_layout()
    fig.savefig(FIG / name)
    plt.close(fig)
    print("wrote", FIG / name)


# 1. Expected Loss — baseline vs stressed (the ~10x headline) ----------------
st = pd.read_csv(OUT / "07_stress_test.csv").set_index("scenario")
order = ["baseline", "mild recession", "severely adverse"]
vals = [st.loc[s, "expected_loss"] / 1e6 for s in order]
sev_uplift = st.loc["severely adverse", "EL_uplift_x"]
fig, ax = plt.subplots(figsize=(6.8, 4.6))
bars = ax.bar(["Baseline\n(calm)", "Mild\nrecession", "Severely\nadverse"], vals,
              color=[CALM, "#f0a500", CRISIS], width=0.62)
for b, v in zip(bars, vals):
    ax.text(b.get_x() + b.get_width() / 2, v, f"${v:,.0f}m",
            ha="center", va="bottom", fontweight="bold")
ax.set_ylabel("portfolio expected loss ($m)")
ax.set_title(f"Expected loss rises ~{sev_uplift:.0f}× under a severe downturn")
ax.set_ylim(0, max(vals) * 1.18)
save(fig, "expected_loss_base_vs_stress.png")

# 2. Default rate by vintage (crisis vs calm) --------------------------------
v = pd.read_csv(OUT / "01_default_lgd_by_vintage.csv")
labels = [f"{y}\n{'crisis' if y != 2015 else 'calm'}" for y in v.vintage_year]
colors = [CRISIS if y != 2015 else CALM for y in v.vintage_year]
fig, ax = plt.subplots(figsize=(6.5, 4.6))
bars = ax.bar(labels, v.default_rate * 100, color=colors, width=0.6)
for b, val in zip(bars, v.default_rate * 100):
    ax.text(b.get_x() + b.get_width() / 2, val, f"{val:.1f}%",
            ha="center", va="bottom", fontweight="bold")
ax.set_ylabel("default rate (%)")
ax.set_title("Default rate by origination year — crisis vs calm")
ax.set_ylim(0, v.default_rate.max() * 100 * 1.18)
save(fig, "default_rate_by_vintage.png")

# 3. LGD — calm vs downturn ---------------------------------------------------
l = pd.read_csv(OUT / "04_lgd_model.csv")
l = l[l.regime != "all"]
fig, ax = plt.subplots(figsize=(6.5, 4.6))
x = range(len(l))
ax.bar([i - 0.2 for i in x], l.observed_lgd * 100, width=0.4, label="observed", color=ACCENT)
ax.bar([i + 0.2 for i in x], l.modelled_lgd * 100, width=0.4, label="modelled", color=CRISIS)
ax.set_xticks(list(x))
ax.set_xticklabels(["calm\n(2015)", "downturn\n(2007-08)"])
for i, (o, mo) in enumerate(zip(l.observed_lgd * 100, l.modelled_lgd * 100)):
    ax.text(i - 0.2, o, f"{o:.0f}%", ha="center", va="bottom", fontsize=11)
    ax.text(i + 0.2, mo, f"{mo:.0f}%", ha="center", va="bottom", fontsize=11)
ax.set_ylabel("loss given default (%)")
ax.set_title("LGD more than doubles in a downturn")
ax.legend(frameon=False)
ax.set_ylim(0, 70)
save(fig, "lgd_calm_vs_downturn.png")

# 4. PD calibration — predicted vs observed by rating grade ------------------
ms = pd.read_csv(OUT / "03b_master_scale.csv")
fig, ax = plt.subplots(figsize=(6.5, 4.6))
x = range(len(ms))
ax.plot(x, ms.predicted_pd * 100, "o-", color=CALM, label="predicted PD", linewidth=2)
ax.plot(x, ms.observed_default_rate * 100, "s--", color=CRISIS, label="observed default rate", linewidth=2)
ax.set_xticks(list(x))
ax.set_xticklabels(ms.grade)
ax.set_xlabel("rating grade (A = safest → H = riskiest)")
ax.set_ylabel("default probability (%)")
ax.set_title("PD calibration: predicted vs observed by grade")
ax.legend(frameon=False)
save(fig, "pd_calibration_by_grade.png")

# 5. Default rate by credit score band (risk rises as expected) --------------
d = pd.read_csv(OUT / "02_risk_by_driver.csv")
cs = d[d.driver == "credit_score"]
fig, ax = plt.subplots(figsize=(6.5, 4.6))
bars = ax.bar(cs.band, cs.default_rate * 100, color=CRISIS, width=0.7)
for b, val in zip(bars, cs.default_rate * 100):
    ax.text(b.get_x() + b.get_width() / 2, val, f"{val:.0f}%",
            ha="center", va="bottom", fontsize=11, fontweight="bold")
ax.set_xlabel("credit score band")
ax.set_ylabel("default rate (%)")
ax.set_title("Default rate falls steeply as credit score rises")
save(fig, "default_rate_by_credit_score.png")

print("\nAll figures written to", FIG)
