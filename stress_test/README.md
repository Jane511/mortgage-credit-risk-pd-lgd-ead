# Statistical Macro-Credit Stress Test (Satellite Model)

> **In one line:** A bank-grade, *statistical* stress test for the Freddie Mac book —
> instead of assuming "PD ×3 in a recession", it **estimates** how default risk responds to
> the economy from ~850k loans observed monthly across the GFC, the recovery and COVID, then
> drives that fitted relationship with recession scenarios to produce stressed PD and
> Expected Loss.

This module sits alongside the parent project's notebook 07. Notebook 07 uses the **simple
multiplier** method (observed GFC-vs-calm ratios — "Method B" of the compliance guidance).
This folder implements the **macro-credit "satellite" model** ("Method C" / the ECB-style
approach a large, data-rich lender would use):

```
macro scenario  →  satellite model  →  stressed PD  →  stressed Expected Loss
```

It is only credible **because the data is rich** — 17 vintages (2006–2022) observed through
2025 give ~79 quarters of default experience spanning two genuine downturns, which is enough
to *estimate and validate* a macro-to-default relationship rather than assume one.

---

## Method — the four steps

| Step | What happens | File / output |
|---|---|---|
| **1. Default panel** | Build a calendar-quarter **point-in-time default-rate** series from the parent's cached loan-level table | `outputs/tables/satellite_panel.csv` |
| **2. Macro data** | US unemployment, house-price growth, GDP, mortgage rate (2006–2024), interpolated to quarterly | `macro/macro_annual.csv` |
| **3. Satellite model** | Fit `logit(default_rate_t) ~ macro_t (+ seasoning)`; validate signs, fit, out-of-time | `outputs/tables/satellite_coefficients.csv`, `satellite_fit.csv` |
| **4. Scenarios** | Run baseline / mild / severe macro paths through the model → stressed PD → stressed EL | `outputs/tables/scenario_stressed_el.csv` |

Run it with `python build_stress.py` (reads the parent's `data/processed/loan_level.parquet`).

---

## 1. Point-in-time default panel (the dependent variable)

For each calendar quarter 2006Q1–2025Q3 the pipeline computes, straight from the loan-level
table, a **marginal default rate**:

```
default_rate(q) = loans first defaulting in quarter q  /  loans at risk in quarter q
```

A loan is "at risk" from origination (`first_payment_date`) until it exits (its first 180+DPD/
credit-event month, or its last observed month). The series also carries the **average
seasoning** of the at-risk pool each quarter — a control that separates *"the economy is bad"*
from *"the book happens to be at its peak-default age"*.

**Result:** 79 quarters; the default rate peaks at **~1.1% per quarter in 2009–2010** (the GFC),
a smaller bump around COVID, and sits near **0.07%** in the calm expansion — a real, observed
credit cycle to fit against.

---

## 2. Macro data

`macro/macro_annual.csv` holds annual US series (unemployment, house-price YoY, real GDP
growth, 30-yr mortgage rate), interpolated to quarterly. The committed values are
**approximate public figures** so the pipeline runs offline; `fetch_macro_fred.py` refreshes
them from FRED (UNRATE, CSUSHPINSA, A191RL1Q225SBEA, MORTGAGE30US) for a production run.

---

## 3. The satellite model — methodology and results

### Methodology

A logit-linear regression of the quarterly default rate on **standardised** macro drivers plus
the seasoning control:

```
logit(default_rate_t) = α + β·[unemployment, ΔHPI, GDP growth, avg_age]_t
```

Standardising makes the coefficients directly comparable (effect per 1 standard deviation).

### Results — coefficients (`satellite_coefficients.csv`)

| Variable | Coefficient (per 1 SD) | Expected sign | OK? |
|---|---:|:---:|:---:|
| **unemployment** | **+0.97** | + | ✅ *(dominant driver)* |
| hpi_yoy (house-price growth) | −0.18 | − | ✅ |
| gdp_growth | −0.12 | − | ✅ |
| avg_age (seasoning control) | +0.50 | + | ✅ |
| ~~mortgage_rate~~ | *excluded* | + | ❌ → dropped |

**Unemployment is the dominant macro driver of mortgage default** — economically exactly
right. **`mortgage_rate` was deliberately dropped:** in this sample it carried a *perverse
negative* coefficient because the Fed **cut** rates during the GFC and COVID, so low rates
coincide with high defaults. Failing the economic-sign check, it is excluded under the
**sign-restriction** principle the guidance requires ("check whether coefficient signs make
economic sense"; ECB economic sign restrictions).

### Results — validation (`satellite_fit.csv`)

| Test | Result |
|---|---|
| In-sample fit (R² on logit default rate) | **0.81** |
| Out-of-time (fit ≤2017, predict 2018–2025 incl. COVID) | corr **0.46** |
| All coefficient signs economic | **Yes** |

The model **tracks the GFC peak well** (2010Q1 fitted 0.73% vs observed 0.81%) and the calm
years closely. Its one honest miss is **COVID-2020**, which it *over-predicts*: the model sees
the unemployment spike and expects high default, but **forbearance suppressed actual defaults**
— the same structural break the parent project flags. A production model would add a
forbearance/disaster control.

---

## 4. Scenarios and stressed Expected Loss

Three 3-year macro paths (year-1 peak shown) are run through the model. Scenario inputs are
**clipped to the estimation sample's observed range** so the model is never used to
extrapolate beyond the data it learned from. EL = stressed PD × stressed LGD × EAD, with the
**no-diversification** rule (APG 113 para 92 — PD and LGD shocks stack, no offset). Stressed
LGD is anchored to the parent's observed regimes (calm ~34% → downturn ~56%), scaled by the
property-price shock.

### Results (`scenario_stressed_el.csv`)

| Scenario | Worst unemp. | Worst ΔHPI | Satellite PD ×base | Stressed PD | Stressed LGD | Stressed EL | EL ×base |
|---|---:|---:|---:|---:|---:|---:|---:|
| **baseline** | 4.6% | +5% | 1.0× | 0.40% | 34% | ~$262m | 1.0× |
| **mild recession** (CRE36.51) | 7.0% | −4% | 5.4× | 2.2% | 38% | ~$1.6bn | 6.0× |
| **severe (GFC-like)** | 9.6% | −7% | 24.7× | 9.9% | 40% | ~$7.7bn | 29× |

Results are **monotonic** (severe > mild > baseline) and the **mild** scenario — which sits
comfortably inside the observed data — gives a very reasonable ~2.2% stressed one-year PD.

---

## 5. Model-risk controls and triangulation (read this before quoting the severe number)

The **severe** satellite result must be read with the model-risk discipline the guidance
demands ("compare with historical downturns", "test whether the model over/understates extreme
losses", "apply conservatism where data is scarce"). The triangulation
(`outputs/tables/triangulation.csv`):

| View | Severe PD multiplier | Why |
|---|---:|---|
| **Satellite (this module)** | **×24.7** | Imposes peak unemployment **and** a house-price crash **simultaneously** |
| Observed worst quarter (data) | ×7.8 | The realised ceiling — worst quarterly default rate ever seen |
| Parent multiplier approach (nb 07) | ×5.7 | The realised GFC-vs-calm one-year-PD ratio |

The satellite runs **hotter than anything realised** — not because it is broken, but because a
*simultaneous* unemployment-plus-house-price shock is **worse than the actual GFC**, where
those shocks peaked at *different* times (house prices fell hardest in 2008 at ~6% unemployment;
unemployment peaked in 2010 after prices had stabilised). The additive logit then compounds a
joint state that never occurred. **This is a genuine property of satellite models in the tail,
and the reason the reported severe figure should be triangulated** — treat the satellite as a
*simultaneous-shock upper bound*, use it directly for mild/moderate scenarios (within joint
support), and cross-check the extreme tail against the realised multiplier and the parent
approach. Other controls applied: inputs clipped to support, sign restriction, monotonicity
check, no-diversification, and the avg-age seasoning control.

---

## Compliance mapping

| Requirement | Source | Where |
|---|---|---|
| IRB credit-risk stress test; ≥ mild recession on PD/LGD/EAD | Basel **CRE36.50–36.53**; APS 113 Att D | mild-recession scenario (§4) |
| Satellite / macro-credit model is an accepted approach | guidance "Method C"; ECB satellite frameworks | the whole module |
| Use ratings-migration / smaller-deterioration / external evidence | **CRE36.52 / APG 113 para 93** | calendar-time default panel (§1) |
| No diversification benefit in stress | **APG 113 para 92** | PD×LGD stack, no offset (§4) |
| Severe but plausible; coefficient signs sensible; OOS testing; conservatism | APS 220 para 72; model-dev controls | §3, §5 |
| Validation independent / annual, compared to historical downturns | APS 220 para 76; APG 113 para 140 | §3 fit + §5 triangulation |

> **Applicability:** APS 113/APS 220 apply to APRA ADIs (IRB). This is a **portfolio
> demonstration** on US agency data — illustrative of the method, not a certified or
> APRA-/IRB-compliant capital stress test. All macro values and scenarios are illustrative.

---

## Limitations

- **Joint-tail extrapolation** — the severe scenario combines macro extremes that never
  co-occurred; the satellite over-states the tail (see §5 — triangulate).
- **COVID over-prediction** — the macro model misses forbearance; a disaster/forbearance
  control is the production fix.
- **Aggregate confound** — the quarterly default rate mixes macro with book seasoning; the
  avg-age control mitigates but does not fully remove this (a production model would use an
  age–period–cohort or vintage-fixed-effects design).
- **Macro data is approximate** — refresh via `fetch_macro_fred.py` for real figures.
- **One-year horizon** — multi-year/lifetime paths and grade-migration matrices (the
  through-the-cycle alternative in the guidance) are noted as the next extension.

---

## How to run

```bash
# from this stress_test/ folder, after the parent pipeline has cached loan_level.parquet
python build_stress.py            # builds panel, fits satellite, runs scenarios, writes outputs
# optional: refresh macro from FRED (needs internet + pandas-datareader)
python fetch_macro_fred.py
```

## Files

```
stress_test/
├── README.md                     # this file
├── build_stress.py               # the full satellite-model pipeline
├── fetch_macro_fred.py           # optional FRED refresh of the macro data
├── macro/macro_annual.csv        # US macro series (committed, approximate)
└── outputs/
    ├── tables/  satellite_panel.csv · satellite_coefficients.csv · satellite_fit.csv
    │            scenario_stressed_el.csv · triangulation.csv
    └── charts/  satellite_fit.png · scenario_expected_loss.png
```
