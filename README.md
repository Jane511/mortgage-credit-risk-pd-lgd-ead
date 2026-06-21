# Mortgage Credit Risk — PD, LGD, EAD, Expected Loss & Stress Testing

An end-to-end IRB-style credit-risk model for a US residential-mortgage portfolio: probability
of default (**PD**), loss given default (**LGD**), exposure at default (**EAD**), Expected Loss
(**EL = PD × LGD × EAD**), IFRS 9 / AASB 9 staging, and two methods of stress testing.

Built on the Freddie Mac Single-Family Loan-Level Dataset: **~850,000 loans across 17
origination vintages (2006–2022)**, observed monthly through 2025-09 — a full cycle covering the
2007–2009 financial crisis, the recovery, and the 2020 COVID shock. All models are interpretable
and every result is reproduced by the pipeline into [outputs/tables/](outputs/tables/).

## Methods used (summary)

| Component | Method |
|---|---|
| **PD** | Logistic regression on origination features → WOE/IV points scorecard → 8 rating grades (A–H). Grades calibrated to a count-weighted long-run one-year default rate; margin of conservatism; revise-upward ratchet; 5 bps floor. |
| **LGD** | Two-stage ("hurdle") model on settled-loss data: **Stage 1 logistic regression** (probability of a loss) × **Stage 2 linear regression** (loss severity). Economic-discounted and APRA-capital variants; downturn LGD. |
| **EAD** | Outstanding balance at default. Supervisory EAD — no credit-conversion factor (a term mortgage has no undrawn limit). |
| **EL** | EL = PD × LGD × EAD per loan, using the calibrated capital PD; IFRS 9 / AASB 9 Stage 1/2/3. |
| **Stress — method A** | Observed downturn multipliers (GFC and COVID) applied to PD and LGD. |
| **Stress — method B** | Macro-credit "satellite" model: logistic regression of the quarterly default rate on macro variables → per-grade stressed PD + grade migration. |
| **Validation** | Discrimination (AUC/Gini/KS), calibration plot, binomial + Hosmer-Lemeshow test, confusion matrix, out-of-time and forward cold holdout, PSI. LGD: loss reconciliation, cohort backtest, segment calibration, benchmarking. |

## Results (summary)

| Period (origination) | One-year default rate | Realised LGD | Notes |
|---|---|---|---|
| GFC crisis (2006–2008) | 7–14% | 54–58% | high default and high severity |
| Recovery (2009–2014) | ~2% | 34–42% | low default, easing severity |
| Calm expansion (2015–2019) | 2–4% | 18–25% | baseline conditions |
| COVID-2020 | 1.2% (forbearance-suppressed) | ~27% | high default risk, mild severity |

- **PD:** held-out AUC 0.83 / Gini 0.65 / KS 0.50; passes a forward cold holdout (train 2006–2019, score the never-seen 2020–2022 loans).
- **LGD:** computed loss reconciles to Freddie Mac's own loss field at 0.99 correlation; GFC downturn LGD ~56.5% vs ~34% outside it.
- **EL:** portfolio 12-month EL ~$236m (12.3 bps of $192bn exposure), rising to ~$323m under IFRS 9 lifetime staging.
- **Stress:** a severe GFC scenario lifts EL ~13× (PD ×5.7, LGD ×2.3); the observed COVID-2020 shape lifts it ~4.7× (PD ×4.3, LGD ×1.1).

---

## 1. Data and inputs

**Coverage.** 17 origination vintages, 2006–2022 (2005 is not in the public sample set), 50,000
loans each (~850,000 total), with monthly performance observed through 2025-09. The one-year PD
window is therefore fully observable for every vintage, including 2022.

**Source.** Freddie Mac Single-Family Loan-Level Dataset (SFLLD) — public US fixed-rate,
fully-amortising single-family loans. Two header-less pipe-delimited files per vintage: an
**origination** file (one row per loan) and a **monthly performance** file (one row per
loan-month, tens of millions of rows). Raw data is **not** redistributed in this repo.
Source: <https://freddiemac.com/research/datasets/sf-loanlevel-dataset>

**Key raw variables.**

| Origination file | Monthly performance file |
|---|---|
| `credit_score`, `original_ltv`, `original_cltv`, `original_dti` | `monthly_reporting_period`, `loan_age` |
| `original_upb` (loan amount), `original_interest_rate` | `current_actual_upb` (running balance) |
| `original_loan_term`, `number_of_borrowers` | `current_loan_delinquency_status` |
| `loan_purpose`, `occupancy_status`, `channel` | `zero_balance_code`, `zero_balance_effective_date` |
| `property_type`, `property_state`, `mi_pct` | `net_sales_proceeds`, `mi_recoveries`, `non_mi_recoveries` |
| | `expenses`, `delinquent_accrued_interest`, `actual_loss_calculation` |

**Cleaning, transformation, feature engineering.**

- **Layout assertion** — the 32-column SFLLD layout is applied in [src/layout.py](src/layout.py)
  and the column count is asserted before any names are attached, preventing a silent mis-mapping.
- **Sentinel handling** — SFLLD missing codes → NA (`credit_score` 9999, `original_dti` 999,
  `original_cltv`/`original_ltv` 999); delinquency text codes (`R`/`RA` for REO) parsed to numbers,
  blanks/`XX` → NA; money fields carrying letters (`C`/`U`) stripped before arithmetic.
- **Loss-field correction** — the aggregate `expenses` field equals the sum of its four
  sub-components and all are stored negative; adding them double-counts costs. Using the aggregate
  once, with the correct sign, makes the computed loss reconcile to `actual_loss_calculation` at
  0.99 correlation (notebook 01).
- **Collapse to loan level** — the monthly file is reduced to one row per loan: default flag and
  date, balance at default, one-time loss/recovery components.
- **Engineered fields** — `default_within_12m` (one-year PD target), `ever_default` (lifetime),
  realised / economic / APRA LGD, EAD, a regime classifier (GFC 2006–09 / COVID 2020 / calm), and a
  WOE/IV transform for the scorecard. → [00_data_quality.csv](outputs/tables/00_data_quality.csv)

---

## 2. PD model

Estimates the 12-month probability that a loan defaults.

**Method.**

- **Default definition** — first month at 180+ days past due (delinquency status ≥ 6) **or** a
  credit-event zero-balance code (third-party sale, short sale/charge-off, REO disposition, note
  sale). Prepaid/matured loans are not defaults. A 90-DPD variant is also computed for the APS 220
  broad-equivalence check.
- **Target** — `default_within_12m`: default within the first 12 months of the loan's life (a fixed
  window, so all 17 vintages are comparable; CRE36.63 / APS 113 Att D PD para 2).
- **Model** — logistic regression on origination features only (a through-the-door scorecard),
  features standardised before fitting.
- **Scorecard** — the logistic score is converted to a WOE/IV points scorecard and loans are sorted
  into 8 rating grades (A safest → H riskiest).
- **Calibration** — each grade is calibrated to its **count-weighted long-run** one-year default
  rate across the 17 vintages, then adjusted by a **risk-sensitive margin of conservatism**, a
  **revise-upward ratchet** (lifts any grade whose realised rate exceeds the prediction; APS 113
  Validation para 6), and the **5 bps regulatory floor**.

**Coefficients** (standardised; `exp(coef)` = odds multiplier per 1 SD)
→ [03_pd_coefficients.csv](outputs/tables/03_pd_coefficients.csv):

| Variable | Coefficient | Odds ×/1 SD | Direction |
|---|---:|---:|---|
| intercept | −6.30 | — | base rate ~0.4% |
| credit_score | −0.64 | 0.53 | higher score → lower default (strongest driver) |
| original_dti | +0.37 | 1.45 | higher debt-to-income → higher default |
| original_interest_rate | +0.33 | 1.39 | higher (risk-priced) rate → higher default |
| original_loan_term | +0.29 | 1.34 | longer term → higher default |
| original_ltv | +0.23 | 1.26 | higher loan-to-value → higher default |
| original_cltv | +0.14 | 1.15 | higher combined LTV → higher default |
| loan_purpose / occupancy / channel | ±0.09 or less | ~1.0 | small categorical adjustments |

All signs are economically consistent: credit score dominates and is protective; leverage
(LTV/CLTV), affordability (DTI) and price (rate) increase default odds.

**Rating grades — master scale** (predicted vs observed default rate per grade)
→ [03b_master_scale.csv](outputs/tables/03b_master_scale.csv):

| Grade | Loans | Predicted PD | Observed | Long-run PD (calibrated) |
|---|---:|---:|---:|---:|
| A | 98,247 | 0.03% | 0.04% | 0.05% |
| B | 114,153 | 0.06% | 0.07% | 0.07% |
| C | 105,933 | 0.11% | 0.09% | 0.09% |
| D | 106,462 | 0.20% | 0.20% | 0.20% |
| E | 105,591 | 0.33% | 0.39% | 0.37% |
| F | 106,384 | 0.47% | 0.41% | 0.40% |
| G | 103,959 | 0.69% | 0.71% | 0.65% |
| H | 109,271 | 1.22% | 1.17% | 1.09% |

Predicted and observed rates rise monotonically and align closely — the scorecard both ranks and
sizes risk. Grades G and H are flagged by the binomial calibration test and ratcheted up to their
realised rate. → [03d_pd_calibration_test.csv](outputs/tables/03d_pd_calibration_test.csv) ·
[03e_grade_pd_moc_floor.csv](outputs/tables/03e_grade_pd_moc_floor.csv)

**Validation.**

| Test | Result | Source |
|---|---|---|
| Discrimination (held-out) | AUC 0.83 · Gini 0.65 · KS 0.50 | [03_pd_metrics.csv](outputs/tables/03_pd_metrics.csv) |
| Calibration | predicted vs observed on the diagonal | [chart](outputs/charts/pd_calibration.png) |
| Calibration test | per-grade binomial traffic-light + portfolio Hosmer-Lemeshow | [03d_pd_calibration_test.csv](outputs/tables/03d_pd_calibration_test.csv) |
| Confusion matrix (cut-off = 0.39% prevalence) | recall 0.75 (743/986 defaults caught); low precision as expected for a ~0.4% base rate | [03_confusion_matrix.csv](outputs/tables/03_confusion_matrix.csv) |
| Out-of-time / out-of-regime | rank-ordering travels across regimes; level/stability is regime-sensitive (PSI) | [03c_oot_validation.csv](outputs/tables/03c_oot_validation.csv) |
| Forward cold holdout (train 2006–2019 → test 2020–2022) | test AUC 0.71; predicted PD 0.21% vs observed 0.36%; PSI 0.16 | same |

---

## 3. LGD model

Estimates, given default, the fraction of EAD that is lost.

**Method.**

- **Loss definition** — economic loss ÷ EAD, computed from settled-loss records (not assumed).
  Realised loss = `EAD + delinquent accrued interest − expenses − (net sales proceeds + MI
  recoveries + non-MI recoveries)`, capped to `[0, 1.10]`. This reconciles to Freddie Mac's own loss
  field at 0.99 correlation.
- **Model** — a two-stage ("hurdle") model on defaulted, disposed (fully resolved) loans, combining
  **two regressions**:
  - **Stage 1 — logistic regression**: probability that a material loss occurs, `P(loss)`.
  - **Stage 2 — linear regression**: loss severity given a loss occurs.
  - `LGD = P(loss) × severity`. (Logistic fits the binary Stage-1 event; the continuous 0–100%
    severity requires a linear regression, not a logistic.)
- **Variables** — original LTV, credit score, loan size (UPB), and a GFC-downturn indicator.
- **Variants** — an **economic (discounted)** LGD (`lgd_econ`, recovery discounted from disposition
  back to default) and an **APRA capital view** (`lgd_apra`: MI recoveries excluded, 20% high-LVR
  reduction, 20% floor), kept separate from the IFRS 9 figures.
- **Downturn LGD** — severity is cyclical (GFC ≈ 1.6× the rest), so the downturn LGD is used for
  capital/EL (APS 113 Att D LGD paras 4–5). An incomplete-workout sensitivity covers unresolved
  recent defaults.

**Coefficients and key drivers** → [04_lgd_coefficients.csv](outputs/tables/04_lgd_coefficients.csv):

| Stage (regression) | Variable | Coefficient | Reading |
|---|---|---:|---|
| 1 — P(loss), logistic | downturn flag | +1.61 | dominant driver — a GFC-era default is far more likely to crystallise a loss |
| | original_ltv | +0.009 | higher LTV → more likely a loss |
| | credit_score | +0.001 | small |
| 2 — severity, linear | downturn flag | +0.18 | a GFC-era loss is ~18 LGD-points more severe |
| | original_ltv | −0.0004 | small |
| | credit_score | −0.0003 | small |

The dominant driver of severity is the downturn/collateral environment, not the individual
borrower — the cyclical signature a downturn LGD is meant to capture.

**Predicted LGD by regime** → [04_lgd_model.csv](outputs/tables/04_lgd_model.csv):

| Regime | Disposed defaults | Observed LGD | Modelled LGD | Economic LGD | APRA-view LGD |
|---|---:|---:|---:|---:|---:|
| downturn (GFC) | 11,222 | 56.5% | 56.5% | 61.9% | 63.2% |
| calm / other | 2,244 | 34.2% | 33.6% | 38.5% | 43.1% |
| all | 13,466 | 52.8% | 52.7% | 58.0% | 59.9% |

By LTV band the model tracks realised severity with no systematic bias (gaps −0.05 to +0.08)
→ [04b_lgd_calibration_by_segment.csv](outputs/tables/04b_lgd_calibration_by_segment.csv).

**Validation.**

| Test | Result | Source |
|---|---|---|
| Reconciliation | computed loss vs Freddie Mac loss field: 0.99 correlation | notebook 01 |
| Out-of-time / out-of-regime | a model trained on calm-only under-predicts the crisis (21% vs 57%) | [04b_lgd_validation.csv](outputs/tables/04b_lgd_validation.csv) |
| Forward cold holdout (train pre-2020 → test 2020–22) | predicted 0.329 vs realised 0.321 (n=65) | same |
| Cohort backtest | predicted-decile means track realised | same |
| Discrimination | Spearman 0.33 (severity is inherently noisy → modest R²) | same |
| Stability | mean LGD moves modestly when any single vintage is dropped | same |
| Benchmarking | downturn ~56% within published US GFC severities (~40–60%); calm ~34% and the APRA 20% floor bracket it | [04b_lgd_benchmarking.csv](outputs/tables/04b_lgd_benchmarking.csv) |

---

## 4. EAD model

Estimates the amount owed at the moment of default.

**Method.**

- **Definition** — the outstanding balance at default (`current_actual_upb` at the first default
  month, plus any deferred balance; falling back to the balance removed at disposition where the
  credit-event row shows zero).
- **No credit-conversion factor (CCF)** — a term mortgage is fully drawn at origination and has no
  undrawn limit, so there is nothing to convert. This is supervisory EAD; own-EAD estimates are
  mandatory only for revolving retail (e.g. credit cards). EAD is floored at the current drawn
  balance, and post-default drawings are assigned to LGD, not EAD.

**Results** → [05_ead_summary.csv](outputs/tables/05_ead_summary.csv): mean EAD rises from ~$179k
(2006) to ~$295k (2022), tracking house-price growth rather than risk; portfolio exposure base
~$192bn. In Expected Loss, the exposure used per loan is the balance at default if the loan
defaulted, otherwise the original loan amount.

---

## 5. Expected Loss

`EL = PD × LGD × EAD`, computed per loan and summed. The PD used is the **calibrated capital grade
PD** (long-run average + margin of conservatism + ratchet + floor), so EL reconciles to the master
scale. Loans are sorted into IFRS 9 / AASB 9 stages — Stage 1 (performing, 12-month ECL), Stage 2
(significant increase in risk, lifetime ECL), Stage 3 (defaulted, lifetime ECL on a best-estimate
basis).

**By rating grade and portfolio total** → [06_el_summary_by_grade.csv](outputs/tables/06_el_summary_by_grade.csv):

| Grade | Loans | Total EAD | Avg PD | Avg LGD | 12-mo Expected Loss | EL rate (bps) |
|---|---:|---:|---:|---:|---:|---:|
| A | 98,247 | $21.4bn | 0.05% | 33.8% | $3.1m | 1.4 |
| B | 114,153 | $25.9bn | 0.08% | 33.2% | $5.8m | 2.2 |
| C | 105,933 | $25.0bn | 0.11% | 33.6% | $7.8m | 3.1 |
| D | 106,462 | $24.7bn | 0.22% | 34.3% | $15.8m | 6.4 |
| E | 105,591 | $24.6bn | 0.40% | 35.3% | $29.2m | 11.9 |
| F | 106,384 | $22.5bn | 0.43% | 36.6% | $30.2m | 13.4 |
| G | 103,959 | $23.3bn | 0.71% | 36.7% | $52.0m | 22.3 |
| H | 109,271 | $24.8bn | 1.17% | 36.7% | $92.1m | 37.1 |
| **PORTFOLIO** | **850,000** | **$192.3bn** | **0.40%** | **35.0%** | **$235.9m** | **12.3** |

The EL rate rises 1.4 → 37 bps from A to H, driven by PD; LGD is broadly flat across grades because
severity is collateral/cycle-driven, not borrower-grade-driven.

**By IFRS 9 stage** → [06_expected_loss.csv](outputs/tables/06_expected_loss.csv):

| Stage | Loans | Avg PD | Avg LGD | Total EAD | 12-month EL | Reported ECL (staged) |
|---|---:|---:|---:|---:|---:|---:|
| 1 — performing | 790,987 | low | 35% | $180.2bn | $206.8m | $206.8m (12-month) |
| 2 — significant ↑ in risk | 26,205 | — | 38% | $5.5bn | $11.3m | $45.2m (lifetime) |
| 3 — defaulted | 32,808 | — | 44% | $6.5bn | $17.8m | $71.2m (lifetime) |

Portfolio 12-month EL ≈ $236m, rising to ≈ $323m once IFRS 9 lifetime ECL applies to Stages 2 and 3.

---

## 6. Stress testing

Two methods are implemented. Both apply the no-diversification rule (PD and LGD shocks stack with
no offset; APG 113 para 92) and both target at least a mild-recession scenario (Basel CRE36.51).

- **Method A — observed multipliers** (notebook 07): multipliers read directly from the data
  (GFC-vs-calm and COVID-vs-calm ratios) and applied to PD and LGD.
- **Method B — statistical macro-credit "satellite" model** ([stress_test/](stress_test/)): a
  logistic regression of the portfolio's quarterly default rate on macro variables, driven by
  scenario macro paths to produce a stressed PD for each rating grade.

### 6.1 Stressed PD

| Method | Severe scenario | Mild / COVID |
|---|---|---|
| Observed multipliers (notebook 07) | PD ×5.7 (portfolio 0.40% → 2.2%) | mild ×2.6 · COVID ×4.3 |
| Satellite model (stress_test/) | PD ×24.7 (simultaneous-shock upper bound, triangulated to the observed ceiling ×7.8) | mild ×5.4 |

### 6.2 Stressed LGD

Severity is collateral-driven, so it is stressed through the house-price shock, anchored to the
measured regimes in §3:

| Scenario | Stressed LGD | Multiplier | Driver |
|---|---:|---:|---|
| baseline (calm) | ~34% | ×1.0 | calm/other regime |
| severe (house prices −25%) | ~56% | ×2.3 | GFC downturn LGD (measured) |
| COVID-2020 | ~27% | ×1.1 | high default but mild severity (house prices rose) |

A downturn raises both PD and LGD, except COVID, where LGD barely moved — the key asymmetry between
the two observed downturns.

### 6.3 Stressed Expected Loss (method A)

→ [07_stress_test.csv](outputs/tables/07_stress_test.csv): on the calm baseline book, a severe GFC
scenario lifts EL ~13× (PD ×5.7, LGD ×2.3); the observed COVID-2020 scenario lifts it ~4.7×
(PD ×4.3, LGD ×1.1).

### 6.4 Statistical satellite model (method B)

**Where the macro data comes from.** The macro drivers are an external overlay of public US series
(they do not come from the loan data, which supplies the default rate). They are held in
[stress_test/macro/macro_annual.csv](stress_test/macro/macro_annual.csv):

| Driver | Source (FRED series) |
|---|---|
| unemployment rate (%) | `UNRATE` |
| house-price growth YoY (%) | `CSUSHPINSA` (Case-Shiller) |
| real GDP growth (%) | `A191RL1Q225SBEA` |
| 30-yr mortgage rate (%) | `MORTGAGE30US` |

Committed values are approximate public figures so the pipeline runs offline;
[fetch_macro_fred.py](stress_test/fetch_macro_fred.py) refreshes them from FRED.

**How the drivers enter the model — a calendar-time join.**

```text
macro_annual.csv ──interpolate to quarterly──┐
                                             merge on calendar quarter ──► logit(default_rate)
loan_level.parquet ──quarterly default rate──┘                            ~ β·[unemp, ΔHPI, GDP, age]
```

1. Read the annual macro CSV and interpolate to quarterly.
2. Build a quarterly point-in-time default rate from the loan panel (79 quarters, 2006Q1–2025Q3).
3. Join the two by calendar quarter (so bad macro aligns with high observed default).
4. Standardise the drivers and fit a logistic regression of the default rate on them — these are the
   satellite coefficients.
5. For a scenario, feed its (clipped-to-support) macro values back through the fitted model →
   stressed default rate.

**Satellite coefficients** (standardised; `mortgage_rate` excluded for a perverse sign):
unemployment +0.97 (dominant), house-price growth −0.18, GDP growth −0.12, seasoning control +0.50.
All signs are economically correct. → [satellite_coefficients.csv](stress_test/outputs/tables/satellite_coefficients.csv)

**Stressed PD by rating grade + grade migration.** The model's systematic macro **log-odds shift**
(mild +1.70, severe +3.24) is added to each grade's base log-odds:
`logit(stressed PD_grade) = logit(base PD_grade) + macro_shift`. Each stressed PD is mapped back to
the master scale to show migration → [scenario_stressed_pd_by_grade.csv](stress_test/outputs/tables/scenario_stressed_pd_by_grade.csv):

| Grade | Base PD | Mild → PD (migrates to) | Severe → PD (migrates to) |
|---|---:|---:|---:|
| A | 0.05% | 0.27% (→ D) | 1.26% (→ H) |
| C | 0.11% | 0.60% (→ G) | 2.74% (→ H+) |
| E | 0.40% | 2.14% (→ H+) | 9.31% (→ H+) |
| H | 1.17% | 6.06% (→ H+) | 23.2% (→ H+) |

The shift is constant in log-odds (a roughly uniform ~5.4× mild / ~25× severe multiplier), but the
absolute PD jump is larger for riskier grades. The severe satellite result is a **simultaneous-shock
upper bound** and is triangulated against the observed ceiling (×7.8) and method A (×5.7); see
[stress_test/README.md](stress_test/README.md) for the full model-risk controls.

---

## Charts

Regenerated from committed result tables by [tools/make_figures.py](tools/make_figures.py)
(aggregated results only, no raw loan records).

| Chart | Content |
|---|---|
| [expected_loss_base_vs_stress.png](outputs/charts/expected_loss_base_vs_stress.png) | Portfolio 12-month EL: calm baseline (~$11m), mild recession (~$38m), severe GFC-calibrated (~$138m) |
| [default_rate_by_vintage.png](outputs/charts/default_rate_by_vintage.png) | One-year default rate by origination year 2006–2022, coloured by regime (GFC / COVID / calm) |
| [lgd_calm_vs_downturn.png](outputs/charts/lgd_calm_vs_downturn.png) | Realised vs modelled LGD: GFC downturn (~56%) vs non-GFC (~34%) |
| [pd_calibration_by_grade.png](outputs/charts/pd_calibration_by_grade.png) | Predicted vs observed one-year default rate per rating grade A–H |
| [default_rate_by_credit_score.png](outputs/charts/default_rate_by_credit_score.png) | One-year default rate by credit-score band |
| [stress_test/.../stressed_pd_by_grade.png](stress_test/outputs/charts/stressed_pd_by_grade.png) | Base vs stressed PD by grade (satellite model) |

---

## Notebooks

| # | Notebook | Output |
|---|---|---|
| 00 | [Load & assemble](notebooks/00_load_and_assemble.ipynb) | [00_data_quality.csv](outputs/tables/00_data_quality.csv) |
| 01 | [Base table](notebooks/01_base_table.ipynb) — default, EAD, realised LGD (0.99 reconcile) | [01_default_lgd_by_vintage.csv](outputs/tables/01_default_lgd_by_vintage.csv) |
| 02 | [EDA](notebooks/02_eda.ipynb) | [02_risk_by_driver.csv](outputs/tables/02_risk_by_driver.csv) |
| 03 | [PD model](notebooks/03_pd_model.ipynb) — logistic regression, coefficients, confusion matrix | [03_pd_metrics.csv](outputs/tables/03_pd_metrics.csv) · [03_pd_coefficients.csv](outputs/tables/03_pd_coefficients.csv) |
| 03b | [PD scorecard](notebooks/03b_PD_Scorecard.ipynb) — WOE/IV, grades, master scale, calibration, MoC, floor | [03b_master_scale.csv](outputs/tables/03b_master_scale.csv) |
| 03c | [PD out-of-time validation](notebooks/03c_PD_OutOfTime_Validation.ipynb) — incl. forward cold holdout | [03c_oot_validation.csv](outputs/tables/03c_oot_validation.csv) |
| 04 | [LGD model](notebooks/04_lgd_model.ipynb) — two-stage (logistic P(loss) × linear severity), variants, downturn | [04_lgd_model.csv](outputs/tables/04_lgd_model.csv) · [04_lgd_coefficients.csv](outputs/tables/04_lgd_coefficients.csv) |
| 04b | [LGD validation](notebooks/04b_LGD_Validation.ipynb) — OOT, forward holdout, LTV calibration, benchmarking | [04b_lgd_validation.csv](outputs/tables/04b_lgd_validation.csv) |
| 05 | [EAD](notebooks/05_ead.ipynb) | [05_ead_summary.csv](outputs/tables/05_ead_summary.csv) |
| 06 | [Expected Loss](notebooks/06_expected_loss.ipynb) — EL = PD×LGD×EAD, by grade/stage, IFRS 9 | [06_expected_loss.csv](outputs/tables/06_expected_loss.csv) · [06_el_summary_by_grade.csv](outputs/tables/06_el_summary_by_grade.csv) |
| 07 | [Stress testing](notebooks/07_stress_testing.ipynb) — observed multipliers (GFC + COVID) | [07_stress_test.csv](outputs/tables/07_stress_test.csv) |
| 08 | [Documentation & monitoring](notebooks/08_documentation_and_monitoring.ipynb) — model pack + PSI | [08_monitoring_psi.csv](outputs/tables/08_monitoring_psi.csv) |
| — | [stress_test/](stress_test/) — statistical macro-credit satellite model (method B) | [stress_test/outputs/tables/](stress_test/outputs/tables/) |

---

## Regulatory alignment (APS 113 / APG 113 / Basel / WP14 / IFRS 9)

The per-component sections state where each rule is applied. In summary: a one-year PD calibrated to
a count-weighted long-run average with a margin of conservatism and the 5 bps floor; economic and
downturn LGD with the APRA floor/MI rules kept separate from IFRS 9; supervisory EAD (no CCF) for a
term product; EL on the same PD as capital, staged under IFRS 9; and a credit-risk stress test
(observed multipliers plus a satellite model) covering at least a mild recession with the
no-diversification assumption. The 17-vintage 2006–2022 window meets the 5-year PD/retail-LGD
minimum. Items documented as out of scope (notebook 08): rating philosophy, override policy, use
test, development/validation independence, cure/probation rules, the 180-vs-90-DPD equivalence, a
full lifetime-PD term structure (a horizon-factor proxy is used), and the recent-vintage
incomplete-workout caveat.

## Limitations

- Demonstration model, not a certified regulatory-capital calculation; the APRA-view overlays are
  illustrative applications of the rules.
- US agency mortgages, not an Australian / APRA IRB portfolio.
- Sample data (50k loans/vintage, ~850k total); illustrative calibration.
- The satellite stress test extrapolates in the joint tail (severe scenario), so its severe figure is
  an upper bound and is triangulated.
- Recent-vintage (2021–2022) LGD workouts are not yet fully resolved (incomplete-workout caveat).

## How to run

```bash
pip install -r requirements.txt
# Place the SFLLD samples under "data/raw data/" — either as sample_orig_YYYY.txt +
# sample_svcg_YYYY.txt directly, or in per-vintage sample_YYYY/ subfolders (both work),
# for origination years 2006–2022.
python build_notebooks.py     # (re)generate the notebooks
python run_notebooks.py all   # execute 00–08; writes tables to outputs/tables/
python tools/make_figures.py  # regenerate the charts
cd stress_test && python build_stress.py   # statistical satellite stress test (method B)
```

Notebook 00 assembles all 17 vintages once (~10 min) and caches a loan-level table; the rest run in
seconds off the cache.

## Repository layout

```
.
├── data/raw data/        # SFLLD files (2006–2022) — gitignored, never committed
├── data/processed/       # cached loan-level tables — gitignored, regenerated by nb 00–01
├── notebooks/            # ordered build 00–08 (+ 03b, 03c, 04b)
├── outputs/tables/       # committed CSV results   ·   outputs/charts/ — committed PNGs
├── src/                  # layout, loaders, definitions, models, metrics, woe, transform, scorecard
├── stress_test/          # statistical macro-credit satellite stress test (self-contained)
├── build_notebooks.py    # regenerates the notebooks
├── run_notebooks.py      # executes the notebooks end-to-end
└── tools/make_figures.py # regenerates the charts
```

## License

MIT License.
