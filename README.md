# Mortgage Credit Risk — PD + LGD + EAD + Expected Loss + Stress Testing

> **In one line:** Built a complete mortgage credit-risk model on 150,000 real US
> home loans — estimating how likely each loan is to default, how much money is
> actually lost when it does, and how much worse losses get in a recession — using
> simple, explainable methods and the 2007/2008 financial crisis as a real downturn.

---

## For a non-technical reader: what is this, and why does it matter?

When a bank lends money for a home, it must set aside a sensible amount for loans
that will go bad. Getting that number right keeps the bank safe **and** keeps
loans affordable. The number is built from three questions:

| Question | Industry term | Plain meaning |
|---|---|---|
| Will the borrower stop paying? | **PD** (Probability of Default) | the *chance* of default |
| If they do, how much is lost? | **LGD** (Loss Given Default) | the *severity* of the loss after the house is sold |
| How much was still owed? | **EAD** (Exposure at Default) | the *amount* at risk |

Multiply them and you get **Expected Loss = PD × LGD × EAD** — the money a lender
should expect to lose. This project estimates all three on real mortgages,
combines them, sorts the loans into the accounting buckets banks must report
(IFRS 9 / AASB 9), and then stress-tests the book against a recession.

**The standout piece** is the LGD: instead of *assuming* a loss rate (as many
projects do), it is **measured from Freddie Mac's real, settled loss records** and
independently validated.

---

## What I achieved (headline results)

**The crisis years are dramatically worse than the calm year — on both how often
loans default and how much is lost each time.**

| Origination year | Loans | Default rate | Avg loss-given-default | Avg exposure |
|---|---|---|---|---|
| **2007** (crisis) | 50,000 | **13.7%** | **58%** | ~$186k |
| **2008** (crisis) | 50,000 | **7.4%** | **54%** | ~$198k |
| **2015** (calm)   | 50,000 | **2.4%** | **25%** | ~$198k |

- **PD model quality:** AUC **0.81**, Gini **0.63**, KS **0.50** on held-out data
  (0.5 AUC = random guessing, so 0.81 is strong, well-separated discrimination).
- **LGD validated:** my computed losses match Freddie Mac's own loss field at
  **0.99 correlation** — the model is grounded in reality, not assumption.
- **Portfolio Expected Loss:** ~**$1.11bn** lifetime across 150,000 loans, of which
  ~**$510m** is the IFRS 9 "reported" figure once accounting staging is applied.
- **Stress test:** under a crisis-calibrated downturn, portfolio Expected Loss
  rises **~10×** — because PD and LGD get worse *at the same time*.

---

## Key charts

*All charts are regenerated from the committed pipeline outputs in [output/](output/)
by [reports/make_figures.py](reports/make_figures.py) — aggregated results only, no
raw loan records.*

### 1. Expected loss: baseline vs stressed (the headline)
![Expected loss rises about ten times under a downturn scenario](reports/figures/expected_loss_base_vs_stress.png)

**What this shows:** the whole portfolio's expected loss in a calm market (~$67m) versus a crisis-like downturn (~$668m).
**Why it matters:** it is the stress story in one picture — losses jump ~10× because the chance of default and the loss per default get worse *together*.

### 2. Default rate by origination year
![Default rate by vintage: 13.7% in 2007 and 7.4% in 2008 versus 2.4% in 2015](reports/figures/default_rate_by_vintage.png)

**What this shows:** how often loans defaulted, split by the year they were written.
**Why it matters:** the 2007/08 crisis cohorts default far more often than the calm 2015 book — a real observed downturn, not an assumption.

### 3. Loss given default: calm vs downturn
![LGD more than doubles in a downturn, from 25% to 57%](reports/figures/lgd_calm_vs_downturn.png)

**What this shows:** the share of the loan actually lost after a default, measured from Freddie Mac's real loss records, and the model's match to it.
**Why it matters:** loss *severity* more than doubles in a downturn — and the modelled line tracks the observed one, so the LGD is grounded in reality.

### 4. PD calibration by rating grade
![Predicted versus observed default rate across rating grades A to H, closely aligned](reports/figures/pd_calibration_by_grade.png)

**What this shows:** for each risk grade (A safest → H riskiest), the predicted default probability against what actually happened.
**Why it matters:** the two lines sit almost on top of each other and rise in order — the scorecard both *ranks* and *sizes* risk correctly.

### 5. Default rate by credit score
![Default rate falls from 29% below 620 to 2% above 780 as credit score rises](reports/figures/default_rate_by_credit_score.png)

**What this shows:** default rate across credit-score bands.
**Why it matters:** risk falls steeply and smoothly as scores rise (29% → 2%) — textbook behaviour that confirms the data and definitions are right.

*Full methodology and code: see the notebooks in [notebooks/](notebooks/).*

---

## What I did (the process, step by step)

Each step is one notebook, written so a non-technical reader can follow the top
and a technical reviewer can check the code. Every notebook saves one clean
results table to [output/](output/).

| # | Step | What I did | Result it produced |
|---|------|-----------|--------------------|
| **00** | [Load & assemble](notebooks/00_load_and_assemble.ipynb) | Took the raw Freddie Mac files (no column headers, ~8.8 million monthly rows), applied the official 32-column layout, verified the counts, joined each loan's facts to its monthly history, and collapsed it to one row per loan across all three years. | [Data-quality summary](output/00_data_quality.csv) |
| **01** | [Base table](notebooks/01_base_table.ipynb) | Defined and flagged **default**, computed **EAD**, and computed **realised LGD from the real loss fields**; reconciled it to the dataset's own loss figure. | [Default & LGD by vintage](output/01_default_lgd_by_vintage.csv) |
| **02** | [EDA](notebooks/02_eda.ipynb) | Showed how risk moves with the two classic drivers — credit score and loan-to-value — with saved charts. | [Risk by driver](output/02_risk_by_driver.csv) + charts |
| **03** | [PD model](notebooks/03_pd_model.ipynb) | Built an interpretable **logistic-regression** default model and graded it (AUC/Gini/KS + a calibration check). | [PD metrics](output/03_pd_metrics.csv) |
| **03b** | [PD scorecard](notebooks/03b_PD_Scorecard.ipynb) | Turned the PD into a **points-based scorecard** (WOE/IV binning + logistic, reusing my consumer-project helpers), sorted loans into **8 rating grades A–H**, and produced a **master scale** with predicted vs observed PD per grade and a downturn view. | [Master scale](output/03b_master_scale.csv) |
| **03c** | [PD out-of-time validation](notebooks/03c_PD_OutOfTime_Validation.ipynb) | **Out-of-time / out-of-regime** test: refit PD on earlier vintages only and scored a held-out later one (no leakage). Shows rank-ordering **travels** across periods while the risk **level/stability shifts** (PSI), and a calm-trained model **under-predicts** a downturn. | [OOT validation](output/03c_oot_validation.csv) |
| **04** | [LGD model](notebooks/04_lgd_model.ipynb) | Built a **two-stage LGD model** (chance of a loss × size of the loss) on disposed defaults and produced a real **downturn LGD**. | [LGD model summary](output/04_lgd_model.csv) |
| **05** | [EAD](notebooks/05_ead.ipynb) | Summarised exposure at default and explained, in one paragraph, why a term mortgage needs **no CCF** (unlike a credit card). | [EAD summary](output/05_ead_summary.csv) |
| **06** | [Expected Loss](notebooks/06_expected_loss.ipynb) | Combined the three models into **EL = PD × LGD × EAD**, added **IFRS 9 / AASB 9 staging**, and walked through the full math for one example loan. | [Expected-loss summary](output/06_expected_loss.csv) |
| **07** | [Stress testing](notebooks/07_stress_testing.ipynb) | Used the *observed* 2007/2008 crisis to calibrate a **downturn scenario** and measured the uplift in PD, LGD and total loss; sketched a climate-risk extension. | [Baseline-vs-stressed](output/07_stress_test.csv) |
| **08** | [Documentation & monitoring](notebooks/08_documentation_and_monitoring.ipynb) | Wrote the model-development pack (objective, data, method, results, limitations, governance) and a **PSI stability** check. | [Monitoring (PSI)](output/08_monitoring_psi.csv) |

**Risk rises exactly as expected** — for example, default rate by credit score
falls from **29% (sub-620)** to **2% (780+)**, and rises with loan-to-value from
**3% (<60%)** to **13% (90%+)** — textbook behaviour that gives confidence the
data and definitions are right.

---

## Analytical rigour worth highlighting

While validating the LGD, I found that the naive loss formula **double-counts
costs**: in this dataset the `expenses` field is *exactly* the sum of its four
sub-components (legal, maintenance, taxes, miscellaneous), and all of them are
stored as **negative** numbers. Adding them all together understated the loss and
produced an implausible downturn LGD of ~33%. Using the aggregate **once**, with
the correct sign, fixed it — the corrected loss reconciles to Freddie Mac's own
loss field at **0.99 correlation** and yields a realistic ~58% downturn LGD. This
kind of cross-check against ground truth is exactly what model-validation work
requires, and it is documented inside notebook 01.

---

## Skills this project demonstrates

- **Credit-risk fundamentals:** PD, LGD, EAD, Expected Loss, IFRS 9 / AASB 9
  staging, and stress testing — the full Expected Credit Loss chain.
- **Real loss modelling:** LGD built and validated from actual settled losses,
  not an assumed rate.
- **Data engineering at scale:** assembling ~8.8 million rows across two files and
  three vintages into a clean loan-level table.
- **Interpretable modelling:** logistic regression, a WOE/IV **points scorecard**
  with rating grades, and a transparent two-stage LGD — methods a reviewer and a
  regulator can follow.
- **Model validation & governance:** reconciliation to ground truth, AUC/Gini/KS,
  calibration, PSI stability monitoring, and a written model-development pack.
- **Clear communication:** every notebook explains itself in plain English first.

---

## Key definitions (stated precisely)

- **Default:** the first month a loan is **180+ days past due** (delinquency
  status ≥ 6) **or** ends in a credit-event zero-balance code (third-party sale,
  short sale/charge-off, REO disposition, note sale). A prepaid/matured loan is
  **not** a default.
- **EAD:** the outstanding balance at default. No CCF — a term mortgage is fully
  drawn at closing and has no undrawn limit (a credit card does, and would need one).
- **Realised LGD:** `Loss / EAD`, where
  `Loss = EAD + delinquent accrued interest − expenses − (net sales proceeds +
  MI recoveries + non-MI recoveries)`, computed from the real loss fields and
  capped to `[0, 1.10]`.

---

## Data source

Uses the Freddie Mac **Single-Family Loan-Level Dataset (SFLLD)**, a public US
mortgage dataset (sample files: 2007, 2008, 2015 vintages — 50,000 loans each).
Downloaded separately via Freddie Mac Clarity Data Intelligence; **raw data is
not redistributed in this repo**.
Source: https://freddiemac.com/research/datasets/sf-loanlevel-dataset

The pipe-delimited files have **no column headers**; the official 32-column SFLLD
layout is applied in [src/layout.py](src/layout.py) and the column count is
asserted before any names are attached, so a wrong mapping can never silently
corrupt the numbers.

## Repo structure

```
.
├── raw data/             # SFLLD files (2007/2008/2015) — GITIGNORED, never committed
├── data/processed/       # cached loan-level tables — GITIGNORED, regenerated by nb 00-01
├── notebooks/            # ordered build 00–08 (+ 03b scorecard, 03c validation)
├── output/               # CSV result snapshots + charts (committed)
├── src/                  # helpers: layout, loaders, definitions, models, metrics,
│                         #          woe, transform, scorecard (WOE/IV points scorecard)
├── build_notebooks.py    # regenerates the notebooks
├── run_notebooks.py      # executes the notebooks end-to-end
├── requirements.txt
└── .gitignore
```

## How to run

```bash
pip install -r requirements.txt
# Place the SFLLD samples under "raw data/sample_2007/", ".../sample_2008/",
# ".../sample_2015/" (each with sample_orig_YYYY.txt and sample_svcg_YYYY.txt).
python build_notebooks.py     # (re)generate the notebooks
python run_notebooks.py all   # execute 00–08; writes tables to output/
```
Notebook 00 does the heavy assembly once (~2 min) and caches a loan-level table;
the rest run in seconds.

## Limitations

- Portfolio demonstration, **not** a production or regulatory-capital model.
- US agency mortgages — **not** an Australian or APRA IRB portfolio.
- Sample data (50k loans/vintage); illustrative calibration.
- The macro stress is scenario-based (crisis-calibrated multipliers), not a
  fitted macroeconomic model.

## Relationship to my consumer-credit project

This is a **separate** project from my Home Credit (consumer credit) repo. Its
reason for existing is the one thing the consumer project could not do: a **real,
modelled LGD from actual loss data**, plus a genuine observed downturn (the
2007/2008 GFC vintages). Where the consumer card project uses a CCF for undrawn
limits, this term-mortgage project deliberately does not — a point, not a gap.
