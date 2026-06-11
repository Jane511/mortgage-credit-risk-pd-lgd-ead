# PROJECT BUILD — Mortgage Credit Risk (PD + LGD + EAD + Expected Loss + Stress Testing)

**For:** Claude Code
**Dataset:** Freddie Mac Single-Family Loan-Level Dataset (SFLLD), sample files
**Goal:** Build a clean, interpretable mortgage credit-risk portfolio project that a
technical interviewer *and* a non-technical HR reviewer can both follow in minutes.

This is a **separate project** from my existing Home Credit (consumer credit) repo. Do not
try to merge them. Its whole reason for existing is the one thing the consumer project
could not do: a **real, modelled LGD** from actual loss data, plus a genuine downturn.

---

## 0. Golden rules (apply to every notebook)

- **Keep it simple.** Prefer clear over clever. Interpretable methods (logistic / linear /
  beta regression, simple averages) over machine learning. No heavy feature engineering.
- **Plain English.** At the top of each notebook, write a 4–5 line summary a non-technical
  HR person could read: what the notebook does and the headline result. Explain any
  technical term in one short sentence.
- **One comment per code cell** saying in plain words what that cell does.
- **One clean results table per notebook**, saved to `output/` as a CSV snapshot.
- **If anything is ambiguous, ask me one question** rather than guessing — especially
  around the column layout (see Section 3).

---

## 1. What this project builds

A staged, end-to-end IFRS 9 / AASB 9-style Expected Credit Loss workflow on mortgages:

1. **PD** — probability a loan defaults
2. **LGD** — loss given default, modelled from **real** loss data (the key feature)
3. **EAD** — exposure at default
4. **Expected Loss** — PD × LGD × EAD
5. **Stress testing** — link PD/LGD to the economy and apply a downturn scenario

---

## 2. Data — what to expect

- Three sample files: **2007, 2008** (downturn) and **2015** (calm). Each is a random
  50,000-loan sample for that origination year.
- Each zip contains **two pipe-delimited (`|`) text files**:
  - **Origination file** — one row per loan, loan characteristics at origination.
  - **Performance (servicing) file** — one row per loan per month, including the
    actual-loss fields.
- Join the two on the **Loan Sequence Number** (the loan ID).

### 2a. COMPLIANCE — do not commit the raw data
Freddie Mac restricts redistribution of this data. Therefore:
- Add the data folder to `.gitignore`. **Never commit the raw `.txt`/`.zip` files.**
- Commit only code, notebooks, and small aggregated output snapshots.
- The README must state the data is downloaded separately and not redistributed here.

---

## 3. CRITICAL — the files have NO column headers

The pipe-delimited files do **not** include a header row. You must apply the official
column layout from the SFLLD **General User Guide** ("File Layout and Data Dictionary"
section), which is included with the download / on the SFLLD Resource Page.

**Before assigning names: count the columns in the actual file and confirm the count
matches the layout you apply.** Freddie Mac has added fields across releases, so the list
below is a reliable *reference* but may not be the newest exactly. If the column count
does not match, STOP and tell me — do not guess, because a wrong mapping silently
corrupts every downstream number.

### Origination file — reference column order
```
credit_score, first_payment_date, first_time_homebuyer_flag, maturity_date, msa,
mi_pct, number_of_units, occupancy_status, original_cltv, original_dti, original_upb,
original_ltv, original_interest_rate, channel, ppm_flag, amortization_type,
property_state, property_type, postal_code, loan_sequence_number, loan_purpose,
original_loan_term, number_of_borrowers, seller_name, servicer_name,
super_conforming_flag, pre_harp_loan_sequence_number, program_indicator, harp_indicator,
property_valuation_method, interest_only_indicator, mi_cancellation_indicator
```

### Performance (servicing) file — reference column order
```
loan_sequence_number, monthly_reporting_period, current_actual_upb,
current_loan_delinquency_status, loan_age, remaining_months_to_legal_maturity,
defect_settlement_date, modification_flag, zero_balance_code,
zero_balance_effective_date, current_interest_rate, current_deferred_upb, ddlpi,
mi_recoveries, net_sales_proceeds, non_mi_recoveries, expenses, legal_costs,
maintenance_preservation_costs, taxes_and_insurance, miscellaneous_expenses,
actual_loss_calculation, modification_cost, step_modification_flag,
deferred_payment_plan, eltv, zero_balance_removal_upb, delinquent_accrued_interest,
delinquency_due_to_disaster, borrower_assistance_status_code,
current_month_modification_cost, interest_bearing_upb
```

### Known data gotchas to handle
- **Delinquency status** is a text code: `0` = current, `1`,`2`,`3`... = months past due,
  and special values like `R`/`RA` (REO) and `XX` (unknown). Convert carefully; treat
  non-numeric statuses explicitly, don't let them coerce to 0.
- **Net sales proceeds** can contain text codes (e.g. `C` = "covered", `U` = "unknown")
  instead of a number. Clean these before any loss arithmetic.
- Loss / recovery fields are only populated around the disposition month, and are blank
  for performing loans. Expect mostly-empty columns — that is normal.

---

## 4. Key definitions — get these exactly right

These three definitions drive every result, so implement them precisely and state each
one in plain English in the relevant notebook.

### Default (for PD)
A loan is **defaulted** the first time **either** of these happens in its performance
history:
- **Current Loan Delinquency Status reaches 180+ days past due** (status `6` or higher), **or**
- a **credit-event Zero Balance Code** appears — i.e. the loan ended in a loss event:
  `02` (third-party sale), `03` (short sale / charge-off), `09` (REO disposition),
  `15` (note sale). Codes like `01` (prepaid/matured) are **not** defaults.

Record the **default date** (first such month) for each defaulted loan.

### EAD (Exposure at Default)
- **EAD = Current Actual UPB at the default month** (add Current Deferred UPB if present).
- This is a term loan with no undrawn limit, so **there is no CCF / drawdown modelling**.
  State this explicitly in one sentence — knowing that CCF applies only to *revolving*
  facilities (like my consumer credit-card project) is a deliberate point, not a gap.

### Realised LGD (for LGD)
For defaulted loans that reached disposition:
```
Loss = EAD + delinquent_accrued_interest - expenses
       - (net_sales_proceeds + mi_recoveries + non_mi_recoveries)

LGD  = Loss / EAD
```
> **Data-verified correction (do NOT use the naive sum of all expense fields).**
> In the SFLLD performance file the aggregate `expenses` column is *exactly* the
> sum of its four sub-components (`legal_costs + maintenance_preservation_costs +
> taxes_and_insurance + miscellaneous_expenses` — confirmed at correlation 1.000,
> zero residual), and **every expense field is stored as a NEGATIVE number**.
> So the loss must (a) use the aggregate `expenses` **once** — adding the
> aggregate *and* its components double-counts costs — and (b) **subtract** it
> (minus a negative adds the cost). With this form, computed loss reconciles to
> the dataset's own `actual_loss_calculation` at **corr ≈ 0.99**; the naive
> all-fields-added version understates LGD badly (≈33% vs the correct ≈58% in the
> downturn). The four sub-component fields are still loaded for EDA, just not for
> the loss math.
- If the dataset's own `actual_loss_calculation` field is populated, **reconcile your
  computed loss against it** and note any difference — that reconciliation is a strong
  validation talking point.
- Winsorise/cap LGD to a sensible range (e.g. floor at 0; cap around 1, or allow a small
  band above 1 and document why). State the rule.
- Build the LGD model **only on defaulted, disposed loans** (that is correct) and say so.

---

## 5. Repo structure (mirror my Home Credit repo's conventions)

```
.
├── data/                 # raw SFLLD files — GITIGNORED, never committed
├── notebooks/            # ordered build, 00–08
├── output/               # CSV snapshots + recruiter-friendly tables
├── src/                  # reusable helpers (loaders, layout, metrics)
├── README.md             # single main entry point
├── requirements.txt
└── .gitignore
```

---

## 6. Build steps — notebook by notebook

Each notebook = HR summary at top, plain comments, one saved results table.

### Notebook 00 — Load & assemble
- Read the pipe-delimited files, apply the column layout (Section 3), confirm column counts.
- Join origination + performance on `loan_sequence_number`.
- Stack the 2007, 2008, 2015 vintages into one table with a `vintage_year` column.
- Output: a small data-quality summary (row counts, default rate by vintage).

### Notebook 01 — Build the modelling base table
- Apply the **default definition** (Section 4); flag defaults and default dates.
- Compute **EAD** at default.
- Compute **realised LGD** for disposed defaults.
- Create the loan-level analysis table: one row per loan with origination features +
  default flag + EAD + LGD (LGD null for non-defaults).
- Output: default rate and average LGD by vintage (downturn vs calm — expect 2007/2008
  much worse than 2015; that contrast is a headline result).

### Notebook 02 — EDA
- A few clear charts: default rate by credit score band, by LTV band, by vintage.
- Output: a one-page summary table of risk by key driver.

### Notebook 03 — PD model
- Logistic regression on origination features (credit score, LTV, CLTV, DTI, loan purpose,
  occupancy, term). Keep it interpretable.
- Report **AUC, Gini, KS, and a calibration plot**.
- Output: PD model metrics table + predicted PD distribution.

### Notebook 04 — LGD model (the key feature)
- Model realised LGD on disposed defaults: a beta regression or a simple two-stage
  (probability of any loss × severity-if-loss) approach. Keep it interpretable.
- Report a **downturn LGD** using the 2007/2008 defaults specifically, and compare to 2015.
- Output: LGD model summary + downturn-vs-benign LGD table.

### Notebook 05 — EAD
- EAD = UPB at default. Show the distribution and average EAD by vintage.
- One short markdown cell: why there is no CCF for amortising term loans (vs revolving).
- Output: EAD summary table.

### Notebook 06 — Expected Loss
- EL = PD × LGD × EAD, combining the three models.
- Add simple IFRS 9 / AASB 9 staging (Stage 1 / 2 / 3) and a 12-month vs lifetime view.
- One worked example showing the full PD × LGD × EAD = EL math for a single loan.
- Output: portfolio EL summary table.

### Notebook 07 — Stress testing
- Link PD (and optionally LGD) to the economy. Use a house-price index and unemployment
  as drivers — pull these from a public macro source (FRED) keyed to the performance dates.
- Apply a **downturn scenario** (replicate the 2008–09 path, or borrow a published Fed
  CCAR severely-adverse path) and show the uplift in PD, LGD, and total EL.
- The 2007/2008 vintage gives you a *real* observed downturn to anchor against — use it.
- Output: baseline-vs-stressed EL table.
- Optional one paragraph: how you'd extend this to a **climate** scenario (e.g. a flood
  shock to house prices feeding LGD). The target role flags climate risk as an interest
  area — a short note is a cheap, high-signal differentiator. Don't build it, just sketch it.

### Notebook 08 — Documentation, validation & monitoring pack
- Short model-development write-up: objective, data, methodology, results, limitations,
  monitoring (PSI / stability), and how each model would be governed.
- This mirrors what a consulting model-validation team produces and is a scored deliverable.

---

## 7. README requirements

- **Title:** "Mortgage Credit Risk — PD + LGD + EAD + Expected Loss + Stress Testing".
- Plain-English intro: what the project shows, in language HR understands.
- **Data source section:**
  > Uses the Freddie Mac **Single-Family Loan-Level Dataset (SFLLD)**, a public US
  > mortgage dataset (sample files: 2007, 2008, 2015 vintages). Downloaded separately via
  > Freddie Mac Clarity Data Intelligence; raw data is **not** redistributed in this repo.
  > Source: https://freddiemac.com/research/datasets/sf-loanlevel-dataset
- **Notebook map** (00–08, one line each).
- **Limitations section:**
  - Portfolio demonstration, not a production or regulatory-capital model.
  - US agency mortgages — not an Australian or APRA IRB portfolio.
  - Sample data (50k loans/vintage), illustrative calibration.
- Suggested repo name in a top comment: `mortgage-credit-risk-pd-lgd-ead`.

---

## 8. One-line portfolio message (for the README)

> Built a mortgage credit-risk project on Freddie Mac loan-level data covering PD, a
> **real LGD modelled from actual loss data**, EAD, Expected Loss, and a downturn stress
> test — using interpretable models and a 2007/2008 GFC vintage for genuine downturn
> severity.
