# Claude Code task (round 2): finish aligning the mortgage PD with the framework

**Repo:** `mortgage-credit-risk-pd-lgd-ead`
**Context:** PD-1…PD-10 are done and correct (12-month PD, long-run calibration, calibration test,
MoC, floor, documentation). This round closes the gaps a second-pass review found — chiefly that
the calibrated/conservative PD does **not** currently reach Expected Loss, a red-flagged grade is
not revised upward, and the stress test still runs on the old default definition.

## Ground rules

1. Don't disturb the parts that already pass (the 12-month flag, master scale, `03d`/`03e`).
2. Plain-English markdown for every change, citing the rule.
3. Small commits, one task per ID; re-run `run_notebooks.py all`; confirm `output/**` regenerates.

---

## PRIORITY 1

### PDR2-1 — Feed the calibrated grade PD (with MoC + floor) into Expected Loss

**Why:** EL framework Part 5.1 — EL must use the **same** PD as capital/RWA. Notebook 06 currently
uses the raw model `pd_hat` (only the 5 bps floor applied); the long-run calibration (PD-3) and the
+25 bps MoC (PD-5) never reach EL, so the dollar loss carries no conservatism. This is the main gap.

**Files:** `notebooks/03b_PD_Scorecard.ipynb` (export the per-loan grade + final PD),
`notebooks/06_expected_loss.ipynb`.

**What to do:**
1. In 03b, persist a per-loan mapping `loan_sequence_number -> grade -> long_run_pd_final`
   (the calibrated + MoC + floored grade PD from `03e`). Save it (e.g.
   `output/03f_loan_grade_pd.csv` or add the column to the cached base table).
2. In 06, attach each loan's `long_run_pd_final` (by grade) and use **that** as the PD in
   `expected_loss = PD x LGD x EAD`. Keep the raw continuous `pd_hat` in the table for ranking/
   comparison, but the EL/capital figure must use the calibrated grade PD.
3. Add a short markdown cell: the regulatory PD that feeds EL is the **calibrated grade PD**
   (long-run, count-weighted, with MoC and floor), not the uncalibrated model score — cite EL Part 5.1.
4. Show baseline-EL old (raw pd_hat) vs new (calibrated grade PD) so the conservatism impact is visible.

**Acceptance:**
- `06_expected_loss.csv` is recomputed on the calibrated grade PD.
- EL on the calibrated PD is **≥** EL on the raw `pd_hat` (the MoC now flows through).
- A note states EL and the master-scale/capital PD now reconcile to the same numbers.

### PDR2-2 — Revise red/amber-flagged grades upward (APS 113 Validation para 6)

**Why:** The `03d` calibration test flags grade H **red** (observed 2.67% > final 2.61%). Para 6:
where realised rates keep exceeding expected, the estimate **must** be revised upward. The flag is
currently raised but not acted on.

**Files:** `notebooks/03b_PD_Scorecard.ipynb` / wherever `long_run_pd_final` is set, and `03e`/`03d`.

**What to do:**
1. Add a **revise-upward rule**: for any grade flagged amber or red by the binomial test, set its
   final PD to at least the **observed default rate** (optionally observed + a small buffer), so the
   calibrated PD is never below realised experience.
2. Re-run the calibration test on the revised PDs and confirm no grade remains red because the
   estimate sits below realised (a grade may still be "watch" for other reasons — that's fine).
3. Markdown: explain the para-6 "ratchet" — estimates move up to meet realised experience but are
   not lowered just because one period looked benign.

**Acceptance:**
- Every grade's final PD ≥ its observed default rate (or the exception is explicitly justified).
- The post-revision calibration-test output shows no grade red on under-estimation grounds.

---

## PRIORITY 2

### PDR2-3 — Make the margin of conservatism risk-sensitive (CRE36.67)

**Why:** CRE36.67: the MoC must be "related to the likely range of errors." A flat +25 bps is a 6×
uplift on grade A but ~10% on grade H — it barely touches the grade that is actually under-predicting.

**Files:** `src/definitions.py` (the MoC helper), `03e`.

**What to do:**
1. Replace the flat additive MoC with a **per-grade, error-related** margin — e.g. half the width of
   the grade's binomial confidence interval, or a multiple of the standard error of the grade default
   rate (`sqrt(p(1-p)/n)`), so thin/volatile grades carry more margin.
2. Keep it transparent: show per-grade `moc_points` varying by grade, with the rationale.
3. Confirm the 5 bps floor still applies after the MoC.

**Acceptance:** `03e` shows MoC that varies by grade and is larger where the data is thin or the
calibration test flags under-estimation.

### PDR2-4 — Save the Hosmer-Lemeshow result

**Why:** PD framework Part 5.3 lists HL as the multi-grade simultaneous calibration test; commit
`00be011` mentions it but `03d` only saves the per-grade binomial p-value.

**What to do:** Compute the portfolio-level Hosmer-Lemeshow chi-square statistic and p-value across
grades and save it (a summary row in `03d` or a small `03d_hl_summary.csv`). State the independence
caveat (HL, like the binomial test, understates Type-I error when defaults are correlated — WP14).

### PDR2-5 — Align the stress test to the Stress framework and the 12-month PD

**Why:** Notebook 07 still computes its baseline PD/multiplier from `ever_default` (the old lifetime
flag), so it is inconsistent with the rest of the model; and against the Stress guide it lacks the
required scenario framing (Basel CRE36.51; APS 220 paras 70–76).

**File:** `notebooks/07_stress_testing.ipynb`.

**What to do:**
1. **Switch the stress base to the one-year/calibrated PD** — use `default_within_12m` (and the
   calibrated grade PD from PDR2-1) so the stress layer matches the EL/capital PD.
2. Frame at least **two named scenarios**: a **mild recession** (Basel CRE36.51's two-consecutive-
   quarters-of-zero-growth example) and the existing **severe** observed-crisis scenario.
3. State the **no-diversification** conservative assumption (APG 113 para 92).
4. Add a short **contingency / management-action** note (APS 220 para 74) and a one-line
   **reverse-stress** framing ("what multiplier pushes EL to X / capital below threshold").
5. Add a one-line note that the stress model would require **independent validation** (APS 220 para 76).

**Acceptance:** 07 runs on the 12-month/calibrated PD; shows a mild and a severe scenario, the
no-diversification assumption, and a contingency + reverse-stress note.

---

## PRIORITY 3 (documentation / optional)

- **PDR2-6 — 90-DPD sensitivity.** Add an optional 90-DPD variant of `default_within_12m` from the
  same delinquency field and report how the one-year default rate and grade PDs shift, to evidence
  the APS 220 broad-equivalence claim already documented in notebook 08.
- **PDR2-7 — Lifetime PD note.** Notebook 06 uses a horizon-factor proxy for Stage 2/3 lifetime ECL.
  Add a one-paragraph note that a production model would estimate a **lifetime PD term structure**
  directly; keep the proxy but name it as such.

---

## Suggested order

PDR2-1 → PDR2-2 → PDR2-3 → PDR2-4 → PDR2-5 → PDR2-6/7.

## Final checks

- [ ] `run_notebooks.py all` runs clean.
- [ ] EL is computed on the calibrated grade PD; EL(calibrated) ≥ EL(raw pd_hat); EL and capital PD reconcile.
- [ ] Every grade's final PD ≥ observed default rate; no grade red on under-estimation.
- [ ] MoC varies by grade and is error-related.
- [ ] Hosmer-Lemeshow statistic saved with the independence caveat.
- [ ] Stress test runs on the 12-month/calibrated PD with mild + severe scenarios, no-diversification,
      contingency, and reverse-stress notes.
- [ ] README/notebook 08 updated to reflect that conservatism now flows through to EL.
- [ ] Each task committed separately with its task ID.
