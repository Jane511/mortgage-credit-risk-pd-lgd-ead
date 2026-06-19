"""Generate the 00-08 notebooks for the Mortgage Credit Risk project.

Each notebook = a plain-English HR summary at the top, code cells with one
comment each, and exactly one results table saved to outputs/tables/. Heavy logic lives
in src/ so the notebooks stay short and readable. Run:  python build_notebooks.py
"""

import os
import sys
import nbformat as nbf

NB_DIR = os.path.join(os.path.dirname(__file__), "notebooks")
os.makedirs(NB_DIR, exist_ok=True)

# Notebooks register here; we write them (optionally filtered by argv) at the end,
# so `python build_notebooks.py 03b` rebuilds just one without touching the rest.
_REGISTRY = []

# Every notebook starts by anchoring to the project root so `from src import ...`
# and the relative data/output paths work whether it is run from the repo root
# or from inside notebooks/.
BOOTSTRAP = """import sys, os
ROOT = os.getcwd()
if not os.path.isdir(os.path.join(ROOT, 'src')):
    ROOT = os.path.dirname(ROOT)
os.chdir(ROOT)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
import warnings; warnings.filterwarnings('ignore')
print('project root:', ROOT)"""


def md(text):
    return nbf.v4.new_markdown_cell(text)


def code(src):
    return nbf.v4.new_code_cell(src)


def write(name, cells):
    _REGISTRY.append((name, cells))


# ======================================================================
# 00 -- Load & assemble
# ======================================================================
write("00_load_and_assemble.ipynb", [
    md("""# 00 -- Load & Assemble the Freddie Mac data

**What this notebook does (plain English):** The raw Freddie Mac files have no
column headings and split each loan across two files -- one row of facts at the
loan's start, and one row *per month* of its life (millions of rows). This
notebook attaches the official column names, checks the layout is right, finds
whether and when each loan defaulted, and boils everything down to **one tidy
row per loan**. It then stacks **17 origination years (2006-2022)** together --
spanning the housing boom, the **2007-2009 financial crisis**, the recovery, the
long expansion, and the **2020 COVID** shock.

**Headline result:** the crisis and COVID vintages default far more often than
the calm expansion years -- a full cycle of good and bad years, which is what the
long-run regulatory calibration needs."""),
    code(BOOTSTRAP),
    code("""# Read all 17 vintages (2006-2022), apply the 32-column layout, and collapse
# the monthly performance files down to one row per loan (this is the heavy step,
# ~minutes: tens of millions of servicing rows across the panel).
from src import loaders
df = loaders.load_all_vintages('data/raw data')
print('assembled loan-level table:', df.shape)"""),
    code("""# Cache the assembled table so every later notebook loads in seconds.
os.makedirs('data/processed', exist_ok=True)
df.to_parquet('data/processed/loan_level.parquet')
print('cached -> data/processed/loan_level.parquet')"""),
    code("""# Build a data-quality + SEASONING summary per vintage (R3-D2). The one-year PD
# target needs >=12 months of performance to be observable, so we evidence how far
# each vintage is observed rather than assume it.
df['_seasoned_12m'] = df['months_observed'].fillna(0) >= 12
dq = df.groupby('vintage_year').agg(
    loans=('loan_sequence_number', 'size'),
    defaults=('ever_default', 'sum'),
    disposed_defaults=('disposed', 'sum'),
    median_credit_score=('credit_score', 'median'),
    median_original_upb=('original_upb', 'median'),
    last_obs_period=('last_period', 'max'),        # data extract horizon (YYYYMM)
    pct_seasoned_12m=('_seasoned_12m', 'mean'),     # share observed >= 12 months
)
dq['default_rate'] = (dq['defaults'] / dq['loans']).round(4)
dq['pct_seasoned_12m'] = dq['pct_seasoned_12m'].round(3)
dq = dq.reset_index()
df.drop(columns='_seasoned_12m', inplace=True)"""),
    code("""# Save the one results table for this notebook.
from src.output import save_csv
save_csv(dq, 'outputs/tables/00_data_quality.csv')
dq"""),
    md("""**Reading the table:** each vintage is a 50,000-loan random sample. The
`default_rate` column is the share of loans that ever hit serious default. The
crisis (2007-09) and COVID (2020) years stand well above the calm expansion years
-- a full good-and-bad cycle, which is what the long-run regulatory calibration needs.

**Seasoning (R3-D2).** `last_obs_period` shows the panel is observed through **2025-09**
for every vintage, and `pct_seasoned_12m` is the share of each vintage seen for >=12
months. Even 2022 is ~96% seasoned, so the one-year PD window is fully observable for
all 17 vintages -- the sub-12-month loans are **early payoffs, not right-censoring**, so
**no vintage is excluded** from the long-run average. (Recent-vintage *LGD* workouts can
still be incomplete; that is handled by the incomplete-workout sensitivity in nb 04.)"""),
])


# ======================================================================
# 01 -- Base table (default flag, EAD, realised LGD)
# ======================================================================
write("01_base_table.ipynb", [
    md("""# 01 -- Build the modelling base table

**What this notebook does (plain English):** This turns the assembled data into
the table every model will use. For each loan it records three things that drive
all later numbers:

- **Default** -- did the loan go badly wrong? (180+ days late, or it ended in a
  loss event such as a foreclosure sale.)
- **EAD (Exposure at Default)** -- how much money was still owed when it defaulted.
- **LGD (Loss Given Default)** -- of that exposure, how much was *actually lost*
  after the property was sold and costs/recoveries settled. This is computed from
  Freddie Mac's **real loss fields**, which is the centrepiece of the project.

**Headline result:** average loss-given-default is far worse in the downturn
(~55-58% in 2007/2008) than in the calm year (~25% in 2015)."""),
    code(BOOTSTRAP),
    code("""# Load the cached loan-level table from notebook 00.
import numpy as np
import pandas as pd
from src import definitions as d
from src.output import save_csv
df = pd.read_parquet('data/processed/loan_level.parquet')
print(df.shape)"""),
    code("""# Realised loss and LGD are only defined for defaulted loans that DISPOSED
# (reached a final sale). For everyone else LGD is left blank, on purpose.
df['realised_loss'] = np.where(df['disposed'], d.realised_loss(df), np.nan)
lgd_raw = df['realised_loss'] / df['ead'].replace(0, np.nan)
df['lgd'] = np.where(df['disposed'], d.winsorise_lgd(lgd_raw), np.nan)"""),
    code("""# Sanity check: reconcile our computed loss against Freddie Mac's own
# actual_loss_calculation field (their number is stored as a negative loss).
rec = df[df['disposed']].dropna(subset=['actual_loss_calculation'])
rec = rec[rec['actual_loss_calculation'] != 0]
corr = np.corrcoef(-rec['actual_loss_calculation'], d.realised_loss(rec))[0, 1]
print(f'loss reconciliation correlation vs dataset field: {corr:.3f}')"""),
    code("""# Add simple risk bands we will reuse in the EDA and models.
df['credit_score_band'] = pd.cut(df['credit_score'], [0, 620, 660, 700, 740, 780, 851],
                                 right=False, labels=['<620', '620-659', '660-699', '700-739', '740-779', '780+'])
df['ltv_band'] = pd.cut(df['original_ltv'], [0, 60, 70, 80, 90, 200],
                        right=False, labels=['<60', '60-69', '70-79', '80-89', '90+'])"""),
    md("""### The workout period (input to the discounting step)

When a loan defaults the money is **not** lost all at once -- the lender works
through foreclosure and sale over many months, and a dollar recovered years later
is worth less than a dollar today. The **workout period** is how long that takes:
from the **default month** to the **disposition month**.

- The **default month** (`default_period`) already comes straight from the data --
  the first month the loan was 180+ days late or hit a loss event.
- The **disposition month** (`disposition_period`) uses the **exact zero-balance
  effective date** where Freddie Mac records it, and falls back automatically to
  the **last servicing month** (`last_period`) when that date is missing.

`months_to_resolution` is that gap in whole months, and it is only meaningful for
**disposed defaults** (NaN everywhere else). Together with `original_interest_rate`
it is one of the two inputs the economic-loss discounting in notebook 01 /
`definitions.economic_loss()` will use (see LGD alignment task P1-1)."""),
    code("""# Disposition month: exact zero-balance date if loaded, else last servicing month.
if 'disposition_period' in df.columns:
    df['disposition_period'] = df['disposition_period'].fillna(df['last_period'])
else:
    df['disposition_period'] = df['last_period']

# Workout length in months, only meaningful for disposed defaults.
df['months_to_resolution'] = np.where(
    df['disposed'],
    d.months_between(df['default_period'], df['disposition_period']),
    np.nan,
)"""),
    md("""### Nominal vs *economic* loss -- discounting (P1-1)

The framework's very first definition of LGD (CRE36.76 / APS 113 Att D LGD para 1)
is **economic loss**, which *must* include "material discount effects". The `lgd`
column above is **nominal** -- it adds up the dollars lost without caring *when*
they were lost. But a mortgage workout takes months or years (`months_to_resolution`),
and a dollar recovered years after default is worth less than a dollar today.

So we add a second, framework-aligned figure, `lgd_econ`:

- We treat the net recovery (sale proceeds + insurance + other recoveries, **minus**
  foreclosure costs) as a single cash flow arriving `months_to_resolution` months
  after default, and **discount it back** to the default date.
- The discount rate is the **facility's own contractual rate** -- the first choice
  in **APG 113 para 122 / Table 8** -- i.e. `original_interest_rate` converted to a
  monthly rate `r_m = (rate/100)/12`.

Discounting shrinks the present value of the recovery, so **economic loss is always
>= nominal loss**, and the gap is widest for the longest workouts. The original
nominal `lgd` (and its ~0.99 reconciliation) is kept untouched."""),
    code("""# Economic (discounted) loss + LGD -- the framework's actual LGD definition.
# Kept SEPARATE from nominal `lgd`; reduces to it when the workout is instant.
df['economic_loss'] = np.where(df['disposed'], d.economic_loss(df), np.nan)
lgd_econ_raw = df['economic_loss'] / df['ead'].replace(0, np.nan)
df['lgd_econ'] = np.where(df['disposed'], d.winsorise_lgd(lgd_econ_raw), np.nan)
print('avg nominal LGD : {:.4f}'.format(df.loc[df['disposed'], 'lgd'].mean()))
print('avg economic LGD: {:.4f}  (>= nominal, as discounting requires)'.format(
    df.loc[df['disposed'], 'lgd_econ'].mean()))"""),
    md("""### IFRS 9 view vs **APRA regulatory-capital view** of LGD (P1-2, P1-3)

The numbers so far answer *"what was actually lost economically?"* -- the **IFRS 9 /
accounting** question, where every real recovery (including mortgage insurance) counts.
APRA's **capital** rules deliberately answer a more conservative question and we keep
them in a **separate** column, `lgd_apra`, never overwriting the IFRS 9 number:

- **No LMI credit (APS 113 Att B para 23).** You may **not** use lender's-mortgage-
  insurance recoveries inside a retail-mortgage LGD. We rebuild the loss with
  `include_mi=False`, then apply the permitted **20% LGD reduction** on the high-LVR
  (LVR > 80) loans that actually carry LMI -- the relief the rule grants in place of
  the recovery.
- **LGD floor (APS 113 Att B paras 19-24, Tables 6-7).** A regulatory minimum LGD is
  then applied -- **20%** for retail residential mortgages where own-LGD estimates are
  not approved (stated as our assumption).

Removing the MI recovery *raises* the loss, so for MI-covered high-LVR loans
`lgd_apra >= lgd`. The floor is a backstop on top. This column is the APRA-view
overlay only; the IFRS 9 `lgd` / `lgd_econ` are left exactly as computed."""),
    code("""# APRA regulatory-capital view: MI excluded, 20% high-LVR+LMI reduction, 20% floor.
loss_no_mi = np.where(df['disposed'], d.economic_loss(df, include_mi=False), np.nan)
lgd_no_mi = d.winsorise_lgd(loss_no_mi / df['ead'].replace(0, np.nan))
mi_present = pd.to_numeric(df['mi_pct'], errors='coerce').fillna(0) > 0
high_lvr = pd.to_numeric(df['original_ltv'], errors='coerce') > 80
# 20% LGD reduction where LVR>80 and LMI is in place (APS 113 Att B para 23).
lgd_apra = np.where(mi_present & high_lvr, lgd_no_mi * (1 - 0.20), lgd_no_mi)
lgd_apra = d.apply_lgd_floor(lgd_apra, floor=0.20)  # APS 113 retail mortgage floor
df['lgd_apra'] = np.where(df['disposed'], lgd_apra, np.nan)
mi_hi = df['disposed'] & mi_present & high_lvr
print('MI-covered high-LVR disposed defaults:', int(mi_hi.sum()))
print('  avg IFRS 9 lgd : {:.4f}'.format(df.loc[mi_hi, 'lgd'].mean()))
print('  avg APRA lgd   : {:.4f}  (>= IFRS 9: MI recovery removed)'.format(
    df.loc[mi_hi, 'lgd_apra'].mean()))"""),
    code("""# Keep one clean analysis row per loan and cache it for later notebooks.
base_cols = [
    'loan_sequence_number', 'vintage_year', 'credit_score', 'original_ltv',
    'original_cltv', 'original_dti', 'original_interest_rate', 'original_loan_term',
    'original_upb', 'loan_purpose', 'occupancy_status', 'channel', 'number_of_borrowers',
    'mi_pct', 'credit_score_band', 'ltv_band', 'ever_default', 'default_within_12m',
    'default_within_12m_90dpd', 'disposed', 'max_delinq_status',
    'ead', 'realised_loss', 'lgd',
    'default_period', 'disposition_period', 'months_to_resolution',
    'economic_loss', 'lgd_econ', 'lgd_apra',
]
base = df[base_cols].copy()
base.to_parquet('data/processed/analysis_base.parquet')
print('analysis base:', base.shape)"""),
    md("""### One-year PD target -- `default_within_12m` (PD-1)

The headline `ever_default` flag asks *"did this loan ever go bad in the history we
observed?"* That is the right lens for lifetime loss and for the LGD population, but it
is **not** how a PD is defined for capital. The framework's foundational definition
(CRE36.63 / APS 113 Att D PD para 2) is a **long-run average of one-year default
rates** -- so the PD target must be measured over a **fixed 12-month window** for every
loan, regardless of how long we happened to watch it.

`default_within_12m` is exactly that: a default (180+ DPD or a credit-event) occurring
within the **first 12 months** of the loan's life (from `loan_age`). Because older books
are observed for far longer than recent ones, their *observed-to-date* rates are not
comparable; the fixed one-year window puts all 17 vintages on the **same footing**.

`ever_default` and `disposed` are kept **unchanged** -- the PD notebooks switch to the
one-year flag, while the lifetime-EL view and the LGD work keep using `ever_default`."""),
    code("""# Sanity-check the one-year PD target: it must be a SUBSET of ever_default, and
# the 12-month default rate is now measured over the same window for all vintages.
assert (df['default_within_12m'] & ~df['ever_default']).sum() == 0, '12m default must imply ever_default'
pd_target = df.groupby('vintage_year').agg(
    loans=('loan_sequence_number', 'size'),
    ever_default_rate=('ever_default', 'mean'),
    one_year_default_rate=('default_within_12m', 'mean'),
).reset_index().round(4)
print(pd_target.to_string(index=False))
print('one-year default is a strict subset of ever-default:',
      bool((df['default_within_12m'] <= df['ever_default']).all()))"""),
    code("""# Sanity-check the new workout-length field on disposed defaults only.
wr = df.loc[df['disposed'], 'months_to_resolution']
print('disposed defaults with a usable months_to_resolution:', int(wr.notna().sum()))
print('min / median / max months: {:.0f} / {:.0f} / {:.0f}'.format(
    wr.min(), wr.median(), wr.max()))
print('share resolved in 0 months:', round(float((wr == 0).mean()), 4))
# Confirm the field is blank for everyone who did NOT dispose-as-default.
print('non-disposed loans with a non-NaN value (should be 0):',
      int(df.loc[~df['disposed'], 'months_to_resolution'].notna().sum()))"""),
    code("""# Results table: default rate and average LGD by vintage (downturn vs calm),
# now showing nominal vs economic (discounted) and the APRA-view LGD side by side.
tbl = df.groupby('vintage_year').agg(
    loans=('loan_sequence_number', 'size'),
    default_rate=('ever_default', 'mean'),
    disposed_defaults=('disposed', 'sum'),
    avg_lgd=('lgd', 'mean'),
    avg_lgd_econ=('lgd_econ', 'mean'),
    avg_lgd_apra=('lgd_apra', 'mean'),
    median_lgd=('lgd', 'median'),
    avg_ead=('ead', 'mean'),
).reset_index().round(4)
save_csv(tbl, 'outputs/tables/01_default_lgd_by_vintage.csv')
tbl"""),
    md("""**Reading the table:** both the chance of default *and* the severity of
loss when it happens are much worse in the crisis vintages -- the two effects
compound, which is exactly why a downturn hurts a mortgage book so much.

Across the new LGD columns: **`avg_lgd_econ` >= `avg_lgd`** in every vintage
(discounting the recovery raises the loss), and **`avg_lgd_apra`** sits higher
again because it strips out mortgage-insurance recoveries and imposes the 20%
regulatory floor. The three columns are deliberately kept separate: nominal IFRS 9,
economic IFRS 9, and the conservative APRA capital view."""),
])


# ======================================================================
# 02 -- EDA
# ======================================================================
write("02_eda.ipynb", [
    md("""# 02 -- Exploratory data analysis

**What this notebook does (plain English):** A few clear pictures of *what makes
a mortgage risky*. We look at how the default rate changes with the borrower's
**credit score**, with the **loan-to-value** ratio (how big the loan is versus
the home's value), and across the 17 **vintages** (2006-2022). Charts are saved for the
README.

**Headline result:** default rate falls steadily as credit score rises and rises
sharply as loan-to-value climbs -- the two classic mortgage risk drivers."""),
    code(BOOTSTRAP),
    code("""# Load the base table and set up plotting (headless backend for saving files).
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from src.output import save_csv
base = pd.read_parquet('data/processed/analysis_base.parquet')
os.makedirs('outputs/charts', exist_ok=True)"""),
    code("""# Default rate by credit-score band.
by_score = base.groupby('credit_score_band', observed=True)['ever_default'].mean()
ax = by_score.plot(kind='bar', color='#4C72B0', title='Default rate by credit-score band')
ax.set_ylabel('default rate'); plt.tight_layout()
plt.savefig('outputs/charts/default_by_credit_score.png', dpi=110); plt.close()"""),
    code("""# Default rate by loan-to-value band.
by_ltv = base.groupby('ltv_band', observed=True)['ever_default'].mean()
ax = by_ltv.plot(kind='bar', color='#C44E52', title='Default rate by loan-to-value band')
ax.set_ylabel('default rate'); plt.tight_layout()
plt.savefig('outputs/charts/default_by_ltv.png', dpi=110); plt.close()"""),
    code("""# One-page risk-by-driver summary table (the saved result for this notebook).
rows = []
for band, v in base.groupby('credit_score_band', observed=True)['ever_default'].mean().items():
    rows.append({'driver': 'credit_score', 'band': band, 'default_rate': round(v, 4)})
for band, v in base.groupby('ltv_band', observed=True)['ever_default'].mean().items():
    rows.append({'driver': 'ltv', 'band': band, 'default_rate': round(v, 4)})
for band, v in base.groupby('vintage_year')['ever_default'].mean().items():
    rows.append({'driver': 'vintage', 'band': str(band), 'default_rate': round(v, 4)})
risk_by_driver = pd.DataFrame(rows)
save_csv(risk_by_driver, 'outputs/tables/02_risk_by_driver.csv')
risk_by_driver"""),
    md("""**Reading the table:** weaker credit scores and higher loan-to-value both
line up with higher default rates, and every band is worse in the crisis years.
These are the drivers we feed into the PD model next."""),
])


# ======================================================================
# 03 -- PD model
# ======================================================================
write("03_pd_model.ipynb", [
    md("""# 03 -- PD model (probability of default)

**What this notebook does (plain English):** Builds a simple, transparent model
that estimates each loan's **chance of defaulting within one year** from facts
known at the start (credit score, loan-to-value, debt-to-income, loan purpose,
etc.). We use **logistic regression** -- the industry-standard interpretable
scorecard method -- and grade it the way a model-validation team would. The target
is the **one-year** default flag (PD-1/PD-2), the framework's PD basis.

**Headline result:** the model separates good from bad loans well, with an **AUC
around 0.86** (a coin-flip would be 0.50), and its predicted one-year default rates
track the actual ones closely."""),
    code(BOOTSTRAP),
    code("""# Load the base table and split into train/test (stratified on default).
import pandas as pd
from sklearn.model_selection import train_test_split
from src import models, metrics
from src.output import save_csv
base = pd.read_parquet('data/processed/analysis_base.parquet')
# PD target = the ONE-YEAR default flag (PD-1/PD-2), not the observed-to-date flag.
PD_TARGET = 'default_within_12m'
train, test = train_test_split(base, test_size=0.30, stratify=base[PD_TARGET], random_state=42)"""),
    code("""# Fit the logistic one-year PD on origination features and score the held-out test set.
model, columns = models.fit_pd(train)
test = test.copy()
test['pd_hat'] = models.predict_pd(model, columns, test)"""),
    code("""# Grade discrimination (AUC / Gini / KS) on the test set.
y = test[PD_TARGET].astype(int)
auc = metrics.auc(y, test['pd_hat'])
gini = metrics.gini(y, test['pd_hat'])
ks = metrics.ks(y, test['pd_hat'])
print(f'AUC={auc:.3f}  Gini={gini:.3f}  KS={ks:.3f}')"""),
    code("""# Calibration: do predicted PDs match observed default rates, decile by decile?
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
cal = metrics.calibration_table(y, test['pd_hat'])
ax = cal.plot(x='predicted_pd', y='observed_default_rate', marker='o', legend=False,
              title='PD calibration (predicted vs observed)')
ax.plot([0, cal['predicted_pd'].max()], [0, cal['predicted_pd'].max()], 'k--', lw=1)
ax.set_xlabel('predicted PD'); ax.set_ylabel('observed default rate'); plt.tight_layout()
os.makedirs('outputs/charts', exist_ok=True)
plt.savefig('outputs/charts/pd_calibration.png', dpi=110); plt.close()"""),
    code("""# Save the metrics + predicted-PD distribution as this notebook's result.
metrics_tbl = pd.DataFrame([
    {'metric': 'AUC', 'value': round(auc, 4)},
    {'metric': 'Gini', 'value': round(gini, 4)},
    {'metric': 'KS', 'value': round(ks, 4)},
    {'metric': 'test_loans', 'value': len(test)},
    {'metric': 'pd_hat_mean', 'value': round(test['pd_hat'].mean(), 4)},
    {'metric': 'pd_hat_p50', 'value': round(test['pd_hat'].median(), 4)},
    {'metric': 'pd_hat_p95', 'value': round(test['pd_hat'].quantile(0.95), 4)},
])
save_csv(metrics_tbl, 'outputs/tables/03_pd_metrics.csv')
metrics_tbl"""),
    code("""# FINAL PD MODEL EQUATION: the logistic-regression coefficient for every variable.
# Features are standardised (zero mean / unit variance) before fitting, so the coefficient
# magnitude is a like-for-like importance and exp(coef) is the odds multiplier per 1 SD move.
import numpy as np
logit = model.named_steps['logit']
coef_tbl = pd.DataFrame({'variable': columns, 'coefficient': logit.coef_[0]})
coef_tbl['odds_ratio_per_1sd'] = np.exp(coef_tbl['coefficient'])
coef_tbl = pd.concat([
    pd.DataFrame([{'variable': 'intercept', 'coefficient': float(logit.intercept_[0]),
                   'odds_ratio_per_1sd': float(np.exp(logit.intercept_[0]))}]),
    coef_tbl.sort_values('coefficient', key=abs, ascending=False),
], ignore_index=True).round(4)
save_csv(coef_tbl, 'outputs/tables/03_pd_coefficients.csv')
coef_tbl"""),
    code("""# CONFUSION MATRIX at a transparent operating point: flag a loan as 'predicted default'
# when its PD exceeds the portfolio's one-year default rate (prevalence threshold). With a
# ~0.4% base rate a naive 0.5 cut-off would predict zero defaults, so the prevalence cut is
# the honest way to show true/false positives and negatives.
thr = float(y.mean())
pred_pos = (test['pd_hat'] >= thr).astype(int)
tp = int(((pred_pos == 1) & (y == 1)).sum()); fp = int(((pred_pos == 1) & (y == 0)).sum())
fn = int(((pred_pos == 0) & (y == 1)).sum()); tn = int(((pred_pos == 0) & (y == 0)).sum())
precision = tp / (tp + fp) if (tp + fp) else 0.0
recall = tp / (tp + fn) if (tp + fn) else 0.0
conf = pd.DataFrame([
    {'metric': 'threshold (PD cut-off)', 'value': round(thr, 4)},
    {'metric': 'true_positives (caught defaults)', 'value': tp},
    {'metric': 'false_positives (false alarms)', 'value': fp},
    {'metric': 'false_negatives (missed defaults)', 'value': fn},
    {'metric': 'true_negatives', 'value': tn},
    {'metric': 'precision', 'value': round(precision, 4)},
    {'metric': 'recall (sensitivity)', 'value': round(recall, 4)},
])
save_csv(conf, 'outputs/tables/03_confusion_matrix.csv')
conf"""),
    md("""**Reading the tables:** AUC/Gini/KS measure how well the model ranks risky loans
above safe ones; higher is better. The **coefficient table** is the final model equation --
each origination variable's logistic weight (standardised, so directly comparable) and its
odds multiplier; a negative coefficient on credit score means higher scores lower the default
odds, exactly as expected. The **confusion matrix** (at the prevalence cut-off) shows the
caught-vs-missed trade-off a portfolio team would tune. The calibration plot (saved to
`outputs/charts/`) shows predicted and actual default rates lining up along the diagonal --
the model is honest, not just discriminating."""),
])


# ======================================================================
# 03b -- PD Scorecard (WOE / IV -> points -> rating master scale)
# ======================================================================
write("03b_PD_Scorecard.ipynb", [
    md("""# 03b -- PD Scorecard (rating grades)

**What this notebook does (plain English):** Notebook 03 estimated each loan's
chance of default. This notebook turns that into the **scorecard** a lender
actually uses: a simple points system (like a credit score) that sorts every
loan into a handful of **rating grades**, from A (safest) to the riskiest. It is
built with the same transparent technique as my consumer-credit scorecard
(Weight-of-Evidence + logistic regression), so each grade has a clear,
defensible meaning.

**Headline result:** the grades line up cleanly -- the model's predicted default
rate for each grade matches the *actual* default rate closely (a calibration
check), and the safest grades hold most of the money at the lowest risk.

**PD horizon (stated up front):** this is a **one-year PD** (PD-1/PD-2) -- a loan
counts as a default if it reached serious default within the **first 12 months** of
its life. A fixed 12-month window is the framework's one-year-PD basis (CRE36.63 /
APS 113 Att D PD para 2) and makes all 17 vintages directly comparable, regardless
of how long each was observed. The model uses only origination facts, so it stays a
clean "through-the-door" scorecard read on that one-year horizon."""),
    code(BOOTSTRAP),
    code("""# Load the loan-level base table; target = 1 means the loan defaulted (a 'bad').
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from src import woe, transform, scorecard, metrics
from src.output import save_csv
base = pd.read_parquet('data/processed/analysis_base.parquet').copy()
# PD target = the ONE-YEAR default flag (PD-1/PD-2).
base['target'] = base['default_within_12m'].astype(int)"""),
    code("""# WOE-bin the main origination predictors on a train split and report each
# feature's Information Value (IV) -- how predictive it is on its own.
features = ['credit_score', 'original_ltv', 'original_cltv', 'original_dti',
            'original_loan_term', 'loan_purpose', 'occupancy_status']
train, test = train_test_split(base, test_size=0.30, stratify=base['target'], random_state=42)
binning, iv_summary, woe_tables = woe.fit_binning(train, features, train['target'], max_bins=5)
save_csv(iv_summary, 'outputs/tables/03b_information_value.csv')
iv_summary"""),
    code("""# Replace raw predictors with their WOE values and fit a logistic regression.
X_train = transform.transform_to_woe(train, binning)
X_test = transform.transform_to_woe(test, binning)
clf = LogisticRegression(max_iter=2000)
clf.fit(X_train, train['target'])"""),
    code("""# Check the scorecard still discriminates well on the held-out test set.
test = test.copy()
test['pd_hat'] = clf.predict_proba(X_test)[:, 1]
y = test['target']
print(f"AUC={metrics.auc(y, test['pd_hat']):.3f}  Gini={metrics.gini(y, test['pd_hat']):.3f}  KS={metrics.ks(y, test['pd_hat']):.3f}")"""),
    code("""# Choose the scorecard scaling and convert the model into points.
# Anchor: 600 points = 50:1 good:bad odds; PDO=20 points doubles the odds.
factor, offset = scorecard.scaling_params(base_points=600, base_odds=50, pdo=20)
print(f"Scaling -> Factor={factor:.2f}, Offset={offset:.2f}  (Score = Offset - Factor x default log-odds)")
X_all = transform.transform_to_woe(base, binning)
base['pd_hat'] = clf.predict_proba(X_all)[:, 1]
base['score'] = scorecard.score_from_logit(clf.decision_function(X_all), factor, offset)"""),
    code("""# Per-bin points table: every predictor band's contribution to the score
# (the points for a loan add up to its total score -- fully transparent).
coefs = dict(zip(X_train.columns, clf.coef_[0]))
points = scorecard.scorecard_points(binning, woe_tables, coefs, clf.intercept_[0], factor, offset)
save_csv(points, 'outputs/tables/03b_scorecard_points.csv')
points.head(15)"""),
    code("""# Sort every loan into 8 rating grades (A safest) and build the MASTER SCALE:
# predicted PD vs observed default rate, with loan count and exposure share.
base['grade'] = scorecard.assign_grades(base['score'], n_grades=8)
master = scorecard.master_scale(base, 'grade', 'pd_hat', 'target', 'original_upb', score_col='score')"""),
    code("""# PD-3: calibrate each grade to its LONG-RUN PD -- the simple average ACROSS the
# 17 vintages (2006-2022) of the per-year one-year default rate (count-weighted within
# year), the framework basis (APG 113 paras 110-114; count-weighted, not exposure-weighted).
lr = scorecard.long_run_grade_pd(base, 'grade', 'target', 'vintage_year', exposure_col='original_upb')
master = master.merge(lr, on='grade', how='left')
master['long_run_pd'] = master['long_run_pd'].round(4)
master['exposure_weighted_pd'] = master['exposure_weighted_pd'].round(4)  # sensitivity only
save_csv(master, 'outputs/tables/03b_master_scale.csv')
master[['grade', 'predicted_pd', 'long_run_pd', 'observed_default_rate',
        'exposure_weighted_pd', 'loans', 'exposure_share']]"""),
    md("""**Long-run grade PD (PD-3).** `predicted_pd` is the model's average per grade;
`long_run_pd` is the framework's calibration figure -- for each grade we take the
one-year default rate **in each vintage** and then **simple-average across the 17
vintages** (each loan counts once *within* a year; each year counts equally *across*
years, per APS 113 Att D PD para 3, which is **count-weighted, not EAD-weighted**).
The two columns are close, confirming the model is well-calibrated in level, not just
in rank. `exposure_weighted_pd` is shown for **sensitivity review only** (APG 113 para
114) and is explicitly *not* the calibration figure.

**A balanced cycle (no longer downturn-heavy).** The panel now spans **2006-2022** --
boom, GFC, recovery, expansion and COVID -- so the simple across-year average sits at a
genuine **through-the-cycle** level rather than being skewed by a couple of crisis years.
This is why the long-run grade PDs are **lower and the margin of conservatism smaller**
than on the old 3-vintage window: more good-and-bad years tighten the estimate (CRE36.67).
The MoC (PD-5) is retained, now data-sized, and the 5-year minimum is comfortably met."""),
    code("""# PD-4: FORMAL calibration test per grade -- a one-sided binomial test for PD
# under-estimation with a green/amber/red traffic-light, plus a portfolio-level
# Hosmer-Lemeshow chi-square. Tests calibration, not just charts it (Part 5.3).
gt = base.groupby('grade', observed=True).agg(
    n=('target', 'size'), observed_defaults=('target', 'sum')).reset_index()
gt = gt.merge(master[['grade', 'long_run_pd']], on='grade')
gt['observed_rate'] = (gt['observed_defaults'] / gt['n']).round(4)
gt['binom_p_underest'] = [round(metrics.binomial_pd_test(p, dft, n), 4)
                          for p, dft, n in zip(gt['long_run_pd'], gt['observed_defaults'], gt['n'])]
gt['flag'] = gt['binom_p_underest'].apply(
    lambda p: 'green' if p > 0.05 else ('amber' if p > 0.01 else 'red'))
hl_stat, hl_p = metrics.hosmer_lemeshow(base['target'], base['pd_hat'], n_bins=10)
print(f'Hosmer-Lemeshow (10 deciles): chi2={hl_stat:.2f}  p={hl_p:.3f}')
save_csv(gt, 'outputs/tables/03d_pd_calibration_test.csv')
gt"""),
    md("""**Reading the calibration test (PD-4).** For each grade we test the assigned
**long-run PD** against the defaults actually observed: `binom_p_underest` is the
one-sided binomial p-value that the grade has **more** defaults than its PD predicts,
and the `flag` turns **amber/red** when that p-value falls below 0.05 / 0.01. The
portfolio **Hosmer-Lemeshow** chi-square does the same across deciles in one number.

**Independence caveat (WP14).** The binomial test assumes defaults are **independent**.
In a mortgage book they are not -- borrowers default together in a downturn -- so the
test **understates** the true Type-I error and will flag amber/red more readily than a
correlation-aware test would. Read any amber/red as a **prompt for review**, not a hard
pass/fail; the margin of conservatism (PD-5) is the deliberate response to exactly this
kind of correlated-tail uncertainty."""),
    code("""# PDR2-3 + PDR2-2 + PD-6: build the FINAL regulatory grade PD in three steps.
#  (a) PDR2-3 risk-sensitive MoC: per-grade margin = 1.645 standard errors of the
#      grade rate, sqrt(p(1-p)/n) -- thin/volatile grades carry MORE margin, as
#      CRE36.67 requires (the margin must relate to the likely range of errors).
#  (b) PDR2-2 ratchet: lift the PD to at least the grade's REALISED rate (APS 113
#      Validation para 6) -- estimates move up to meet experience, never down.
#  (c) PD-6: the 5 bps regulatory floor (APS 113 Att B para 1) as a backstop.
from src import definitions as d
gp = master[['grade', 'long_run_pd']].merge(gt[['grade', 'n', 'observed_rate', 'flag']], on='grade')
gp['moc_points'] = d.risk_sensitive_moc(gp['long_run_pd'].values, gp['n'].values, z=1.645).round(4)
gp['pd_after_moc'] = (gp['long_run_pd'] + gp['moc_points']).round(4)
gp['pd_revised'] = np.maximum(gp['pd_after_moc'], gp['observed_rate']).round(4)  # ratchet
gp['long_run_pd_final'] = d.apply_pd_floor(gp['pd_revised'].values, floor=0.0005).round(4)
save_csv(gp[['grade', 'long_run_pd', 'moc_points', 'pd_after_moc', 'observed_rate',
             'flag', 'pd_revised', 'long_run_pd_final']], 'outputs/tables/03e_grade_pd_moc_floor.csv')
gp[['grade', 'long_run_pd', 'moc_points', 'pd_after_moc', 'observed_rate', 'flag', 'long_run_pd_final']]"""),
    md("""**Risk-sensitive MoC + ratchet + floor (PDR2-3, PDR2-2, PD-6).** The final
regulatory grade PD is built transparently in three steps:

- **`moc_points` (PDR2-3)** -- a *risk-sensitive* margin of conservatism: 1.645 standard
  errors of each grade's default rate (`sqrt(p(1-p)/n)`). Unlike the earlier flat +25 bps
  (which was a 6x uplift on grade A but barely touched the under-predicting grade H), this
  margin is **larger where the data is thin or the rate is volatile**, exactly as CRE36.67
  requires -- the margin must relate to the likely range of errors.
- **`pd_revised` (PDR2-2 ratchet)** -- the PD is then lifted to **at least the grade's
  realised default rate** (APS 113 Validation para 6). Where experience keeps exceeding the
  estimate, the estimate must be revised **up** and is never lowered just because one period
  looked benign. This is what acts on the grade-H red flag from the calibration test.
- **`long_run_pd_final` (PD-6)** -- the 5 bps floor (APS 113 Att B para 1) as a backstop.

Crucially, **`long_run_pd_final` is the regulatory PD that now feeds Expected Loss**
(PDR2-1), so EL and the master-scale/capital PD reconcile to the same numbers."""),
    code("""# PDR2-2 check + PDR2-4: re-run the binomial calibration test on the REVISED final
# PD (no grade should remain red on under-estimation), and save the portfolio
# Hosmer-Lemeshow result across grades with the independence caveat.
post = gp[['grade', 'n', 'observed_rate', 'long_run_pd_final']].copy()
post['observed_defaults'] = (post['observed_rate'] * post['n']).round().astype(int)
post['binom_p_underest'] = [round(metrics.binomial_pd_test(p, dft, n), 4)
                            for p, dft, n in zip(post['long_run_pd_final'], post['observed_defaults'], post['n'])]
post['flag'] = post['binom_p_underest'].apply(
    lambda p: 'green' if p > 0.05 else ('amber' if p > 0.01 else 'red'))
save_csv(post, 'outputs/tables/03d_pd_calibration_test_post_revision.csv')
loan_final_pd = base[['grade']].merge(gp[['grade', 'long_run_pd_final']], on='grade', how='left')['long_run_pd_final']
hl_stat2, hl_p2 = metrics.hosmer_lemeshow(base['target'].values, loan_final_pd.values, n_bins=8)
hl = pd.DataFrame([{'test': 'hosmer_lemeshow_across_grades', 'chi2': round(hl_stat2, 2),
                    'p_value': round(hl_p2, 4), 'n_grades': int(post.shape[0]),
                    'caveat': 'assumes independent defaults; understates Type-I error under correlation (WP14)'}])
save_csv(hl, 'outputs/tables/03d_hl_summary.csv')
print('post-revision flags:', dict(post['flag'].value_counts()))
print('Hosmer-Lemeshow across grades: chi2={:.2f}  p={:.3f}'.format(hl_stat2, hl_p2))
post"""),
    md("""**Post-revision check (PDR2-2) + Hosmer-Lemeshow (PDR2-4).** After the ratchet,
every grade's final PD sits **at or above** its realised rate, so the binomial test shows
**no grade red on under-estimation** -- the grade-H flag is now acted on, not just raised.
The portfolio **Hosmer-Lemeshow** chi-square across grades is saved to `03d_hl_summary.csv`
(the multi-grade simultaneous calibration test, Part 5.3). Both tests assume **independent**
defaults; in a mortgage book defaults are correlated in a downturn, so they **understate**
Type-I error (WP14) -- read them as prompts, and note the risk-sensitive MoC above is the
deliberate buffer for that correlated-tail uncertainty."""),
    code("""# PDR2-1: persist the per-loan calibrated regulatory PD (each loan -> its grade ->
# the grade's PDs) so Expected Loss uses the SAME PD as capital (EL Part 5.1). We
# carry both the pre-MoC long-run PD and the final PD so notebook 06 can show the
# margin-of-conservatism uplift on a like-for-like (pooled) basis.
loan_grade_pd = base[['loan_sequence_number', 'grade']].merge(
    gp[['grade', 'long_run_pd', 'long_run_pd_final']], on='grade', how='left').rename(
    columns={'long_run_pd': 'grade_pd_longrun', 'long_run_pd_final': 'grade_pd_final'})
save_csv(loan_grade_pd, 'outputs/tables/03f_loan_grade_pd.csv')
print('exported per-loan calibrated regulatory PD for', len(loan_grade_pd), 'loans')
loan_grade_pd.head()"""),
    code("""# PDR2-6: 90-DPD sensitivity. Re-measure the one-year default rate and each grade's
# observed rate under the broader 90-DPD trigger (vs the repo's 180-DPD definition),
# evidencing the APS 220 broad-equivalence note in notebook 08. The grades are held
# fixed -- only the default *definition* is swapped -- so this isolates its effect.
dpd = pd.DataFrame({
    'basis': ['180-DPD (model definition)', '90-DPD (APS 220 / Basel)'],
    'one_year_default_rate': [round(base['target'].mean(), 4),
                              round(base['default_within_12m_90dpd'].mean(), 4)],
})
by_grade = base.groupby('grade', observed=True).agg(
    rate_180dpd=('target', 'mean'),
    rate_90dpd=('default_within_12m_90dpd', 'mean')).reset_index().round(4)
by_grade['uplift_x'] = (by_grade['rate_90dpd'] / by_grade['rate_180dpd'].replace(0, np.nan)).round(2)
save_csv(by_grade, 'outputs/tables/03g_dpd_sensitivity.csv')
print(dpd.to_string(index=False))
by_grade"""),
    md("""**90-DPD sensitivity (PDR2-6).** The model defines default at **180-DPD** (a common
mortgage convention and what the data cleanly supports); APS 220 / Basel reference **90-DPD**.
Swapping only the trigger, the one-year default rate **rises** (more loans cross 90 than 180
days late within the first year) and every grade's observed rate steps up by a similar factor
-- the rank-ordering is preserved, so the grades and scorecard would still hold under a
90-DPD definition, with the PD **level** re-anchored upward. This quantifies the "broad
equivalence" adjustment documented in notebook 08 (it is a sensitivity, not the model
target; nothing downstream is re-pointed to it)."""),
    code("""# Downturn view: reuse the stress logic (PD multiplier = GFC crisis vs calm default
# rate) to show how each grade's predicted PD shifts in a recession. Regime via the
# documented classifier (R3-C2): downturn = GFC vintages, calm = the reference book.
from src import definitions as d
calm = base[base['vintage_year'] == d.CALM_REFERENCE_VINTAGE]
downturn = base[d.is_downturn_vintage(base['vintage_year'])]
pd_mult = downturn['target'].mean() / calm['target'].mean()
grade_pd = base.groupby('grade', observed=True)['pd_hat'].mean().reset_index().rename(columns={'pd_hat': 'base_pd'})
grade_pd['stressed_pd'] = np.minimum(grade_pd['base_pd'] * pd_mult, 1.0).round(4)
grade_pd['base_pd'] = grade_pd['base_pd'].round(4)
grade_pd['pd_multiplier'] = round(pd_mult, 2)
save_csv(grade_pd, 'outputs/tables/03b_downturn_by_grade.csv')
grade_pd"""),
    md("""**Reading the master scale:** `predicted_pd` is what the scorecard expects;
`observed_default_rate` is what actually happened. They track closely down the
grades -- the scorecard is well-calibrated, not just well-ranked. `exposure_share`
shows where the lending is concentrated. The downturn table then takes each
grade's PD and applies the crisis multiplier, showing how every grade's risk
steps up together in a recession -- the same stress lens as notebook 07, now read
grade-by-grade.

**Saved tables:** `03b_information_value.csv` (feature IV), `03b_scorecard_points.csv`
(points per band), `03b_master_scale.csv` (the rating master scale -- the key
deliverable), and `03b_downturn_by_grade.csv` (base vs stressed PD per grade)."""),
])


# ======================================================================
# 03c -- PD out-of-time / out-of-regime validation
# ======================================================================
write("03c_PD_OutOfTime_Validation.ipynb", [
    md("""# 03c -- PD out-of-time / out-of-regime validation

**What this notebook does (plain English):** A model that looks good on the data
it was *trained* on can still fail on *new* loans. The honest test is to train on
one period and check the model on a **different, later** period it has never seen.
We use the three origination years for exactly this:

- **Split A (out-of-time, same conditions):** train on **2007**, test on **2008**
  -- both crisis years.
- **Split B (out-of-regime):** train on the **crisis (2007+2008)**, test on the
  **calm 2015** book -- a deliberately harder test across very different conditions.

**No leakage:** for each split the PD model is **re-fitted on the training years
only**, then used to score the held-out year. The pooled all-vintage model is
*not* used here -- that would let the test data sneak into training.

**Headline result:** the model's **rank-ordering holds up** out-of-time (it still
sorts risky from safe), but **risk levels are regime-sensitive** -- the crisis books
average ~1% one-year default versus ~0.14% in the calm year, the score distribution
moves wholesale (high PSI), and a level fitted in one regime cannot be trusted in
another. That is exactly why PD models are recalibrated
through the cycle."""),
    code(BOOTSTRAP),
    code("""# Load the base table and reuse the existing PD model + metrics helpers.
import pandas as pd
from src import models, metrics
from src.output import save_csv
base = pd.read_parquet('data/processed/analysis_base.parquet')"""),
    code("""# One split = refit PD on the TRAIN vintages only, then score the held-out test
# vintage (this is what prevents the test data leaking into training).
def evaluate_split(train_years, test_years, label):
    tr = base[base['vintage_year'].isin(train_years)]
    te = base[base['vintage_year'].isin(test_years)]
    model, cols = models.fit_pd(tr)                 # fit on training years ONLY
    tr_pd = models.predict_pd(model, cols, tr)
    te_pd = models.predict_pd(model, cols, te)
    ytr = tr['default_within_12m'].astype(int)      # one-year PD target (PD-1/PD-2)
    yte = te['default_within_12m'].astype(int)
    detail = {'label': label, 'arrays': (ytr, tr_pd, yte, te_pd)}
    row = {
        'split': label,
        'train_auc': round(metrics.auc(ytr, tr_pd), 4),
        'test_auc': round(metrics.auc(yte, te_pd), 4),
        'train_avg_predicted_pd': round(float(tr_pd.mean()), 4),
        'test_avg_predicted_pd': round(float(te_pd.mean()), 4),
        'test_observed_default_rate': round(float(yte.mean()), 4),
        'psi_train_vs_test': round(metrics.psi(tr_pd, te_pd), 4),
    }
    return row, detail"""),
    code("""# Run the classic regime splits, plus the genuine FORWARD holdout (R3-V3): fit on
# everything up to 2019 and score the never-seen 2020-2022 vintages cold -- the strongest
# out-of-time test the panel allows (the held-out era even contains the COVID shock).
splits = [
    ([2007], [2008], 'A) out-of-time, same regime: train 2007 -> test 2008'),
    ([2007, 2008], [2015], 'B) out-of-regime: train crisis 2007+08 -> test calm 2015'),
    ([2015], [2007, 2008], 'C) reverse what-if (NOT a forward test): train calm 2015 -> test crisis'),
    (list(range(2006, 2020)), [2020, 2021, 2022], 'D) FORWARD holdout: train 2006-2019 -> test 2020-2022 (cold)'),
]
rows, details = [], []
for tr_y, te_y, label in splits:
    r, d = evaluate_split(tr_y, te_y, label)
    rows.append(r); details.append(d)"""),
    code("""# The one comparison table (the saved deliverable).
comparison = pd.DataFrame(rows)[[
    'split', 'train_auc', 'test_auc', 'train_avg_predicted_pd',
    'test_avg_predicted_pd', 'test_observed_default_rate', 'psi_train_vs_test']]
save_csv(comparison, 'outputs/tables/03c_oot_validation.csv')
comparison"""),
    code("""# Full discrimination detail (AUC / Gini / KS, train vs test) for the record.
for d in details:
    ytr, tr_pd, yte, te_pd = d['arrays']
    print(d['label'])
    print(f"   train: AUC={metrics.auc(ytr,tr_pd):.3f} Gini={metrics.gini(ytr,tr_pd):.3f} KS={metrics.ks(ytr,tr_pd):.3f}")
    print(f"   test : AUC={metrics.auc(yte,te_pd):.3f} Gini={metrics.gini(yte,te_pd):.3f} KS={metrics.ks(yte,te_pd):.3f}")"""),
    md("""## Interpretation (plain English)

- **Discrimination held out-of-time.** Across all splits the test AUC stays strong
  (~0.79-0.82) -- the model still clearly **rank-orders** risky loans above safe ones,
  so sorting power travels across periods.
- **Levels travel well *within reach* of the training data.** Same-regime (Split A,
  train 2007 -> test 2008) the average predicted one-year PD lands almost exactly on the
  observed rate (~0.9% vs ~0.9%), and even crisis -> calm (Split B) lands close (~0.14%
  vs ~0.14%) because the origination features (credit score, LTV) carry the level.
- **But the population shift is huge.** Split B **PSI ~2.0** and Split C **PSI ~4.8**,
  far above the 0.25 "material shift" line -- the score distributions barely overlap
  across regimes, so a level that happens to land well is not something to rely on.
- **The reverse what-if (Split C, train calm 2015 -> test crisis)** shows the level
  breaking: keyed on the calm book, the model reads the crisis vintages' weak credit
  features and predicts ~4% one-year PD against an observed ~1.1% -- it **over-states**
  the one-year rate (most crisis defaults actually fall in years 2-4, outside the
  one-year window). Either way, a level fitted in one regime is untrustworthy in another.
- **The forward holdout (Split D, train 2006-2019 -> test 2020-2022)** is the honest
  production test: the model is fitted on the past and scored on genuinely later loans it
  has never seen, including the COVID era. Rank-ordering again **travels** (test AUC stays
  strong), confirming the origination-feature scorecard generalises forward; the level is
  read against the recent vintages' own observed one-year rate, with PSI quantifying how far
  the population has drifted since training.
- **Takeaway:** rank-ordering travels, but the *level* and *stability* do not. This
  is precisely why PD models are **recalibrated through the cycle** or carry a
  **macro overlay** -- the same lesson the stress test in notebook 07 makes
  quantitatively."""),
])


# ======================================================================
# 04 -- LGD model (the key feature)
# ======================================================================
write("04_lgd_model.ipynb", [
    md("""# 04 -- LGD model (loss given default) -- the key feature

**What this notebook does (plain English):** When a mortgage defaults, the lender
doesn't lose everything -- it sells the house and recovers most of the money.
**LGD** is the slice that is actually lost. Unlike a typical consumer-credit
project (where LGD is an assumption), here we model LGD from Freddie Mac's
**real, settled loss figures**. We use a simple **two-stage** model: the chance
of *any* loss, times the *size* of the loss when it happens.

**Headline result:** modelled LGD is roughly **double in the downturn** (~55%)
versus the calm year (~25%) -- a real, data-driven downturn LGD, which is the
single thing this project exists to demonstrate."""),
    code(BOOTSTRAP),
    code("""# Load the base table; LGD is modelled ONLY on defaulted, disposed loans.
import pandas as pd
import numpy as np
from src.models import TwoStageLGD
from src.output import save_csv
base = pd.read_parquet('data/processed/analysis_base.parquet')
disposed = base[base['disposed'] & base['lgd'].notna()].copy()
print('disposed defaults used for LGD:', len(disposed))"""),
    code("""# Fit the two-stage LGD model (P(loss) x severity) and predict back on them.
lgd_model = TwoStageLGD().fit(disposed)
disposed['lgd_hat'] = lgd_model.predict(disposed)"""),
    code("""# LGD MODEL EQUATION: coefficients for BOTH stages and the key drivers of loss severity.
# Stage 1 = logistic 'is there any material loss?'; stage 2 = linear 'how big is the loss?'.
# Variables: original LTV, credit score, loan size (UPB) and the GFC-downturn flag.
cols = lgd_model.columns
coef_rows = [{'stage': '1: P(loss) logistic', 'variable': 'intercept',
              'coefficient': round(float(lgd_model.p_model.intercept_[0]), 6)}]
for c, b in zip(cols, lgd_model.p_model.coef_[0]):
    coef_rows.append({'stage': '1: P(loss) logistic', 'variable': c, 'coefficient': round(float(b), 6)})
coef_rows.append({'stage': '2: severity linear', 'variable': 'intercept',
                  'coefficient': round(float(lgd_model.sev_model.intercept_), 6)})
for c, b in zip(cols, lgd_model.sev_model.coef_):
    coef_rows.append({'stage': '2: severity linear', 'variable': c, 'coefficient': round(float(b), 6)})
lgd_coef = pd.DataFrame(coef_rows)
save_csv(lgd_coef, 'outputs/tables/04_lgd_coefficients.csv')
lgd_coef"""),
    code("""# Compare observed vs modelled LGD, downturn (GFC) vs calm (non-GFC), and
# show the three LGD lenses side by side: nominal IFRS 9, economic IFRS 9, APRA.
# Regime via the documented classifier (R3-C2): downturn = GFC housing-crisis
# vintages; everything else (incl. low-severity COVID) sits in 'calm/other'.
from src import definitions as d
disposed['regime'] = np.where(d.is_downturn_vintage(disposed['vintage_year']), 'downturn (GFC)', 'calm/other')
tbl = disposed.groupby('regime').agg(
    disposed_defaults=('lgd', 'size'),
    observed_lgd=('lgd', 'mean'),
    modelled_lgd=('lgd_hat', 'mean'),
    observed_lgd_econ=('lgd_econ', 'mean'),
    lgd_apra=('lgd_apra', 'mean'),
).reset_index().round(4)"""),
    code("""# Add an "all vintages" row and save as this notebook's result table.
overall = pd.DataFrame([{
    'regime': 'all', 'disposed_defaults': len(disposed),
    'observed_lgd': round(disposed['lgd'].mean(), 4),
    'modelled_lgd': round(disposed['lgd_hat'].mean(), 4),
    'observed_lgd_econ': round(disposed['lgd_econ'].mean(), 4),
    'lgd_apra': round(disposed['lgd_apra'].mean(), 4),
}])
lgd_summary = pd.concat([tbl, overall], ignore_index=True)
save_csv(lgd_summary, 'outputs/tables/04_lgd_model.csv')
lgd_summary"""),
    md("""**Reading the table:** `observed_lgd` is what actually happened (nominal
IFRS 9); `modelled_lgd` is the two-stage model's fit. The downturn (GFC) row sits about
**1.6x** the calm/other row (~56% vs ~34%), and ~2.3x the calmest 2015 book the stress
test baselines on -- the **downturn LGD** a stress test needs. ("calm/other" pools every
non-GFC vintage, including the moderate-severity 2009-2014 recovery, so it sits above the
single calmest year.)

The last two columns are the framework views built in notebook 01, carried through
here so a reviewer sees them next to the model:
- **`observed_lgd_econ`** -- the *economic* (discounted) IFRS 9 loss; >= nominal
  because the recovery is discounted over the workout (APS 113 Att D LGD para 1).
- **`lgd_apra`** -- the **APRA regulatory-capital view**: mortgage-insurance
  recoveries excluded (APS 113 Att B para 23), the 20% high-LVR+LMI reduction
  applied, then floored at 20% (APS 113 Att B paras 19-24). It is deliberately the
  most conservative column and is **never** mixed into the IFRS 9 figures.

The model is built only on loans that truly disposed, so every number is grounded
in a real settled loss."""),
    code("""# Cyclicality test (APS 113 Att D LGD paras 4-5): is loss severity materially
# higher in bad years than good? If so, a DOWNTURN LGD is required, not optional.
cyc = disposed.groupby('regime').agg(
    n=('lgd', 'size'), realised_lgd=('lgd', 'mean')).reset_index()
calm_lgd = float(cyc.loc[cyc['regime'].str.startswith('calm'), 'realised_lgd'].iloc[0])
down_lgd = float(cyc.loc[cyc['regime'].str.startswith('downturn'), 'realised_lgd'].iloc[0])
print('calm LGD     : {:.4f}'.format(calm_lgd))
print('downturn LGD : {:.4f}'.format(down_lgd))
print('downturn / calm ratio: {:.2f}x'.format(down_lgd / calm_lgd))
print('=> severity is strongly cyclical, so the LGD ESTIMATE must reflect downturn '
      'conditions (APS 113 Att D LGD para 4-5), not the through-the-cycle average.')"""),
    md("""**Cyclicality (P2-3).** Realised severity is far higher in the GFC crisis books
than outside them (~1.6x, and ~2.3x versus the calmest 2015 book), the textbook signature of a
**cyclical** LGD. Under APS 113 Att D LGD paras 4-5, where loss severity is cyclical
the LGD *estimate* used for capital/EL must reflect **downturn** conditions rather
than the long-run average. Notebook 06 therefore carries an explicit downturn-LGD
variant of Expected Loss alongside the through-the-cycle one."""),
    md("""### Incomplete workouts -- resolution bias (P2-1)

The model above uses only **disposed** (fully resolved) loans. But APG 113 para 126
says LGD must also reflect **defaulted-but-not-yet-resolved** loans, with *estimated*
future recoveries, a **sensitivity** on that estimate, and a **maximum workout
period**. Leaving open workouts out biases LGD because the quick, clean resolutions
finish first and the messy ones are still open. The next cells quantify that bias."""),
    code("""# P2-1: count open workouts and estimate their LGD with a documented cap.
# APG 113 para 126: include incomplete workouts with estimated future recoveries.
from src import definitions as d
MAX_WORKOUT_MONTHS = 36  # documented cap: assume no further recovery beyond this.
defaulted = base[base['ever_default']].copy()
open_wf = defaulted[~defaulted['disposed']].copy()
frac_open = len(open_wf) / max(len(defaulted), 1)
print(f'defaulted loans: {len(defaulted)}')
print(f'open workouts (defaulted, not yet disposed): {len(open_wf)} = {frac_open:.1%} of defaults')"""),
    md("""**Important nuance:** here "default" = first month at 180+ DPD *or* a loss
disposition. Most defaulted-but-not-disposed loans are 180-DPD loans that later
**cured or prepaid** with little or no loss -- they are not all genuine open
foreclosures. So assigning every one of them the full ~56% segment severity is an
*upper bound*, not a best estimate. We show both, and a cure-aware best estimate in
between, so the bias is bracketed honestly."""),
    code("""# Best estimate vs conservative upper bound for the open workouts.
open_wf['regime'] = np.where(d.is_downturn_vintage(open_wf['vintage_year']), 'downturn (GFC)', 'calm/other')
seg_lgd = disposed.groupby('regime')['lgd'].mean()
open_wf['age_months'] = d.months_between(open_wf['default_period'], open_wf['disposition_period'])
# P(eventually disposes WITH a loss | defaulted): the empirical loss-disposition rate.
loss_disp_rate = len(disposed) / max(len(defaulted), 1)
seg_sev = open_wf['regime'].map(seg_lgd).fillna(disposed['lgd'].mean())
# Best estimate: expected severity = P(loss disposition) x segment severity (most cure).
open_wf['lgd_best'] = loss_disp_rate * seg_sev
# Conservative upper bound: assume every open default disposes at full segment
# severity; loans already open beyond the cap recover nothing further (LGD -> 1).
open_wf['lgd_upper'] = seg_sev
open_wf.loc[open_wf['age_months'] > MAX_WORKOUT_MONTHS, 'lgd_upper'] = 1.0
print(f'P(loss disposition | default) = {loss_disp_rate:.3f}  (the rest cure/prepay)')"""),
    code("""# Sensitivity: portfolio mean LGD with vs without the open workouts (the 'with
# and without' APG 113 para 126 asks for), bracketed best-estimate to upper-bound.
disposed_only = disposed['lgd'].mean()
incl_best = pd.concat([disposed['lgd'], open_wf['lgd_best']]).mean()
incl_upper = pd.concat([disposed['lgd'], open_wf['lgd_upper']]).mean()
sens = pd.DataFrame([
    {'basis': 'disposed only (current LGD, resolution-biased)', 'mean_lgd': round(float(disposed_only), 4)},
    {'basis': 'incl. open workouts @ cure-aware best estimate', 'mean_lgd': round(float(incl_best), 4)},
    {'basis': 'incl. open workouts @ conservative upper bound', 'mean_lgd': round(float(incl_upper), 4)},
])
sens['open_workout_share_of_defaults'] = round(frac_open, 4)
sens['max_workout_months'] = MAX_WORKOUT_MONTHS
save_csv(sens, 'outputs/tables/04_incomplete_workouts.csv')
sens"""),
    md("""**Reading the sensitivity (P2-1).** The first row is today's disposed-only
LGD. The second folds the open workouts back in at a **cure-aware best estimate**
(expected severity = P(eventual loss disposition) x segment severity), and the third
at a **conservative upper bound** (every open default disposes at full severity; any
loan already open beyond the **36-month maximum workout period** is assumed to recover
nothing further). The true unbiased LGD sits inside that band. Because a large share of
180-DPD "defaults" cure or prepay, the best estimate stays close to the disposed-only
figure while the upper bound shows how much resolution bias *could* matter -- the
bracketed disclosure APG 113 para 126 asks for, rather than dropping the open loans."""),
    md("""### Margin of conservatism overlay (P2-2)

CRE36.67 / Step 11 require an explicit **margin of conservatism (MoC)** where the
data is thin and the observation window short -- which is exactly this setup: three
discrete vintages, ~7k disposed defaults, and the open-workout uncertainty just shown.
The MoC is an **overlay**: it sits on the **APRA capital view only**, on top of the
model, and is **never** mixed into the model itself or the IFRS 9 figures."""),
    code("""# P2-2: +5 LGD-point margin of conservatism, APRA view only (an overlay).
MOC_PP = 0.05  # documented add-on for thin data + incomplete-workout uncertainty.
moc = disposed.groupby('regime').agg(lgd_apra=('lgd_apra', 'mean')).reset_index()
moc['lgd_apra_with_moc'] = d.add_moc(moc['lgd_apra'].values, MOC_PP).round(4)
moc['lgd_apra'] = moc['lgd_apra'].round(4)
moc['moc_points'] = MOC_PP
save_csv(moc, 'outputs/tables/04_moc_overlay.csv')
moc"""),
    md("""**Reading the MoC table (P2-2).** `lgd_apra_with_moc` is simply the APRA-view
LGD plus a documented **+5 LGD-point** margin. It is deliberately small and explicit,
and it lives outside the model so it can be reviewed, dialled, or removed without
re-fitting anything. Justification: although the panel now spans a full cycle (2006-2022),
the **recent vintages' workouts are not yet fully resolved** and the thinner severity cells
still carry estimation uncertainty; the MoC is the conservative buffer the framework expects
for that (and would be dialled down further as those workouts complete)."""),
])


# ======================================================================
# 04b -- LGD validation (out-of-time, backtest, discrimination, stability)
# ======================================================================
write("04b_LGD_Validation.ipynb", [
    md("""# 04b -- LGD validation

**What this notebook does (plain English):** Part 5 of the framework (APS 113
Validation paras 1-6; APG 113 para 140's eight elements; WP14 Section IV) says an
LGD model must be **independently validated**, just like a PD model. The repo
already validates PD (notebook 03c + PSI) but had **no LGD validation** -- the most
visible gap to a credit-risk reviewer. This notebook closes it, mirroring the style
of `03c_PD_OutOfTime_Validation.ipynb`:

1. **Out-of-time / out-of-regime** -- fit the two-stage LGD on the crisis vintages
   and predict the calm one, then the reverse.
2. **Predicted-vs-realised backtest at cohort level** -- by predicted-LGD decile,
   not loan-by-loan.
3. **Discrimination** -- how well predicted severity rank-orders realised severity.
4. **Stability** -- drop each vintage in turn and watch the estimate move.
5. **Benchmarking note** -- because internal data is thin, benchmarking and
   qualitative review carry more weight than backtesting (APG 113 para 140(c); WP14).

**Headline result:** the LGD model **rank-orders** severity but, like PD, its
**level is regime-dependent** -- a model trained only on the calm 2015 book badly
**under-predicts** downturn severity, which is exactly why a downturn LGD is used."""),
    code(BOOTSTRAP),
    code("""# LGD is validated only on defaulted, DISPOSED loans (the loans with a real
# settled loss). Reuse the same two-stage model the production notebook 04 uses.
import pandas as pd
import numpy as np
from src.models import TwoStageLGD
from src.output import save_csv
base = pd.read_parquet('data/processed/analysis_base.parquet')
disposed = base[base['disposed'] & base['lgd'].notna()].copy()
print('disposed defaults available for LGD validation:', len(disposed))
print('by vintage:'); print(disposed['vintage_year'].value_counts().sort_index())"""),
    code("""# One out-of-time split: fit the LGD model on the TRAIN vintages only, then
# predict the held-out TEST vintage. No leakage -- the test year never trains.
def oot_lgd(train_years, test_years, label):
    tr = disposed[disposed['vintage_year'].isin(train_years)]
    te = disposed[disposed['vintage_year'].isin(test_years)]
    model = TwoStageLGD().fit(tr)
    pred = model.predict(te)
    return {
        'split': label,
        'n_test': len(te),
        'observed_lgd': round(float(te['lgd'].mean()), 4),
        'predicted_lgd': round(float(np.mean(pred)), 4),
        'pred_minus_obs': round(float(np.mean(pred) - te['lgd'].mean()), 4),
    }"""),
    code("""# Out-of-regime tests, the reverse 'what-if', and the genuine FORWARD holdout (R3-V3):
# fit on pre-2020 and score the never-seen 2020-2022 disposed defaults cold. The forward
# LGD test is thin (few recent workouts are fully resolved), so it is read alongside the
# cross-regime splits rather than on its own.
oot_rows = [
    oot_lgd([2007, 2008], [2015], 'A) train crisis 2007+08 -> test calm 2015'),
    oot_lgd([2015], [2007, 2008], 'B) reverse: train calm 2015 -> test crisis (under-predicts)'),
    oot_lgd(list(range(2006, 2020)), [2020, 2021, 2022], 'C) FORWARD holdout: train pre-2020 -> test 2020-22 (thin)'),
]
oot = pd.DataFrame(oot_rows)
oot"""),
    code("""# Cohort backtest: fit on everything, bucket disposed defaults by PREDICTED-LGD
# decile, and compare mean predicted vs mean realised in each bucket (cohort-level,
# never loan-by-loan -- WP14 warns point-in-time realised LGD is noisy per loan).
full = TwoStageLGD().fit(disposed)
disposed['lgd_hat'] = full.predict(disposed)
disposed['pred_decile'] = pd.qcut(disposed['lgd_hat'], 10, duplicates='drop', labels=False) + 1
backtest = disposed.groupby('pred_decile').agg(
    n=('lgd', 'size'),
    mean_predicted=('lgd_hat', 'mean'),
    mean_realised=('lgd', 'mean'),
).reset_index().round(4)
backtest['gap'] = (backtest['mean_predicted'] - backtest['mean_realised']).round(4)
backtest"""),
    code("""# R3-LGD5: realised-vs-predicted CALIBRATION by an INDEPENDENT risk segment (original-LTV
# band) -- the LGD analogue of the PD calibration test (APS 113 Att D Validation para 3;
# APG 113 para 140 element 3). Calibration is judged WITHIN business-recognised segments,
# not just by the model's own output decile, so a reviewer can see where it over/under-shoots.
disposed['ltv_band'] = pd.cut(
    pd.to_numeric(disposed['original_ltv'], errors='coerce'),
    [0, 60, 70, 80, 90, 200], labels=['<=60', '60-70', '70-80', '80-90', '90+'])
seg_cal = disposed.groupby('ltv_band', observed=True).agg(
    n=('lgd', 'size'),
    realised_lgd=('lgd', 'mean'),
    predicted_lgd=('lgd_hat', 'mean'),
).reset_index().round(4)
seg_cal['gap_pred_minus_real'] = (seg_cal['predicted_lgd'] - seg_cal['realised_lgd']).round(4)
save_csv(seg_cal, 'outputs/tables/04b_lgd_calibration_by_segment.csv')
seg_cal"""),
    code("""# R3-LGD5: external BENCHMARKING of the modelled LGD against published mortgage
# severities (qualitative anchors -- APG 113 para 140(c) / WP14 Section IV). These are
# documented reference ranges, NOT fitted, used only to sanity-check the level is plausible.
from src import definitions as d_def
down_lgd = float(disposed.loc[d_def.is_downturn_vintage(disposed['vintage_year']), 'lgd'].mean())
calm_lgd = float(disposed.loc[~d_def.is_downturn_vintage(disposed['vintage_year']), 'lgd'].mean())
bench = pd.DataFrame([
    {'source': 'This model -- downturn (GFC) realised LGD', 'lgd_ref': round(down_lgd, 3),
     'note': 'GFC 2006-09 disposed defaults'},
    {'source': 'This model -- calm/other realised LGD', 'lgd_ref': round(calm_lgd, 3),
     'note': 'non-GFC vintages'},
    {'source': 'APRA APS 113 retail-mortgage LGD floor', 'lgd_ref': 0.20,
     'note': 'regulatory minimum where own-LGD not approved (Att B)'},
    {'source': 'Published US GFC residential severities (indicative)', 'lgd_ref': '0.40-0.60',
     'note': 'distressed dispositions 2008-2011 -- reference range, not fitted'},
])
save_csv(bench, 'outputs/tables/04b_lgd_benchmarking.csv')
bench"""),
    code("""# Discrimination on the loss-only loans: does higher predicted severity line up
# with higher realised severity? Spearman rank correlation + R^2.
loss_only = disposed[disposed['lgd'] > 0.05]
spearman = float(loss_only['lgd_hat'].corr(loss_only['lgd'], method='spearman'))
ss_res = float(((loss_only['lgd'] - loss_only['lgd_hat']) ** 2).sum())
ss_tot = float(((loss_only['lgd'] - loss_only['lgd'].mean()) ** 2).sum())
r2 = 1 - ss_res / ss_tot
print(f'Spearman(predicted, realised) on loss-only loans: {spearman:.3f}')
print(f'R^2 of predicted vs realised severity            : {r2:.3f}')"""),
    code("""# Stability: re-fit dropping each vintage in turn and see how the overall mean
# predicted LGD moves -- the 'stability analysis' WP14 asks for.
all_pred = float(TwoStageLGD().fit(disposed).predict(disposed).mean())
stab_rows = [{'configuration': 'all vintages', 'mean_predicted_lgd': round(all_pred, 4),
              'shift_vs_all': 0.0}]
for y in sorted(disposed['vintage_year'].unique()):
    sub = disposed[disposed['vintage_year'] != y]
    mp = float(TwoStageLGD().fit(sub).predict(sub).mean())
    stab_rows.append({'configuration': f'drop {y}', 'mean_predicted_lgd': round(mp, 4),
                      'shift_vs_all': round(mp - all_pred, 4)})
stability = pd.DataFrame(stab_rows)
stability"""),
    code("""# Combine the headline validation results into one saved table.
val = pd.concat([
    oot.assign(section='out_of_time').rename(columns={'split': 'detail'})[
        ['section', 'detail', 'n_test', 'observed_lgd', 'predicted_lgd', 'pred_minus_obs']],
    stability.assign(section='stability', n_test=np.nan).rename(
        columns={'configuration': 'detail', 'mean_predicted_lgd': 'predicted_lgd'})[
        ['section', 'detail', 'n_test', 'predicted_lgd']],
    pd.DataFrame([{'section': 'discrimination', 'detail': 'spearman / R2 on loss-only',
                   'observed_lgd': round(spearman, 4), 'predicted_lgd': round(r2, 4)}]),
], ignore_index=True)
save_csv(val, 'outputs/tables/04b_lgd_validation.csv')
val"""),
    md("""## Interpretation (plain English)

- **Out-of-time / out-of-regime.** A model trained on the **crisis** books
  **over-predicts** the calm 2015 book (predicted ~43% vs realised ~25%) -- i.e. it is
  *conservative* out-of-regime, which is the safe direction. The **reverse** is the
  dangerous one: training only on calm 2015 and predicting the crisis **under-predicts**
  downturn severity badly (predicted ~21% vs realised ~57%). A model built only in good
  times is blind to a downturn; this is the headline out-of-time finding and the reason
  the downturn LGD (notebook 04 / 06) is used for the conservative estimate.
- **Forward holdout (R3-V3).** Fitting on **pre-2020** and scoring the never-seen
  **2020-2022** loans cold is the honest production test. For LGD it is deliberately
  read with caution -- only a handful of recent defaults are fully worked out -- so the
  cross-regime splits and the cohort backtest carry the weight; the PD forward holdout
  (notebook 03c, split D) is the stronger of the two because it has far more test loans.
- **Cohort backtest.** Read by predicted-LGD decile, mean predicted and mean realised
  track in the same direction -- the model is **calibrated in rank**. Per the WP14
  caveat, this is a cohort comparison; a single point-in-time realised LGD must **not**
  be compared directly to a long-run estimate loan-by-loan.
- **Segment calibration (R3-LGD5).** Realised vs predicted LGD is also compared **within
  original-LTV bands** (an independent risk segment, not the model's own output): the
  `gap` column shows the model tracks realised severity across LTV without a systematic
  bias -- the LGD analogue of the PD calibration test (APS 113 Att D Validation para 3).
- **Discrimination.** A positive Spearman correlation between predicted and realised
  severity on the loss-only loans confirms the model **rank-orders** loss size, though
  mortgage LGD is inherently noisy so the R^2 is modest -- normal for severity models.
- **Stability.** Dropping any single vintage moves the overall mean predicted LGD only
  modestly, **except** when the crisis volume is removed, which pulls the estimate down
  -- consistent with severity being driven by the downturn cohorts.

## Benchmarking note (APG 113 para 140(c); WP14)

The internal sample is now far deeper -- **17 vintages and ~13k disposed defaults** across
a full cycle -- so backtesting carries real weight, and the `04b_lgd_benchmarking.csv` table
anchors the level against external references. The modelled **downturn (GFC) severity ~56%**
sits squarely within **published US agency mortgage loss severities** for 2008-2011 (broadly
~40-60% on distressed dispositions), and the **calm/other ~34%** and the **APRA 20% floor**
bracket it sensibly. Where the sample is still thin -- the **2020-2022 workouts not yet fully
resolved** -- benchmarking and the incomplete-workout sensitivity (notebook 04) carry more
weight than the raw recent realised numbers. In production this external comparison would be
refreshed annually alongside an expert-judgement overlay."""),
])


# ======================================================================
# 05 -- EAD
# ======================================================================
write("05_ead.ipynb", [
    md("""# 05 -- EAD (exposure at default)

**What this notebook does (plain English):** **EAD** is simply how much money was
still owed on a loan at the moment it defaulted -- the amount truly at risk. For
a mortgage this is just the outstanding balance, because a mortgage is fully
drawn on day one. (Contrast a credit card, which has an undrawn limit the
borrower can run up before defaulting -- that needs an extra "credit conversion
factor"; a mortgage does not.)

**Headline result:** average exposure at default is around **$190k**, and it is
broadly similar across vintages -- EAD is a balance, not a risk gauge."""),
    code(BOOTSTRAP),
    code("""# Load the base table and keep defaulted loans (EAD is the balance at default).
import pandas as pd
from src.output import save_csv
base = pd.read_parquet('data/processed/analysis_base.parquet')
defaulted = base[base['ever_default']].copy()
print('defaulted loans:', len(defaulted))"""),
    code("""# EAD summary: average and spread of exposure at default, by vintage.
ead_summary = defaulted.groupby('vintage_year').agg(
    defaults=('ead', 'size'),
    mean_ead=('ead', 'mean'),
    median_ead=('ead', 'median'),
    p95_ead=('ead', lambda s: s.quantile(0.95)),
    total_ead=('ead', 'sum'),
).reset_index().round(2)
save_csv(ead_summary, 'outputs/tables/05_ead_summary.csv')
ead_summary"""),
    md("""**Why there is no CCF here:** A credit conversion factor models how much of
an *undrawn* limit a borrower draws before defaulting. A term mortgage has no
undrawn limit -- the full principal is advanced at closing and only ever
amortises down -- so EAD is just the outstanding balance and **no CCF/drawdown
modelling applies**. Knowing CCF belongs to *revolving* products (like a credit
card) and deliberately *not* using it here is the point, not a gap."""),
])


# ======================================================================
# 06 -- Expected Loss
# ======================================================================
write("06_expected_loss.ipynb", [
    md("""# 06 -- Expected Loss (EL = PD x LGD x EAD)

**What this notebook does (plain English):** Brings the three pieces together.
**Expected Loss** is the average loss a lender should budget for:

> **Expected Loss = chance of default (PD) x loss if it defaults (LGD) x amount
> owed (EAD)**

We score every loan, total it into a portfolio number, and sort loans into the
accounting **IFRS 9 / AASB 9 stages** (1 = healthy, 2 = deteriorating, 3 =
defaulted). We also walk through the full sum for one example loan.

**Headline result:** a single portfolio Expected Loss figure, dominated by the
Stage 3 (already-defaulted) loans and by the crisis vintages."""),
    code(BOOTSTRAP),
    code("""# Load the base table and build the three components for EVERY loan.
import pandas as pd
import numpy as np
from src import models
from src.output import save_csv
base = pd.read_parquet('data/processed/analysis_base.parquet').copy()"""),
    code("""# One-year PD for every loan (logistic model fit on the whole book). pd_hat is the
# raw, continuous 12-month model score (PD-1/PD-2), kept for ranking/comparison.
from src import definitions as d
pd_model, pd_cols = models.fit_pd(base)
base['pd_hat'] = d.apply_pd_floor(models.predict_pd(pd_model, pd_cols, base), floor=0.0005)
# PDR2-1: the PD that feeds EL must be the SAME PD as capital (EL framework Part 5.1) --
# the CALIBRATED grade PD (long-run + risk-sensitive MoC + ratchet + floor) from 03b.
grade_pd_map = pd.read_csv('outputs/tables/03f_loan_grade_pd.csv')[
    ['loan_sequence_number', 'grade', 'grade_pd_longrun', 'grade_pd_final']]
base['loan_sequence_number'] = base['loan_sequence_number'].astype(str)
grade_pd_map['loan_sequence_number'] = grade_pd_map['loan_sequence_number'].astype(str)
base = base.merge(grade_pd_map, on='loan_sequence_number', how='left')
base['pd_capital'] = base['grade_pd_final'].fillna(base['pd_hat'])
base['pd_longrun'] = base['grade_pd_longrun'].fillna(base['pd_hat'])"""),
    code("""# LGD for every loan (two-stage model trained on disposed defaults).
disposed = base[base['disposed'] & base['lgd'].notna()]
lgd_model = models.TwoStageLGD().fit(disposed)
base['lgd_hat'] = lgd_model.predict(base)"""),
    code("""# EAD for every loan: balance at default if it defaulted, else the original
# loan amount as the exposure proxy for a still-performing loan.
base['ead_loan'] = np.where(base['ever_default'], base['ead'], base['original_upb'])
# Expected loss per loan = PD x LGD x EAD, using the CALIBRATED capital PD (PDR2-1).
base['expected_loss'] = base['pd_capital'] * base['lgd_hat'] * base['ead_loan']"""),
    md("""### The EL/capital PD reconcile (PDR2-1)

EL framework Part 5.1 requires Expected Loss to use the **same PD as capital/RWA**. Earlier
the dollar loss used only the raw model score with the 5 bps floor, so the long-run
calibration and the margin of conservatism never reached EL. Now `expected_loss` uses
**`pd_capital`** -- the calibrated grade PD (long-run average + risk-sensitive MoC + ratchet
+ floor) exported from notebook 03b -- so EL and the master-scale/capital PD are one and the
same number. The raw `pd_hat` is retained only for ranking and the comparison below."""),
    code("""# PDR2-1 impact: EL on three PD bases. The honest comparison is LIKE-FOR-LIKE at the
# pooled grade level (rows 2 vs 3): the MoC + ratchet flow straight through, so EL on the
# calibrated capital PD is the HIGHER (safer) one. Row 1 (the raw continuous score) is
# shown for context -- see the note on why pooling changes the level.
el_raw = float((base['pd_hat'] * base['lgd_hat'] * base['ead_loan']).sum())
el_lr = float((base['pd_longrun'] * base['lgd_hat'] * base['ead_loan']).sum())
el_cap = float(base['expected_loss'].sum())
pd_compare = pd.DataFrame([
    {'pd_basis': '1. raw continuous model PD (pre-calibration, floored)',
     'mean_pd': round(float(base['pd_hat'].mean()), 5), 'portfolio_EL': round(el_raw, 0)},
    {'pd_basis': '2. pooled grade PD, long-run only (no MoC)',
     'mean_pd': round(float(base['pd_longrun'].mean()), 5), 'portfolio_EL': round(el_lr, 0)},
    {'pd_basis': '3. calibrated capital PD (long-run + MoC + ratchet + floor)',
     'mean_pd': round(float(base['pd_capital'].mean()), 5), 'portfolio_EL': round(el_cap, 0)},
])
pd_compare['EL_vs_longrun_x'] = (pd_compare['portfolio_EL'] / el_lr).round(3)
save_csv(pd_compare, 'outputs/tables/06_pd_basis_el_compare.csv')
print('MoC flows through (EL capital >= EL long-run, like-for-like):', el_cap >= el_lr)
pd_compare"""),
    md("""**Reading the PD-basis comparison (PDR2-1).** On the **like-for-like pooled basis**
(rows 2 vs 3) the margin of conservatism and the grade-H ratchet flow straight through:
the calibrated **capital PD raises EL** above the bare long-run calibration -- the
conservatism now reaches the dollar loss, which was the whole gap this task closes.

Row 1 (the raw continuous score) actually sits a little **above** the calibrated capital
EL. That is **not** missing conservatism -- the mean calibrated PD is higher -- it is a
**pooling effect**: collapsing 150,000 continuous scores into 8 grade PDs removes the
within-grade correlation between the model score and exposure (the riskiest, largest loans
inside a grade no longer carry an individually higher PD). Capital frameworks pool exposures
into grades/pools by design (the grade PD *is* the regulatory PD), so the calibrated capital
EL is the correct figure to report, and it now reconciles exactly with the master scale."""),
    code("""# IFRS 9 / AASB 9 staging: 3 = defaulted (credit-impaired), 2 = significant
# increase in risk (ever 60+ days late but not defaulted), 1 = performing.
stage2 = (~base['ever_default']) & (base['max_delinq_status'].fillna(0) >= 2)
base['ifrs9_stage'] = np.where(base['ever_default'], 3, np.where(stage2, 2, 1))
# pd_capital is a genuine 12-MONTH PD, which is exactly the Stage 1 (12-month ECL)
# input -- so Stage 1 reported EL is the 12-month EL directly (no ad-hoc 0.25 factor
# any more). Stages 2 & 3 need LIFETIME ECL; with only a one-year PD modelled here we
# scale by a transparent multi-year horizon factor as a lifetime proxy (PDR2-7: a
# production model would estimate a lifetime PD term structure directly).
LIFETIME_HORIZON = 4
base['el_reported'] = np.where(base['ifrs9_stage'] == 1, base['expected_loss'],
                               base['expected_loss'] * LIFETIME_HORIZON)"""),
    code("""# Portfolio Expected Loss summary by IFRS 9 stage (the saved result). With a
# one-year PD, `expected_loss_12m` is a 12-MONTH EL; `reported_expected_loss` applies
# the IFRS 9 staging (12-month for Stage 1, lifetime proxy for Stages 2 & 3).
el_summary = base.groupby('ifrs9_stage').agg(
    loans=('loan_sequence_number', 'size'),
    avg_pd=('pd_capital', 'mean'),
    avg_lgd=('lgd_hat', 'mean'),
    total_ead=('ead_loan', 'sum'),
    expected_loss_12m=('expected_loss', 'sum'),
    reported_expected_loss=('el_reported', 'sum'),
).reset_index().round(2)
save_csv(el_summary, 'outputs/tables/06_expected_loss.csv')
el_summary"""),
    code("""# Portfolio Expected Loss summary BY RATING GRADE (A safest -> H riskiest) plus a
# whole-portfolio total row -- the transparent EL build a reviewer asked for: how many
# loans, how much exposure (EAD), the average PD and LGD, and the 12-month dollar EL in
# each grade. EL = avg_pd x avg_lgd x total_ead reconciles row by row.
by_grade = base[base['grade'].notna()].groupby('grade').agg(
    loans=('loan_sequence_number', 'size'),
    total_ead=('ead_loan', 'sum'),
    avg_pd=('pd_capital', 'mean'),
    avg_lgd=('lgd_hat', 'mean'),
    total_expected_loss_12m=('expected_loss', 'sum'),
).reset_index()
portfolio = pd.DataFrame([{
    'grade': 'PORTFOLIO', 'loans': len(base), 'total_ead': base['ead_loan'].sum(),
    'avg_pd': base['pd_capital'].mean(), 'avg_lgd': base['lgd_hat'].mean(),
    'total_expected_loss_12m': base['expected_loss'].sum(),
}])
el_by_grade = pd.concat([by_grade, portfolio], ignore_index=True)
el_by_grade['el_rate_bps'] = (el_by_grade['total_expected_loss_12m'] / el_by_grade['total_ead'] * 1e4).round(1)
for c in ['avg_pd', 'avg_lgd']:
    el_by_grade[c] = el_by_grade[c].round(4)
for c in ['total_ead', 'total_expected_loss_12m']:
    el_by_grade[c] = el_by_grade[c].round(0)
save_csv(el_by_grade, 'outputs/tables/06_el_summary_by_grade.csv')
el_by_grade"""),
    md("""**Lifetime PD note (PDR2-7).** `expected_loss_12m` is a genuine **12-month** EL,
the correct IFRS 9 **Stage 1** input. The `reported_expected_loss` column then needs
**lifetime** ECL for Stages 2 and 3, and here we approximate it by scaling the 12-month
figure by a flat multi-year **horizon factor** (`LIFETIME_HORIZON = 4`). This is a
deliberate, named **proxy**: a production model would instead estimate a full **lifetime
PD term structure** (cumulative one-year PDs across the remaining life, conditioned on
age and macro path) rather than a single scalar. The proxy is kept here only to show the
staged-ECL shape end to end."""),
    md("""### Downturn-LGD variant of Expected Loss (P2-3)

Notebook 04 showed loss severity is strongly **cyclical** (~25% calm vs ~57% crisis).
APS 113 Att D LGD paras 4-5 say that where severity is cyclical, the LGD *estimate*
must reflect **downturn** conditions, not the through-the-cycle average. So alongside
the baseline EL we compute a **downturn-LGD variant**, lifting every loan's LGD to at
least the crisis-regime realised severity. This is the conservative figure the
framework expects a capital/EL report to show."""),
    code("""# P2-3: downturn-LGD variant of EL. Lift each loan's LGD to >= the crisis-regime
# realised severity (the observed downturn LGD), then recompute Expected Loss.
# Downturn population via the documented classifier (R3-C2): GFC housing-crisis vintages.
downturn_lgd = float(base.loc[base['disposed'] & d.is_downturn_vintage(base['vintage_year']), 'lgd'].mean())
base['lgd_downturn'] = np.maximum(base['lgd_hat'], downturn_lgd)
base['expected_loss_downturn'] = base['pd_capital'] * base['lgd_downturn'] * base['ead_loan']
el_variant = pd.DataFrame([
    {'view': 'through-the-cycle (baseline)', 'lgd_basis': 'modelled lgd_hat',
     'total_expected_loss': round(float(base['expected_loss'].sum()), 0)},
    {'view': 'downturn LGD (APS 113 Att D LGD 4-5)', 'lgd_basis': f'max(lgd_hat, {downturn_lgd:.3f})',
     'total_expected_loss': round(float(base['expected_loss_downturn'].sum()), 0)},
])
el_variant['uplift_x'] = (el_variant['total_expected_loss'] /
                          el_variant['total_expected_loss'].iloc[0]).round(2)
save_csv(el_variant, 'outputs/tables/06_el_downturn_variant.csv')
el_variant"""),
    md("""### Best estimate of EL for already-defaulted (Stage 3) loans (P2-4)

APS 113 Att D para 11 / Part 4.3: for loans **already in default** (Stage 3), you must
form a **best estimate of expected loss for that loan given current conditions** --
mechanically applying the model's average LGD is "not acceptable". We replace the
mechanical PD x LGD with: the loan's **realised** LGD where its workout is materially
complete (disposed), otherwise the **segment downturn LGD**; since the loan is already
in default its PD is 1, so EL = best-estimate LGD x EAD."""),
    code("""# P2-4: best-estimate EL for Stage 3 (already-defaulted) loans.
stage3 = base['ifrs9_stage'] == 3
best_lgd = np.where(base['disposed'] & base['lgd'].notna(), base['lgd'], downturn_lgd)
base['el_stage3_bestestimate'] = np.where(stage3, best_lgd * base['ead_loan'], np.nan)
s3 = pd.DataFrame([{
    'stage3_loans': int(stage3.sum()),
    'el_mechanical_pd_x_lgd': round(float(base.loc[stage3, 'expected_loss'].sum()), 0),
    'el_best_estimate': round(float(np.nansum(base['el_stage3_bestestimate'])), 0),
}])
s3['ratio_best_vs_mechanical'] = round(s3['el_best_estimate'] / s3['el_mechanical_pd_x_lgd'], 2)
save_csv(s3, 'outputs/tables/06_stage3_best_estimate.csv')
s3"""),
    md("""**Reading the Stage 3 table (P2-4).** The mechanical column applies the model
PD x LGD even to loans that have *already* defaulted (so its PD < 1 understates the
loss); the best-estimate column uses each defaulted loan's realised loss where the
workout is complete and the downturn LGD otherwise, with PD = 1. The best estimate is
materially larger -- which is the point: a defaulted loan's expected loss should be
built from its own resolution, not a portfolio-average model output."""),
    code("""# Worked example: show PD x LGD x EAD = EL for a single representative loan,
# using the calibrated capital PD (the same PD the portfolio EL is built on).
ex = base.sort_values('expected_loss', ascending=False).iloc[100]
print('Worked example loan:', ex['loan_sequence_number'])
print(f"  PD  (calibrated 1yr)    = {ex['pd_capital']:.3f}")
print(f"  LGD (loss if default)   = {ex['lgd_hat']:.3f}")
print(f"  EAD (amount owed)       = ${ex['ead_loan']:,.0f}")
print(f"  Expected Loss = {ex['pd_capital']:.3f} x {ex['lgd_hat']:.3f} x ${ex['ead_loan']:,.0f} = ${ex['expected_loss']:,.0f}")"""),
    md("""**Reading the table:** Stage 3 holds the already-defaulted loans and
carries most of the loss; Stage 1 is the large healthy book on a 12-month view.
The worked example shows the headline equation end-to-end for one loan."""),
])


# ======================================================================
# 07 -- Stress testing
# ======================================================================
write("07_stress_testing.ipynb", [
    md("""# 07 -- Stress testing (downturn scenario)

**What this notebook does (plain English):** Asks the key risk question: *how
much worse would losses get in a recession?* Instead of guessing, we use a real
one. The 2007/2008 vintages **lived through** the global financial crisis, so the
jump from the calm 2015 book to the crisis books gives an **observed** downturn
multiplier for both PD and LGD. We apply that downturn to the calm-year portfolio
and read off the increase in Expected Loss.

**Headline result:** under the crisis-calibrated downturn, portfolio Expected
Loss rises several-fold versus the calm baseline -- driven by PD and LGD getting
worse *at the same time*.

**Consistency (PDR2-5):** the stress now runs on the **same one-year / calibrated
capital PD** as the rest of the model (notebooks 03b/06), with **two named scenarios**
(a mild recession and the severe observed crisis), the **no-diversification** assumption,
and a contingency + reverse-stress note -- aligning it to the Stress framework (Basel
CRE36.51; APS 220 paras 70-76)."""),
    code(BOOTSTRAP),
    code("""# Load the base table + the calibrated capital PD (03f), and split calm vs downturn.
import pandas as pd
import numpy as np
from src.output import save_csv
base = pd.read_parquet('data/processed/analysis_base.parquet').copy()
base['loan_sequence_number'] = base['loan_sequence_number'].astype(str)
cap = pd.read_csv('outputs/tables/03f_loan_grade_pd.csv')[['loan_sequence_number', 'grade_pd_final']]
cap['loan_sequence_number'] = cap['loan_sequence_number'].astype(str)
base = base.merge(cap, on='loan_sequence_number', how='left')
# Regime via the documented classifier (R3-C2): calm reference book, the GFC
# severity downturn, and COVID-2020 as a separate (non-housing) scenario.
from src import definitions as d
calm = base[base['vintage_year'] == d.CALM_REFERENCE_VINTAGE]
downturn = base[d.is_downturn_vintage(base['vintage_year'])]
covid = base[d.is_covid_vintage(base['vintage_year'])]"""),
    code("""# Observed SEVERE multipliers on the ONE-YEAR PD (PDR2-5: same target as PD/EL) and
# on realised LGD; a MILD recession is framed as a documented fraction of that path
# (Basel CRE36.51's two-consecutive-quarters-of-zero-growth example).
pd_calm_1yr, pd_down_1yr = calm['default_within_12m'].mean(), downturn['default_within_12m'].mean()
lgd_calm = calm.loc[calm['disposed'], 'lgd'].mean()
lgd_down = downturn.loc[downturn['disposed'], 'lgd'].mean()
pd_mult_sev, lgd_mult_sev = pd_down_1yr / pd_calm_1yr, lgd_down / lgd_calm
MILD_FRACTION = 0.33  # mild recession ~ one third of the GFC severity (documented)
pd_mult_mild = 1 + (pd_mult_sev - 1) * MILD_FRACTION
lgd_mult_mild = 1 + (lgd_mult_sev - 1) * MILD_FRACTION
# COVID-2020 (R3-STR2): a SECOND, observed downturn -- high default but much milder
# loss severity (house prices rose, forbearance), so its LGD multiplier is near 1.
# Empirically grounded in the 2020 vintage, it shows a different downturn shape.
pd_mult_covid = covid['default_within_12m'].mean() / pd_calm_1yr
lgd_mult_covid = covid.loc[covid['disposed'], 'lgd'].mean() / lgd_calm
print(f'severe(GFC): PD x{pd_mult_sev:.2f}  LGD x{lgd_mult_sev:.2f}   mild: PD x{pd_mult_mild:.2f}  LGD x{lgd_mult_mild:.2f}')
print(f'COVID-2020 : PD x{pd_mult_covid:.2f}  LGD x{lgd_mult_covid:.2f}  (high default, mild severity)')"""),
    code("""# Macro context per scenario (documented assumptions; production would pull these live
# from FRED -- unemployment UNRATE, house prices CSUSHPINSA).
macro = pd.DataFrame([
    {'scenario': 'baseline (2015 calm)', 'unemployment_pct': 5.3, 'hpi_change_pct': 5.0},
    {'scenario': 'mild recession (CRE36.51: 2 quarters ~0 growth)', 'unemployment_pct': 7.0, 'hpi_change_pct': -8.0},
    {'scenario': 'severely adverse (observed GFC 2008-09)', 'unemployment_pct': 10.0, 'hpi_change_pct': -30.0},
    {'scenario': 'COVID-2020 (observed: income shock, HPI up)', 'unemployment_pct': 8.1, 'hpi_change_pct': 10.0},
])
macro"""),
    code("""# Stress the calm-2015 book: baseline PD = the CALIBRATED CAPITAL PD (matches EL),
# then apply each scenario's PD and LGD multipliers TOGETHER (no diversification, APG
# 113 para 92 -- the shocks are not allowed to offset; they stack multiplicatively).
ead_calm = np.where(calm['ever_default'], calm['ead'], calm['original_upb'])
base_pd = float(calm['grade_pd_final'].fillna(calm['default_within_12m'].mean()).mean())  # calibrated capital PD
base_lgd = float(lgd_calm)
baseline_el = (base_pd * base_lgd * ead_calm).sum()

def stressed_el(pd_m, lgd_m):
    return (min(base_pd * pd_m, 1.0) * min(base_lgd * lgd_m, 1.0) * ead_calm).sum()

rows = []
for name, pm, lm in [('baseline', 1.0, 1.0),
                     ('mild recession', pd_mult_mild, lgd_mult_mild),
                     ('severely adverse', pd_mult_sev, lgd_mult_sev),
                     ('COVID-2020 (observed)', pd_mult_covid, lgd_mult_covid)]:
    el = stressed_el(pm, lm)
    rows.append({'scenario': name, 'pd_mult': round(pm, 2), 'lgd_mult': round(lm, 2),
                 'stressed_pd': round(min(base_pd * pm, 1.0), 4),
                 'stressed_lgd': round(min(base_lgd * lm, 1.0), 4),
                 'expected_loss': round(el, 0), 'EL_uplift_x': round(el / baseline_el, 2)})
stress_tbl = pd.DataFrame(rows)
save_csv(stress_tbl, 'outputs/tables/07_stress_test.csv')
stress_tbl"""),
    code("""# Reverse stress (APS 220): what COMBINED PD x LGD multiplier drives EL to a chosen
# severity multiple? Since EL scales with the product of the two shocks, it is that
# product. Here: the shock that would QUADRUPLE baseline Expected Loss.
TARGET_UPLIFT = 4.0
print(f'Reverse stress: EL reaches {TARGET_UPLIFT:.0f}x baseline at a combined '
      f'PD x LGD multiplier of ~{TARGET_UPLIFT:.1f}x '
      f'(e.g. PD x{TARGET_UPLIFT**0.5:.1f} and LGD x{TARGET_UPLIFT**0.5:.1f} together).')
print(f'For reference the severe observed scenario already reaches '
      f'{stress_tbl.loc[stress_tbl.scenario==\"severely adverse\",\"EL_uplift_x\"].iloc[0]:.1f}x.')"""),
    md("""**Reading the table (PDR2-5).** We take the calm 2015 portfolio at its **calibrated
capital PD** (the same PD behind notebook 06's EL) and push PD and LGD up by each
scenario's multipliers. Two named scenarios are shown: a **mild recession** (framed on
Basel CRE36.51's two-quarters-of-near-zero-growth example, ~one third of the GFC path) and
the **severely adverse** observed 2008-09 crisis. Because the shocks **stack
multiplicatively with no diversification offset** (APG 113 para 92), Expected Loss rises
far more than either driver alone -- the core lesson of downturn stress testing.

**Management actions / contingency (APS 220 para 74).** A breach of the severe scenario
would trigger documented management actions -- tightening new-origination cut-offs and
high-LVR lending, raising provisions and the capital buffer, and re-pricing -- which are
*not* modelled here but would form the contingency plan in production.

**Reverse stress (APS 220).** The cell above frames the inverse question -- the combined
shock that drives EL to a chosen multiple (here 4x) -- the starting point for identifying
the scenarios that would threaten viability.

**Independent validation (APS 220 para 76).** Like the PD and LGD models, this stress
framework would require **independent validation** of its scenarios, multipliers and
assumptions before use; here development and validation are separated only by notebook.

**Extension -- climate scenario (sketch, not built):** the same machinery extends to
physical climate risk -- a flood/wildfire shock lowers house prices in exposed postcodes,
raising **LGD** (smaller recovery) and, via negative equity, **PD**. One would overlay a
hazard map on the property postcode, apply a region-specific house-price haircut, and
re-run this exact PD/LGD/EL stress."""),
])


# ======================================================================
# 08 -- Documentation, validation & monitoring
# ======================================================================
write("08_documentation_and_monitoring.ipynb", [
    md("""# 08 -- Documentation, validation & monitoring pack

**What this notebook does (plain English):** The write-up a model-validation or
consulting team would hand over: what we built, how, what the results were, the
limitations, and how we'd **monitor** the models once live. It includes a
**stability check (PSI)** -- a standard early-warning gauge that flags when the
loans coming through the door no longer look like the ones a model was built on.

**Headline result:** the population shifts materially between the calm and crisis
books (high PSI), exactly the kind of drift monitoring is designed to catch."""),
    md("""## Model development summary

**Objective.** Quantify mortgage credit risk end-to-end -- PD, LGD, EAD,
Expected Loss and a downturn stress test -- on the Freddie Mac Single-Family
Loan-Level Dataset.

**Data.** 50,000-loan samples for **17 origination years (2006-2022)** -- spanning the
housing boom, the GFC, the recovery, the long expansion and COVID-2020, observed through
2025-09; origination characteristics joined to monthly performance and collapsed to one
row per loan (~850k loans). Raw data is not redistributed in this repo.

**Methodology.**
- *Default* = first month at 180+ days past due, or a credit-event zero-balance
  code (third-party sale, short sale/charge-off, REO disposition, note sale).
- *PD* = logistic regression on origination features (interpretable scorecard), on a
  **one-year** default target, calibrated to a count-weighted long-run average per grade,
  with a formal calibration test, a margin of conservatism, and the 5 bps floor
  -- see the PD framework-alignment notes below.
- *LGD* = two-stage model (P(loss) x severity) on **realised** losses from
  defaulted, disposed loans; reconciled to Freddie Mac's own loss field (corr ~0.99).
  Extended to **economic (discounted) loss**, an **APRA capital view** (MI excluded,
  20% high-LVR reduction, 20% floor), a margin of conservatism, a downturn LGD, and an
  **independent LGD validation** (notebook 04b) -- see the framework-alignment notes below.
- *EAD* = outstanding balance at default (no CCF -- a term loan has no undrawn limit).
- *EL* = PD x LGD x EAD, staged under IFRS 9 / AASB 9.
- *Stress* = downturn multipliers observed in the **GFC (2006-09)** vintages, plus a
  separate **observed COVID-2020** scenario (high default, mild severity).

**Results.** Across the cycle, GFC origination years run ~7-14% default with ~54-58% LGD
versus ~2-3% / ~20-35% in the calm expansion; COVID-2020 shows the **lowest** default
(~1.2%, forbearance-suppressed) at moderate severity. Long-run grade PDs calibrate to a
**count-weighted full-cycle average**, the downturn LGD is ~56% (GFC) vs ~34% (non-GFC),
portfolio 12-month EL is ~$236m rising to ~$323m under IFRS 9 lifetime staging, and the
severe stress lifts EL ~13x (COVID-shape ~4.7x). PD model AUC ~0.75-0.80.

**Limitations.** Portfolio demonstration, not a regulatory-capital model; US
agency mortgages, not an APRA IRB portfolio; 50k-loan samples, illustrative
calibration; macro stress is scenario-based, not a fitted macroeconomic model.

**Governance / monitoring.** Track discrimination (AUC/Gini/KS), calibration, and
**population stability (PSI)** over time; re-fit on a trigger; maintain model
documentation and an owner for each model."""),
    code(BOOTSTRAP),
    code("""# Load the base table and re-fit the PD model to get a score to monitor.
import pandas as pd
from src import models, metrics
from src.output import save_csv
base = pd.read_parquet('data/processed/analysis_base.parquet')
pd_model, pd_cols = models.fit_pd(base)
base = base.copy()
base['pd_hat'] = models.predict_pd(pd_model, pd_cols, base)"""),
    code("""# PSI: compare the calm reference book (expected) to the GFC crisis books (actual)
# on both a raw driver (credit score) and the model output (PD). Regime via the
# documented classifier (R3-C2).
from src import definitions as d
calm = base[base['vintage_year'] == d.CALM_REFERENCE_VINTAGE]
crisis = base[d.is_downturn_vintage(base['vintage_year'])]
psi_tbl = pd.DataFrame([
    {'feature': 'credit_score', 'psi_calm_vs_crisis': round(metrics.psi(calm['credit_score'], crisis['credit_score']), 4)},
    {'feature': 'pd_hat', 'psi_calm_vs_crisis': round(metrics.psi(calm['pd_hat'], crisis['pd_hat']), 4)},
])
psi_tbl['interpretation'] = psi_tbl['psi_calm_vs_crisis'].apply(
    lambda v: 'stable (<0.10)' if v < 0.10 else ('watch (0.10-0.25)' if v < 0.25 else 'material shift (>0.25)'))
save_csv(psi_tbl, 'outputs/tables/08_monitoring_psi.csv')
psi_tbl"""),
    md("""**Reading the table:** PSI above 0.25 signals the incoming population has
shifted materially from the reference book -- here, the crisis vintages look
very different from the calm one, which would trigger a model review in
production. This is the kind of monitoring that keeps a deployed model honest."""),
    md("""## Framework alignment notes (APS 113 / APG 113 / Basel / WP14)

These notes record where the build sits against the regulatory framework. The LGD
work (notebooks 01, 04, 04b, 06) now implements economic-loss discounting, the
APRA-view overlays (MI exclusion, 20% reduction, 20% floor), a margin of
conservatism, downturn LGD, a Stage 3 best estimate, and an independent LGD
validation. The remaining items below are **documentation choices**, stated
explicitly so a reviewer can see they were considered.

**Default definition (Step 2 / APS 113 Att D para 5).** This project defines default
at **180 days past due** (status 6+) or a credit-event zero-balance code. 180 DPD is a
common *mortgage* convention and is what the Freddie Mac data cleanly supports. The
APS 220 / Basel reference point is **90 DPD**; we treat the 180-DPD choice as a "broad
equivalence" adjustment and note that a 90-DPD definition would classify more loans as
defaulted earlier (higher PD, generally lower average LGD as more cures are captured).
A **90-DPD sensitivity is already built** (`default_within_12m_90dpd`, from the same
delinquency field) and quantified per grade in notebook 03b -- the one-year rate roughly
doubles under the broader trigger while the rank-ordering holds.

**Cure rules (Step 2).** A loan is counted as defaulted if it *ever* reached 180+ DPD
within its observed history; we do **not** net out subsequent cures in the default flag
(an observed-to-date definition). The resolution-bias analysis in notebook 04 (P2-1)
shows the practical effect: a large share of 180-DPD "defaults" cure or prepay with
little loss, which is why the cure-aware best estimate sits well below the disposed-only
LGD. A production model would add an explicit cure/re-default (probation) window.

**Borrower/collateral correlation (Part 4.1).** In mortgages, falling house prices both
*trigger* defaults (negative equity) and *deepen* losses (smaller recovery on sale), so
PD and LGD are positively correlated in a downturn. We do not model that correlation
parametrically; instead the **downturn LGD** (notebook 04/06) and the joint PD-and-LGD
stress (notebook 07) are the conservative treatment of it -- both drivers worsen together.

**Observation period (Step 7).** The panel now spans **17 origination vintages (2006-2022)**
-- boom-peak, the GFC, the recovery, the long expansion and COVID-2020 -- a **full economic
cycle with two distinct downturns**, observed through 2025-09. This **comfortably exceeds the
framework's 5-year minimum** for PD and retail LGD (APS 113 Att D PD para 4 / LGD para; CRE36.88)
and spans good and bad years as the long-run calibration requires. The **margin of conservatism**
(P2-2) is retained but is now **sized by the data** -- per-grade error bars shrink as observations
accumulate, so the MoC is materially smaller than on the old 3-vintage window while still erring
conservative (CRE36.67). Benchmarking/qualitative review still support the thinnest recent-vintage
LGD cells, whose workouts are not yet fully resolved.

**Segmentation (Step 1).** Current severity drivers are LTV, credit score, loan size
(UPB) and the downturn indicator. In production the LGD segmentation would extend to
**loan purpose, occupancy, and with/without-LMI** (and likely geography/house-price
region), each of which plausibly shifts recovery; they are noted here as the natural
next segments rather than fitted, given the sample size."""),
    md("""## PD framework alignment notes (APS 113 / APG 113 / Basel / WP14)

The PD work (notebooks 01, 03, 03b, 03c) now uses a **one-year default target**
(`default_within_12m`), calibrates each grade to a **count-weighted long-run average**
across vintages, applies a **formal calibration test**, a **risk-sensitive margin of
conservatism**, a **revise-upward ratchet** on under-predicting grades, and the **5 bps
PD floor**. Crucially, this calibrated grade PD now **flows through to Expected Loss
(notebook 06) and the stress test (notebook 07)**, so EL/capital and the stress layer use
the **same PD** as the master scale (EL Part 5.1) -- conservatism reaches the dollar loss,
not just the rating. The items below are the required **documentation** elements.

**Rating philosophy (Step 3 / APG 113 para 73).** This is a **point-in-time-leaning,
through-the-door** scorecard: it uses **origination features only** (credit score, LTV,
DTI, purpose, occupancy, channel) with **no behavioural inputs**, so a loan's grade is
fixed at booking. Through-the-cycle behaviour is *approximated* by calibrating grade PDs
to the **long-run average** of yearly one-year rates (PD-3). We explicitly flag APG 113
para 73's warning that **calibrating PIT ratings to a long-run average does not by itself
make them TTC** -- a genuinely TTC rating would also dampen the rating *migration* through
the cycle, which an origination-only scorecard does not attempt.

**Override policy (Step 8).** No overrides are applied in this demonstration. In
production, rating overrides would be governed by a **written policy** with defined
approval authorities, a documented rationale per override, and **separate tracking and
monitoring** of override rates and their subsequent performance.

**Use test (Part 4.1).** This is a demonstration model and is **not used in live credit
approval, pricing or provisioning**. The "use test" requires that the same ratings drive
real decisions (origination cut-offs, limit-setting, pricing, capital, ECL). Here we show
the *capability* (grades, master scale, EL) but make no claim of operational use.

**Independence (Part 5.8).** Model **development and validation would be independent
functions** in production. In this repo they are separated only by notebook (03/03b build,
03c/04b validate); a real governance setup would place validation in a separate team with
its own sign-off.

**Observation window (Step 5 / PD-8).** Three vintages (2007, 2008, 2015) is **short of a
full cycle**, though it deliberately spans a severe downturn and a calm year -- which is
exactly why the **margin of conservatism** (PD-5) is applied. The framework's one-year
default rate is `(D - E_D) / (N - E_N)` (APG 113 para 110), where `E_D`/`E_N` exclude
zero-exposure and purely technical defaults; in this clean agency sample those exclusions
are **immaterial** (no undrawn/zero-exposure facilities, and defaults are real 180-DPD /
loss events, not technical), so `D/N` is used directly.

**Retail pool framing (Part 2.4 / PD-9).** Residential mortgages are a **retail** asset
class, so the A-H grades are best read as **pools** of homogeneous risk rather than
obligor ratings. A production retail pool system would also separate **delinquent vs
current** exposures and reflect **LGD and EAD pooling**, not PD alone. (The 180-vs-90-DPD
default-definition equivalence under Step 2 is covered in the LGD notes above.)

**Annual review (Part 4.4 / PD-10).** The intended governance cycle is an **annual
revalidation** of the PD (discrimination, calibration test, PSI) and a re-sizing of the
margin of conservatism as more vintages accumulate, plus an out-of-cycle review on a PSI
or calibration-flag trigger."""),
])

def _flush():
    sel = sys.argv[1:]
    for name, cells in _REGISTRY:
        if sel and not any(name.startswith(s) or s in name for s in sel):
            continue
        nb = nbf.v4.new_notebook()
        nb.cells = cells
        nb.metadata = {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
        }
        path = os.path.join(NB_DIR, name)
        with open(path, "w", encoding="utf-8") as fh:
            nbf.write(nb, fh)
        print("wrote", path)


_flush()
print("\\nDone.")
