"""Generate the 00-08 notebooks for the Mortgage Credit Risk project.

Each notebook = a plain-English HR summary at the top, code cells with one
comment each, and exactly one results table saved to output/. Heavy logic lives
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
row per loan**. It then stacks three origination years together: **2007 and
2008** (the financial-crisis "downturn" years) and **2015** (a calm year).

**Headline result:** the crisis vintages default far more often than the calm
one -- about **14% (2007)** and **7% (2008)** versus roughly **2% (2015)**."""),
    code(BOOTSTRAP),
    code("""# Read all three vintages, apply the 32-column layout, and collapse the
# monthly performance files down to one row per loan (this is the heavy step).
from src import loaders
df = loaders.load_all_vintages('raw data')
print('assembled loan-level table:', df.shape)"""),
    code("""# Cache the assembled table so every later notebook loads in seconds.
os.makedirs('data/processed', exist_ok=True)
df.to_parquet('data/processed/loan_level.parquet')
print('cached -> data/processed/loan_level.parquet')"""),
    code("""# Build a small data-quality summary: loan counts and default rate per vintage.
dq = df.groupby('vintage_year').agg(
    loans=('loan_sequence_number', 'size'),
    defaults=('ever_default', 'sum'),
    disposed_defaults=('disposed', 'sum'),
    median_credit_score=('credit_score', 'median'),
    median_original_upb=('original_upb', 'median'),
)
dq['default_rate'] = (dq['defaults'] / dq['loans']).round(4)
dq = dq.reset_index()"""),
    code("""# Save the one results table for this notebook.
from src.output import save_csv
save_csv(dq, 'output/00_data_quality.csv')
dq"""),
    md("""**Reading the table:** each vintage is a 50,000-loan random sample. The
`default_rate` column is the share of loans that ever hit serious default. The
crisis years dwarf 2015 -- the contrast this whole project is built to show."""),
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
    'disposed', 'max_delinq_status',
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
within the **first 12 months** of the loan's life (from `loan_age`). Because the 2007/08
books are observed for far longer than 2015, their *observed-to-date* rates are not
comparable; the fixed one-year window puts all three vintages on the **same footing**.

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
save_csv(tbl, 'output/01_default_lgd_by_vintage.csv')
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
the home's value), and across the three **vintages**. Charts are saved for the
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
os.makedirs('output/readme_assets', exist_ok=True)"""),
    code("""# Default rate by credit-score band.
by_score = base.groupby('credit_score_band', observed=True)['ever_default'].mean()
ax = by_score.plot(kind='bar', color='#4C72B0', title='Default rate by credit-score band')
ax.set_ylabel('default rate'); plt.tight_layout()
plt.savefig('output/readme_assets/default_by_credit_score.png', dpi=110); plt.close()"""),
    code("""# Default rate by loan-to-value band.
by_ltv = base.groupby('ltv_band', observed=True)['ever_default'].mean()
ax = by_ltv.plot(kind='bar', color='#C44E52', title='Default rate by loan-to-value band')
ax.set_ylabel('default rate'); plt.tight_layout()
plt.savefig('output/readme_assets/default_by_ltv.png', dpi=110); plt.close()"""),
    code("""# One-page risk-by-driver summary table (the saved result for this notebook).
rows = []
for band, v in base.groupby('credit_score_band', observed=True)['ever_default'].mean().items():
    rows.append({'driver': 'credit_score', 'band': band, 'default_rate': round(v, 4)})
for band, v in base.groupby('ltv_band', observed=True)['ever_default'].mean().items():
    rows.append({'driver': 'ltv', 'band': band, 'default_rate': round(v, 4)})
for band, v in base.groupby('vintage_year')['ever_default'].mean().items():
    rows.append({'driver': 'vintage', 'band': str(band), 'default_rate': round(v, 4)})
risk_by_driver = pd.DataFrame(rows)
save_csv(risk_by_driver, 'output/02_risk_by_driver.csv')
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
os.makedirs('output/readme_assets', exist_ok=True)
plt.savefig('output/readme_assets/pd_calibration.png', dpi=110); plt.close()"""),
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
save_csv(metrics_tbl, 'output/03_pd_metrics.csv')
metrics_tbl"""),
    md("""**Reading the table:** AUC/Gini/KS measure how well the model ranks risky
loans above safe ones; higher is better. The calibration plot (saved to
`output/readme_assets/`) shows predicted and actual default rates lining up along
the diagonal -- the model is honest, not just discriminating."""),
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
APS 113 Att D PD para 2) and makes the three vintages directly comparable, regardless
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
save_csv(iv_summary, 'output/03b_information_value.csv')
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
save_csv(points, 'output/03b_scorecard_points.csv')
points.head(15)"""),
    code("""# Sort every loan into 8 rating grades (A safest) and build the MASTER SCALE:
# predicted PD vs observed default rate, with loan count and exposure share.
base['grade'] = scorecard.assign_grades(base['score'], n_grades=8)
master = scorecard.master_scale(base, 'grade', 'pd_hat', 'target', 'original_upb', score_col='score')"""),
    code("""# PD-3: calibrate each grade to its LONG-RUN PD -- the simple average ACROSS the
# three vintages of the per-year one-year default rate (count-weighted within year),
# the framework basis (APG 113 paras 110-114; count-weighted, not exposure-weighted).
lr = scorecard.long_run_grade_pd(base, 'grade', 'target', 'vintage_year', exposure_col='original_upb')
master = master.merge(lr, on='grade', how='left')
master['long_run_pd'] = master['long_run_pd'].round(4)
master['exposure_weighted_pd'] = master['exposure_weighted_pd'].round(4)  # sensitivity only
save_csv(master, 'output/03b_master_scale.csv')
master[['grade', 'predicted_pd', 'long_run_pd', 'observed_default_rate',
        'exposure_weighted_pd', 'loans', 'exposure_share']]"""),
    md("""**Long-run grade PD (PD-3).** `predicted_pd` is the model's average per grade;
`long_run_pd` is the framework's calibration figure -- for each grade we take the
one-year default rate **in each vintage** and then **simple-average across the three
vintages** (each loan counts once *within* a year; each year counts equally *across*
years, per APS 113 Att D PD para 3, which is **count-weighted, not EAD-weighted**).
The two columns are close, confirming the model is well-calibrated in level, not just
in rank. `exposure_weighted_pd` is shown for **sensitivity review only** (APG 113 para
114) and is explicitly *not* the calibration figure.

**Downturn-heavy caveat.** Only three vintages are available and **two are crisis
years**, so this simple across-year average is skewed toward downturn conditions. That
conservatism is appropriate for capital, but it is a real limitation -- it is exactly
why the **margin of conservatism** (PD-5) and the documented short-observation-window
note (PD-8) exist, and a fuller cycle of vintages would dilute the crisis weighting."""),
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
save_csv(gt, 'output/03d_pd_calibration_test.csv')
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
    code("""# PD-5 + PD-6: margin of conservatism (additive overlay) then the 5 bps regulatory
# floor, on each grade's calibrated long-run PD. MoC covers the thin, downturn-heavy
# window (CRE36.67 / APG 113 para 115a); the floor is the APS 113 Att B para 1 backstop.
from src import definitions as d
PD_MOC_PP = 0.0025  # +25 bps additive overlay on each grade PD (an overlay, not the model).
grade_pd = master[['grade', 'long_run_pd']].copy()
grade_pd['long_run_pd_moc'] = d.add_moc(grade_pd['long_run_pd'].values, PD_MOC_PP, cap=1.0).round(4)
grade_pd['long_run_pd_final'] = d.apply_pd_floor(grade_pd['long_run_pd_moc'].values, floor=0.0005).round(4)
grade_pd['moc_points'] = PD_MOC_PP
grade_pd['floor_binds'] = grade_pd['long_run_pd_moc'] < 0.0005
save_csv(grade_pd, 'output/03e_grade_pd_moc_floor.csv')
grade_pd"""),
    md("""**Margin of conservatism + floor (PD-5, PD-6).** `long_run_pd` is the calibrated
estimate; `long_run_pd_moc` adds a documented **+25 bps additive margin of conservatism**;
`long_run_pd_final` then applies the **5 bps regulatory PD floor** (`max(PD, 0.0005)`).

- **Why this MoC, and where it sits.** Only three vintages, two of them crisis, plus the
  correlated-default caveat from the calibration test, mean the grade PDs carry real
  estimation uncertainty. The MoC is an explicit **overlay on each grade** (not inside the
  model), deliberately additive and modest; it bites hardest on the safest grades, which is
  the conservative direction for thin data. It would be **reviewed annually** (PD-10) and
  re-sized as more vintages accumulate.
- **The floor** (APS 113 Att B para 1, the 5 bps minimum) is applied for completeness.
  At grade level it does **not** bind here once the +25 bps MoC is added (`floor_binds`
  is all False, min grade PD 0.30%). But because this is a low one-year PD, the floor
  **does bite at the per-loan level** in the Expected-Loss step (notebook 06), where it
  lifts roughly **one in five** loans whose raw model PD sits below 5 bps. Sovereign
  exposures are exempt from the floor, which is irrelevant to a mortgage book."""),
    code("""# Downturn view: reuse the stress logic (PD multiplier = crisis vs calm default
# rate) to show how each grade's predicted PD shifts in a recession.
calm = base[base['vintage_year'] == 2015]
downturn = base[base['vintage_year'].isin([2007, 2008])]
pd_mult = downturn['target'].mean() / calm['target'].mean()
grade_pd = base.groupby('grade', observed=True)['pd_hat'].mean().reset_index().rename(columns={'pd_hat': 'base_pd'})
grade_pd['stressed_pd'] = np.minimum(grade_pd['base_pd'] * pd_mult, 1.0).round(4)
grade_pd['base_pd'] = grade_pd['base_pd'].round(4)
grade_pd['pd_multiplier'] = round(pd_mult, 2)
save_csv(grade_pd, 'output/03b_downturn_by_grade.csv')
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
    code("""# Run the two forward splits, plus a clearly-labelled reverse 'what-if'.
splits = [
    ([2007], [2008], 'A) out-of-time, same regime: train 2007 -> test 2008'),
    ([2007, 2008], [2015], 'B) out-of-regime: train crisis 2007+08 -> test calm 2015'),
    ([2015], [2007, 2008], 'C) reverse what-if (NOT a forward test): train calm 2015 -> test crisis'),
]
rows, details = [], []
for tr_y, te_y, label in splits:
    r, d = evaluate_split(tr_y, te_y, label)
    rows.append(r); details.append(d)"""),
    code("""# The one comparison table (the saved deliverable).
comparison = pd.DataFrame(rows)[[
    'split', 'train_auc', 'test_auc', 'train_avg_predicted_pd',
    'test_avg_predicted_pd', 'test_observed_default_rate', 'psi_train_vs_test']]
save_csv(comparison, 'output/03c_oot_validation.csv')
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
    code("""# Compare observed vs modelled LGD, downturn (2007/2008) vs calm (2015), and
# show the three LGD lenses side by side: nominal IFRS 9, economic IFRS 9, APRA.
disposed['regime'] = np.where(disposed['vintage_year'].isin([2007, 2008]), 'downturn (2007-08)', 'calm (2015)')
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
save_csv(lgd_summary, 'output/04_lgd_model.csv')
lgd_summary"""),
    md("""**Reading the table:** `observed_lgd` is what actually happened (nominal
IFRS 9); `modelled_lgd` is the two-stage model's fit. The downturn row sits roughly
twice as high as the calm row -- the **downturn LGD** a stress test needs.

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
    md("""**Cyclicality (P2-3).** Realised severity is far higher in the crisis books
than the calm one (roughly a 2x ratio), which is the textbook signature of a
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
open_wf['regime'] = np.where(open_wf['vintage_year'].isin([2007, 2008]), 'downturn (2007-08)', 'calm (2015)')
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
save_csv(sens, 'output/04_incomplete_workouts.csv')
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
save_csv(moc, 'output/04_moc_overlay.csv')
moc"""),
    md("""**Reading the MoC table (P2-2).** `lgd_apra_with_moc` is simply the APRA-view
LGD plus a documented **+5 LGD-point** margin. It is deliberately small and explicit,
and it lives outside the model so it can be reviewed, dialled, or removed without
re-fitting anything. Justification: only three vintages (short of a full cycle) and a
modest disposed-default count mean the point estimate carries real estimation
uncertainty; the MoC is the conservative buffer the framework expects for that."""),
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
    code("""# Forward out-of-regime test + the reverse 'what-if' that exposes the blind spot.
oot_rows = [
    oot_lgd([2007, 2008], [2015], 'A) train crisis 2007+08 -> test calm 2015'),
    oot_lgd([2015], [2007, 2008], 'B) reverse: train calm 2015 -> test crisis (under-predicts)'),
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
save_csv(val, 'output/04b_lgd_validation.csv')
val"""),
    md("""## Interpretation (plain English)

- **Out-of-time / out-of-regime.** A model trained on the **crisis** books
  **over-predicts** the calm 2015 book (predicted ~43% vs realised ~25%) -- i.e. it is
  *conservative* out-of-regime, which is the safe direction. The **reverse** is the
  dangerous one: training only on calm 2015 and predicting the crisis **under-predicts**
  downturn severity badly (predicted ~21% vs realised ~57%). A model built only in good
  times is blind to a downturn; this is the headline out-of-time finding and the reason
  the downturn LGD (notebook 04 / 06) is used for the conservative estimate.
- **Cohort backtest.** Read by predicted-LGD decile, mean predicted and mean realised
  track in the same direction -- the model is **calibrated in rank**. Per the WP14
  caveat, this is a cohort comparison; a single point-in-time realised LGD must **not**
  be compared directly to a long-run estimate loan-by-loan.
- **Discrimination.** A positive Spearman correlation between predicted and realised
  severity on the loss-only loans confirms the model **rank-orders** loss size, though
  mortgage LGD is inherently noisy so the R^2 is modest -- normal for severity models.
- **Stability.** Dropping any single vintage moves the overall mean predicted LGD only
  modestly, **except** when the crisis volume is removed, which pulls the estimate down
  -- consistent with severity being driven by the downturn cohorts.

## Benchmarking note (APG 113 para 140(c); WP14)

Internal loss data here is **thin** -- three discrete vintages, only ~7k disposed
defaults, and the calm year has barely 100 -- so the framework expects **benchmarking
and qualitative review** to carry more weight than pure backtesting. The modelled
downturn severity (~55-58%) is in line with **published US agency mortgage loss
severities** for the 2008-09 period (broadly ~50-60% on distressed dispositions),
which supports the magnitude even where the internal sample is too small to backtest
tightly. In production this would be supplemented with external severity benchmarks and
an expert-judgement overlay rather than relying on the internal backtest alone."""),
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
save_csv(ead_summary, 'output/05_ead_summary.csv')
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
    code("""# One-year PD for every loan (logistic model fit on the whole book). pd_hat is a
# 12-month PD (PD-1/PD-2) -- exactly the IFRS 9 Stage 1 (12-month ECL) input.
from src import definitions as d
pd_model, pd_cols = models.fit_pd(base)
base['pd_hat'] = models.predict_pd(pd_model, pd_cols, base)
# PD-6: apply the 5 bps regulatory PD floor to the per-loan PD used in EL.
base['pd_hat'] = d.apply_pd_floor(base['pd_hat'], floor=0.0005)"""),
    code("""# LGD for every loan (two-stage model trained on disposed defaults).
disposed = base[base['disposed'] & base['lgd'].notna()]
lgd_model = models.TwoStageLGD().fit(disposed)
base['lgd_hat'] = lgd_model.predict(base)"""),
    code("""# EAD for every loan: balance at default if it defaulted, else the original
# loan amount as the exposure proxy for a still-performing loan.
base['ead_loan'] = np.where(base['ever_default'], base['ead'], base['original_upb'])
# Expected loss per loan = PD x LGD x EAD.
base['expected_loss'] = base['pd_hat'] * base['lgd_hat'] * base['ead_loan']"""),
    code("""# IFRS 9 / AASB 9 staging: 3 = defaulted (credit-impaired), 2 = significant
# increase in risk (ever 60+ days late but not defaulted), 1 = performing.
stage2 = (~base['ever_default']) & (base['max_delinq_status'].fillna(0) >= 2)
base['ifrs9_stage'] = np.where(base['ever_default'], 3, np.where(stage2, 2, 1))
# pd_hat is now a genuine 12-MONTH PD, which is exactly the Stage 1 (12-month ECL)
# input -- so Stage 1 reported EL is the 12-month EL directly (no ad-hoc 0.25 factor
# any more). Stages 2 & 3 need LIFETIME ECL; with only a one-year PD modelled here we
# scale by a transparent multi-year horizon factor as a lifetime proxy (a production
# model would estimate lifetime PD directly).
LIFETIME_HORIZON = 4
base['el_reported'] = np.where(base['ifrs9_stage'] == 1, base['expected_loss'],
                               base['expected_loss'] * LIFETIME_HORIZON)"""),
    code("""# Portfolio Expected Loss summary by IFRS 9 stage (the saved result).
el_summary = base.groupby('ifrs9_stage').agg(
    loans=('loan_sequence_number', 'size'),
    avg_pd=('pd_hat', 'mean'),
    avg_lgd=('lgd_hat', 'mean'),
    total_ead=('ead_loan', 'sum'),
    lifetime_expected_loss=('expected_loss', 'sum'),
    reported_expected_loss=('el_reported', 'sum'),
).reset_index().round(2)
save_csv(el_summary, 'output/06_expected_loss.csv')
el_summary"""),
    md("""### Downturn-LGD variant of Expected Loss (P2-3)

Notebook 04 showed loss severity is strongly **cyclical** (~25% calm vs ~57% crisis).
APS 113 Att D LGD paras 4-5 say that where severity is cyclical, the LGD *estimate*
must reflect **downturn** conditions, not the through-the-cycle average. So alongside
the baseline EL we compute a **downturn-LGD variant**, lifting every loan's LGD to at
least the crisis-regime realised severity. This is the conservative figure the
framework expects a capital/EL report to show."""),
    code("""# P2-3: downturn-LGD variant of EL. Lift each loan's LGD to >= the crisis-regime
# realised severity (the observed downturn LGD), then recompute Expected Loss.
downturn_lgd = float(base.loc[base['disposed'] & base['vintage_year'].isin([2007, 2008]), 'lgd'].mean())
base['lgd_downturn'] = np.maximum(base['lgd_hat'], downturn_lgd)
base['expected_loss_downturn'] = base['pd_hat'] * base['lgd_downturn'] * base['ead_loan']
el_variant = pd.DataFrame([
    {'view': 'through-the-cycle (baseline)', 'lgd_basis': 'modelled lgd_hat',
     'total_expected_loss': round(float(base['expected_loss'].sum()), 0)},
    {'view': 'downturn LGD (APS 113 Att D LGD 4-5)', 'lgd_basis': f'max(lgd_hat, {downturn_lgd:.3f})',
     'total_expected_loss': round(float(base['expected_loss_downturn'].sum()), 0)},
])
el_variant['uplift_x'] = (el_variant['total_expected_loss'] /
                          el_variant['total_expected_loss'].iloc[0]).round(2)
save_csv(el_variant, 'output/06_el_downturn_variant.csv')
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
save_csv(s3, 'output/06_stage3_best_estimate.csv')
s3"""),
    md("""**Reading the Stage 3 table (P2-4).** The mechanical column applies the model
PD x LGD even to loans that have *already* defaulted (so its PD < 1 understates the
loss); the best-estimate column uses each defaulted loan's realised loss where the
workout is complete and the downturn LGD otherwise, with PD = 1. The best estimate is
materially larger -- which is the point: a defaulted loan's expected loss should be
built from its own resolution, not a portfolio-average model output."""),
    code("""# Worked example: show PD x LGD x EAD = EL for a single representative loan.
ex = base.sort_values('expected_loss', ascending=False).iloc[100]
print('Worked example loan:', ex['loan_sequence_number'])
print(f"  PD  (chance of default) = {ex['pd_hat']:.3f}")
print(f"  LGD (loss if default)   = {ex['lgd_hat']:.3f}")
print(f"  EAD (amount owed)       = ${ex['ead_loan']:,.0f}")
print(f"  Expected Loss = {ex['pd_hat']:.3f} x {ex['lgd_hat']:.3f} x ${ex['ead_loan']:,.0f} = ${ex['expected_loss']:,.0f}")"""),
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
worse *at the same time*."""),
    code(BOOTSTRAP),
    code("""# Load the base table and split into calm (2015) and downturn (2007-08) books.
import pandas as pd
import numpy as np
from src.output import save_csv
base = pd.read_parquet('data/processed/analysis_base.parquet')
calm = base[base['vintage_year'] == 2015]
downturn = base[base['vintage_year'].isin([2007, 2008])]"""),
    code("""# Observed downturn multipliers: how much PD and LGD worsened in the crisis.
pd_calm, pd_down = calm['ever_default'].mean(), downturn['ever_default'].mean()
lgd_calm = calm.loc[calm['disposed'], 'lgd'].mean()
lgd_down = downturn.loc[downturn['disposed'], 'lgd'].mean()
pd_mult = pd_down / pd_calm
lgd_mult = lgd_down / lgd_calm
print(f'PD multiplier  = {pd_mult:.2f}x   LGD multiplier = {lgd_mult:.2f}x')"""),
    code("""# Macro context for the downturn (documented assumptions; production would pull
# these live from FRED: unemployment UNRATE, house prices CSUSHPINSA). Values
# reflect the 2008-09 GFC path, comparable to a CCAR severely-adverse scenario.
macro = pd.DataFrame([
    {'scenario': 'baseline (2015 calm)', 'unemployment_pct': 5.3, 'hpi_change_pct': 5.0},
    {'scenario': 'severely adverse (GFC 2008-09)', 'unemployment_pct': 10.0, 'hpi_change_pct': -30.0},
])
macro"""),
    code("""# Apply the downturn to the calm-year book: stress PD and LGD, recompute EL.
ead_calm = np.where(calm['ever_default'], calm['ead'], calm['original_upb'])
base_pd = calm['ever_default'].mean()
base_lgd = lgd_calm
baseline_el = (base_pd * base_lgd * ead_calm).sum()
stressed_el = (min(base_pd * pd_mult, 1.0) * min(base_lgd * lgd_mult, 1.0) * ead_calm).sum()
stress_tbl = pd.DataFrame([
    {'measure': 'PD', 'baseline': round(base_pd, 4), 'stressed': round(min(base_pd * pd_mult, 1.0), 4)},
    {'measure': 'LGD', 'baseline': round(base_lgd, 4), 'stressed': round(min(base_lgd * lgd_mult, 1.0), 4)},
    {'measure': 'expected_loss', 'baseline': round(baseline_el, 0), 'stressed': round(stressed_el, 0)},
    {'measure': 'EL_uplift_x', 'baseline': 1.0, 'stressed': round(stressed_el / baseline_el, 2)},
])
save_csv(stress_tbl, 'output/07_stress_test.csv')
stress_tbl"""),
    md("""**Reading the table:** we take the calm 2015 portfolio and push PD and LGD
up by the multipliers the crisis actually produced. Because the two stack
multiplicatively, Expected Loss rises far more than either driver alone -- the
core lesson of downturn stress testing.

**Extension -- climate scenario (sketch, not built):** the same machinery extends
to physical climate risk. A flood or wildfire shock lowers house prices in
exposed postcodes, which raises **LGD** (smaller recovery on sale) and, via
negative equity, raises **PD**. One would overlay a hazard map on the property
postcode, apply a region-specific house-price haircut, and re-run this exact
PD/LGD/EL stress -- a cheap, high-signal differentiator for a climate-risk role."""),
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

**Data.** 50,000-loan samples for the 2007, 2008 (crisis) and 2015 (calm)
origination years; origination characteristics joined to monthly performance and
collapsed to one row per loan. Raw data is not redistributed in this repo.

**Methodology.**
- *Default* = first month at 180+ days past due, or a credit-event zero-balance
  code (third-party sale, short sale/charge-off, REO disposition, note sale).
- *PD* = logistic regression on origination features (interpretable scorecard).
- *LGD* = two-stage model (P(loss) x severity) on **realised** losses from
  defaulted, disposed loans; reconciled to Freddie Mac's own loss field (corr ~0.99).
  Extended to **economic (discounted) loss**, an **APRA capital view** (MI excluded,
  20% high-LVR reduction, 20% floor), a margin of conservatism, a downturn LGD, and an
  **independent LGD validation** (notebook 04b) -- see the framework-alignment notes below.
- *EAD* = outstanding balance at default (no CCF -- a term loan has no undrawn limit).
- *EL* = PD x LGD x EAD, staged under IFRS 9 / AASB 9.
- *Stress* = downturn multipliers observed in the 2007/2008 crisis vintages.

**Results.** Default rate ~14% / 7% / 2% and LGD ~58% / 54% / 25% across
2007 / 2008 / 2015; PD model AUC ~0.75-0.80; Expected Loss concentrated in the
crisis vintages and Stage 3.

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
    code("""# PSI: compare the calm 2015 book (expected) to the crisis books (actual) on
# both a raw driver (credit score) and the model output (PD).
calm = base[base['vintage_year'] == 2015]
crisis = base[base['vintage_year'].isin([2007, 2008])]
psi_tbl = pd.DataFrame([
    {'feature': 'credit_score', 'psi_2015_vs_crisis': round(metrics.psi(calm['credit_score'], crisis['credit_score']), 4)},
    {'feature': 'pd_hat', 'psi_2015_vs_crisis': round(metrics.psi(calm['pd_hat'], crisis['pd_hat']), 4)},
])
psi_tbl['interpretation'] = psi_tbl['psi_2015_vs_crisis'].apply(
    lambda v: 'stable (<0.10)' if v < 0.10 else ('watch (0.10-0.25)' if v < 0.25 else 'material shift (>0.25)'))
save_csv(psi_tbl, 'output/08_monitoring_psi.csv')
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
A 90-DPD sensitivity flag could be added from the same delinquency field if required.

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

**Observation period (Step 7).** Three discrete vintages (2007, 2008, 2015) is **short
of a full economic cycle**, though it deliberately spans both a severe downturn and a
calm year. This short window is precisely why the **margin of conservatism** (P2-2) is
applied and why benchmarking/qualitative review carry extra weight (APG 113 para 140(c)).

**Segmentation (Step 1).** Current severity drivers are LTV, credit score, loan size
(UPB) and the downturn indicator. In production the LGD segmentation would extend to
**loan purpose, occupancy, and with/without-LMI** (and likely geography/house-price
region), each of which plausibly shifts recovery; they are noted here as the natural
next segments rather than fitted, given the sample size."""),
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
