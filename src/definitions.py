"""The risk definitions that drive every result -- kept in one place so the
default / EAD / LGD logic is identical across notebooks.

Plain English:
- DEFAULT: a loan defaults the first month it is 180+ days past due (delinquency
  status >= 6) OR ends in a credit-event zero-balance code (a loss disposition).
- EAD (exposure at default): the unpaid balance owed at the default month.
- REALISED LGD (loss given default): actual money lost on a disposed default,
  as a fraction of EAD, built from the real loss/recovery fields.
"""

import numpy as np
import pandas as pd

# Zero-balance codes that mean the loan ended in a CREDIT EVENT (a loss).
# 02 third-party sale, 03 short sale / charge-off, 09 REO disposition, 15 note sale.
# (01 prepaid/matured is NOT a default.)
CREDIT_EVENT_ZB_CODES = {"02", "03", "09", "15"}

# 180+ days past due -> delinquency status 6 or higher.
DEFAULT_DPD_STATUS = 6

# --- Realised-loss components (signs verified against the data, see note) ---
# IMPORTANT reconciliation finding: in the SFLLD performance file the four
# expense sub-fields (legal_costs, maintenance_preservation_costs,
# taxes_and_insurance, miscellaneous_expenses) SUM EXACTLY to the aggregate
# `expenses` field, and every expense field is stored as a NEGATIVE number.
# So the realised loss must (a) use the aggregate `expenses` ONCE -- never the
# aggregate plus its components, which double-counts -- and (b) subtract it
# (minus a negative adds the cost). With this, our computed loss matches the
# dataset's own `actual_loss_calculation` field at correlation ~0.99.
EXPENSE_FIELD = "expenses"                       # aggregate, stored negative
ACCRUED_INTEREST_FIELD = "delinquent_accrued_interest"   # stored positive
RECOVERY_FIELDS = ["net_sales_proceeds", "mi_recoveries", "non_mi_recoveries"]

# Expense sub-components: summed only for EDA/transparency, NOT for the loss math.
EXPENSE_SUBCOMPONENTS = [
    "legal_costs", "maintenance_preservation_costs",
    "taxes_and_insurance", "miscellaneous_expenses",
]


def parse_delinquency(series):
    """Delinquency status is a TEXT code, not a number.

    '0' = current, '1'..'n' = months past due. Special non-numeric values
    ('R'/'RA' = REO, 'XX'/space = unknown) must NOT silently coerce to 0, so
    we map them to NaN and treat them explicitly.
    """
    s = series.astype(str).str.strip().str.upper()
    # REO statuses behave like a deep default; map to a high numeric so the
    # 180+ DPD rule catches them. Unknowns ('XX', blank) -> NaN.
    s = s.replace({"R": "99", "RA": "99"})
    return pd.to_numeric(s, errors="coerce")


def clean_money(series):
    """Loss/recovery columns can carry text codes (e.g. net_sales_proceeds 'C'
    = covered, 'U' = unknown) instead of a number. Strip those to NaN before
    any arithmetic so a stray letter never poisons a sum."""
    s = series.astype(str).str.strip()
    s = s.replace({"": np.nan, "C": np.nan, "U": np.nan})
    return pd.to_numeric(s, errors="coerce")


def is_credit_event(zb_code_series):
    """True where the zero-balance code is a loss disposition (02/03/09/15)."""
    z = zb_code_series.astype(str).str.strip().str.zfill(2)
    return z.isin(CREDIT_EVENT_ZB_CODES)


def realised_loss(df):
    """Realised loss on disposed defaults, mirroring the SFLLD actual-loss definition:

        Loss = EAD
             + delinquent_accrued_interest      (interest owed but unpaid)
             - expenses                          (stored negative => subtract = add cost)
             - net_sales_proceeds - mi_recoveries - non_mi_recoveries   (money recovered)

    Component columns are assumed already cleaned to numbers (blanks -> 0, since
    loss/recovery fields are only populated around disposition).
    """
    ead = df["ead"].fillna(0)
    accrued = df[ACCRUED_INTEREST_FIELD].fillna(0)
    expenses = df[EXPENSE_FIELD].fillna(0)              # negative in the data
    recoveries = df[RECOVERY_FIELDS].fillna(0).sum(axis=1)
    return ead + accrued - expenses - recoveries


def winsorise_lgd(lgd, floor=0.0, cap=1.10):
    """Cap realised LGD to a sensible band. We floor at 0 (a 'gain' is treated
    as no loss) and cap slightly above 1 (1.10) -- costs can push true loss just
    past the exposure, and clipping hard at 1 would hide that. Documented choice.
    """
    return np.clip(lgd, floor, cap)
