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


def yyyymm_to_months(s):
    """Convert YYYYMM integers to an absolute month count (year*12 + month)."""
    s = pd.to_numeric(s, errors="coerce")
    year = (s // 100)
    month = (s % 100)
    return year * 12 + month


def months_between(start_yyyymm, end_yyyymm):
    """Whole months from start to end (YYYYMM each), floored at 0; NaN-safe.

    YYYYMM integers cannot be subtracted directly (200901 - 200812 must equal 1,
    not 89), so convert both to an absolute month count first.
    """
    diff = yyyymm_to_months(end_yyyymm) - yyyymm_to_months(start_yyyymm)
    return diff.clip(lower=0)


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


def _recovery_fields(include_mi=True):
    """The recovery columns to sum. APS 113 Att B para 23 forbids using LMI
    (mortgage-insurance) recoveries inside a retail-mortgage LGD, so the APRA
    view drops `mi_recoveries` by passing include_mi=False."""
    if include_mi:
        return list(RECOVERY_FIELDS)
    return [c for c in RECOVERY_FIELDS if c != "mi_recoveries"]


def net_recovery(df, include_mi=True):
    """Net cash the lender gets back at disposition, as a single lump sum:
    recoveries MINUS foreclosure costs. Expenses are stored NEGATIVE in the
    SFLLD file, so adding them subtracts the cost. This is the cash flow the
    economic-loss calculation discounts back to the default date."""
    recoveries = df[_recovery_fields(include_mi)].fillna(0).sum(axis=1)
    expenses = df[EXPENSE_FIELD].fillna(0)              # negative in the data
    return recoveries + expenses


def realised_loss(df, include_mi=True):
    """Realised (NOMINAL, undiscounted) loss on disposed defaults, mirroring the
    SFLLD actual-loss definition:

        Loss = EAD
             + delinquent_accrued_interest      (interest owed but unpaid)
             - expenses                          (stored negative => subtract = add cost)
             - net_sales_proceeds - mi_recoveries - non_mi_recoveries   (money recovered)

    This is the reconciliation anchor (corr ~0.99 to Freddie Mac's own loss
    field) -- the DEFAULT (include_mi=True) is byte-for-byte the original
    formula, so the reconciliation is untouched. include_mi=False drops LMI
    recoveries for the separate APRA view (APS 113 Att B para 23).

    Component columns are assumed already cleaned to numbers (blanks -> 0, since
    loss/recovery fields are only populated around disposition).
    """
    ead = df["ead"].fillna(0)
    accrued = df[ACCRUED_INTEREST_FIELD].fillna(0)
    expenses = df[EXPENSE_FIELD].fillna(0)              # negative in the data
    recoveries = df[_recovery_fields(include_mi)].fillna(0).sum(axis=1)
    return ead + accrued - expenses - recoveries


def economic_loss(df, include_mi=True, rate_col="original_interest_rate", max_months=None):
    """ECONOMIC (discounted) loss on disposed defaults -- the framework's actual
    LGD definition (CRE36.76 / APS 113 Att D LGD para 1), which must include
    "material discount effects".

    Because the SFLLD file lumps all recoveries/expenses at disposition, we treat
    the net recovery as a single cash flow occurring `months_to_resolution` months
    after default and discount it back to the default date:

        r_m             = (original_interest_rate / 100) / 12      # APG 113 Table 8: facility's own rate
        discount_factor = 1 / (1 + r_m) ** months_to_resolution
        economic_loss   = EAD + accrued_interest - discount_factor * net_recovery

    Discounting shrinks the present value of the recovery, so economic loss is
    >= nominal loss; the gap grows with the workout length. At
    months_to_resolution = 0 it reduces exactly to realised_loss().

    NaN-safe: a missing rate or month-gap falls back to no discounting
    (discount_factor = 1) rather than crashing -- those few rows then equal the
    undiscounted loss (documented limitation, see LGD task P1-1 / NOT-to-do).
    """
    ead = df["ead"].fillna(0)
    accrued = df[ACCRUED_INTEREST_FIELD].fillna(0)
    rec = net_recovery(df, include_mi=include_mi)
    r_m = (pd.to_numeric(df[rate_col], errors="coerce") / 100.0 / 12.0).fillna(0.0)
    months = pd.to_numeric(df["months_to_resolution"], errors="coerce")
    if max_months is not None:
        months = months.clip(upper=max_months)
    months = months.fillna(0.0)
    discount_factor = 1.0 / (1.0 + r_m) ** months
    return ead + accrued - discount_factor * rec


def winsorise_lgd(lgd, floor=0.0, cap=1.10):
    """Cap realised LGD to a sensible band. We floor at 0 (a 'gain' is treated
    as no loss) and cap slightly above 1 (1.10) -- costs can push true loss just
    past the exposure, and clipping hard at 1 would hide that. Documented choice.
    """
    return np.clip(lgd, floor, cap)


def apply_lgd_floor(lgd, floor=0.20):
    """APS 113 Att B paras 19-24 / Tables 6-7 prescribe minimum (floored) LGDs as
    a regulatory-capital backstop. 0.20 is the retail residential-mortgage floor
    used when own-LGD estimates are not approved. NaN-safe and only applied to the
    separate APRA-view column -- never to the IFRS 9 realised/modelled LGD."""
    lgd = np.asarray(lgd, dtype=float)
    return np.where(np.isnan(lgd), lgd, np.maximum(lgd, floor))


def apply_pd_floor(pd_values, floor=0.0005):
    """Regulatory PD floor (PD-6): PD = max(estimate, 0.0005) per APS 113 Att B para 1
    / Step 11 (the 5 bps minimum). Sovereigns are exempt, which is irrelevant to a
    mortgage book. NaN-safe."""
    pd_values = np.asarray(pd_values, dtype=float)
    return np.where(np.isnan(pd_values), pd_values, np.maximum(pd_values, floor))


def add_moc(lgd, add_pp=0.05, cap=1.10):
    """Margin of conservatism overlay (CRE36.67 / Step 11). An ADDITIVE add-on, in
    LGD points, to cover the thin data and short observation window (3 vintages) and
    the incomplete-workout uncertainty. It is an OVERLAY applied only to the APRA
    capital view -- never inside the model and never on the IFRS 9 figures. NaN-safe."""
    lgd = np.asarray(lgd, dtype=float)
    return np.where(np.isnan(lgd), lgd, np.minimum(lgd + add_pp, cap))
