"""Read the pipe-delimited SFLLD files, apply the official layout, and collapse
the huge month-by-month performance file down to ONE row per loan with the
fields the credit-risk models need (default flag/date, EAD, loss components).

The performance files are millions of rows; we only ever load the columns we
use and reduce immediately to loan level, so the whole pipeline fits in memory.
"""

import os
import pandas as pd

from . import layout
from . import definitions as d


def _count_columns(path):
    with open(path, "r", encoding="latin-1") as fh:
        first = fh.readline()
    return first.count("|") + 1


def load_origination(path):
    """One row per loan: loan characteristics at origination, properly named
    and type-cast. Validates the 32-column layout before naming anything."""
    n = _count_columns(path)
    layout.assert_columns(n, layout.ORIG_N_COLS, "Origination")
    df = pd.read_csv(
        path, sep="|", header=None, names=layout.ORIG_COLUMNS,
        dtype=str, na_values=[""], keep_default_na=True, encoding="latin-1",
    )
    # Cast the numeric origination drivers we actually model on.
    numeric = [
        "credit_score", "mi_pct", "number_of_units", "original_cltv",
        "original_dti", "original_upb", "original_ltv", "original_interest_rate",
        "original_loan_term", "number_of_borrowers",
    ]
    for c in numeric:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    # SFLLD sentinels: credit_score 9999, DTI 999, CLTV/LTV 999 mean "missing".
    df.loc[df["credit_score"] >= 9999, "credit_score"] = pd.NA
    df.loc[df["original_dti"] >= 999, "original_dti"] = pd.NA
    df.loc[df["original_cltv"] >= 999, "original_cltv"] = pd.NA
    df.loc[df["original_ltv"] >= 999, "original_ltv"] = pd.NA
    return df


# Performance columns we actually read (by name); the rest are skipped for speed.
_SVCG_USECOLS = [
    "loan_sequence_number", "monthly_reporting_period", "current_actual_upb",
    "current_loan_delinquency_status", "zero_balance_code", "current_deferred_upb",
    "mi_recoveries", "net_sales_proceeds", "non_mi_recoveries", "expenses",
    "legal_costs", "maintenance_preservation_costs", "taxes_and_insurance",
    "miscellaneous_expenses", "actual_loss_calculation", "zero_balance_removal_upb",
    "delinquent_accrued_interest", "zero_balance_effective_date", "loan_age",
]

# Money fields that appear once (at disposition) -> safe to sum to loan level.
_MONEY_ONE_TIME = (
    [d.EXPENSE_FIELD, d.ACCRUED_INTEREST_FIELD]
    + d.EXPENSE_SUBCOMPONENTS + d.RECOVERY_FIELDS
    + ["actual_loss_calculation", "zero_balance_removal_upb"]
)


def summarize_performance(path):
    """Collapse a performance file to one row per loan.

    Returns, per loan: default flag + default date, EAD, the disposition code,
    and the (cleaned) realised loss/recovery components, plus a couple of
    data-quality fields.
    """
    n = _count_columns(path)
    layout.assert_columns(n, layout.SVCG_N_COLS, "Performance")

    df = pd.read_csv(
        path, sep="|", header=None, names=layout.SVCG_COLUMNS, usecols=_SVCG_USECOLS,
        dtype=str, na_values=[""], keep_default_na=True, encoding="latin-1",
    )

    # --- type conversions / cleaning ---
    df["monthly_reporting_period"] = pd.to_numeric(df["monthly_reporting_period"], errors="coerce")
    df["current_actual_upb"] = pd.to_numeric(df["current_actual_upb"], errors="coerce")
    df["current_deferred_upb"] = pd.to_numeric(df["current_deferred_upb"], errors="coerce")
    df["delinq_num"] = d.parse_delinquency(df["current_loan_delinquency_status"])
    df["credit_event"] = d.is_credit_event(df["zero_balance_code"])
    for c in _MONEY_ONE_TIME:
        df[c] = d.clean_money(df[c])

    # A row marks default if 180+ DPD (status >= 6) OR a credit-event disposition.
    df["is_default_row"] = (df["delinq_num"] >= d.DEFAULT_DPD_STATUS) | df["credit_event"]

    # ONE-YEAR default flag (PD-1): a default occurring within the first 12 months
    # of the loan's life (loan_age in months). A fixed window makes vintages with
    # different observation lengths directly comparable -- the framework's long-run
    # one-year PD basis (CRE36.63 / APS 113 Att D PD para 2).
    df["loan_age_num"] = pd.to_numeric(df["loan_age"], errors="coerce")
    within_12m = df["loan_age_num"].between(0, 12)
    df["default_12m_row"] = within_12m & df["is_default_row"]

    # First defaulting month per loan -> default date + EAD (UPB at that month).
    dft = (
        df[df["is_default_row"]]
        .sort_values(["loan_sequence_number", "monthly_reporting_period"])
        .groupby("loan_sequence_number", as_index=False)
        .first()
    )
    first_def = dft[[
        "loan_sequence_number", "monthly_reporting_period",
        "current_actual_upb", "current_deferred_upb",
    ]].rename(columns={
        "monthly_reporting_period": "default_period",
        "current_actual_upb": "upb_at_default",
        "current_deferred_upb": "deferred_at_default",
    })

    # Loan-level one-time aggregates (loss/recovery fields appear once, at disposition).
    agg = df.groupby("loan_sequence_number").agg(
        disposed=("credit_event", "max"),
        default_within_12m=("default_12m_row", "max"),
        max_delinq_status=("delinq_num", "max"),
        last_period=("monthly_reporting_period", "max"),
        months_observed=("monthly_reporting_period", "size"),
        **{c: (c, "sum") for c in _MONEY_ONE_TIME},
    ).reset_index()

    # Disposition (zero-balance) code + exact effective date on the credit-event
    # row, if any. The effective date is the disposition month (YYYYMM); P0 surfaces
    # it so the LGD discounting step has an exact workout end-point, falling back to
    # last_period downstream when it is sparse/garbled.
    ce = df[df["credit_event"]].sort_values(
        ["loan_sequence_number", "monthly_reporting_period"]
    ).groupby("loan_sequence_number", as_index=False).first()[
        ["loan_sequence_number", "zero_balance_code", "zero_balance_effective_date"]
    ].rename(columns={
        "zero_balance_code": "disposition_zb_code",
        "zero_balance_effective_date": "disposition_period",
    })
    ce["disposition_period"] = pd.to_numeric(ce["disposition_period"], errors="coerce")

    out = agg.merge(first_def, on="loan_sequence_number", how="left")
    out = out.merge(ce, on="loan_sequence_number", how="left")

    out["ever_default"] = out["default_period"].notna()
    # EAD = UPB at default (+ deferred). At a credit-event month UPB is often 0,
    # so fall back to the balance removed at disposition.
    upb = out["upb_at_default"].fillna(0)
    upb = upb.where(upb > 0, out["zero_balance_removal_upb"].fillna(0))
    out["ead"] = upb + out["deferred_at_default"].fillna(0)
    # EAD is "exposure AT DEFAULT" -- only meaningful for defaulted loans. Blank
    # it for non-defaults (e.g. prepaid loans, whose payoff balance is irrelevant
    # to credit loss) so EAD never mixes defaults with healthy paid-off loans.
    out["ead"] = out["ead"].where(out["ever_default"])
    return out


def assemble_vintage(orig_path, svcg_path, vintage_year):
    """Join one vintage's origination + summarised performance into a loan-level
    table and tag it with the origination year."""
    orig = load_origination(orig_path)
    perf = summarize_performance(svcg_path)
    df = orig.merge(perf, on="loan_sequence_number", how="left")
    df["vintage_year"] = vintage_year
    # Loans with no performance rows at all never defaulted.
    df["ever_default"] = df["ever_default"].fillna(False)
    df["disposed"] = df["disposed"].fillna(False)
    df["default_within_12m"] = df["default_within_12m"].fillna(False).astype(bool)
    return df


def load_all_vintages(data_dir, vintages=(2007, 2008, 2015)):
    """Stack the requested vintages into one loan-level table."""
    frames = []
    for y in vintages:
        orig = os.path.join(data_dir, f"sample_{y}", f"sample_orig_{y}.txt")
        svcg = os.path.join(data_dir, f"sample_{y}", f"sample_svcg_{y}.txt")
        frames.append(assemble_vintage(orig, svcg, y))
    return pd.concat(frames, ignore_index=True)
