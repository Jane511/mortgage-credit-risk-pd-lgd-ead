"""Official SFLLD column layout (no header rows in the raw files).

The pipe-delimited Freddie Mac files ship WITHOUT a header. We apply the
column order from the SFLLD General User Guide ("File Layout and Data
Dictionary"). Both the origination and performance files in these sample
vintages have exactly 32 columns -- `assert_columns` enforces that so a
silent mis-mapping can never corrupt downstream numbers.
"""

# Origination file -- one row per loan, characteristics at origination.
ORIG_COLUMNS = [
    "credit_score",
    "first_payment_date",
    "first_time_homebuyer_flag",
    "maturity_date",
    "msa",
    "mi_pct",
    "number_of_units",
    "occupancy_status",
    "original_cltv",
    "original_dti",
    "original_upb",
    "original_ltv",
    "original_interest_rate",
    "channel",
    "ppm_flag",
    "amortization_type",
    "property_state",
    "property_type",
    "postal_code",
    "loan_sequence_number",
    "loan_purpose",
    "original_loan_term",
    "number_of_borrowers",
    "seller_name",
    "servicer_name",
    "super_conforming_flag",
    "pre_harp_loan_sequence_number",
    "program_indicator",
    "harp_indicator",
    "property_valuation_method",
    "interest_only_indicator",
    "mi_cancellation_indicator",
]

# Performance (servicing) file -- one row per loan per month.
SVCG_COLUMNS = [
    "loan_sequence_number",
    "monthly_reporting_period",
    "current_actual_upb",
    "current_loan_delinquency_status",
    "loan_age",
    "remaining_months_to_legal_maturity",
    "defect_settlement_date",
    "modification_flag",
    "zero_balance_code",
    "zero_balance_effective_date",
    "current_interest_rate",
    "current_deferred_upb",
    "ddlpi",
    "mi_recoveries",
    "net_sales_proceeds",
    "non_mi_recoveries",
    "expenses",
    "legal_costs",
    "maintenance_preservation_costs",
    "taxes_and_insurance",
    "miscellaneous_expenses",
    "actual_loss_calculation",
    "modification_cost",
    "step_modification_flag",
    "deferred_payment_plan",
    "eltv",
    "zero_balance_removal_upb",
    "delinquent_accrued_interest",
    "delinquency_due_to_disaster",
    "borrower_assistance_status_code",
    "current_month_modification_cost",
    "interest_bearing_upb",
]

ORIG_N_COLS = len(ORIG_COLUMNS)   # 32
SVCG_N_COLS = len(SVCG_COLUMNS)   # 32


def assert_columns(n_found, expected, kind):
    """Stop loudly if the real file's column count != the layout we apply."""
    if n_found != expected:
        raise ValueError(
            f"{kind} file has {n_found} columns but the layout expects {expected}. "
            "STOP: do not guess a mapping -- a wrong layout silently corrupts every "
            "downstream number. Re-check the SFLLD General User Guide layout."
        )
