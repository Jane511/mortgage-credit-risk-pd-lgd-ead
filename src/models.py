"""Shared, deliberately simple/interpretable models so the PD, LGD and
Expected-Loss notebooks all build them the same way.

- PD: one logistic regression on origination features.
- LGD: a two-stage model (probability of any loss x severity-if-loss), which
  is easy to explain and naturally keeps predictions in [0, ~1].
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from . import definitions as d

# Origination drivers used for PD (kept small and interpretable).
PD_NUMERIC = [
    "credit_score", "original_ltv", "original_cltv", "original_dti",
    "original_interest_rate", "original_loan_term",
]
PD_CATEGORICAL = ["loan_purpose", "occupancy_status", "channel"]

# Drivers for LGD severity.
LGD_NUMERIC = ["original_ltv", "credit_score", "original_upb"]


def build_pd_design(df, dummy_columns=None):
    """Turn the loan table into a numeric design matrix for logistic PD.

    Median-imputes numeric drivers and one-hot encodes the categoricals.
    Pass `dummy_columns` to align a new sample to the training columns.
    """
    X = df[PD_NUMERIC].apply(pd.to_numeric, errors="coerce")
    X = X.fillna(X.median())
    cats = pd.get_dummies(df[PD_CATEGORICAL].astype(str), prefix=PD_CATEGORICAL)
    X = pd.concat([X.reset_index(drop=True), cats.reset_index(drop=True)], axis=1)
    if dummy_columns is not None:
        X = X.reindex(columns=dummy_columns, fill_value=0)
    return X


def fit_pd(df):
    """Fit the logistic PD on origination features. Returns (pipeline, design
    columns) so the same encoding can be reused to score any sample."""
    X = build_pd_design(df)
    y = df["ever_default"].astype(int).values
    model = Pipeline([
        ("scale", StandardScaler()),
        ("logit", LogisticRegression(max_iter=2000)),
    ])
    model.fit(X, y)
    return model, list(X.columns)


def predict_pd(model, columns, df):
    X = build_pd_design(df, dummy_columns=columns)
    return model.predict_proba(X)[:, 1]


class TwoStageLGD:
    """Probability-of-loss (logistic) x severity-if-loss (linear).

    Built only on defaulted, disposed loans -- the only loans with a realised
    LGD -- exactly as the loss-given-default literature recommends.
    """

    def __init__(self):
        self.p_model = LogisticRegression(max_iter=2000)
        self.sev_model = LinearRegression()
        self.columns = None
        self.loss_threshold = 0.05  # "any material loss"

    def _design(self, df):
        X = df[LGD_NUMERIC].apply(pd.to_numeric, errors="coerce")
        X = X.fillna(X.median())
        # A downturn indicator lets severity lift in the 2007/2008 vintages.
        X = X.assign(downturn=df["vintage_year"].isin([2007, 2008]).astype(int))
        return X

    def fit(self, disposed):
        X = self._design(disposed)
        self.columns = list(X.columns)
        lgd = disposed["lgd"].clip(0, 1.1)
        any_loss = (lgd > self.loss_threshold).astype(int)
        self.p_model.fit(X, any_loss)
        mask = any_loss == 1
        self.sev_model.fit(X[mask.values], lgd[mask.values])
        return self

    def predict(self, df):
        X = self._design(df).reindex(columns=self.columns, fill_value=0)
        p_loss = self.p_model.predict_proba(X)[:, 1]
        severity = np.clip(self.sev_model.predict(X), 0, 1.1)
        return np.clip(p_loss * severity, 0, 1.1)
