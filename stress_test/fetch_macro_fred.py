"""Refresh macro/macro_annual.csv from FRED (optional, for production use).

The committed macro_annual.csv holds **approximate** annual US values so the stress
test runs offline. For a real run, pull the official series from FRED and overwrite it.

FRED series used:
  - UNRATE            : civilian unemployment rate (%), monthly -> annual mean
  - CSUSHPINSA        : S&P/Case-Shiller US National Home Price Index -> YoY % change
  - A191RL1Q225SBEA   : real GDP, % change from preceding period (annual rate) -> annual mean
  - MORTGAGE30US      : 30-year fixed mortgage rate (%), weekly -> annual mean

Requires an internet connection and `pandas_datareader` (pip install pandas-datareader).
Run:  python fetch_macro_fred.py
"""
import os
import pandas as pd

SERIES = {
    "unemployment": "UNRATE",
    "hpi_index": "CSUSHPINSA",
    "gdp_growth": "A191RL1Q225SBEA",
    "mortgage_rate": "MORTGAGE30US",
}
START, END = "2005-01-01", "2025-12-31"
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "macro", "macro_annual.csv")


def main():
    from pandas_datareader import data as web  # imported here so the file parses without the dep

    raw = {k: web.DataReader(code, "fred", START, END) for k, code in SERIES.items()}
    # Annual means for level/rate series; YoY for the house-price index.
    unemployment = raw["unemployment"].resample("YE").mean().iloc[:, 0]
    gdp_growth = raw["gdp_growth"].resample("YE").mean().iloc[:, 0]
    mortgage_rate = raw["mortgage_rate"].resample("YE").mean().iloc[:, 0]
    hpi = raw["hpi_index"].resample("YE").last().iloc[:, 0]
    hpi_yoy = hpi.pct_change() * 100.0

    out = pd.DataFrame({
        "year": unemployment.index.year,
        "unemployment": unemployment.round(1).values,
        "hpi_yoy": hpi_yoy.round(1).values,
        "gdp_growth": gdp_growth.round(1).values,
        "mortgage_rate": mortgage_rate.round(1).values,
    }).dropna()
    out.to_csv(OUT, index=False)
    print("wrote", OUT, "->", len(out), "years")


if __name__ == "__main__":
    main()
