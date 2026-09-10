"""OddAlerts historical data ingestion package.

Scope of this package, deliberately: acquire raw fixture data from the
OddAlerts API and persist it (both untouched and in a normalized form)
for later feature engineering. It does NOT compute predictive features,
attack/defence ratings, Poisson simulations, or predictions of any kind.
"""

__version__ = "0.1.0"
