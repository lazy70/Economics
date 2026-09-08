from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from statsmodels.tsa.api import VAR

REQUIRED_COLUMNS = ["date", "crude_oil_price", "inflation", "gdp_growth"]


def _create_sample_data(periods: int = 120) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    date_index = pd.date_range(start="2015-01-01", periods=periods, freq="MS")
    oil = 65 + np.cumsum(rng.normal(0, 2.2, size=periods))
    oil_shock = np.concatenate([[0.0], np.diff(oil) / np.clip(oil[:-1], 1.0, None)])
    inflation = 3.0 + 0.4 * oil_shock + rng.normal(0, 0.2, size=periods)
    growth = 2.2 - 0.3 * oil_shock + rng.normal(0, 0.25, size=periods)
    return pd.DataFrame(
        {
            "date": date_index,
            "crude_oil_price": oil,
            "inflation": inflation,
            "gdp_growth": growth,
        }
    )


def _load_data(data_path: str | None) -> pd.DataFrame:
    if data_path is None:
        frame = _create_sample_data()
    else:
        frame = pd.read_csv(data_path)
    missing = [col for col in REQUIRED_COLUMNS if col not in frame.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    frame = frame[REQUIRED_COLUMNS].copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame = frame.dropna(subset=["date"]).sort_values("date")
    return frame


def _build_features(frame: pd.DataFrame) -> pd.DataFrame:
    work = frame.copy()
    work["oil_shock"] = work["crude_oil_price"].pct_change().replace([np.inf, -np.inf], np.nan)
    for column in ["oil_shock", "inflation", "gdp_growth"]:
        work[f"{column}_lag1"] = work[column].shift(1)
    return work.dropna().reset_index(drop=True)


def run_pipeline(data_path: str | None = None, output_path: str | None = None) -> dict:
    data = _load_data(data_path)
    modeled = _build_features(data)
    if len(modeled) < 36:
        raise ValueError("Dataset is too small after feature engineering; need at least 36 observations.")

    split_index = int(len(modeled) * 0.8)
    train = modeled.iloc[:split_index].copy()
    test = modeled.iloc[split_index:].copy()

    var_train = train[["oil_shock", "inflation", "gdp_growth"]]
    var_model = VAR(var_train)
    var_result = var_model.fit(maxlags=2, ic="aic")
    lag_order = max(var_result.k_ar, 1)
    lag_values = var_train.values[-lag_order:]
    var_forecast = var_result.forecast(lag_values, steps=len(test))
    var_forecast_frame = pd.DataFrame(var_forecast, columns=["oil_shock", "inflation", "gdp_growth"])

    feature_cols = ["oil_shock", "oil_shock_lag1", "inflation_lag1", "gdp_growth_lag1"]
    x_train = train[feature_cols]
    x_test = test[feature_cols]

    inflation_model = RandomForestRegressor(n_estimators=200, random_state=42)
    growth_model = RandomForestRegressor(n_estimators=200, random_state=42)
    inflation_model.fit(x_train, train["inflation"])
    growth_model.fit(x_train, train["gdp_growth"])

    inflation_pred = inflation_model.predict(x_test)
    growth_pred = growth_model.predict(x_test)

    metrics = {
        "rows_analyzed": int(len(modeled)),
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
        "var": {
            "lag_order": int(var_result.k_ar),
            "inflation_mae": float(mean_absolute_error(test["inflation"], var_forecast_frame["inflation"])),
            "growth_mae": float(mean_absolute_error(test["gdp_growth"], var_forecast_frame["gdp_growth"])),
        },
        "random_forest": {
            "inflation_mae": float(mean_absolute_error(test["inflation"], inflation_pred)),
            "growth_mae": float(mean_absolute_error(test["gdp_growth"], growth_pred)),
        },
    }

    if output_path is not None:
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        output_file.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    return metrics
