"""Machine-learning energy and SoC prediction for the CAEV-BMS dashboard."""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split


def _rmse_score(y_true: pd.Series | np.ndarray, y_pred: np.ndarray) -> float:
    """Return RMSE across sklearn versions with safe compatibility."""
    try:
        return float(mean_squared_error(y_true, y_pred, squared=False))
    except TypeError:
        return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def build_synthetic_dataset(samples: int = 500, seed: int = 42) -> pd.DataFrame:
    """Build a synthetic dataset for demonstration and model evaluation.

    This intentionally does not claim real-world predictive accuracy. It exists to
    provide a working demonstration pipeline when no real measurement data is available.
    """
    rng = np.random.default_rng(seed)
    mode_code = rng.integers(0, 2, size=samples)
    current = rng.uniform(5.0, 35.0, size=samples)
    current = np.where(mode_code == 0, current, -current * 0.8)

    voltage = np.clip(48.0 - 0.08 * np.abs(current) + rng.normal(0.0, 0.8, size=samples), 35.0, 55.0)
    soc = np.clip(
        100.0 - np.cumsum(np.abs(current) / 10.0) + rng.normal(0.0, 2.0, size=samples),
        5.0,
        95.0,
    )
    temperature = np.clip(23.0 + 0.12 * np.abs(current) + rng.normal(0.0, 2.0, size=samples), 15.0, 60.0)
    energy_wh = np.maximum(
        0.0,
        np.abs(current) * voltage * (0.6 + rng.random(samples) * 0.7) / 60.0,
    )

    df = pd.DataFrame(
        {
            "current_a": current,
            "voltage_v": voltage,
            "soc_percent": soc,
            "temperature_c": temperature,
            "mode_code": mode_code,
            "energy_consumed_wh": energy_wh,
            "time_minutes": np.arange(samples),
        }
    )
    return df


def _prepare_features(df: pd.DataFrame, include_soc: bool = True) -> pd.DataFrame:
    required = ["current_a", "voltage_v", "temperature_c", "mode_code"]
    if include_soc:
        required.insert(2, "soc_percent")
    missing = [key for key in required if key not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")
    return df[required].copy()


def save_model(model, model_path: str | Path) -> Path:
    path = Path(model_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    return path


def load_model(model_path: str | Path):
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {path}")
    return joblib.load(path)


def train_energy_predictor(
    dataframe: pd.DataFrame | None = None,
    future_horizon_minutes: int = 10,
    model_path: str | Path = "models/energy_predictor.joblib",
):
    """Train and compare a few regression models for energy prediction."""
    if dataframe is None:
        dataframe = build_synthetic_dataset()

    features = _prepare_features(dataframe, include_soc=True)
    target = dataframe["energy_consumed_wh"].copy()
    X_train, X_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=0.2,
        random_state=42,
        shuffle=True,
    )

    models = {
        "baseline": DummyRegressor(strategy="mean"),
        "linear_regression": LinearRegression(),
        "random_forest": RandomForestRegressor(
            n_estimators=200,
            random_state=42,
            max_depth=12,
            min_samples_leaf=2,
        ),
        "gradient_boosting": GradientBoostingRegressor(random_state=42),
    }

    results = {}
    evaluation_df = pd.DataFrame({"actual_energy_wh": y_test.to_numpy()})
    for name, model in models.items():
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        results[name] = {
            "mae_wh": float(mean_absolute_error(y_test, pred)),
            "rmse_wh": _rmse_score(y_test, pred),
            "predictions": pred,
        }
        evaluation_df[f"{name}_pred_wh"] = pred

    champion_name = min(results, key=lambda key: results[key]["rmse_wh"])
    champion_model = models[champion_name]
    model_file = save_model(champion_model, model_path)

    metrics = {
        "baseline_mae_wh": float(results["baseline"]["mae_wh"]),
        "baseline_rmse_wh": float(results["baseline"]["rmse_wh"]),
        "rf_mae_wh": float(results["random_forest"]["mae_wh"]),
        "rf_rmse_wh": float(results["random_forest"]["rmse_wh"]),
        "best_model": champion_name,
        "future_horizon_minutes": int(future_horizon_minutes),
        "model_path": str(model_file),
    }
    return {
        "model": champion_model,
        "baseline": models["baseline"],
        "evaluation_df": evaluation_df,
        "metrics": metrics,
        "feature_columns": list(features.columns),
        "dataset": dataframe,
        "model_comparison": results,
    }


def train_soc_predictor(
    dataframe: pd.DataFrame | None = None,
    model_path: str | Path = "models/soc_predictor.joblib",
):
    """Train and compare a few models for SoC prediction from battery telemetry."""
    if dataframe is None:
        dataframe = build_synthetic_dataset()

    features = _prepare_features(dataframe, include_soc=False)
    target = dataframe["soc_percent"].copy()
    X_train, X_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=0.2,
        random_state=42,
        shuffle=True,
    )

    models = {
        "baseline": DummyRegressor(strategy="mean"),
        "linear_regression": LinearRegression(),
        "random_forest": RandomForestRegressor(
            n_estimators=200,
            random_state=42,
            max_depth=12,
            min_samples_leaf=2,
        ),
        "gradient_boosting": GradientBoostingRegressor(random_state=42),
    }

    results = {}
    evaluation_df = pd.DataFrame({"actual_soc_percent": y_test.to_numpy()})
    for name, model in models.items():
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        results[name] = {
            "mae_pct": float(mean_absolute_error(y_test, pred)),
            "rmse_pct": _rmse_score(y_test, pred),
            "predictions": pred,
        }
        evaluation_df[f"{name}_pred_soc"] = pred

    champion_name = min(results, key=lambda key: results[key]["rmse_pct"])
    champion_model = models[champion_name]
    save_model(champion_model, model_path)
    return {
        "model": champion_model,
        "comparison": results,
        "evaluation_df": evaluation_df,
        "feature_columns": list(features.columns),
        "best_model": champion_name,
    }


def predict_energy(model, feature_row: dict) -> float:
    """Predict energy for a single feature row."""
    required = ["current_a", "voltage_v", "soc_percent", "temperature_c", "mode_code"]
    missing = [key for key in required if key not in feature_row]
    if missing:
        raise ValueError(f"Missing model inputs: {missing}")
    feature_vector = np.array([[feature_row[key] for key in required]], dtype=float)
    return float(model.predict(feature_vector)[0])


def predict_soc(model, feature_row: dict) -> float:
    """Predict SoC for a single feature row."""
    required = ["current_a", "voltage_v", "temperature_c", "mode_code"]
    missing = [key for key in required if key not in feature_row]
    if missing:
        raise ValueError(f"Missing SoC model inputs: {missing}")
    feature_vector = np.array([[feature_row[key] for key in required]], dtype=float)
    return float(model.predict(feature_vector)[0])
