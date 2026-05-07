from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ml_features import build_impulse_feature_frame, make_prediction_sequences, make_supervised_sequences
from ml_models import predict_model, train_model


@dataclass
class Standardizer:
    mean_: np.ndarray
    scale_: np.ndarray

    @classmethod
    def fit(cls, values: np.ndarray):
        mean = values.mean(axis=0)
        scale = values.std(axis=0)
        scale = np.where(scale == 0.0, 1.0, scale)
        return cls(mean, scale)

    def transform(self, values: np.ndarray) -> np.ndarray:
        return (values - self.mean_) / self.scale_

    def inverse_transform(self, values: np.ndarray) -> np.ndarray:
        return values * self.scale_ + self.mean_


def metric_summary(observed: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    observed = np.asarray(observed, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    mask = np.isfinite(observed) & np.isfinite(predicted)
    if mask.sum() < 2:
        return {"R2": np.nan, "RMSE": np.nan, "EVP": np.nan}

    obs = observed[mask]
    pred = predicted[mask]
    residual = obs - pred
    ss_res = float(np.sum(residual**2))
    ss_tot = float(np.sum((obs - obs.mean()) ** 2))
    r2 = np.nan if ss_tot == 0 else 1.0 - ss_res / ss_tot
    rmse = float(np.sqrt(np.mean(residual**2)))
    evp = np.nan
    obs_var = float(np.var(obs))
    if obs_var > 0:
        evp = max(0.0, 100.0 * (1.0 - float(np.var(residual)) / obs_var))
    return {"R2": float(r2), "RMSE": rmse, "EVP": float(evp)}


def train_evaluate_hybrid(
    frame: pd.DataFrame,
    feature_columns: list[str],
    model_type: str,
    window_size: int,
    horizon: int,
    train_fraction: float = 0.6,
    epochs: int = 60,
    learning_rate: float = 0.001,
    hidden_size: int = 32,
    target_mode: str = "hybrid",
):
    target_mode = str(target_mode or "hybrid").lower()
    if target_mode not in {"hybrid", "direct"}:
        raise ValueError(f"Unsupported ML target mode: {target_mode}")
    target_column = "target_residual" if target_mode == "hybrid" else "observed"
    x_values, y_values, target_dates = make_supervised_sequences(
        frame,
        feature_columns,
        target_column,
        window_size,
        horizon,
    )
    if not feature_columns:
        raise ValueError("Bitte mindestens ein Feature für das ML-Training aktivieren.")
    if len(x_values) < 30:
        available_targets = int(np.isfinite(frame[target_column].astype(float).to_numpy()).sum())
        raise ValueError(
            "Nicht genug nutzbare Sequenzen für 60/20/20-Training, Test und Validierung. "
            f"Diese Kombination braucht mindestens {window_size + horizon} Kalendertage "
            f"Vorlauf bis zum Zielwert und genügend Messwerte danach. "
            f"Nutzbare Zielwerte in der Station: {available_targets}, erzeugte Sequenzen: {len(x_values)}."
        )

    n_sequences = len(x_values)
    train_end = int(n_sequences * 0.6)
    test_end = int(n_sequences * 0.8)
    if train_end < 5 or test_end - train_end < 5 or n_sequences - test_end < 5:
        raise ValueError("Nicht genug Sequenzen für je mindestens 5 Test- und Validierungspunkte.")

    x_train = x_values[:train_end]
    y_train = y_values[:train_end]
    x_test = x_values[train_end:test_end]
    y_test = y_values[train_end:test_end]
    test_dates = target_dates[train_end:test_end]
    x_valid = x_values[test_end:]
    y_valid = y_values[test_end:]
    valid_dates = target_dates[test_end:]

    x_scaler = Standardizer.fit(x_train.reshape(-1, x_train.shape[-1]))
    y_scaler = Standardizer.fit(y_train.reshape(-1, 1))
    x_train_scaled = x_scaler.transform(x_train.reshape(-1, x_train.shape[-1])).reshape(x_train.shape)
    y_train_scaled = y_scaler.transform(y_train.reshape(-1, 1)).ravel()

    model, train_info = train_model(
        x_train_scaled,
        y_train_scaled,
        model_type=model_type,
        epochs=epochs,
        learning_rate=learning_rate,
        hidden_size=hidden_size,
    )

    def evaluate_split(x_split, y_split, date_split, label):
        x_split_scaled = x_scaler.transform(x_split.reshape(-1, x_split.shape[-1])).reshape(x_split.shape)
        target_prediction_scaled = predict_model(model, x_split_scaled)
        target_prediction = y_scaler.inverse_transform(
            target_prediction_scaled.reshape(-1, 1)
        ).ravel()
        split_df = frame.loc[date_split, ["observed", "pastas_sim"]].copy()
        split_df["split"] = label
        split_df["target_mode"] = target_mode
        split_df["ml_prediction"] = target_prediction
        split_df["target_value"] = y_split
        if target_mode == "hybrid":
            split_df["ml_residual_pred"] = target_prediction
            split_df["ml_direct_pred"] = np.nan
            if split_df["pastas_sim"].notna().any():
                split_df["final_prediction"] = split_df["pastas_sim"] + split_df["ml_residual_pred"]
            else:
                split_df["final_prediction"] = split_df["ml_residual_pred"]
        else:
            split_df["ml_direct_pred"] = target_prediction
            split_df["ml_residual_pred"] = split_df["ml_direct_pred"] - split_df["pastas_sim"]
            split_df["final_prediction"] = split_df["ml_direct_pred"]
        split_df["hybrid_prediction"] = split_df["final_prediction"]
        metrics = metric_summary(split_df["observed"].values, split_df["final_prediction"].values)
        return split_df.reset_index().rename(columns={"index": "date"}), metrics

    test_df, test_metrics = evaluate_split(x_test, y_test, test_dates, "test")
    validation_df, validation_metrics = evaluate_split(x_valid, y_valid, valid_dates, "validation")
    evaluation_df = pd.concat([test_df, validation_df], ignore_index=True)
    return {
        "model": model,
        "x_scaler": x_scaler,
        "y_scaler": y_scaler,
        "train_info": train_info,
        "test": test_df,
        "validation": validation_df,
        "evaluation": evaluation_df,
        "metrics": validation_metrics,
        "test_metrics": test_metrics,
        "validation_metrics": validation_metrics,
        "n_train": int(len(x_train)),
        "n_test": int(len(x_test)),
        "n_valid": int(len(x_valid)),
        "feature_columns": feature_columns,
        "target_mode": target_mode,
        "target_column": target_column,
        "split": {"train": 0.6, "test": 0.2, "validation": 0.2},
    }


def build_artifacts(
    result: dict,
    model_type: str,
    window_size: int,
    horizon: int,
    hidden_size: int,
    target_mode: str = "hybrid",
) -> dict:
    return {
        "model": result["model"],
        "x_scaler": result["x_scaler"],
        "y_scaler": result["y_scaler"],
        "feature_columns": list(result["feature_columns"]),
        "model_type": model_type,
        "target_mode": result.get("target_mode", target_mode),
        "window_size": int(window_size),
        "horizon": int(horizon),
        "hidden_size": int(hidden_size),
    }


def predict_ml_values(frame: pd.DataFrame, artifacts: dict) -> pd.DataFrame:
    feature_columns = list(artifacts.get("feature_columns", []))
    if not feature_columns:
        return pd.DataFrame()

    x_values, dates = make_prediction_sequences(
        frame,
        feature_columns,
        int(artifacts.get("window_size", 30)),
        int(artifacts.get("horizon", 1)),
    )
    if len(x_values) == 0:
        return pd.DataFrame()

    x_scaler = artifacts["x_scaler"]
    y_scaler = artifacts["y_scaler"]
    x_scaled = x_scaler.transform(x_values.reshape(-1, x_values.shape[-1])).reshape(x_values.shape)
    prediction_scaled = predict_model(artifacts["model"], x_scaled)
    prediction = y_scaler.inverse_transform(prediction_scaled.reshape(-1, 1)).ravel()
    target_mode = str(artifacts.get("target_mode", "hybrid") or "hybrid").lower()
    result = pd.DataFrame({"date": dates, "ml_prediction": prediction, "target_mode": target_mode})
    if target_mode == "hybrid":
        result["ml_residual_pred"] = prediction
    else:
        result["ml_direct_pred"] = prediction
    return result


def predict_hybrid_residuals(frame: pd.DataFrame, artifacts: dict) -> pd.DataFrame:
    prediction = predict_ml_values(frame, artifacts)
    if prediction.empty:
        return prediction
    if "ml_residual_pred" not in prediction.columns:
        prediction["ml_residual_pred"] = prediction.get("ml_prediction", np.nan)
    return prediction


def predict_impulse_response(
    artifacts: dict,
    impulse_mm: float = 25.0,
    response_days: int = 730,
) -> pd.DataFrame:
    feature_columns = list(artifacts.get("feature_columns", []))
    if not feature_columns:
        return pd.DataFrame()

    impulse_frame, impulse_date = build_impulse_feature_frame(
        feature_columns,
        int(artifacts.get("window_size", 30)),
        int(artifacts.get("horizon", 1)),
        int(response_days),
        float(impulse_mm),
    )
    baseline_frame, _ = build_impulse_feature_frame(
        feature_columns,
        int(artifacts.get("window_size", 30)),
        int(artifacts.get("horizon", 1)),
        int(response_days),
        0.0,
    )
    impulse_prediction = predict_hybrid_residuals(impulse_frame, artifacts)
    baseline_prediction = predict_hybrid_residuals(baseline_frame, artifacts)
    if impulse_prediction.empty or baseline_prediction.empty:
        return pd.DataFrame()

    response = impulse_prediction.merge(
        baseline_prediction.rename(columns={"ml_residual_pred": "baseline_residual_pred"}),
        on="date",
        how="inner",
    )
    response["lag_days"] = (pd.to_datetime(response["date"]) - impulse_date).dt.days
    response = response[(response["lag_days"] >= 0) & (response["lag_days"] <= int(response_days))]
    response["ml_residual_response"] = (
        response["ml_residual_pred"] - response["baseline_residual_pred"]
    )
    response["impulse_mm"] = float(impulse_mm)
    return response.reset_index(drop=True)
