from __future__ import annotations

import io
import json
import zipfile
from datetime import datetime

import pandas as pd

from ml_models import build_model
from ml_training import Standardizer


def ml_results_to_csv_bytes(df: pd.DataFrame) -> bytes:
    if df is None or df.empty:
        return b""
    return df.to_csv(index=False, sep=";").encode("utf-8")


def _serialize_artifact(artifact: dict) -> dict:
    return {
        "model_type": artifact.get("model_type"),
        "station": artifact.get("station"),
        "window_size": artifact.get("window_size"),
        "horizon": artifact.get("horizon"),
        "hidden_size": artifact.get("hidden_size"),
        "feature_columns": artifact.get("feature_columns", []),
        "x_scaler": {
            "mean": artifact["x_scaler"].mean_.tolist(),
            "scale": artifact["x_scaler"].scale_.tolist(),
        },
        "y_scaler": {
            "mean": artifact["y_scaler"].mean_.tolist(),
            "scale": artifact["y_scaler"].scale_.tolist(),
        },
        "model_state_dict": artifact["model"].state_dict(),
    }


def save_ml_artifacts_bytes(artifacts: dict) -> bytes:
    try:
        import torch
    except Exception:
        return b""

    if not artifacts:
        return b""

    if "model" in artifacts:
        payload = _serialize_artifact(artifacts)
    else:
        payload = {
            "format": "multi",
            "artifacts": {
                str(name): _serialize_artifact(artifact)
                for name, artifact in artifacts.items()
                if isinstance(artifact, dict) and "model" in artifact
            },
        }
        if not payload["artifacts"]:
            return b""

    buffer = io.BytesIO()
    torch.save(payload, buffer)
    return buffer.getvalue()


def _deserialize_artifact(payload: dict) -> dict:
    import numpy as np

    feature_columns = list(payload.get("feature_columns", []))
    model = build_model(
        payload.get("model_type", "CNN"),
        n_features=len(feature_columns),
        hidden_size=int(payload.get("hidden_size", 32) or 32),
    )
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    return {
        "model": model,
        "x_scaler": Standardizer(
            np.asarray(payload["x_scaler"]["mean"], dtype=float),
            np.asarray(payload["x_scaler"]["scale"], dtype=float),
        ),
        "y_scaler": Standardizer(
            np.asarray(payload["y_scaler"]["mean"], dtype=float),
            np.asarray(payload["y_scaler"]["scale"], dtype=float),
        ),
        "feature_columns": feature_columns,
        "model_type": payload.get("model_type"),
        "station": payload.get("station"),
        "window_size": int(payload.get("window_size", 30)),
        "horizon": int(payload.get("horizon", 1)),
        "hidden_size": int(payload.get("hidden_size", 32) or 32),
    }


def load_ml_artifacts_bytes(data: bytes) -> dict:
    import torch

    buffer = io.BytesIO(data)
    try:
        payload = torch.load(buffer, map_location="cpu", weights_only=False)
    except TypeError:
        buffer.seek(0)
        payload = torch.load(buffer, map_location="cpu")
    if payload.get("format") == "multi":
        return {
            name: _deserialize_artifact(artifact_payload)
            for name, artifact_payload in payload.get("artifacts", {}).items()
        }
    return _deserialize_artifact(payload)


def create_ml_run_package(
    validation_df: pd.DataFrame,
    metadata: dict,
    forecast_df: pd.DataFrame | None = None,
    impulse_df: pd.DataFrame | None = None,
    artifacts: dict | None = None,
    summary_df: pd.DataFrame | None = None,
) -> bytes:
    buffer = io.BytesIO()
    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "metadata": metadata,
        "files": {},
    }
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        if validation_df is not None and not validation_df.empty:
            manifest["files"]["validation"] = "validation.csv"
            archive.writestr("validation.csv", ml_results_to_csv_bytes(validation_df))
        if forecast_df is not None and not forecast_df.empty:
            manifest["files"]["forecast"] = "forecast.csv"
            archive.writestr("forecast.csv", ml_results_to_csv_bytes(forecast_df))
        if impulse_df is not None and not impulse_df.empty:
            manifest["files"]["impulse_response"] = "impulse_response.csv"
            archive.writestr("impulse_response.csv", ml_results_to_csv_bytes(impulse_df))
        if summary_df is not None and not summary_df.empty:
            manifest["files"]["summary"] = "summary.csv"
            archive.writestr("summary.csv", ml_results_to_csv_bytes(summary_df))
        artifact_bytes = save_ml_artifacts_bytes(artifacts or {})
        if artifact_bytes:
            manifest["files"]["model"] = "model_state.pt"
            archive.writestr("model_state.pt", artifact_bytes)
        archive.writestr("manifest.json", json.dumps(manifest, ensure_ascii=False, indent=2))
    buffer.seek(0)
    return buffer.getvalue()
