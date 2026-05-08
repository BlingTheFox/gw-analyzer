from __future__ import annotations

import io
import json
import zipfile
from datetime import datetime
from pathlib import PurePosixPath

import pandas as pd

from ml_models import build_model
from ml_training import Standardizer


MAX_ML_PACKAGE_BYTES = 250 * 1024 * 1024
MAX_ML_ZIP_ENTRY_BYTES = 50 * 1024 * 1024
MAX_ML_ZIP_ENTRIES = 100


def validate_ml_zip_archive(archive: zipfile.ZipFile) -> None:
    infos = archive.infolist()
    if len(infos) > MAX_ML_ZIP_ENTRIES:
        raise ValueError("ML package contains too many files.")

    total_size = 0
    for info in infos:
        member_path = PurePosixPath(info.filename)
        if member_path.is_absolute() or ".." in member_path.parts:
            raise ValueError("ML package contains an unsafe path.")
        if info.file_size > MAX_ML_ZIP_ENTRY_BYTES:
            raise ValueError("ML package contains a file that is too large.")
        total_size += int(info.file_size)
        if total_size > MAX_ML_PACKAGE_BYTES:
            raise ValueError("ML package is too large.")


def ml_results_to_csv_bytes(df: pd.DataFrame) -> bytes:
    if df is None or df.empty:
        return b""
    return df.to_csv(index=False, sep=";").encode("utf-8")


def _serialize_artifact(artifact: dict) -> dict:
    return {
        "model_type": artifact.get("model_type"),
        "target_mode": artifact.get("target_mode", "hybrid"),
        "display_name": artifact.get("display_name"),
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
        "target_mode": payload.get("target_mode", "hybrid"),
        "display_name": payload.get("display_name"),
        "station": payload.get("station"),
        "window_size": int(payload.get("window_size", 30)),
        "horizon": int(payload.get("horizon", 1)),
        "hidden_size": int(payload.get("hidden_size", 32) or 32),
    }


def load_ml_artifacts_bytes(data: bytes) -> dict:
    import torch

    buffer = io.BytesIO(data)
    try:
        payload = torch.load(buffer, map_location="cpu", weights_only=True)
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


def load_ml_run_package_bytes(data: bytes) -> dict:
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        validate_ml_zip_archive(archive)
        manifest = json.loads(archive.read("manifest.json").decode("utf-8"))
        files = manifest.get("files", {})

        def read_csv(name: str) -> pd.DataFrame:
            archive_path = files.get(name)
            if not archive_path:
                return pd.DataFrame()
            with archive.open(archive_path) as handle:
                df = pd.read_csv(handle, sep=";")
            if "date" in df.columns:
                df["date"] = pd.to_datetime(df["date"], errors="coerce")
            return df

        artifacts = {}
        if files.get("model"):
            artifacts = load_ml_artifacts_bytes(archive.read(files["model"]))

        return {
            "manifest": manifest,
            "metadata": manifest.get("metadata", {}),
            "validation_df": read_csv("validation"),
            "forecast_df": read_csv("forecast"),
            "impulse_df": read_csv("impulse_response"),
            "summary_df": read_csv("summary"),
            "artifacts": artifacts,
        }
