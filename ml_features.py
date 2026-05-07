from __future__ import annotations

import numpy as np
import pandas as pd


def _daily_series(series: pd.Series, index: pd.DatetimeIndex, fill_value=None) -> pd.Series:
    values = series.copy()
    values.index = pd.to_datetime(values.index)
    values = values.sort_index().resample("D").mean()
    values = values.reindex(index)
    if fill_value is not None:
        return values.fillna(fill_value)
    return values.interpolate(limit=14).ffill().bfill()


def build_hybrid_feature_frame(
    station: str,
    gw_df: pd.DataFrame,
    rain: pd.Series,
    evap: pd.Series,
    pastas_model=None,
    include_head=False,
    include_weather=True,
    include_rollings=True,
    include_season=False,
) -> pd.DataFrame:
    """Build daily features for residual learning.

    Target is `observed - pastas_sim` when a Pastas model is available. This keeps
    the neural model focused on correcting the physically interpretable baseline.
    For the PASTAS comparison setup, previous groundwater heads and seasonal
    sin/cos proxies are intentionally excluded from the feature set.
    """

    include_head = False
    include_season = False

    observed = gw_df[station].dropna().astype(float)
    observed.index = pd.to_datetime(observed.index)
    observed = observed.sort_index().resample("D").mean()
    if observed.empty:
        return pd.DataFrame()

    index = pd.date_range(observed.index.min(), observed.index.max(), freq="D")
    frame = pd.DataFrame(index=index)
    frame["observed"] = observed.reindex(index)
    frame["head_filled"] = frame["observed"].interpolate(limit=14).ffill().bfill()

    if pastas_model is not None:
        simulation = pastas_model.simulate(tmin=index.min(), tmax=index.max())
        simulation = simulation.reindex(index).interpolate(limit=14).ffill().bfill()
        frame["pastas_sim"] = simulation.astype(float)
        frame["target_residual"] = frame["observed"] - frame["pastas_sim"]
    else:
        frame["pastas_sim"] = np.nan
        frame["target_residual"] = frame["observed"]

    rain_daily = _daily_series(rain, index, fill_value=0.0).astype(float)
    evap_daily = _daily_series(evap, index, fill_value=0.0).astype(float)

    if include_weather:
        frame["rain"] = rain_daily
        frame["evap"] = evap_daily

    if include_rollings:
        for window in (7, 30, 90):
            frame[f"rain_sum_{window}"] = rain_daily.rolling(window, min_periods=1).sum()
            frame[f"evap_mean_{window}"] = evap_daily.rolling(window, min_periods=1).mean()

    if include_season:
        frame = pd.concat([frame, _season_features(index)], axis=1)

    feature_columns = [
        column
        for column in frame.columns
        if column not in {"observed", "pastas_sim", "target_residual", "head_filled"}
    ]
    # Keep the daily weather history. Groundwater observations may be weekly or
    # irregular, so targets are filtered later when supervised sequences are built.
    if feature_columns:
        frame = frame.dropna(subset=feature_columns)
    return frame


def build_hybrid_forecast_feature_frame(
    station: str,
    gw_df: pd.DataFrame,
    forecast_df: pd.DataFrame,
) -> pd.DataFrame:
    """Build feature rows for Pastas forecast periods.

    This mirrors the restricted training setup: weather-derived inputs only,
    no previous groundwater heads and no seasonal sin/cos proxies.
    """

    if forecast_df is None or forecast_df.empty:
        return pd.DataFrame()

    source = forecast_df.copy()
    source["date"] = pd.to_datetime(source["date"], errors="coerce")
    source = source.dropna(subset=["date"]).sort_values("date")
    source = source.drop_duplicates(subset=["date"], keep="last")
    source = source.set_index("date")

    if "simulated_head" in source.columns:
        pastas_sim = source["simulated_head"].astype(float)
    elif "pastas_sim" in source.columns:
        pastas_sim = source["pastas_sim"].astype(float)
    else:
        return pd.DataFrame()

    index = pd.date_range(source.index.min(), source.index.max(), freq="D")
    frame = pd.DataFrame(index=index)
    frame["pastas_sim"] = pastas_sim.reindex(index).interpolate(limit=14).ffill().bfill()

    observed = pd.Series(index=index, dtype=float)
    if station in gw_df.columns:
        observed = gw_df[station].dropna().astype(float)
        observed.index = pd.to_datetime(observed.index)
        observed = observed.sort_index().resample("D").mean().reindex(index)

    frame["observed"] = observed
    frame["target_residual"] = frame["observed"] - frame["pastas_sim"]
    frame["head_filled"] = frame["observed"].combine_first(frame["pastas_sim"])

    rain_daily = _daily_series(source.get("rain", pd.Series(dtype=float)), index, fill_value=0.0).astype(float)
    evap_daily = _daily_series(source.get("evap", pd.Series(dtype=float)), index, fill_value=0.0).astype(float)
    frame["rain"] = rain_daily
    frame["evap"] = evap_daily
    for window in (7, 30, 90):
        frame[f"rain_sum_{window}"] = rain_daily.rolling(window, min_periods=1).sum()
        frame[f"evap_mean_{window}"] = evap_daily.rolling(window, min_periods=1).mean()

    return frame


def build_impulse_feature_frame(
    feature_columns: list[str],
    window_size: int,
    horizon: int,
    response_days: int,
    impulse_mm: float,
) -> tuple[pd.DataFrame, pd.Timestamp]:
    window_size = int(window_size)
    horizon = int(horizon)
    response_days = int(response_days)
    length = window_size + horizon + response_days + 1
    index = pd.date_range("2000-01-01", periods=length, freq="D")
    impulse_date = index[window_size - 1]

    rain_daily = pd.Series(0.0, index=index)
    evap_daily = pd.Series(0.0, index=index)
    rain_daily.loc[impulse_date] = float(impulse_mm)

    frame = pd.DataFrame(index=index)
    frame["observed"] = np.nan
    frame["pastas_sim"] = 0.0
    frame["target_residual"] = np.nan
    frame["head_filled"] = 0.0
    if "rain" in feature_columns:
        frame["rain"] = rain_daily
    if "evap" in feature_columns:
        frame["evap"] = evap_daily
    for window in (7, 30, 90):
        if f"rain_sum_{window}" in feature_columns:
            frame[f"rain_sum_{window}"] = rain_daily.rolling(window, min_periods=1).sum()
        if f"evap_mean_{window}" in feature_columns:
            frame[f"evap_mean_{window}"] = evap_daily.rolling(window, min_periods=1).mean()

    for column in feature_columns:
        if column not in frame.columns:
            frame[column] = 0.0

    return frame, impulse_date


def make_supervised_sequences(
    frame: pd.DataFrame,
    feature_columns: list[str],
    target_column: str,
    window_size: int,
    horizon: int,
):
    window_size = int(window_size)
    horizon = int(horizon)
    if frame.empty or len(frame) < window_size + horizon:
        return (
            np.empty((0, window_size, len(feature_columns))),
            np.empty((0,)),
            pd.DatetimeIndex([]),
        )

    values = frame[feature_columns].astype(float).to_numpy()
    target = frame[target_column].astype(float).to_numpy()
    dates = []
    x_values = []
    y_values = []

    target_indices = np.flatnonzero(np.isfinite(target))
    for target_index in target_indices:
        start = int(target_index) - window_size - horizon + 1
        if start < 0:
            continue
        end = start + window_size
        feature_window = values[start:end]
        if len(feature_window) != window_size or not np.isfinite(feature_window).all():
            continue
        x_values.append(feature_window)
        y_values.append(target[int(target_index)])
        dates.append(frame.index[target_index])

    return (
        np.asarray(x_values, dtype=np.float32),
        np.asarray(y_values, dtype=np.float32),
        pd.DatetimeIndex(dates),
    )


def make_prediction_sequences(
    frame: pd.DataFrame,
    feature_columns: list[str],
    window_size: int,
    horizon: int,
):
    window_size = int(window_size)
    horizon = int(horizon)
    frame = frame.dropna(subset=feature_columns)
    if frame.empty or len(frame) < window_size + horizon:
        return np.empty((0, window_size, len(feature_columns))), pd.DatetimeIndex([])

    values = frame[feature_columns].astype(float).to_numpy()
    dates = []
    x_values = []

    for start in range(0, len(frame) - window_size - horizon + 1):
        end = start + window_size
        target_index = end + horizon - 1
        x_values.append(values[start:end])
        dates.append(frame.index[target_index])

    return np.asarray(x_values, dtype=np.float32), pd.DatetimeIndex(dates)
