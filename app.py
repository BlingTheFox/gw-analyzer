import io
from copy import deepcopy
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import pastas as ps
import streamlit as st
from pandas.api.types import is_bool_dtype, is_numeric_dtype

# Project concept and application lead: Robin Carow / RCnet (development started 20.04.2026)
# Project made for the ZALF
# Modeling framework credit: Pastas by Collenteur et al. (2019), https://doi.org/10.1111/gwat.12925


st.set_page_config(page_title="GW Analyzer", layout="wide")


if "lang" not in st.session_state:
    st.session_state.lang = "Deutsch"
if "history" not in st.session_state:
    st.session_state.history = pd.DataFrame()
if "last_run_results" not in st.session_state:
    st.session_state.last_run_results = pd.DataFrame()
if "last_run_models" not in st.session_state:
    st.session_state.last_run_models = {}
if "last_run_label" not in st.session_state:
    st.session_state.last_run_label = None
if "last_extra_match" not in st.session_state:
    st.session_state.last_extra_match = {}


TEXT = {
    "Deutsch": {
        "title": "Grundwasser Analyse Tool",
        "sidebar": "Einstellungen",
        "lang": "Sprache / Language",
        "upload_gw": "1. Grundwasserdaten hochladen (CSV, XLSX, ODS)",
        "upload_weather": "2. Wetterdaten hochladen (CSV, XLSX, ODS)",
        "upload_extra": "3. Zusatzdaten hochladen (optional: CPC/PC CSV, XLSX, ODS)",
        "waiting": "Bitte zuerst die Dateien links in der Sidebar hochladen.",
        "load_error": "Daten konnten nicht geladen werden",
        "config_heading": "Konfiguration",
        "mode": "Auswertungsmodus",
        "mode_single": "Einzelne Messstelle",
        "mode_multi": "Mehrere Messstellen",
        "mode_all": "Alle Messstellen",
        "station": "Messstelle",
        "stations": "Messstellen",
        "available_stations": "Messstellen im Datensatz",
        "selection_count": "Ausgewaehlte Messstellen",
        "model": "Reaktionsmodell",
        "use_flex": "FlexModel verwenden",
        "use_noise": "NoiseModel verwenden",
        "parameter_box": "Modellparameter anpassen",
        "parameter_help": "Hier kannst du Startwerte und Optimierung einzelner Parameter steuern.",
        "vary": "optimieren",
        "response_cutoff": "Response cutoff",
        "noise_norm": "Noise normalisieren",
        "run": "Analyse starten",
        "choose_station": "Bitte mindestens eine Messstelle auswaehlen.",
        "computing": "Berechne Modelle...",
        "progress": "Aktuelle Messstelle",
        "success": "Berechnung abgeschlossen.",
        "summary": "Zusammenfassung des letzten Laufs",
        "run_label": "Lauf",
        "successful_runs": "Erfolgreiche Modelle",
        "failed_runs": "Fehler oder Uebersprungen",
        "best_station": "Beste Messstelle (R2)",
        "mean_r2": "Mittleres R2",
        "extra_match": "Zusatzdaten gematcht",
        "extra_missing": "Messstellen ohne Zusatzdaten",
        "extra_unused": "Zusatzdaten ohne passenden Lauf",
        "tab_plot": "Visualisierung",
        "tab_compare": "Vergleich",
        "tab_save": "Speichern und Laden",
        "no_run": "Fuehre zuerst eine Analyse aus, um Modelle zu visualisieren.",
        "plot_station": "Messstelle fuer die Visualisierung",
        "plot_type": "Visualisierungsmethode",
        "plot_overview": "Pastas Uebersicht",
        "plot_obs_sim": "Beobachtet vs. simuliert",
        "plot_residuals": "Residuale ueber Zeit",
        "plot_res_hist": "Residual-Histogramm",
        "plot_step": "Schrittantwort",
        "observed": "Beobachtet",
        "simulated": "Simuliert",
        "residuals": "Residuale",
        "head_axis": "Grundwasserstand",
        "count_axis": "Anzahl",
        "days_axis": "Tage",
        "response_axis": "Antwort",
        "download_plot": "Diagramm herunterladen",
        "result_table": "Ergebnistabelle",
        "compact_table_note": "Die Standardansicht zeigt nur die wichtigsten Spalten.",
        "extended_table": "Erweiterte Tabelle mit Parametern anzeigen",
        "compare_scope": "Datenbasis",
        "scope_last": "Letzter Lauf",
        "scope_all": "Gesamte Historie",
        "chart_type": "Diagrammtyp",
        "chart_ranking": "Ranking Balkendiagramm",
        "chart_scatter": "Scatterplot",
        "chart_hist": "Histogramm",
        "chart_box": "Boxplot",
        "chart_corr": "Korrelationsmatrix",
        "metric": "Kennzahl",
        "x_axis": "X-Achse",
        "y_axis": "Y-Achse",
        "box_cols": "Spalten fuer den Boxplot",
        "corr_cols": "Spalten fuer die Korrelationsmatrix",
        "no_numeric": "Es sind nicht genug numerische Spalten fuer diese Visualisierung vorhanden.",
        "export_csv": "Ergebnisse als CSV exportieren",
        "import_csv": "Ergebnisse importieren",
        "import_button": "Import starten",
        "import_success": "Ergebnisse wurden importiert.",
        "error_table": "Fehlermeldungen",
        "too_few_obs": "Zu wenige Beobachtungen (<50).",
        "no_valid_stations": "Es wurden keine gueltigen Messstellen mit mehr als 50 Beobachtungen gefunden.",
        "accessibility": "Barrierefreiheit",
        "font_profile": "Schriftprofil",
        "font_standard": "Standard",
        "font_readable": "Lesefreundlich",
        "font_dyslexia": "Dyslexia/LRS-freundlich",
        "font_note": "Die Dyslexia-Option nutzt OpenDyslexic mit Atkinson-Fallback, falls verfuegbar.",
        "text_scale": "Schriftgroesse (%)",
        "high_contrast": "Hoher Kontrast",
        "strong_focus": "Starke Fokusmarkierung",
    },
    "English": {
        "title": "Groundwater Analysis Tool",
        "sidebar": "Settings",
        "lang": "Sprache / Language",
        "upload_gw": "1. Upload groundwater data (CSV, XLSX, ODS)",
        "upload_weather": "2. Upload weather data (CSV, XLSX, ODS)",
        "upload_extra": "3. Upload extra station data (optional: CPC/PC CSV, XLSX, ODS)",
        "waiting": "Please upload the files in the sidebar first.",
        "load_error": "Data could not be loaded",
        "config_heading": "Configuration",
        "mode": "Evaluation mode",
        "mode_single": "Single station",
        "mode_multi": "Multiple stations",
        "mode_all": "All stations",
        "station": "Station",
        "stations": "Stations",
        "available_stations": "Stations in dataset",
        "selection_count": "Selected stations",
        "model": "Response model",
        "use_flex": "Use FlexModel",
        "use_noise": "Use noise model",
        "parameter_box": "Adjust model parameters",
        "parameter_help": "Set initial values and control which parameters remain free during optimization.",
        "vary": "optimize",
        "response_cutoff": "Response cutoff",
        "noise_norm": "Normalize noise",
        "run": "Run analysis",
        "choose_station": "Please select at least one station.",
        "computing": "Computing models...",
        "progress": "Current station",
        "success": "Calculation finished.",
        "summary": "Last run summary",
        "run_label": "Run",
        "successful_runs": "Successful models",
        "failed_runs": "Failed or skipped",
        "best_station": "Best station (R2)",
        "mean_r2": "Mean R2",
        "extra_match": "Matched extra data",
        "extra_missing": "Stations without extra data",
        "extra_unused": "Extra-data rows without matching run",
        "tab_plot": "Visualization",
        "tab_compare": "Comparison",
        "tab_save": "Save and Load",
        "no_run": "Run an analysis first to visualize models.",
        "plot_station": "Station for visualization",
        "plot_type": "Visualization method",
        "plot_overview": "Pastas overview",
        "plot_obs_sim": "Observed vs simulated",
        "plot_residuals": "Residuals over time",
        "plot_res_hist": "Residual histogram",
        "plot_step": "Step response",
        "observed": "Observed",
        "simulated": "Simulated",
        "residuals": "Residuals",
        "head_axis": "Groundwater head",
        "count_axis": "Count",
        "days_axis": "Days",
        "response_axis": "Response",
        "download_plot": "Download plot",
        "result_table": "Result table",
        "compact_table_note": "The default view only shows the most important columns.",
        "extended_table": "Show extended table with parameters",
        "compare_scope": "Data scope",
        "scope_last": "Last run",
        "scope_all": "Full history",
        "chart_type": "Chart type",
        "chart_ranking": "Ranking bar chart",
        "chart_scatter": "Scatter plot",
        "chart_hist": "Histogram",
        "chart_box": "Box plot",
        "chart_corr": "Correlation matrix",
        "metric": "Metric",
        "x_axis": "X axis",
        "y_axis": "Y axis",
        "box_cols": "Columns for box plot",
        "corr_cols": "Columns for correlation matrix",
        "no_numeric": "There are not enough numeric columns for this visualization.",
        "export_csv": "Export results as CSV",
        "import_csv": "Import results",
        "import_button": "Start import",
        "import_success": "Results were imported.",
        "error_table": "Error messages",
        "too_few_obs": "Too few observations (<50).",
        "no_valid_stations": "No valid stations with more than 50 observations were found.",
        "accessibility": "Accessibility",
        "font_profile": "Font profile",
        "font_standard": "Standard",
        "font_readable": "Readable",
        "font_dyslexia": "Dyslexia-friendly",
        "font_note": "The dyslexia option uses OpenDyslexic with Atkinson fallback when available.",
        "text_scale": "Text size (%)",
        "high_contrast": "High contrast",
        "strong_focus": "Strong focus highlight",
    },
}


PARAMETER_TEMPLATES = {
    ("Gamma", False): [
        {"name": "recharge_A", "label": "Recharge A", "initial": 0.220341, "vary": True},
        {"name": "recharge_n", "label": "Gamma n", "initial": 1.0, "vary": True},
        {"name": "recharge_a", "label": "Gamma a", "initial": 10.0, "vary": True},
        {"name": "recharge_f", "label": "Evaporation factor f", "initial": -1.0, "vary": True},
    ],
    ("Exponential", False): [
        {"name": "recharge_A", "label": "Recharge A", "initial": 0.220341, "vary": True},
        {"name": "recharge_a", "label": "Exponential a", "initial": 10.0, "vary": True},
        {"name": "recharge_f", "label": "Evaporation factor f", "initial": -1.0, "vary": True},
    ],
    ("Gamma", True): [
        {"name": "recharge_A", "label": "Recharge A", "initial": 0.906149, "vary": True},
        {"name": "recharge_n", "label": "Gamma n", "initial": 1.0, "vary": True},
        {"name": "recharge_a", "label": "Gamma a", "initial": 10.0, "vary": True},
        {"name": "recharge_srmax", "label": "Root zone storage srmax", "initial": 250.0, "vary": True},
        {"name": "recharge_lp", "label": "Soil moisture lp", "initial": 0.25, "vary": False},
        {"name": "recharge_ks", "label": "Conductivity ks", "initial": 100.0, "vary": True},
        {"name": "recharge_gamma", "label": "Recharge gamma", "initial": 2.0, "vary": True},
        {"name": "recharge_kv", "label": "Percolation factor kv", "initial": 1.0, "vary": True},
        {"name": "recharge_simax", "label": "Interception simax", "initial": 2.0, "vary": False},
    ],
    ("Exponential", True): [
        {"name": "recharge_A", "label": "Recharge A", "initial": 0.906149, "vary": True},
        {"name": "recharge_a", "label": "Exponential a", "initial": 10.0, "vary": True},
        {"name": "recharge_srmax", "label": "Root zone storage srmax", "initial": 250.0, "vary": True},
        {"name": "recharge_lp", "label": "Soil moisture lp", "initial": 0.25, "vary": False},
        {"name": "recharge_ks", "label": "Conductivity ks", "initial": 100.0, "vary": True},
        {"name": "recharge_gamma", "label": "Recharge gamma", "initial": 2.0, "vary": True},
        {"name": "recharge_kv", "label": "Percolation factor kv", "initial": 1.0, "vary": True},
        {"name": "recharge_simax", "label": "Interception simax", "initial": 2.0, "vary": False},
    ],
}


def normalize_station_name(value):
    return str(value).replace('"', "").strip()


def infer_step(value):
    value = abs(float(value))
    if value >= 100:
        return 1.0
    if value >= 10:
        return 0.5
    if value >= 1:
        return 0.1
    return 0.01


def find_matching_column(columns, keywords):
    lowered = [str(column).lower() for column in columns]
    for keyword in keywords:
        for column, lowered_name in zip(columns, lowered):
            if keyword in lowered_name:
                return column
    raise ValueError(f"No matching column found for keywords: {keywords}")


def read_table(source):
    if source is None:
        return None

    if isinstance(source, (str, Path)):
        path = Path(source)
        ext = path.suffix.lower()
        if ext == ".csv":
            with path.open("r", encoding="utf-8-sig", errors="ignore") as handle:
                first_line = handle.readline()
            separator = ";" if first_line.count(";") >= first_line.count(",") else ","
            return pd.read_csv(path, sep=separator)
        if ext == ".ods":
            return pd.read_excel(path, engine="odf")
        return pd.read_excel(path)

    ext = Path(source.name).suffix.lower()
    raw_bytes = source.getvalue()
    if ext == ".csv":
        content = raw_bytes.decode("utf-8-sig", errors="ignore")
        first_line = content.splitlines()[0] if content.splitlines() else ""
        separator = ";" if first_line.count(";") >= first_line.count(",") else ","
        return pd.read_csv(io.StringIO(content), sep=separator)
    if ext == ".ods":
        return pd.read_excel(io.BytesIO(raw_bytes), engine="odf")
    return pd.read_excel(io.BytesIO(raw_bytes))


@st.cache_data
def load_data(gw_source, weather_source, extra_source):
    gw_df = read_table(gw_source)
    weather_df = read_table(weather_source)
    extra_df = read_table(extra_source) if extra_source is not None else None

    gw_df.columns = [normalize_station_name(column) for column in gw_df.columns]
    weather_df.columns = [normalize_station_name(column) for column in weather_df.columns]

    gw_date_col = find_matching_column(gw_df.columns, ["datum", "date"])
    gw_df[gw_date_col] = pd.to_datetime(gw_df[gw_date_col], errors="coerce")
    gw_df = gw_df.dropna(subset=[gw_date_col]).set_index(gw_date_col).sort_index()
    gw_df = gw_df.loc["1983":].dropna(axis=1, how="all")
    for column in gw_df.columns:
        gw_df[column] = pd.to_numeric(gw_df[column], errors="coerce")

    weather_date_col = find_matching_column(weather_df.columns, ["datum", "date"])
    weather_df[weather_date_col] = pd.to_datetime(
        weather_df[weather_date_col], errors="coerce", dayfirst=True
    )
    weather_df = weather_df.dropna(subset=[weather_date_col]).set_index(weather_date_col).sort_index()

    rain_col = find_matching_column(weather_df.columns, ["niederschlag", "rain", "prec"])
    evap_col = find_matching_column(weather_df.columns, ["verdunstung", "evap"])
    weather_df[rain_col] = pd.to_numeric(weather_df[rain_col], errors="coerce")
    weather_df[evap_col] = pd.to_numeric(weather_df[evap_col], errors="coerce")

    rain = weather_df[rain_col].resample("D").mean().fillna(0.0)
    evap = weather_df[evap_col].resample("D").mean().fillna(0.0)

    if extra_df is not None:
        extra_df.columns = [normalize_station_name(column) for column in extra_df.columns]
        site_candidates = [column for column in extra_df.columns if column.lower() in {"site", "messstelle", "station"}]
        site_col = site_candidates[0] if site_candidates else extra_df.columns[0]
        extra_df[site_col] = extra_df[site_col].map(normalize_station_name)
        extra_df = extra_df.rename(columns={site_col: "Messstelle"})
        for column in extra_df.columns:
            if column != "Messstelle":
                extra_df[column] = pd.to_numeric(extra_df[column], errors="coerce")

    return gw_df, rain, evap, extra_df


def get_parameter_specs(model_type, use_flex):
    return deepcopy(PARAMETER_TEMPLATES[(model_type, use_flex)])


def build_model(
    head,
    rain,
    evap,
    model_type,
    use_flex,
    use_noise,
    parameter_values,
    vary_flags,
    response_cutoff,
    noise_norm,
):
    model = ps.Model(head, name=head.name)
    response_function = ps.Gamma(cutoff=response_cutoff) if model_type == "Gamma" else ps.Exponential(cutoff=response_cutoff)
    recharge_model = ps.rch.FlexModel() if use_flex else ps.rch.Linear()

    stress_model = ps.RechargeModel(rain, evap, rfunc=response_function, recharge=recharge_model)
    model.add_stressmodel(stress_model)

    if use_noise:
        model.add_noisemodel(ps.ArNoiseModel(norm=noise_norm))

    for name, value in parameter_values.items():
        if name in model.parameters.index:
            model.set_parameter(name, initial=float(value), vary=bool(vary_flags.get(name, True)))

    model.solve(tmin="1983", report=False)
    return model


def get_selected_stations(mode_label, single_station, multi_stations, stations, t):
    if mode_label == t["mode_single"]:
        return [single_station] if single_station else []
    if mode_label == t["mode_multi"]:
        return list(multi_stations)
    return list(stations)


def create_result_row(
    station,
    mode_label,
    model_type,
    use_flex,
    use_noise,
    response_cutoff,
    noise_norm,
    head,
    parameter_values,
    vary_flags,
    run_label,
    model=None,
    error_message=None,
):
    row = {
        "Run": run_label,
        "Messstelle": station,
        "Modus": mode_label,
        "Modell": model_type,
        "Flex": use_flex,
        "Noise": use_noise,
        "Cutoff": round(float(response_cutoff), 4),
        "NoiseNorm": bool(noise_norm) if use_noise else None,
        "n_obs": int(head.shape[0]),
        "Status": "ok" if model is not None and error_message is None else "error",
        "R2": None,
        "RMSE": None,
        "EVP": None,
        "AIC": None,
        "Fehler": error_message,
    }

    for name, value in parameter_values.items():
        row[f"requested_{name}"] = float(value)
        row[f"requested_vary_{name}"] = bool(vary_flags.get(name, True))

    if model is None:
        return row

    row["R2"] = round(model.stats.rsq(), 3)
    row["RMSE"] = round(model.stats.rmse(), 3)
    row["EVP"] = round(model.stats.evp(), 1)
    row["AIC"] = round(model.stats.aic(), 1)

    for name, parameter_row in model.parameters.iterrows():
        initial_value = parameter_row.get("initial")
        optimal_value = parameter_row.get("optimal")
        vary_value = parameter_row.get("vary")
        row[f"param_init_{name}"] = None if pd.isna(initial_value) else float(initial_value)
        row[f"param_vary_{name}"] = None if pd.isna(vary_value) else bool(vary_value)
        row[f"param_opt_{name}"] = None if pd.isna(optimal_value) else float(optimal_value)

    return row


def merge_extra_data(result_df, extra_df):
    info = {"matched": 0, "missing": 0, "unused": 0}
    if extra_df is None or result_df.empty:
        return result_df, info

    merged = result_df.copy()
    merged["_station_key"] = merged["Messstelle"].map(normalize_station_name)

    extra_copy = extra_df.copy()
    extra_copy["_station_key"] = extra_copy["Messstelle"].map(normalize_station_name)

    result_keys = set(merged["_station_key"])
    extra_keys = set(extra_copy["_station_key"])
    info["matched"] = len(result_keys & extra_keys)
    info["missing"] = len(result_keys - extra_keys)
    info["unused"] = len(extra_keys - result_keys)

    merged = merged.merge(extra_copy.drop(columns=["Messstelle"]), on="_station_key", how="left")
    merged = merged.drop(columns=["_station_key"])
    return merged, info


def get_numeric_columns(df):
    numeric_columns = []
    for column in df.columns:
        if is_numeric_dtype(df[column]) and not is_bool_dtype(df[column]):
            numeric_columns.append(column)
    return numeric_columns


def create_station_figure(model, plot_type, t):
    if plot_type == t["plot_overview"]:
        axes = model.plot()
        figure = axes.figure
        figure.set_size_inches(12, 7)
        figure.tight_layout()
        return figure

    observations = model.observations()
    simulation = model.simulate(tmin=observations.index.min(), tmax=observations.index.max())
    aligned = pd.concat(
        [
            observations.rename(t["observed"]),
            simulation.rename(t["simulated"]),
        ],
        axis=1,
    ).dropna()

    if plot_type == t["plot_obs_sim"]:
        figure, axis = plt.subplots(figsize=(12, 5))
        axis.plot(aligned.index, aligned[t["observed"]], label=t["observed"], linewidth=1.2)
        axis.plot(aligned.index, aligned[t["simulated"]], label=t["simulated"], linewidth=1.0)
        axis.set_title(f"{model.name}: {t['plot_obs_sim']}")
        axis.set_ylabel(t["head_axis"])
        axis.grid(alpha=0.3)
        axis.legend()
        figure.tight_layout()
        return figure

    residuals = model.residuals()

    if plot_type == t["plot_residuals"]:
        figure, axis = plt.subplots(figsize=(12, 4))
        axis.plot(residuals.index, residuals.values, linewidth=0.9, color="#b22222")
        axis.axhline(0.0, color="black", linewidth=1.0, linestyle="--")
        axis.set_title(f"{model.name}: {t['plot_residuals']}")
        axis.set_ylabel(t["residuals"])
        axis.grid(alpha=0.3)
        figure.tight_layout()
        return figure

    if plot_type == t["plot_res_hist"]:
        figure, axis = plt.subplots(figsize=(8, 4))
        axis.hist(residuals.dropna().values, bins=40, color="#2f6db3", edgecolor="white")
        axis.axvline(0.0, color="black", linewidth=1.0, linestyle="--")
        axis.set_title(f"{model.name}: {t['plot_res_hist']}")
        axis.set_xlabel(t["residuals"])
        axis.set_ylabel(t["count_axis"])
        figure.tight_layout()
        return figure

    response = model.get_step_response("recharge")
    figure, axis = plt.subplots(figsize=(8, 4))
    axis.plot(response.index, response.values, color="#1f7a1f", linewidth=1.5)
    axis.set_title(f"{model.name}: {t['plot_step']}")
    axis.set_xlabel(t["days_axis"])
    axis.set_ylabel(t["response_axis"])
    axis.grid(alpha=0.3)
    figure.tight_layout()
    return figure


def figure_to_png_bytes(figure):
    buffer = io.BytesIO()
    figure.savefig(buffer, format="png", dpi=200, bbox_inches="tight")
    buffer.seek(0)
    return buffer.getvalue()


def default_numeric_columns(numeric_columns):
    preferred = [
        "R2",
        "RMSE",
        "EVP",
        "AIC",
        "CPC",
        "PC1",
        "PC2",
        "PC3",
        "PC4",
        "PC5",
        "PC6",
        "PC7",
    ]
    ordered = [column for column in preferred if column in numeric_columns]
    ordered.extend(column for column in numeric_columns if column not in ordered)
    return ordered


def ranking_ascending(metric_name):
    return metric_name in {"RMSE", "AIC"}


def get_accessibility_css(font_profile, text_scale, high_contrast, strong_focus, t):
    font_map = {
        "standard": '"Segoe UI", "Helvetica Neue", Arial, sans-serif',
        "readable": '"Atkinson Hyperlegible", "Trebuchet MS", Verdana, sans-serif',
        "dyslexia": '"OpenDyslexic", "Atkinson Hyperlegible", Verdana, sans-serif',
    }
    font_family = font_map.get(font_profile, font_map["standard"])
    text_color = "#111111"
    background = "#ffffff" if high_contrast else "#f8fafc"
    surface = "#ffffff"
    border = "#1f2937" if high_contrast else "#94a3b8"
    accent = "#005fcc" if high_contrast else "#1d4ed8"
    focus_rule = (
        f"""
        .stApp *:focus {{
            outline: 3px solid {accent} !important;
            outline-offset: 2px !important;
        }}
        """
        if strong_focus
        else ""
    )
    return f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Atkinson+Hyperlegible:wght@400;700&display=swap');
    .stApp {{
        font-family: {font_family} !important;
    }}
    .stApp button,
    .stApp input,
    .stApp textarea,
    .stApp select,
    .stApp label,
    .stApp table,
    .stApp [data-testid="stMarkdownContainer"],
    .stApp [data-testid="stMarkdownContainer"] * {{
        font-family: {font_family} !important;
    }}
    [class^="material-symbols"],
    [class*=" material-symbols"],
    .material-icons,
    .material-icons-round,
    .material-icons-outlined {{
        font-family: "Material Symbols Rounded", "Material Symbols Outlined", "Material Icons Round", "Material Icons" !important;
        font-weight: normal !important;
        font-style: normal !important;
        letter-spacing: normal !important;
        text-transform: none !important;
        white-space: nowrap !important;
        direction: ltr !important;
    }}
    html, body, [data-testid="stAppViewContainer"], [data-testid="stSidebar"] {{
        font-size: {int(text_scale)}%;
        color: {text_color};
    }}
    .stApp {{
        background-color: {background};
        color: {text_color};
    }}
    [data-testid="stSidebar"] {{
        background-color: {surface};
    }}
    .stButton > button,
    .stDownloadButton > button,
    div[data-baseweb="select"] > div,
    .stMultiSelect [data-baseweb="select"] > div,
    .stTextInput input,
    .stNumberInput input,
    .stTextArea textarea,
    [data-testid="stFileUploader"] section {{
        border: 2px solid {border} !important;
        border-radius: 0.5rem !important;
    }}
    .stButton > button,
    .stDownloadButton > button {{
        background: {surface};
        color: {text_color};
    }}
    [data-testid="stMarkdownContainer"] p,
    [data-testid="stMarkdownContainer"] li,
    label,
    .stMetric,
    table {{
        line-height: 1.6 !important;
        letter-spacing: 0.01em;
        word-spacing: 0.03em;
    }}
    thead tr th {{
        background-color: #e2e8f0 !important;
        color: {text_color} !important;
    }}
    {focus_rule}
    </style>
    """


def get_compact_result_columns(df):
    preferred = [
        "Run",
        "Messstelle",
        "Modus",
        "Modell",
        "Flex",
        "Noise",
        "Status",
        "n_obs",
        "R2",
        "RMSE",
        "EVP",
        "AIC",
        "CPC",
        "PC1",
        "PC2",
        "PC3",
        "PC4",
        "PC5",
        "PC6",
        "PC7",
        "Fehler",
    ]
    return [column for column in preferred if column in df.columns]


def get_display_column_labels(t):
    return {
        "Run": t["run_label"],
        "Messstelle": t["station"],
        "Modus": t["mode"],
        "Modell": t["model"],
        "Flex": "Flex",
        "Noise": "Noise",
        "Status": "Status",
        "n_obs": "n_obs",
        "R2": "R2",
        "RMSE": "RMSE",
        "EVP": "EVP",
        "AIC": "AIC",
        "CPC": "CPC",
        "PC1": "PC1",
        "PC2": "PC2",
        "PC3": "PC3",
        "PC4": "PC4",
        "PC5": "PC5",
        "PC6": "PC6",
        "PC7": "PC7",
        "Fehler": t["error_table"],
    }


def sort_results_for_display(df):
    if "R2" in df.columns:
        return df.sort_values("R2", ascending=False, na_position="last")
    return df


t = TEXT[st.session_state.lang]


with st.sidebar:
    st.title(t["sidebar"])
    language = st.radio(
        t["lang"],
        ["Deutsch", "English"],
        index=0 if st.session_state.lang == "Deutsch" else 1,
    )
    if language != st.session_state.lang:
        st.session_state.lang = language
        st.rerun()

    with st.expander(t["accessibility"], expanded=False):
        font_profile_labels = {
            "standard": t["font_standard"],
            "readable": t["font_readable"],
            "dyslexia": t["font_dyslexia"],
        }
        font_profile = st.selectbox(
            t["font_profile"],
            options=list(font_profile_labels.keys()),
            format_func=lambda key: font_profile_labels[key],
            key="font_profile",
        )
        text_scale = st.slider(
            t["text_scale"],
            min_value=90,
            max_value=140,
            value=100,
            step=5,
            key="text_scale",
        )
        high_contrast = st.checkbox(t["high_contrast"], value=False, key="high_contrast")
        strong_focus = st.checkbox(t["strong_focus"], value=True, key="strong_focus")
        st.caption(t["font_note"])

    st.divider()
    gw_file = st.file_uploader(t["upload_gw"], type=["csv", "xlsx", "ods"])
    weather_file = st.file_uploader(t["upload_weather"], type=["csv", "xlsx", "ods"])
    extra_file = st.file_uploader(t["upload_extra"], type=["csv", "xlsx", "ods"])

st.markdown(
    get_accessibility_css(font_profile, text_scale, high_contrast, strong_focus, t),
    unsafe_allow_html=True,
)


st.title(t["title"])


if not (gw_file and weather_file):
    st.info(t["waiting"])
    st.stop()


try:
    gw_df, rain, evap, extra_df = load_data(gw_file, weather_file, extra_file)
except Exception as exc:
    st.error(f"{t['load_error']}: {exc}")
    st.stop()


stations = [column for column in gw_df.columns if gw_df[column].dropna().shape[0] > 50]
if not stations:
    st.error(t["no_valid_stations"])
    st.stop()


st.markdown(f"### {t['config_heading']}")

control_col1, control_col2, control_col3, control_col4, control_col5 = st.columns(5)

with control_col1:
    mode = st.selectbox(
        t["mode"],
        [t["mode_single"], t["mode_multi"], t["mode_all"]],
    )

single_station = None
multi_stations = []

with control_col2:
    if mode == t["mode_single"]:
        single_station = st.selectbox(t["station"], stations)
    elif mode == t["mode_multi"]:
        multi_stations = st.multiselect(t["stations"], stations)
    else:
        st.markdown(f"**{len(stations)}** {t['available_stations'].lower()}")

with control_col3:
    model_type = st.selectbox(t["model"], ["Gamma", "Exponential"])

with control_col4:
    use_flex = st.checkbox(t["use_flex"], value=True)

with control_col5:
    use_noise = st.checkbox(t["use_noise"], value=True)


selected_station_preview = get_selected_stations(mode, single_station, multi_stations, stations, t)
st.caption(f"{t['selection_count']}: {len(selected_station_preview)}")


parameter_specs = get_parameter_specs(model_type, use_flex)
parameter_values = {}
vary_flags = {}

with st.expander(t["parameter_box"], expanded=False):
    st.caption(t["parameter_help"])
    advanced_col1, advanced_col2 = st.columns(2)
    with advanced_col1:
        response_cutoff = st.number_input(
            t["response_cutoff"],
            min_value=0.9000,
            max_value=0.9999,
            value=0.9990,
            step=0.0001,
            format="%.4f",
        )
    with advanced_col2:
        noise_norm = st.checkbox(t["noise_norm"], value=True, disabled=not use_noise)

    for spec in parameter_specs:
        param_col1, param_col2 = st.columns([3, 1])
        with param_col1:
            parameter_values[spec["name"]] = st.number_input(
                f"{spec['label']} ({spec['name']})",
                value=float(spec["initial"]),
                step=float(infer_step(spec["initial"])),
                format="%.4f",
                key=f"value_{model_type}_{use_flex}_{spec['name']}",
            )
        with param_col2:
            vary_flags[spec["name"]] = st.checkbox(
                t["vary"],
                value=bool(spec["vary"]),
                key=f"vary_{model_type}_{use_flex}_{spec['name']}",
            )


if st.button(t["run"], type="primary"):
    selected_stations = get_selected_stations(mode, single_station, multi_stations, stations, t)

    if not selected_stations:
        st.warning(t["choose_station"])
    else:
        run_label = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        results = []
        models = {}
        progress_bar = st.progress(0)
        progress_text = st.empty()

        with st.spinner(t["computing"]):
            total = len(selected_stations)
            for index, station_name in enumerate(selected_stations, start=1):
                head = gw_df[station_name].dropna()
                progress_text.write(f"{t['progress']}: {station_name} ({index}/{total})")

                if head.shape[0] < 50:
                    results.append(
                        create_result_row(
                            station=station_name,
                            mode_label=mode,
                            model_type=model_type,
                            use_flex=use_flex,
                            use_noise=use_noise,
                            response_cutoff=response_cutoff,
                            noise_norm=noise_norm,
                            head=head,
                            parameter_values=parameter_values,
                            vary_flags=vary_flags,
                            run_label=run_label,
                            error_message=t["too_few_obs"],
                        )
                    )
                    progress_bar.progress(index / total)
                    continue

                try:
                    model = build_model(
                        head=head,
                        rain=rain,
                        evap=evap,
                        model_type=model_type,
                        use_flex=use_flex,
                        use_noise=use_noise,
                        parameter_values=parameter_values,
                        vary_flags=vary_flags,
                        response_cutoff=response_cutoff,
                        noise_norm=noise_norm,
                    )
                    models[station_name] = model
                    results.append(
                        create_result_row(
                            station=station_name,
                            mode_label=mode,
                            model_type=model_type,
                            use_flex=use_flex,
                            use_noise=use_noise,
                            response_cutoff=response_cutoff,
                            noise_norm=noise_norm,
                            head=head,
                            parameter_values=parameter_values,
                            vary_flags=vary_flags,
                            run_label=run_label,
                            model=model,
                        )
                    )
                except Exception as exc:
                    results.append(
                        create_result_row(
                            station=station_name,
                            mode_label=mode,
                            model_type=model_type,
                            use_flex=use_flex,
                            use_noise=use_noise,
                            response_cutoff=response_cutoff,
                            noise_norm=noise_norm,
                            head=head,
                            parameter_values=parameter_values,
                            vary_flags=vary_flags,
                            run_label=run_label,
                            error_message=str(exc),
                        )
                    )

                progress_bar.progress(index / total)

        progress_bar.empty()
        progress_text.empty()

        result_df = pd.DataFrame(results)
        result_df, extra_match_info = merge_extra_data(result_df, extra_df)

        if st.session_state.history.empty:
            st.session_state.history = result_df
        else:
            st.session_state.history = pd.concat(
                [st.session_state.history, result_df],
                ignore_index=True,
                sort=False,
            )

        st.session_state.last_run_results = result_df
        st.session_state.last_run_models = models
        st.session_state.last_run_label = run_label
        st.session_state.last_extra_match = extra_match_info
        st.success(t["success"])


last_run_results = st.session_state.last_run_results
last_run_models = st.session_state.last_run_models

if not last_run_results.empty:
    valid_last_run = last_run_results[last_run_results["Status"] == "ok"].copy()
    failed_last_run = last_run_results[last_run_results["Status"] != "ok"].copy()

    summary_col1, summary_col2, summary_col3, summary_col4 = st.columns(4)
    summary_col1.metric(t["successful_runs"], int(valid_last_run.shape[0]))
    summary_col2.metric(t["failed_runs"], int(failed_last_run.shape[0]))

    if not valid_last_run.empty and valid_last_run["R2"].notna().any():
        best_row = valid_last_run.sort_values("R2", ascending=False).iloc[0]
        summary_col3.metric(t["best_station"], f"{best_row['Messstelle']} ({best_row['R2']:.3f})")
        summary_col4.metric(t["mean_r2"], f"{valid_last_run['R2'].mean():.3f}")
    else:
        summary_col3.metric(t["best_station"], "-")
        summary_col4.metric(t["mean_r2"], "-")

    if st.session_state.last_run_label is not None:
        st.caption(f"{t['run_label']}: {st.session_state.last_run_label}")

    if extra_file is not None and st.session_state.last_extra_match:
        match_info = st.session_state.last_extra_match
        st.caption(
            f"{t['extra_match']}: {match_info.get('matched', 0)} | "
            f"{t['extra_missing']}: {match_info.get('missing', 0)} | "
            f"{t['extra_unused']}: {match_info.get('unused', 0)}"
        )


tab_plot, tab_compare, tab_save = st.tabs([t["tab_plot"], t["tab_compare"], t["tab_save"]])


with tab_plot:
    if not last_run_models:
        st.info(t["no_run"])
    else:
        plot_station_options = list(last_run_models.keys())
        plot_station = st.selectbox(t["plot_station"], plot_station_options)
        plot_type = st.selectbox(
            t["plot_type"],
            [
                t["plot_overview"],
                t["plot_obs_sim"],
                t["plot_residuals"],
                t["plot_res_hist"],
                t["plot_step"],
            ],
        )

        selected_model = last_run_models[plot_station]
        figure = create_station_figure(selected_model, plot_type, t)
        st.pyplot(figure)
        image_bytes = figure_to_png_bytes(figure)
        st.download_button(
            t["download_plot"],
            data=image_bytes,
            file_name=f"{plot_station}_{plot_type.replace(' ', '_')}.png",
            mime="image/png",
            key=f"download_{plot_station}_{plot_type}",
        )
        plt.close(figure)


with tab_compare:
    st.subheader(t["result_table"])

    if st.session_state.history.empty:
        st.info(t["no_run"])
    else:
        scope = st.radio(
            t["compare_scope"],
            [t["scope_last"], t["scope_all"]],
            horizontal=True,
        )

        compare_df = last_run_results.copy() if scope == t["scope_last"] and not last_run_results.empty else st.session_state.history.copy()
        compare_df = sort_results_for_display(compare_df)

        compact_columns = get_compact_result_columns(compare_df)
        compact_df = compare_df[compact_columns].copy() if compact_columns else compare_df.copy()
        compact_df = compact_df.rename(columns=get_display_column_labels(t))
        st.dataframe(compact_df, use_container_width=True, hide_index=True)
        st.caption(t["compact_table_note"])

        with st.expander(t["extended_table"], expanded=False):
            st.dataframe(compare_df, use_container_width=True, hide_index=True)

        numeric_columns = default_numeric_columns(get_numeric_columns(compare_df))
        if len(numeric_columns) < 1:
            st.info(t["no_numeric"])
        else:
            chart_type = st.selectbox(
                t["chart_type"],
                [
                    t["chart_ranking"],
                    t["chart_scatter"],
                    t["chart_hist"],
                    t["chart_box"],
                    t["chart_corr"],
                ],
            )

            if chart_type == t["chart_ranking"]:
                metric_default = numeric_columns.index("R2") if "R2" in numeric_columns else 0
                metric = st.selectbox(t["metric"], numeric_columns, index=metric_default)
                plot_df = compare_df[["Messstelle", metric]].dropna()
                plot_df = plot_df.sort_values(metric, ascending=ranking_ascending(metric))
                figure, axis = plt.subplots(figsize=(10, max(4, 0.35 * len(plot_df))))
                axis.barh(plot_df["Messstelle"], plot_df[metric], color="#2f6db3")
                axis.set_title(metric)
                axis.set_xlabel(metric)
                axis.grid(axis="x", alpha=0.3)
                figure.tight_layout()
                st.pyplot(figure)
                plt.close(figure)

            elif chart_type == t["chart_scatter"]:
                if len(numeric_columns) < 2:
                    st.info(t["no_numeric"])
                else:
                    x_default = numeric_columns.index("CPC") if "CPC" in numeric_columns else 0
                    y_default = numeric_columns.index("R2") if "R2" in numeric_columns else min(1, len(numeric_columns) - 1)
                    scatter_col1, scatter_col2 = st.columns(2)
                    with scatter_col1:
                        x_axis = st.selectbox(t["x_axis"], numeric_columns, index=x_default)
                    with scatter_col2:
                        y_axis = st.selectbox(t["y_axis"], numeric_columns, index=y_default)
                    plot_df = compare_df.dropna(subset=[x_axis, y_axis])
                    figure, axis = plt.subplots(figsize=(7, 5))
                    axis.scatter(plot_df[x_axis], plot_df[y_axis], alpha=0.75, color="#b24c3d")
                    axis.set_xlabel(x_axis)
                    axis.set_ylabel(y_axis)
                    axis.set_title(f"{y_axis} vs {x_axis}")
                    axis.grid(alpha=0.3)
                    figure.tight_layout()
                    st.pyplot(figure)
                    plt.close(figure)

            elif chart_type == t["chart_hist"]:
                metric_default = numeric_columns.index("R2") if "R2" in numeric_columns else 0
                metric = st.selectbox(t["metric"], numeric_columns, index=metric_default, key="hist_metric")
                figure, axis = plt.subplots(figsize=(8, 4))
                axis.hist(compare_df[metric].dropna(), bins=30, color="#4b8f29", edgecolor="white")
                axis.set_title(metric)
                axis.set_xlabel(metric)
                axis.set_ylabel(t["count_axis"])
                figure.tight_layout()
                st.pyplot(figure)
                plt.close(figure)

            elif chart_type == t["chart_box"]:
                default_selection = [column for column in ["R2", "RMSE", "EVP", "AIC"] if column in numeric_columns]
                selected_box_columns = st.multiselect(
                    t["box_cols"],
                    numeric_columns,
                    default=default_selection[:4],
                )
                if not selected_box_columns:
                    st.info(t["no_numeric"])
                else:
                    figure, axis = plt.subplots(figsize=(10, 5))
                    data = [compare_df[column].dropna() for column in selected_box_columns]
                    axis.boxplot(data, labels=selected_box_columns, vert=True)
                    axis.set_title(t["chart_box"])
                    axis.tick_params(axis="x", rotation=30)
                    figure.tight_layout()
                    st.pyplot(figure)
                    plt.close(figure)

            else:
                default_selection = [column for column in ["R2", "RMSE", "EVP", "AIC", "CPC", "PC1"] if column in numeric_columns]
                selected_corr_columns = st.multiselect(
                    t["corr_cols"],
                    numeric_columns,
                    default=default_selection[:6],
                )
                if len(selected_corr_columns) < 2:
                    st.info(t["no_numeric"])
                else:
                    corr = compare_df[selected_corr_columns].corr()
                    figure, axis = plt.subplots(figsize=(8, 6))
                    image = axis.imshow(corr, cmap="RdBu", vmin=-1, vmax=1)
                    axis.set_xticks(range(len(corr.columns)))
                    axis.set_yticks(range(len(corr.index)))
                    axis.set_xticklabels(corr.columns, rotation=45, ha="right")
                    axis.set_yticklabels(corr.index)
                    for row_index, row_name in enumerate(corr.index):
                        for col_index, col_name in enumerate(corr.columns):
                            axis.text(
                                col_index,
                                row_index,
                                f"{corr.loc[row_name, col_name]:.2f}",
                                ha="center",
                                va="center",
                                color="black",
                            )
                    figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
                    axis.set_title(t["chart_corr"])
                    figure.tight_layout()
                    st.pyplot(figure)
                    plt.close(figure)

        if "Fehler" in compare_df.columns:
            error_df = compare_df[compare_df["Fehler"].notna() & (compare_df["Fehler"].astype(str) != "")]
            if not error_df.empty:
                with st.expander(t["error_table"], expanded=False):
                    error_display = error_df[["Run", "Messstelle", "Status", "Fehler"]].rename(
                        columns=get_display_column_labels(t)
                    )
                    st.dataframe(
                        error_display,
                        use_container_width=True,
                        hide_index=True,
                    )


with tab_save:
    if not st.session_state.history.empty:
        export_data = st.session_state.history.to_csv(index=False, sep=";").encode("utf-8")
        st.download_button(
            t["export_csv"],
            data=export_data,
            file_name="gw_results.csv",
            mime="text/csv",
        )

    imported_file = st.file_uploader(t["import_csv"], type=["csv"], key="import_results")
    if imported_file is not None and st.button(t["import_button"], key="import_results_button"):
        imported_df = pd.read_csv(imported_file, sep=";")
        if st.session_state.history.empty:
            st.session_state.history = imported_df
        else:
            st.session_state.history = pd.concat(
                [st.session_state.history, imported_df],
                ignore_index=True,
                sort=False,
            )
        st.success(t["import_success"])
