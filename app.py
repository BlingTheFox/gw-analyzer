import io
import html
import ipaddress
import json
import socket
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from threading import Lock, Thread
from urllib.parse import urlparse
from uuid import uuid4

import matplotlib.pyplot as plt
import pandas as pd
import pastas as ps
import pydeck as pdk
import requests
import streamlit as st
from pandas.api.types import is_bool_dtype, is_numeric_dtype

# Project concept and application lead: Robin Carow / RCnet
# Modeling framework credit: Pastas by Collenteur et al. (2019), https://doi.org/10.1111/gwat.12925


st.set_page_config(page_title="GW Analyzer Beta", layout="wide")

APP_DIR = Path(__file__).resolve().parent
UPLOAD_CACHE_DIR = APP_DIR / ".upload_cache"
UPLOAD_CACHE_MANIFEST = UPLOAD_CACHE_DIR / "manifest.json"
MAX_REMOTE_DOWNLOAD_BYTES = 25 * 1024 * 1024
MAX_ZIP_ENTRY_BYTES = 50 * 1024 * 1024
MAX_PLOT_POINTS = 2500
MAX_PARALLEL_WORKERS = 4
SPREADSHEET_FORMULA_PREFIXES = ("=", "+", "-", "@")


class JobCancelled(Exception):
    pass


TEXT = {
    "Deutsch": {
        "title": "Grundwasser Analyse Tool Beta",
        "beta_note": "Beta-Arbeitskopie fuer neue Funktionen.",
        "sidebar": "Einstellungen",
        "lang": "Sprache / Language",
        "accessibility": "Barrierefreiheit",
        "font_profile": "Schriftprofil",
        "font_standard": "Standard",
        "font_readable": "Lesefreundlich",
        "font_dyslexia": "Dyslexia/LRS-freundlich",
        "font_note": "Die Dyslexia-Option nutzt OpenDyslexic mit Atkinson-Fallback, falls verfuegbar.",
        "text_scale": "Schriftgroesse (%)",
        "high_contrast": "Hoher Kontrast",
        "strong_focus": "Starke Fokusmarkierung",
        "upload_gw": "1. Grundwasserdaten hochladen (CSV, XLSX, ODS)",
        "upload_weather": "2. Wetterdaten hochladen (CSV, XLSX, ODS)",
        "upload_extra": "3. Zusatzdaten hochladen (optional: CPC/PC CSV, XLSX, ODS)",
        "upload_coords": "4. Stationskoordinaten hochladen (optional: CSV, XLSX, ODS)",
        "cache_heading": "Temporärer Upload-Cache",
        "cache_help": "Speichert aktuelle Uploads/URLs lokal, damit sie nach Reload oder Sprachwechsel weiter nutzbar bleiben.",
        "cache_save": "Aktuelle Quellen cachen",
        "cache_clear": "Cache leeren",
        "cache_saved": "Temporärer Cache wurde gespeichert.",
        "cache_cleared": "Temporärer Cache wurde geleert.",
        "cache_empty": "Es gibt noch keine hochgeladenen oder verlinkten Quellen zum Cachen.",
        "cache_active": "Aktiver Temp-Cache",
        "cache_used": "Folgende Quellen werden gerade aus dem Temp-Cache verwendet:",
        "cache_lang_saved": "Aktuelle Quellen wurden vor dem Sprachwechsel temporaer gesichert.",
        "source_summary": "Aktive Quellen",
        "source_uploaded": "Upload",
        "source_url_label": "URL",
        "source_cache_label": "Temp-Cache",
        "source_gw": "Grundwasser",
        "source_weather": "Wetter",
        "source_extra": "Zusatzdaten",
        "source_coords": "Koordinaten",
        "remote_heading": "Remote-Datenabruf (Beta)",
        "remote_help": "Direkte oeffentliche URLs zu CSV/XLSX/ODS oder ZIP-Dateien. Geeignet als Grundlage fuer DWD- oder andere API/Open-Data-Quellen.",
        "remote_gw": "Grundwasser-URL",
        "remote_weather": "Wetter-URL",
        "remote_extra": "Zusatzdaten-URL",
        "remote_coords": "Koordinaten-URL",
        "reload_sources": "Datenquellen neu laden",
        "reload_sources_done": "Datenquellen wurden neu eingelesen.",
        "waiting": "Bitte zuerst die Dateien links in der Sidebar hochladen.",
        "load_error": "Daten konnten nicht geladen werden",
        "no_valid_stations": "Es wurden keine gueltigen Messstellen mit mehr als 50 Beobachtungen gefunden.",
        "config_heading": "Konfiguration",
        "mode": "Auswertungsmodus",
        "mode_single": "Einzelne Messstelle",
        "mode_multi": "Mehrere Messstellen",
        "mode_all": "Alle Messstellen",
        "station": "Messstelle",
        "stations": "Messstellen",
        "available_stations": "Messstellen im Datensatz",
        "selection_count": "Ausgewaehlte Messstellen",
        "run_strategy": "Berechnungsstrategie",
        "strategy_manual": "Ausgewaehlte Konfiguration berechnen",
        "strategy_auto": "Bestes Modell automatisch suchen",
        "strategy_auto_note": "Die automatische Suche testet Gamma/Exponential, Flex/ohne Flex und Noise/ohne Noise fuer jede Messstelle.",
        "auto_combo_count": "Modellkombinationen pro Messstelle",
        "performance_box": "Performance-Optionen",
        "parallel_workers": "Parallele Auto-Suche (Worker)",
        "parallel_note": "Parallelisierung laeuft stationsweise und ist fuer groessere Auto-Laeufe gedacht.",
        "early_stop": "Fruehstopp bei Ziel-R2",
        "early_stop_r2": "Ziel-R2 fuer Fruehstopp",
        "model": "Reaktionsmodell",
        "use_flex": "FlexModel verwenden",
        "use_noise": "NoiseModel verwenden",
        "parameter_box": "Modellparameter anpassen",
        "parameter_help_manual": "Hier kannst du Startwerte und Optimierung einzelner Parameter fuer die ausgewaehlte Konfiguration steuern.",
        "parameter_help_auto": "Die automatische Modellsuche nutzt fuer jede Modellstruktur passende Standard-Startwerte.",
        "vary": "optimieren",
        "response_cutoff": "Response cutoff",
        "noise_norm": "Noise normalisieren",
        "run": "Batch starten",
        "cancel_run": "Lauf abbrechen",
        "choose_station": "Bitte mindestens eine Messstelle auswaehlen.",
        "computing": "Berechne Modelle...",
        "progress": "Aktuelle Messstelle",
        "progress_config": "Aktuelle Konfiguration",
        "progress_steps": "Fortschritt",
        "cancel_requested": "Abbruch angefordert. Der Lauf stoppt nach dem aktuellen Modellschritt.",
        "cancelled": "Lauf wurde abgebrochen.",
        "success": "Berechnung abgeschlossen.",
        "job_failed": "Batch-Lauf ist fehlgeschlagen",
        "too_few_obs": "Zu wenige Beobachtungen (<50).",
        "summary": "Zusammenfassung des letzten Laufs",
        "run_label": "Lauf",
        "successful_runs": "Erfolgreiche Modelle",
        "failed_runs": "Fehler oder Uebersprungen",
        "best_station": "Beste Messstelle (R2)",
        "mean_r2": "Mittleres R2",
        "extra_match": "Zusatzdaten gematcht",
        "extra_missing": "Messstellen ohne Zusatzdaten",
        "extra_unused": "Zusatzdaten ohne passenden Lauf",
        "coords_match": "Koordinaten gematcht",
        "coords_missing": "Messstellen ohne Koordinaten",
        "coords_unused": "Koordinaten ohne passenden Lauf",
        "data_status": "Datenstand und Update-Assistent",
        "data_status_note": "Prueft Datenluecken, fehlende Werte und ob Forecasts mit altem Wetterantrieb laufen.",
        "data_gw_until": "Grundwasser bis",
        "data_weather_until": "Wetter bis",
        "data_gap_days": "Luecke bis heute",
        "data_station_count": "Messstellen",
        "data_quality_table": "Datenqualitaet je Messstelle",
        "data_update_hint": "Die lokalen Daten sind nicht aktuell. Fuer belastbare Forecasts sollten historische Wetter-/Grundwasserluecken vor dem Szenario geschlossen werden; Open-Meteo deckt hier nur die naechsten Forecast-Tage ab.",
        "data_fresh": "Datenstand wirkt aktuell genug fuer einen Kurzforecast.",
        "tab_diagnostics": "Diagnose",
        "tab_plot": "Visualisierung",
        "tab_compare": "Vergleich",
        "tab_map": "Karte",
        "tab_forecast": "Forecast",
        "tab_save": "Speichern und Laden",
        "no_run": "Fuehre zuerst eine Analyse aus, um Ergebnisse anzuzeigen.",
        "plot_station": "Messstelle fuer die Visualisierung",
        "plot_type": "Visualisierungsmethode",
        "plot_overview": "Pastas Uebersicht",
        "plot_obs_sim": "Beobachtet vs. simuliert",
        "plot_residuals": "Residuale ueber Zeit",
        "plot_res_hist": "Residual-Histogramm",
        "plot_step": "Schrittantwort",
        "plot_pulse": "Pulse Response",
        "plot_downsampled": "Diagramm fuer schnelle Darstellung ausgeduennt",
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
        "best_only": "Nur bestes Modell je Messstelle anzeigen",
        "r2_filter_toggle": "R2-Filter aktivieren",
        "r2_threshold": "Mindestens R2",
        "run_filter": "Laufe filtern",
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
        "error_table": "Fehlermeldungen",
        "run_compare": "Laeufe nebeneinander vergleichen",
        "run_compare_note": "Vergleich mehrerer Laeufe fuer eine Kennzahl in einer Pivot-Tabelle.",
        "run_compare_runs": "Laeufe fuer den Vergleich",
        "run_compare_metric": "Kennzahl fuer den Laufvergleich",
        "run_compare_summary": "Zusammenfassung pro Lauf",
        "run_compare_station": "Vergleich pro Messstelle",
        "map_scope": "Datenbasis fuer die Karte",
        "map_metric": "Kennzahl fuer die Kartentabelle",
        "map_no_coords": "Keine Koordinaten vorhanden. Lade eine optionale Stationsdatei mit Messstelle/Site und Lat/Lon hoch.",
        "map_note": "Die Kartenansicht ist vorbereitet. Sobald Koordinaten vorliegen, werden die Messstellen direkt angezeigt.",
        "map_table": "Messstellen mit Koordinaten",
        "map_size_metric": "Markergroesse",
        "map_details": "Kartendetails",
        "map_color_note": "Markerfarbe folgt der ausgewaehlten Kennzahl, Markergroesse folgt der Datenlaenge oder einer zweiten Kennzahl.",
        "diag_station": "Messstelle fuer Diagnose",
        "diag_heading": "Modelldiagnose",
        "diag_bias": "Bias",
        "diag_resid_std": "Residuen-Std.",
        "diag_resid_acf": "Residuen-ACF Lag 1",
        "diag_missing_pct": "Fehlende Werte",
        "diag_quality": "Qualitaet",
        "diag_green": "Gruen",
        "diag_yellow": "Gelb",
        "diag_red": "Rot",
        "diag_seasonal_error": "Saisonaler Fehler",
        "diag_cluster": "Response-Cluster",
        "diag_cluster_note": "Gruppiert Messstellen grob nach Peak-Zeit der Pulse Response.",
        "forecast_station": "Messstelle fuer Forecast",
        "forecast_preset": "Szenario-Preset",
        "forecast_apply_preset": "Preset anwenden",
        "preset_custom": "Benutzerdefiniert",
        "preset_normal": "Normaljahr",
        "preset_dry": "Trocken",
        "preset_wet": "Nass",
        "preset_hot": "Heisser Sommer",
        "preset_open_meteo": "Open-Meteo kurz + Muster lang",
        "forecast_years": "Forecast-Horizont (Jahre)",
        "forecast_pattern_years": "Wiederholte Basisjahre",
        "forecast_weather_source": "Wetterquelle",
        "forecast_source_pattern": "Historisches Muster",
        "forecast_source_open_meteo": "Open-Meteo + Muster",
        "forecast_api_days": "Open-Meteo-Tage",
        "forecast_timezone": "Zeitzone",
        "forecast_use_manual_coords": "Koordinaten manuell setzen",
        "forecast_latitude": "Latitude",
        "forecast_longitude": "Longitude",
        "forecast_rain_factor": "Niederschlagsfaktor",
        "forecast_evap_factor": "Verdunstungsfaktor",
        "forecast_rain_offset": "Niederschlag Offset (mm/Tag)",
        "forecast_evap_offset": "Verdunstung Offset (mm/Tag)",
        "forecast_clip_nonnegative": "Negative Wetterwerte auf 0 begrenzen",
        "forecast_uncertainty": "Unsicherheitsband (%)",
        "forecast_run": "Forecast-Szenario berechnen",
        "forecast_note": "Das Szenario nutzt historische Wetterjahre als Muster und kann aktuelle Open-Meteo-Tageswerte in den Zukunftshorizont einfuegen.",
        "forecast_download": "Forecast als CSV herunterladen",
        "forecast_missing": "Forecast braucht ein erfolgreiches Modell aus dem letzten Lauf.",
        "forecast_results": "Forecast-Ergebnisse",
        "forecast_axis": "Simulierter Grundwasserstand",
        "forecast_weather_forcing": "Wetterantrieb",
        "forecast_weather_table": "Forecast-Wetterwerte",
        "forecast_rain_axis": "Niederschlag (mm/Tag)",
        "forecast_evap_axis": "Verdunstung (mm/Tag)",
        "forecast_api_missing_coords": "Open-Meteo braucht Stationskoordinaten oder manuelle Koordinaten.",
        "forecast_api_inserted": "Open-Meteo-Werte eingefuegt",
        "forecast_api_outside_horizon": "Die Open-Meteo-Tage liegen ausserhalb des gewaehlten Modellhorizonts. Erhoehe den Forecast-Horizont oder aktualisiere die Eingangsdaten.",
        "forecast_stale_weather": "Der Wetterantrieb ist alt. Nutze Open-Meteo fuer die naechsten Tage oder aktualisiere die Wetterdatei.",
        "forecast_band_label": "Unsicherheitsband",
        "export_csv": "Ergebnisse als CSV exportieren",
        "export_excel": "Ergebnisse als Excel exportieren",
        "export_report": "Kurzreport als HTML exportieren",
        "import_csv": "Ergebnisse importieren",
        "import_button": "Import starten",
        "import_success": "Ergebnisse wurden importiert.",
        "save_note": "Excel-Datei enthaelt mehrere Sheets fuer Historie, letzten Lauf, Bestmodelle und Fehler.",
        "history_heading": "Historie",
        "history_limit": "Maximal gespeicherte Laeufe",
        "history_limit_note": "Begrenzt die Session-Historie auf die zuletzt gespeicherten Laeufe.",
        "history_clear": "Historie leeren",
        "history_cleared": "Historie wurde geleert.",
        "config_column": "Konfiguration",
        "search_mode": "Suchmodus",
        "best_model_flag": "Bestes Modell",
        "status": "Status",
        "running_info": "Ein Batch-Lauf ist aktiv. Die Ansicht aktualisiert sich automatisch.",
    },
    "English": {
        "title": "Groundwater Analysis Tool Beta",
        "beta_note": "Beta work copy for new features.",
        "sidebar": "Settings",
        "lang": "Sprache / Language",
        "accessibility": "Accessibility",
        "font_profile": "Font profile",
        "font_standard": "Standard",
        "font_readable": "Readable",
        "font_dyslexia": "Dyslexia-friendly",
        "font_note": "The dyslexia option uses OpenDyslexic with Atkinson fallback when available.",
        "text_scale": "Text size (%)",
        "high_contrast": "High contrast",
        "strong_focus": "Strong focus highlight",
        "upload_gw": "1. Upload groundwater data (CSV, XLSX, ODS)",
        "upload_weather": "2. Upload weather data (CSV, XLSX, ODS)",
        "upload_extra": "3. Upload extra station data (optional: CPC/PC CSV, XLSX, ODS)",
        "upload_coords": "4. Upload station coordinates (optional: CSV, XLSX, ODS)",
        "cache_heading": "Temporary upload cache",
        "cache_help": "Stores current uploads/URLs locally so they remain usable after reload or language changes.",
        "cache_save": "Cache current sources",
        "cache_clear": "Clear cache",
        "cache_saved": "Temporary cache was saved.",
        "cache_cleared": "Temporary cache was cleared.",
        "cache_empty": "There are no uploaded or linked sources to cache yet.",
        "cache_active": "Active temp cache",
        "cache_used": "The following sources are currently loaded from temp cache:",
        "cache_lang_saved": "Current sources were temporarily preserved before the language change.",
        "source_summary": "Active sources",
        "source_uploaded": "Upload",
        "source_url_label": "URL",
        "source_cache_label": "Temp cache",
        "source_gw": "Groundwater",
        "source_weather": "Weather",
        "source_extra": "Extra data",
        "source_coords": "Coordinates",
        "remote_heading": "Remote data fetch (Beta)",
        "remote_help": "Direct public URLs to CSV/XLSX/ODS or ZIP files. Useful as a base for DWD or other API/open-data sources.",
        "remote_gw": "Groundwater URL",
        "remote_weather": "Weather URL",
        "remote_extra": "Extra-data URL",
        "remote_coords": "Coordinates URL",
        "reload_sources": "Reload data sources",
        "reload_sources_done": "Data sources were reloaded.",
        "waiting": "Please upload the files in the sidebar first.",
        "load_error": "Data could not be loaded",
        "no_valid_stations": "No valid stations with more than 50 observations were found.",
        "config_heading": "Configuration",
        "mode": "Evaluation mode",
        "mode_single": "Single station",
        "mode_multi": "Multiple stations",
        "mode_all": "All stations",
        "station": "Station",
        "stations": "Stations",
        "available_stations": "Stations in dataset",
        "selection_count": "Selected stations",
        "run_strategy": "Run strategy",
        "strategy_manual": "Run selected configuration",
        "strategy_auto": "Automatically search for best model",
        "strategy_auto_note": "Automatic search tests Gamma/Exponential, Flex/no Flex and Noise/no Noise for each station.",
        "auto_combo_count": "Model combinations per station",
        "performance_box": "Performance options",
        "parallel_workers": "Parallel auto search (workers)",
        "parallel_note": "Parallelization runs by station and is intended for larger auto runs.",
        "early_stop": "Early stop at target R2",
        "early_stop_r2": "Target R2 for early stop",
        "model": "Response model",
        "use_flex": "Use FlexModel",
        "use_noise": "Use noise model",
        "parameter_box": "Adjust model parameters",
        "parameter_help_manual": "Set initial values and optimization flags for the selected configuration.",
        "parameter_help_auto": "Automatic model search uses suitable default start values for each model structure.",
        "vary": "optimize",
        "response_cutoff": "Response cutoff",
        "noise_norm": "Normalize noise",
        "run": "Start batch",
        "cancel_run": "Cancel run",
        "choose_station": "Please select at least one station.",
        "computing": "Computing models...",
        "progress": "Current station",
        "progress_config": "Current configuration",
        "progress_steps": "Progress",
        "cancel_requested": "Cancellation requested. The run will stop after the current model step.",
        "cancelled": "Run was cancelled.",
        "success": "Calculation finished.",
        "job_failed": "Batch job failed",
        "too_few_obs": "Too few observations (<50).",
        "summary": "Last run summary",
        "run_label": "Run",
        "successful_runs": "Successful models",
        "failed_runs": "Failed or skipped",
        "best_station": "Best station (R2)",
        "mean_r2": "Mean R2",
        "extra_match": "Matched extra data",
        "extra_missing": "Stations without extra data",
        "extra_unused": "Extra-data rows without matching run",
        "coords_match": "Matched coordinates",
        "coords_missing": "Stations without coordinates",
        "coords_unused": "Coordinate rows without matching run",
        "data_status": "Data status and update assistant",
        "data_status_note": "Checks data gaps, missing values and whether forecasts use stale weather forcing.",
        "data_gw_until": "Groundwater until",
        "data_weather_until": "Weather until",
        "data_gap_days": "Gap to today",
        "data_station_count": "Stations",
        "data_quality_table": "Data quality by station",
        "data_update_hint": "The local data are not current. For robust forecasts, close historical weather/groundwater gaps before running scenarios; Open-Meteo only covers the next forecast days here.",
        "data_fresh": "Data status looks current enough for a short forecast.",
        "tab_diagnostics": "Diagnostics",
        "tab_plot": "Visualization",
        "tab_compare": "Comparison",
        "tab_map": "Map",
        "tab_forecast": "Forecast",
        "tab_save": "Save and Load",
        "no_run": "Run an analysis first to show results.",
        "plot_station": "Station for visualization",
        "plot_type": "Visualization method",
        "plot_overview": "Pastas overview",
        "plot_obs_sim": "Observed vs simulated",
        "plot_residuals": "Residuals over time",
        "plot_res_hist": "Residual histogram",
        "plot_step": "Step response",
        "plot_pulse": "Pulse response",
        "plot_downsampled": "Plot downsampled for faster rendering",
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
        "best_only": "Show only the best model per station",
        "r2_filter_toggle": "Enable R2 filter",
        "r2_threshold": "Minimum R2",
        "run_filter": "Filter runs",
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
        "error_table": "Error messages",
        "run_compare": "Compare runs side by side",
        "run_compare_note": "Compare multiple runs for one metric in a pivot table.",
        "run_compare_runs": "Runs for comparison",
        "run_compare_metric": "Metric for run comparison",
        "run_compare_summary": "Summary by run",
        "run_compare_station": "Comparison by station",
        "map_scope": "Data scope for the map",
        "map_metric": "Metric for the map table",
        "map_no_coords": "No coordinates available. Upload an optional station file with Messstelle/Site and Lat/Lon.",
        "map_note": "The map view is ready. As soon as coordinates are available, the stations will be shown directly.",
        "map_table": "Stations with coordinates",
        "map_size_metric": "Marker size",
        "map_details": "Map details",
        "map_color_note": "Marker color follows the selected metric; marker size follows data length or a second metric.",
        "diag_station": "Station for diagnostics",
        "diag_heading": "Model diagnostics",
        "diag_bias": "Bias",
        "diag_resid_std": "Residual std.",
        "diag_resid_acf": "Residual ACF lag 1",
        "diag_missing_pct": "Missing values",
        "diag_quality": "Quality",
        "diag_green": "Green",
        "diag_yellow": "Yellow",
        "diag_red": "Red",
        "diag_seasonal_error": "Seasonal error",
        "diag_cluster": "Response clusters",
        "diag_cluster_note": "Groups stations roughly by pulse-response peak timing.",
        "forecast_station": "Station for forecast",
        "forecast_preset": "Scenario preset",
        "forecast_apply_preset": "Apply preset",
        "preset_custom": "Custom",
        "preset_normal": "Normal year",
        "preset_dry": "Dry",
        "preset_wet": "Wet",
        "preset_hot": "Hot summer",
        "preset_open_meteo": "Open-Meteo short + long pattern",
        "forecast_years": "Forecast horizon (years)",
        "forecast_pattern_years": "Repeated baseline years",
        "forecast_weather_source": "Weather source",
        "forecast_source_pattern": "Historical pattern",
        "forecast_source_open_meteo": "Open-Meteo + pattern",
        "forecast_api_days": "Open-Meteo days",
        "forecast_timezone": "Timezone",
        "forecast_use_manual_coords": "Set coordinates manually",
        "forecast_latitude": "Latitude",
        "forecast_longitude": "Longitude",
        "forecast_rain_factor": "Precipitation factor",
        "forecast_evap_factor": "Evaporation factor",
        "forecast_rain_offset": "Precipitation offset (mm/day)",
        "forecast_evap_offset": "Evaporation offset (mm/day)",
        "forecast_clip_nonnegative": "Clip negative weather values to 0",
        "forecast_uncertainty": "Uncertainty band (%)",
        "forecast_run": "Compute forecast scenario",
        "forecast_note": "The scenario uses historical weather years as a repeating pattern and can insert current Open-Meteo daily values into the future horizon.",
        "forecast_download": "Download forecast as CSV",
        "forecast_missing": "Forecast requires a successful model from the last run.",
        "forecast_results": "Forecast results",
        "forecast_axis": "Simulated groundwater head",
        "forecast_weather_forcing": "Weather forcing",
        "forecast_weather_table": "Forecast weather values",
        "forecast_rain_axis": "Precipitation (mm/day)",
        "forecast_evap_axis": "Evaporation (mm/day)",
        "forecast_api_missing_coords": "Open-Meteo requires station coordinates or manual coordinates.",
        "forecast_api_inserted": "Open-Meteo values inserted",
        "forecast_api_outside_horizon": "The Open-Meteo days are outside the selected model horizon. Increase the forecast horizon or update the input data.",
        "forecast_stale_weather": "The weather forcing is stale. Use Open-Meteo for the next days or update the weather file.",
        "forecast_band_label": "Uncertainty band",
        "export_csv": "Export results as CSV",
        "export_excel": "Export results as Excel",
        "export_report": "Export short report as HTML",
        "import_csv": "Import results",
        "import_button": "Start import",
        "import_success": "Results were imported.",
        "save_note": "The Excel file includes multiple sheets for history, last run, best models and errors.",
        "history_heading": "History",
        "history_limit": "Maximum stored runs",
        "history_limit_note": "Limits the session history to the most recently stored runs.",
        "history_clear": "Clear history",
        "history_cleared": "History was cleared.",
        "config_column": "Configuration",
        "search_mode": "Search mode",
        "best_model_flag": "Best model",
        "status": "Status",
        "running_info": "A batch job is active. The view refreshes automatically.",
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
if "last_coord_match" not in st.session_state:
    st.session_state.last_coord_match = {}
if "active_job_id" not in st.session_state:
    st.session_state.active_job_id = None
if "job_notice" not in st.session_state:
    st.session_state.job_notice = None
if "session_id" not in st.session_state:
    st.session_state.session_id = uuid4().hex
if "last_forecast_df" not in st.session_state:
    st.session_state.last_forecast_df = pd.DataFrame()
if "last_forecast_info" not in st.session_state:
    st.session_state.last_forecast_info = {}
if "upload_cache_notice" not in st.session_state:
    st.session_state.upload_cache_notice = None
if "data_refresh_token" not in st.session_state:
    st.session_state.data_refresh_token = 0
if "data_refresh_notice" not in st.session_state:
    st.session_state.data_refresh_notice = None
if "history_run_limit" not in st.session_state:
    st.session_state.history_run_limit = 15
if "history_notice" not in st.session_state:
    st.session_state.history_notice = None


def normalize_station_name(value):
    return str(value).replace('"', "").strip()


def is_remote_source(value):
    return isinstance(value, str) and value.lower().startswith(("http://", "https://"))


def mask_url_for_display(value):
    if not is_remote_source(value):
        return str(value)
    parsed = urlparse(value)
    host = parsed.hostname or ""
    if parsed.port:
        host = f"{host}:{parsed.port}"
    path = parsed.path or ""
    masked = f"{parsed.scheme}://{host}{path}"
    if parsed.query:
        masked = f"{masked}?..."
    if parsed.fragment:
        masked = f"{masked}#..."
    return masked


def can_cache_remote_url(value):
    parsed = urlparse(value)
    return not (parsed.username or parsed.password or parsed.query or parsed.fragment)


def normalize_ip_for_compare(value):
    ip_obj = ipaddress.ip_address(str(value).split("%")[0])
    if getattr(ip_obj, "ipv4_mapped", None) is not None:
        ip_obj = ip_obj.ipv4_mapped
    return ip_obj, ip_obj.compressed


def escape_spreadsheet_formula(value):
    if isinstance(value, str) and value[:1] in SPREADSHEET_FORMULA_PREFIXES:
        return f"'{value}"
    return value


def sanitize_export_df(df):
    if df is None or df.empty:
        return df
    sanitized = df.copy()
    for column in sanitized.columns:
        if sanitized[column].dtype == object or pd.api.types.is_string_dtype(sanitized[column]):
            sanitized[column] = sanitized[column].map(escape_spreadsheet_formula)
    return sanitized


def ensure_upload_cache_dir():
    UPLOAD_CACHE_DIR.mkdir(parents=True, exist_ok=True)


def load_upload_cache_manifest():
    if not UPLOAD_CACHE_MANIFEST.exists():
        return {}
    try:
        return json.loads(UPLOAD_CACHE_MANIFEST.read_text(encoding="utf-8"))
    except Exception:
        return {}


def clear_upload_cache():
    if UPLOAD_CACHE_DIR.exists():
        for path in UPLOAD_CACHE_DIR.iterdir():
            if path.is_file():
                path.unlink()


def save_sources_to_upload_cache(source_entries):
    ensure_upload_cache_dir()
    clear_upload_cache()
    manifest = {}

    for source_key, source_value in source_entries.items():
        if source_value is None:
            continue
        if is_remote_source(source_value):
            if can_cache_remote_url(source_value):
                manifest[source_key] = {"type": "url", "value": source_value}
            continue
        if hasattr(source_value, "getvalue") and hasattr(source_value, "name"):
            suffix = Path(source_value.name).suffix or ".bin"
            cache_filename = f"{source_key}{suffix}"
            cache_path = UPLOAD_CACHE_DIR / cache_filename
            cache_path.write_bytes(source_value.getvalue())
            manifest[source_key] = {
                "type": "file",
                "path": str(cache_path),
                "name": source_value.name,
            }

    UPLOAD_CACHE_MANIFEST.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return manifest


def resolve_cached_sources():
    manifest = load_upload_cache_manifest()
    resolved = {}
    active_labels = []
    for source_key, entry in manifest.items():
        if entry.get("type") == "url" and entry.get("value"):
            resolved[source_key] = entry["value"]
            active_labels.append(source_key)
        elif entry.get("type") == "file" and entry.get("path"):
            cache_path = Path(entry["path"])
            if cache_path.exists():
                resolved[source_key] = cache_path
                active_labels.append(source_key)
    return resolved, active_labels


def collect_current_source_entries():
    def pick_source(file_key, url_key):
        file_value = st.session_state.get(file_key)
        if file_value is not None:
            return file_value
        url_value = st.session_state.get(url_key, "")
        if isinstance(url_value, str):
            url_value = url_value.strip()
        return url_value or None

    return {
        "gw": pick_source("gw_upload_file", "remote_gw_url"),
        "weather": pick_source("weather_upload_file", "remote_weather_url"),
        "extra": pick_source("extra_upload_file", "remote_extra_url"),
        "coords": pick_source("coords_upload_file", "remote_coords_url"),
    }


def persist_current_sources():
    source_entries = collect_current_source_entries()
    if not any(source_entries.values()):
        return {}
    return save_sources_to_upload_cache(source_entries)


def describe_cached_sources(t):
    manifest = load_upload_cache_manifest()
    descriptions = []
    label_map = {
        "gw": t["source_gw"],
        "weather": t["source_weather"],
        "extra": t["source_extra"],
        "coords": t["source_coords"],
    }
    for source_key in ["gw", "weather", "extra", "coords"]:
        entry = manifest.get(source_key)
        if not entry:
            continue
        label = label_map.get(source_key, source_key)
        if entry.get("type") == "file":
            descriptions.append(f"{label}: {entry.get('name', Path(entry.get('path', '')).name)}")
        elif entry.get("type") == "url":
            descriptions.append(f"{label}: {mask_url_for_display(entry.get('value', ''))}")
    return descriptions


def describe_active_sources(source_mapping, active_cached_source_keys, t):
    descriptions = []
    manifest = load_upload_cache_manifest()
    label_map = {
        "gw": t["source_gw"],
        "weather": t["source_weather"],
        "extra": t["source_extra"],
        "coords": t["source_coords"],
    }
    for source_key in ["gw", "weather", "extra", "coords"]:
        source_value = source_mapping.get(source_key)
        if source_value is None:
            continue

        origin = t["source_uploaded"]
        detail = ""
        if source_key in active_cached_source_keys:
            manifest_entry = manifest.get(source_key, {})
            origin = t["source_cache_label"]
            if manifest_entry.get("type") == "file":
                detail = manifest_entry.get("name", "")
            elif manifest_entry.get("type") == "url":
                detail = mask_url_for_display(manifest_entry.get("value", ""))
            elif isinstance(source_value, Path):
                detail = source_value.name
            else:
                detail = str(source_value)
        elif is_remote_source(source_value):
            origin = t["source_url_label"]
            detail = mask_url_for_display(source_value)
        elif hasattr(source_value, "name"):
            detail = source_value.name
        elif isinstance(source_value, Path):
            origin = t["source_cache_label"]
            detail = source_value.name
        else:
            detail = str(source_value)

        descriptions.append(f"{label_map.get(source_key, source_key)}: {detail} ({origin})")
    return descriptions


def limit_history_runs(history_df, max_runs):
    if history_df.empty or "Run" not in history_df.columns:
        return history_df
    max_runs = max(int(max_runs), 1)
    ordered_runs = list(dict.fromkeys(history_df["Run"].astype(str).tolist()))
    if len(ordered_runs) <= max_runs:
        return history_df
    keep_runs = set(ordered_runs[-max_runs:])
    trimmed = history_df[history_df["Run"].astype(str).isin(keep_runs)].copy()
    return trimmed.reset_index(drop=True)


def append_history_entries(history_df, new_entries_df, max_runs):
    if new_entries_df is None or new_entries_df.empty:
        return history_df.copy() if isinstance(history_df, pd.DataFrame) else pd.DataFrame()
    if history_df is None or history_df.empty:
        combined = new_entries_df.copy()
    else:
        combined = pd.concat(
            [history_df, new_entries_df],
            ignore_index=True,
            sort=False,
        )
    return limit_history_runs(combined, max_runs)


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


def validate_remote_url(source):
    parsed = urlparse(source)
    if parsed.scheme.lower() not in {"http", "https"}:
        raise ValueError("Remote sources must use http or https.")

    hostname = parsed.hostname
    if not hostname:
        raise ValueError("Remote source URL is missing a valid host.")
    if parsed.username or parsed.password:
        raise ValueError("Remote source URLs must not contain usernames or passwords.")

    try:
        port = parsed.port or (443 if parsed.scheme.lower() == "https" else 80)
    except ValueError as exc:
        raise ValueError("Remote source URL uses an invalid port.") from exc

    try:
        addr_infos = socket.getaddrinfo(hostname, port, proto=socket.IPPROTO_TCP)
    except socket.gaierror as exc:
        raise ValueError(f"Remote host could not be resolved: {hostname}") from exc

    allowed_ips = set()
    for addr_info in addr_infos:
        ip_obj, ip_string = normalize_ip_for_compare(addr_info[4][0])
        if (
            ip_obj.is_private
            or ip_obj.is_loopback
            or ip_obj.is_link_local
            or ip_obj.is_reserved
            or ip_obj.is_multicast
            or ip_obj.is_unspecified
        ):
            raise ValueError(
                "Remote sources on local, private or reserved networks are not allowed."
            )
        allowed_ips.add(ip_string)

    return parsed, allowed_ips


def get_response_peer_ip(response):
    connection = getattr(response.raw, "_connection", None)
    sock = getattr(connection, "sock", None)
    if sock is None:
        return None
    try:
        _, ip_string = normalize_ip_for_compare(sock.getpeername()[0])
        return ip_string
    except Exception:
        return None


def download_remote_bytes(source):
    parsed, allowed_ips = validate_remote_url(source)
    with requests.get(
        source,
        timeout=(10, 60),
        stream=True,
        allow_redirects=False,
        headers={"User-Agent": "GW-Analyzer-Beta/1.0"},
    ) as response:
        if 300 <= response.status_code < 400:
            raise ValueError("Redirecting remote URLs are not supported. Please use the final direct URL.")
        response.raise_for_status()
        peer_ip = get_response_peer_ip(response)
        if peer_ip is not None and peer_ip not in allowed_ips:
            raise ValueError("Remote host resolved to a different address during download.")

        content_length = response.headers.get("content-length")
        if content_length:
            try:
                content_length_value = int(content_length)
            except ValueError:
                content_length_value = None
            if content_length_value is not None and content_length_value > MAX_REMOTE_DOWNLOAD_BYTES:
                raise ValueError("Remote file is too large.")

        chunks = []
        total_bytes = 0
        for chunk in response.iter_content(chunk_size=65536):
            if not chunk:
                continue
            total_bytes += len(chunk)
            if total_bytes > MAX_REMOTE_DOWNLOAD_BYTES:
                raise ValueError("Remote file is too large.")
            chunks.append(chunk)

        content = b"".join(chunks)
        content_type = response.headers.get("content-type", "").lower()
        extension_hint = Path(parsed.path).suffix.lower()
        if "zip" in content_type and extension_hint != ".zip":
            extension_hint = ".zip"
        elif not extension_hint:
            if "csv" in content_type or "text/plain" in content_type:
                extension_hint = ".csv"
            elif "spreadsheetml" in content_type or "excel" in content_type:
                extension_hint = ".xlsx"
            elif "opendocument" in content_type:
                extension_hint = ".ods"
        return content, extension_hint


def read_table_from_bytes(raw_bytes, extension_hint=""):
    ext = extension_hint.lower()
    if ext == ".zip":
        with zipfile.ZipFile(io.BytesIO(raw_bytes)) as archive:
            candidates = [
                name
                for name in archive.namelist()
                if not name.endswith("/")
                and Path(name).suffix.lower() in {".csv", ".txt", ".xlsx", ".ods"}
            ]
            if not candidates:
                raise ValueError("ZIP file does not contain a supported table.")
            first_name = candidates[0]
            first_info = archive.getinfo(first_name)
            if first_info.file_size > MAX_ZIP_ENTRY_BYTES:
                raise ValueError("ZIP table is too large.")
            nested_bytes = archive.read(first_name)
            return read_table_from_bytes(nested_bytes, Path(first_name).suffix.lower())
    if ext in {".csv", ".txt"}:
        content = raw_bytes.decode("utf-8-sig", errors="ignore")
        first_line = content.splitlines()[0] if content.splitlines() else ""
        separator = ";" if first_line.count(";") >= first_line.count(",") else ","
        return pd.read_csv(io.StringIO(content), sep=separator)
    if ext == ".ods":
        return pd.read_excel(io.BytesIO(raw_bytes), engine="odf")
    return pd.read_excel(io.BytesIO(raw_bytes))


def read_table(source):
    if source is None:
        return None

    if is_remote_source(source):
        raw_bytes, extension_hint = download_remote_bytes(source)
        return read_table_from_bytes(raw_bytes, extension_hint)

    if isinstance(source, (str, Path)):
        path = Path(source)
        ext = path.suffix.lower()
        if ext in {".csv", ".txt"}:
            with path.open("r", encoding="utf-8-sig", errors="ignore") as handle:
                first_line = handle.readline()
            separator = ";" if first_line.count(";") >= first_line.count(",") else ","
            return pd.read_csv(path, sep=separator)
        if ext == ".ods":
            return pd.read_excel(path, engine="odf")
        return pd.read_excel(path)

    ext = Path(source.name).suffix.lower()
    raw_bytes = source.getvalue()
    return read_table_from_bytes(raw_bytes, ext)


def prepare_extra_df(extra_df):
    if extra_df is None:
        return None
    extra_df = extra_df.copy()
    extra_df.columns = [normalize_station_name(column) for column in extra_df.columns]
    site_candidates = [
        column
        for column in extra_df.columns
        if column.lower() in {"site", "messstelle", "station"}
    ]
    site_col = site_candidates[0] if site_candidates else extra_df.columns[0]
    extra_df[site_col] = extra_df[site_col].map(normalize_station_name)
    extra_df = extra_df.rename(columns={site_col: "Messstelle"})
    for column in extra_df.columns:
        if column != "Messstelle":
            extra_df[column] = pd.to_numeric(extra_df[column], errors="coerce")
    return extra_df


def prepare_coords_df(coords_df):
    if coords_df is None:
        return None
    coords_df = coords_df.copy()
    coords_df.columns = [normalize_station_name(column) for column in coords_df.columns]
    site_candidates = [
        column
        for column in coords_df.columns
        if column.lower() in {"site", "messstelle", "station"}
    ]
    site_col = site_candidates[0] if site_candidates else coords_df.columns[0]
    lat_col = find_matching_column(coords_df.columns, ["latitude", "lat", "breite", "y"])
    lon_col = find_matching_column(coords_df.columns, ["longitude", "long", "lon", "lng", "laenge", "x"])
    coords_df = coords_df.rename(
        columns={site_col: "Messstelle", lat_col: "Latitude", lon_col: "Longitude"}
    )
    coords_df["Messstelle"] = coords_df["Messstelle"].map(normalize_station_name)
    coords_df["Latitude"] = pd.to_numeric(coords_df["Latitude"], errors="coerce")
    coords_df["Longitude"] = pd.to_numeric(coords_df["Longitude"], errors="coerce")
    coords_df = coords_df.dropna(subset=["Messstelle", "Latitude", "Longitude"])
    coords_df = coords_df.drop_duplicates(subset=["Messstelle"])
    return coords_df


@st.cache_data
def load_data(gw_source, weather_source, extra_source, coords_source, refresh_token=0):
    gw_df = read_table(gw_source)
    weather_df = read_table(weather_source)
    extra_df = prepare_extra_df(read_table(extra_source)) if extra_source is not None else None
    coords_df = prepare_coords_df(read_table(coords_source)) if coords_source is not None else None

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
    weather_df = (
        weather_df.dropna(subset=[weather_date_col])
        .set_index(weather_date_col)
        .sort_index()
    )
    rain_col = find_matching_column(weather_df.columns, ["niederschlag", "rain", "prec"])
    evap_col = find_matching_column(weather_df.columns, ["verdunstung", "evap"])
    weather_df[rain_col] = pd.to_numeric(weather_df[rain_col], errors="coerce")
    weather_df[evap_col] = pd.to_numeric(weather_df[evap_col], errors="coerce")
    rain = weather_df[rain_col].resample("D").mean().fillna(0.0)
    evap = weather_df[evap_col].resample("D").mean().fillna(0.0)

    return gw_df, rain, evap, extra_df, coords_df


def get_parameter_specs(model_type, use_flex):
    return deepcopy(PARAMETER_TEMPLATES[(model_type, use_flex)])


def create_config_label(model_type, use_flex, use_noise):
    recharge_label = "Flex" if use_flex else "Linear"
    noise_label = "Noise" if use_noise else "NoNoise"
    return f"{model_type} | {recharge_label} | {noise_label}"


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
    response_function = (
        ps.Gamma(cutoff=response_cutoff)
        if model_type == "Gamma"
        else ps.Exponential(cutoff=response_cutoff)
    )
    recharge_model = ps.rch.FlexModel() if use_flex else ps.rch.Linear()
    stress_model = ps.RechargeModel(rain, evap, rfunc=response_function, recharge=recharge_model)
    model.add_stressmodel(stress_model)

    if use_noise:
        model.add_noisemodel(ps.ArNoiseModel(norm=noise_norm))

    for name, value in parameter_values.items():
        if name in model.parameters.index:
            model.set_parameter(
                name,
                initial=float(value),
                vary=bool(vary_flags.get(name, True)),
            )

    model.solve(tmin="1983", report=False)
    return model


def get_selected_stations(mode_label, single_station, multi_stations, stations, t):
    if mode_label == t["mode_single"]:
        return [single_station] if single_station else []
    if mode_label == t["mode_multi"]:
        return list(multi_stations)
    return list(stations)


def get_search_configurations(
    run_strategy_label,
    manual_model_type,
    manual_use_flex,
    manual_use_noise,
    manual_parameter_values,
    manual_vary_flags,
    response_cutoff,
    noise_norm,
    t,
):
    if run_strategy_label == t["strategy_manual"]:
        return [
            {
                "model_type": manual_model_type,
                "use_flex": manual_use_flex,
                "use_noise": manual_use_noise,
                "parameter_values": deepcopy(manual_parameter_values),
                "vary_flags": deepcopy(manual_vary_flags),
                "response_cutoff": response_cutoff,
                "noise_norm": noise_norm,
                "configuration_label": create_config_label(
                    manual_model_type, manual_use_flex, manual_use_noise
                ),
            }
        ]

    configurations = []
    for model_type in ["Gamma", "Exponential"]:
        for use_flex in [False, True]:
            for use_noise in [False, True]:
                specs = get_parameter_specs(model_type, use_flex)
                configurations.append(
                    {
                        "model_type": model_type,
                        "use_flex": use_flex,
                        "use_noise": use_noise,
                        "parameter_values": {
                            spec["name"]: spec["initial"] for spec in specs
                        },
                        "vary_flags": {
                            spec["name"]: spec["vary"] for spec in specs
                        },
                        "response_cutoff": response_cutoff,
                        "noise_norm": noise_norm,
                        "configuration_label": create_config_label(
                            model_type, use_flex, use_noise
                        ),
                    }
                )
    return configurations


def create_result_row(
    station,
    mode_label,
    run_strategy_label,
    configuration_label,
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
    best_station_model=False,
):
    row = {
        "Run": run_label,
        "Messstelle": station,
        "Modus": mode_label,
        "Suchmodus": run_strategy_label,
        "Konfiguration": configuration_label,
        "Modell": model_type,
        "Flex": use_flex,
        "Noise": use_noise,
        "Cutoff": round(float(response_cutoff), 4),
        "NoiseNorm": bool(noise_norm) if use_noise else None,
        "n_obs": int(head.shape[0]),
        "Status": "ok" if model is not None and error_message is None else "error",
        "BestStationModel": bool(best_station_model),
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


def merge_coordinate_data(result_df, coords_df):
    info = {"matched": 0, "missing": 0, "unused": 0}
    if coords_df is None or result_df.empty:
        return result_df, info

    merged = result_df.copy()
    merged["_station_key"] = merged["Messstelle"].map(normalize_station_name)
    coords_copy = coords_df.copy()
    coords_copy["_station_key"] = coords_copy["Messstelle"].map(normalize_station_name)

    result_keys = set(merged["_station_key"])
    coord_keys = set(coords_copy["_station_key"])
    info["matched"] = len(result_keys & coord_keys)
    info["missing"] = len(result_keys - coord_keys)
    info["unused"] = len(coord_keys - result_keys)

    merged = merged.merge(
        coords_copy.drop(columns=["Messstelle"]),
        on="_station_key",
        how="left",
    )
    merged = merged.drop(columns=["_station_key"])
    return merged, info


def get_numeric_columns(df):
    numeric_columns = []
    for column in df.columns:
        if is_numeric_dtype(df[column]) and not is_bool_dtype(df[column]):
            numeric_columns.append(column)
    return numeric_columns


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


def get_best_only_rows(df):
    if "BestStationModel" in df.columns and df["BestStationModel"].astype(bool).any():
        return df[df["BestStationModel"].astype(bool)].copy()
    return df.copy()


def get_summary_base(df):
    valid = df[df["Status"] == "ok"].copy() if "Status" in df.columns else df.copy()
    return get_best_only_rows(valid)


def sort_results_for_display(df):
    if "R2" in df.columns:
        return df.sort_values("R2", ascending=False, na_position="last")
    return df


def get_compact_result_columns(df):
    preferred = [
        "Run",
        "Messstelle",
        "Modus",
        "Suchmodus",
        "Konfiguration",
        "Modell",
        "Flex",
        "Noise",
        "BestStationModel",
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
        "Suchmodus": t["search_mode"],
        "Konfiguration": t["config_column"],
        "Modell": t["model"],
        "Flex": "Flex",
        "Noise": "Noise",
        "BestStationModel": t["best_model_flag"],
        "Status": t["status"],
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
        "Latitude": "Lat",
        "Longitude": "Lon",
    }


def downsample_series_for_plot(series, max_points=MAX_PLOT_POINTS):
    if series is None or len(series) <= max_points:
        return series, False
    step = max(1, int(len(series) / max_points))
    return series.iloc[::step], True


def downsample_frame_for_plot(df, max_points=MAX_PLOT_POINTS):
    if df is None or len(df) <= max_points:
        return df, False
    step = max(1, int(len(df) / max_points))
    return df.iloc[::step], True


def format_date_value(value):
    if value is None or pd.isna(value):
        return "-"
    return pd.Timestamp(value).strftime("%Y-%m-%d")


def data_gap_days(last_date):
    if last_date is None or pd.isna(last_date):
        return None
    today = pd.Timestamp(datetime.now().date())
    return max(int((today - pd.Timestamp(last_date).normalize()).days), 0)


def build_data_status(gw_df, rain, evap):
    gw_last = gw_df.index.max() if gw_df is not None and not gw_df.empty else None
    weather_last = rain.index.max() if rain is not None and not rain.empty else None
    return {
        "gw_last": gw_last,
        "weather_last": weather_last,
        "gw_gap_days": data_gap_days(gw_last),
        "weather_gap_days": data_gap_days(weather_last),
        "station_count": int(gw_df.shape[1]) if gw_df is not None else 0,
        "weather_days": int(rain.dropna().shape[0]) if rain is not None else 0,
    }


@st.cache_data
def build_data_quality_table(gw_df):
    rows = []
    total_rows = max(len(gw_df.index), 1)
    for station in gw_df.columns:
        series = gw_df[station]
        valid_count = int(series.notna().sum())
        missing_pct = round(100.0 * (1.0 - valid_count / total_rows), 2)
        rows.append(
            {
                "Messstelle": station,
                "n_obs": valid_count,
                "missing_pct": missing_pct,
                "first_date": format_date_value(series.dropna().index.min()) if valid_count else "-",
                "last_date": format_date_value(series.dropna().index.max()) if valid_count else "-",
            }
        )
    return pd.DataFrame(rows).sort_values(["missing_pct", "n_obs"], ascending=[True, False])


def get_station_result_row(last_run_df, station):
    row = get_station_metadata_row(last_run_df, station)
    return row if row is not None else pd.Series(dtype=object)


def residual_lag1_autocorr(residuals):
    clean = residuals.dropna()
    if clean.shape[0] < 3:
        return None
    return clean.autocorr(lag=1)


def classify_model_quality(row, residual_acf):
    r2 = pd.to_numeric(pd.Series([row.get("R2")]), errors="coerce").iloc[0]
    rmse = pd.to_numeric(pd.Series([row.get("RMSE")]), errors="coerce").iloc[0]
    acf = abs(float(residual_acf)) if residual_acf is not None and pd.notna(residual_acf) else 0.0
    if pd.notna(r2) and r2 >= 0.75 and acf < 0.35:
        return "green"
    if pd.notna(r2) and r2 >= 0.55 and (pd.isna(rmse) or rmse < 1.0) and acf < 0.6:
        return "yellow"
    return "red"


def quality_label(quality_key, t):
    return {
        "green": t["diag_green"],
        "yellow": t["diag_yellow"],
        "red": t["diag_red"],
    }.get(quality_key, t["diag_red"])


def build_seasonal_error_table(model):
    residuals = model.residuals().dropna()
    if residuals.empty:
        return pd.DataFrame()
    seasonal = (
        residuals.abs()
        .groupby(residuals.index.month)
        .agg(["mean", "median", "count"])
        .reset_index()
        .rename(columns={"index": "month", "mean": "mean_abs_residual", "median": "median_abs_residual"})
    )
    seasonal = seasonal.rename(columns={seasonal.columns[0]: "month"})
    return seasonal.round(4)


def build_response_cluster_table(models):
    rows = []
    for station, model in models.items():
        try:
            response = model.get_block_response("recharge")
        except Exception:
            continue
        if response is None or response.empty:
            continue
        values = response.dropna()
        if values.empty:
            continue
        peak_index = values.abs().idxmax()
        area = float(values.sum())
        peak = float(values.loc[peak_index])
        rows.append(
            {
                "Messstelle": station,
                "peak_day": float(peak_index),
                "response_peak": peak,
                "response_area": area,
            }
        )
    cluster_df = pd.DataFrame(rows)
    if cluster_df.empty:
        return cluster_df
    if cluster_df["peak_day"].nunique() >= 3:
        cluster_df["cluster"] = pd.qcut(
            cluster_df["peak_day"],
            q=3,
            labels=["fast", "medium", "slow"],
            duplicates="drop",
        ).astype(str)
    else:
        cluster_df["cluster"] = "similar"
    return cluster_df.sort_values(["cluster", "peak_day"]).reset_index(drop=True)


def normalize_metric_to_color(values, higher_is_better=True):
    numeric = pd.to_numeric(values, errors="coerce")
    if numeric.notna().sum() == 0:
        return [[96, 96, 96, 180] for _ in values]
    low = numeric.min()
    high = numeric.max()
    span = high - low
    colors = []
    for value in numeric:
        if pd.isna(value) or span == 0:
            score = 0.5
        else:
            score = (value - low) / span
            if not higher_is_better:
                score = 1.0 - score
        red = int(220 * (1.0 - score) + 35 * score)
        green = int(75 * (1.0 - score) + 145 * score)
        blue = int(65 * (1.0 - score) + 85 * score)
        colors.append([red, green, blue, 190])
    return colors


def normalize_metric_to_radius(values, min_radius=450, max_radius=1700):
    numeric = pd.to_numeric(values, errors="coerce").fillna(0.0)
    if numeric.max() == numeric.min():
        return [900 for _ in values]
    scaled = (numeric - numeric.min()) / (numeric.max() - numeric.min())
    return (min_radius + scaled * (max_radius - min_radius)).round().astype(int).tolist()


def get_forecast_presets(t):
    return {
        t["preset_custom"]: {},
        t["preset_normal"]: {
            "forecast_years": 10,
            "pattern_years": 3,
            "weather_source": t["forecast_source_pattern"],
            "rain_factor": 1.0,
            "rain_offset": 0.0,
            "evap_factor": 1.0,
            "evap_offset": 0.0,
            "clip_nonnegative": True,
            "api_days": 7,
            "uncertainty": 10,
        },
        t["preset_dry"]: {
            "forecast_years": 10,
            "pattern_years": 3,
            "weather_source": t["forecast_source_pattern"],
            "rain_factor": 0.75,
            "rain_offset": 0.0,
            "evap_factor": 1.15,
            "evap_offset": 0.0,
            "clip_nonnegative": True,
            "api_days": 7,
            "uncertainty": 15,
        },
        t["preset_wet"]: {
            "forecast_years": 10,
            "pattern_years": 3,
            "weather_source": t["forecast_source_pattern"],
            "rain_factor": 1.25,
            "rain_offset": 0.0,
            "evap_factor": 0.95,
            "evap_offset": 0.0,
            "clip_nonnegative": True,
            "api_days": 7,
            "uncertainty": 15,
        },
        t["preset_hot"]: {
            "forecast_years": 10,
            "pattern_years": 3,
            "weather_source": t["forecast_source_pattern"],
            "rain_factor": 0.7,
            "rain_offset": 0.0,
            "evap_factor": 1.25,
            "evap_offset": 0.1,
            "clip_nonnegative": True,
            "api_days": 7,
            "uncertainty": 20,
        },
        t["preset_open_meteo"]: {
            "forecast_years": 10,
            "pattern_years": 3,
            "weather_source": t["forecast_source_open_meteo"],
            "rain_factor": 1.0,
            "rain_offset": 0.0,
            "evap_factor": 1.0,
            "evap_offset": 0.0,
            "clip_nonnegative": True,
            "api_days": 7,
            "uncertainty": 10,
        },
    }


def apply_forecast_preset(preset):
    key_map = {
        "forecast_years": "forecast_years_control",
        "pattern_years": "forecast_pattern_years_control",
        "weather_source": "forecast_weather_source_control",
        "rain_factor": "forecast_rain_factor_control",
        "rain_offset": "forecast_rain_offset_control",
        "evap_factor": "forecast_evap_factor_control",
        "evap_offset": "forecast_evap_offset_control",
        "clip_nonnegative": "forecast_clip_control",
        "api_days": "forecast_api_days_control",
        "uncertainty": "forecast_uncertainty_control",
    }
    for preset_key, state_key in key_map.items():
        if preset_key in preset:
            st.session_state[state_key] = preset[preset_key]


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
        plot_aligned, was_downsampled = downsample_frame_for_plot(aligned)
        figure, axis = plt.subplots(figsize=(12, 5))
        axis.plot(plot_aligned.index, plot_aligned[t["observed"]], label=t["observed"], linewidth=1.2)
        axis.plot(plot_aligned.index, plot_aligned[t["simulated"]], label=t["simulated"], linewidth=1.0)
        axis.set_title(f"{model.name}: {t['plot_obs_sim']}")
        axis.set_ylabel(t["head_axis"])
        axis.grid(alpha=0.3)
        axis.legend()
        if was_downsampled:
            axis.text(0.01, 0.02, t["plot_downsampled"], transform=axis.transAxes, fontsize=9)
        figure.tight_layout()
        return figure

    residuals = model.residuals()

    if plot_type == t["plot_residuals"]:
        plot_residuals, was_downsampled = downsample_series_for_plot(residuals)
        figure, axis = plt.subplots(figsize=(12, 4))
        axis.plot(plot_residuals.index, plot_residuals.values, linewidth=0.9, color="#b22222")
        axis.axhline(0.0, color="black", linewidth=1.0, linestyle="--")
        axis.set_title(f"{model.name}: {t['plot_residuals']}")
        axis.set_ylabel(t["residuals"])
        axis.grid(alpha=0.3)
        if was_downsampled:
            axis.text(0.01, 0.02, t["plot_downsampled"], transform=axis.transAxes, fontsize=9)
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

    is_pulse_response = plot_type == t["plot_pulse"]
    response = (
        model.get_block_response("recharge")
        if is_pulse_response
        else model.get_step_response("recharge")
    )
    figure, axis = plt.subplots(figsize=(8, 4))
    axis.plot(
        response.index,
        response.values,
        color="#7a3db2" if is_pulse_response else "#1f7a1f",
        linewidth=1.5,
    )
    axis.axhline(0.0, color="black", linewidth=0.8, alpha=0.7)
    response_label = t["plot_pulse"] if is_pulse_response else t["plot_step"]
    axis.set_title(f"{model.name}: {response_label}")
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


def get_station_metadata_row(last_run_df, station):
    if last_run_df.empty:
        return None
    station_df = last_run_df[last_run_df["Messstelle"] == station].copy()
    if station_df.empty:
        return None
    if "BestStationModel" in station_df.columns and station_df["BestStationModel"].astype(bool).any():
        station_df = station_df[station_df["BestStationModel"].astype(bool)]
    if "R2" in station_df.columns:
        station_df = station_df.sort_values("R2", ascending=False, na_position="last")
    return station_df.iloc[0]


def coerce_bool(value):
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "ja", "y"}
    return bool(value)


def get_station_coordinates(station_row):
    if station_row is None:
        return None, None
    if {"Latitude", "Longitude"}.issubset(set(station_row.index)):
        latitude = pd.to_numeric(pd.Series([station_row["Latitude"]]), errors="coerce").iloc[0]
        longitude = pd.to_numeric(pd.Series([station_row["Longitude"]]), errors="coerce").iloc[0]
        if pd.notna(latitude) and pd.notna(longitude):
            return float(latitude), float(longitude)
    return None, None


def repeat_series_pattern(series, future_index, pattern_years):
    pattern_days = max(365, int(pattern_years * 365))
    pattern_values = series.dropna().tail(pattern_days).tolist()
    if not pattern_values:
        raise ValueError("Not enough source data for forecast pattern.")
    repeated = [pattern_values[i % len(pattern_values)] for i in range(len(future_index))]
    return pd.Series(repeated, index=future_index)


@st.cache_data(ttl=1800, show_spinner=False)
def fetch_open_meteo_daily_forecast(latitude, longitude, forecast_days, timezone_name):
    latitude = float(latitude)
    longitude = float(longitude)
    if not -90.0 <= latitude <= 90.0 or not -180.0 <= longitude <= 180.0:
        raise ValueError("Coordinates are outside WGS84 bounds.")

    response = requests.get(
        "https://api.open-meteo.com/v1/forecast",
        params={
            "latitude": latitude,
            "longitude": longitude,
            "daily": "precipitation_sum,et0_fao_evapotranspiration,temperature_2m_mean",
            "timezone": timezone_name,
            "forecast_days": int(forecast_days),
            "temperature_unit": "celsius",
            "precipitation_unit": "mm",
            "wind_speed_unit": "kmh",
            "timeformat": "iso8601",
        },
        timeout=20,
    )
    response.raise_for_status()
    payload = response.json()
    daily = payload.get("daily", {})
    forecast_df = pd.DataFrame(
        {
            "date": pd.to_datetime(daily.get("time", []), errors="coerce"),
            "rain": pd.to_numeric(
                pd.Series(daily.get("precipitation_sum", [])), errors="coerce"
            ),
            "evap": pd.to_numeric(
                pd.Series(daily.get("et0_fao_evapotranspiration", [])), errors="coerce"
            ),
            "temperature_2m_mean": pd.to_numeric(
                pd.Series(daily.get("temperature_2m_mean", [])), errors="coerce"
            ),
        }
    )
    forecast_df = forecast_df.dropna(subset=["date", "rain", "evap"])
    forecast_df["date"] = forecast_df["date"].dt.normalize()
    forecast_df = forecast_df.drop_duplicates(subset=["date"]).sort_values("date")
    return forecast_df


def create_future_weather(
    rain,
    evap,
    future_index,
    pattern_years,
    rain_factor,
    evap_factor,
    rain_offset,
    evap_offset,
    clip_nonnegative,
    external_weather_df=None,
):
    future_rain = repeat_series_pattern(rain, future_index, pattern_years)
    future_evap = repeat_series_pattern(evap, future_index, pattern_years)
    weather_source = pd.Series("pattern", index=future_index)
    inserted_external_rows = 0
    external_rows = 0

    if external_weather_df is not None and not external_weather_df.empty:
        external = external_weather_df.copy()
        external["date"] = pd.to_datetime(external["date"], errors="coerce").dt.normalize()
        external = external.dropna(subset=["date", "rain", "evap"])
        external = external.drop_duplicates(subset=["date"]).set_index("date").sort_index()
        external_rows = int(external.shape[0])
        aligned_index = future_index.intersection(external.index)
        inserted_external_rows = int(len(aligned_index))
        if inserted_external_rows:
            future_rain.loc[aligned_index] = external.loc[aligned_index, "rain"].astype(float)
            future_evap.loc[aligned_index] = external.loc[aligned_index, "evap"].astype(float)
            weather_source.loc[aligned_index] = "open_meteo"

    future_rain = future_rain.astype(float) * float(rain_factor) + float(rain_offset)
    future_evap = future_evap.astype(float) * float(evap_factor) + float(evap_offset)
    if clip_nonnegative:
        future_rain = future_rain.clip(lower=0.0)
        future_evap = future_evap.clip(lower=0.0)

    future_weather = pd.DataFrame(
        {
            "date": future_index,
            "rain": future_rain.values,
            "evap": future_evap.values,
            "weather_source": weather_source.values,
        }
    )
    forecast_info = {
        "external_rows": external_rows,
        "inserted_external_rows": inserted_external_rows,
    }
    return future_weather, forecast_info


def create_forecast_simulation(
    station,
    station_row,
    rain,
    evap,
    gw_df,
    forecast_years,
    pattern_years,
    rain_factor,
    evap_factor,
    rain_offset=0.0,
    evap_offset=0.0,
    clip_nonnegative=True,
    external_weather_df=None,
):
    future_days = int(max(1, forecast_years) * 365)
    future_index = pd.date_range(
        start=rain.index.max() + pd.Timedelta(days=1),
        periods=future_days,
        freq="D",
    )
    future_weather, forecast_info = create_future_weather(
        rain=rain,
        evap=evap,
        future_index=future_index,
        pattern_years=pattern_years,
        rain_factor=rain_factor,
        evap_factor=evap_factor,
        rain_offset=rain_offset,
        evap_offset=evap_offset,
        clip_nonnegative=clip_nonnegative,
        external_weather_df=external_weather_df,
    )
    future_rain = pd.Series(
        future_weather["rain"].values,
        index=future_index,
        name=rain.name,
    )
    future_evap = pd.Series(
        future_weather["evap"].values,
        index=future_index,
        name=evap.name,
    )

    rain_extended = pd.concat([rain, future_rain])
    evap_extended = pd.concat([evap, future_evap])

    head = gw_df[station].dropna()
    model = ps.Model(head, name=station)
    response_function = (
        ps.Gamma(cutoff=float(station_row["Cutoff"]))
        if station_row["Modell"] == "Gamma"
        else ps.Exponential(cutoff=float(station_row["Cutoff"]))
    )
    recharge_model = ps.rch.FlexModel() if coerce_bool(station_row["Flex"]) else ps.rch.Linear()
    stress_model = ps.RechargeModel(
        rain_extended,
        evap_extended,
        rfunc=response_function,
        recharge=recharge_model,
    )
    model.add_stressmodel(stress_model)

    if coerce_bool(station_row["Noise"]):
        noise_norm_value = True
        if "NoiseNorm" in station_row.index and pd.notna(station_row["NoiseNorm"]):
            noise_norm_value = coerce_bool(station_row["NoiseNorm"])
        model.add_noisemodel(ps.ArNoiseModel(norm=noise_norm_value))

    optimal_values = {}
    for name in model.parameters.index:
        optimal_col = f"param_opt_{name}"
        if optimal_col in station_row.index and pd.notna(station_row[optimal_col]):
            optimal_values[name] = float(station_row[optimal_col])
        elif name in model.parameters.index:
            optimal_values[name] = float(model.parameters.loc[name, "initial"])

    for name, value in optimal_values.items():
        model.set_parameter(name, initial=float(value), vary=False)

    simulation = model.simulate(
        p=model.parameters["initial"].values,
        tmin=head.index.min(),
        tmax=future_index.max(),
    )

    forecast_df = pd.DataFrame(
        {
            "date": simulation.index,
            "simulated_head": simulation.values,
        }
    )
    forecast_df["is_forecast"] = forecast_df["date"] > rain.index.max()
    weather_lookup = pd.DataFrame(
        {
            "date": rain_extended.index,
            "rain": rain_extended.values,
            "evap": evap_extended.values,
        }
    )
    source_lookup = pd.DataFrame(
        {
            "date": rain.index,
            "weather_source": "historical",
        }
    )
    source_lookup = pd.concat(
        [source_lookup, future_weather[["date", "weather_source"]]],
        ignore_index=True,
    )
    forecast_df = forecast_df.merge(weather_lookup, on="date", how="left")
    forecast_df = forecast_df.merge(source_lookup, on="date", how="left")
    return model, forecast_df, forecast_info


def build_run_summary_table(df):
    if df.empty or "Run" not in df.columns:
        return pd.DataFrame()
    summary_base = get_best_only_rows(df[df["Status"] == "ok"].copy())
    if summary_base.empty:
        return pd.DataFrame()
    summary = (
        summary_base.groupby("Run")
        .agg(
            Modelle=("Messstelle", "count"),
            Mean_R2=("R2", "mean"),
            Median_R2=("R2", "median"),
            Mean_RMSE=("RMSE", "mean"),
            Mean_AIC=("AIC", "mean"),
        )
        .reset_index()
    )
    return summary.round(3)


@st.cache_data
def create_excel_export(history_df, last_run_df):
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        if not history_df.empty:
            safe_history = sanitize_export_df(sort_results_for_display(history_df))
            safe_history.to_excel(
                writer,
                sheet_name="history",
                index=False,
            )
            sanitize_export_df(get_best_only_rows(history_df)).to_excel(
                writer,
                sheet_name="best_models",
                index=False,
            )
            errors = history_df[
                history_df["Fehler"].notna() & (history_df["Fehler"].astype(str) != "")
            ].copy()
            if not errors.empty:
                sanitize_export_df(errors).to_excel(writer, sheet_name="errors", index=False)
            run_summary = build_run_summary_table(history_df)
            if not run_summary.empty:
                run_summary.to_excel(writer, sheet_name="run_summary", index=False)
        if not last_run_df.empty:
            sanitize_export_df(sort_results_for_display(last_run_df)).to_excel(
                writer,
                sheet_name="last_run",
                index=False,
            )
    buffer.seek(0)
    return buffer.getvalue()


@st.cache_data
def create_html_report(history_df, last_run_df, data_status, t):
    summary = build_run_summary_table(history_df)
    best_models = get_best_only_rows(last_run_df) if not last_run_df.empty else pd.DataFrame()
    html_parts = [
        "<!doctype html><html><head><meta charset='utf-8'>",
        "<style>body{font-family:Arial,sans-serif;margin:32px;color:#111}",
        "table{border-collapse:collapse;width:100%;margin:16px 0}",
        "th,td{border:1px solid #ccc;padding:6px;text-align:left}",
        "th{background:#eef2f7}</style></head><body>",
        f"<h1>{html.escape(t['title'])}</h1>",
        f"<p>{html.escape(t['data_gw_until'])}: {html.escape(format_date_value(data_status.get('gw_last')))}<br>",
        f"{html.escape(t['data_weather_until'])}: {html.escape(format_date_value(data_status.get('weather_last')))}<br>",
        f"{html.escape(t['data_station_count'])}: {html.escape(str(data_status.get('station_count', '-')))}</p>",
    ]
    if not summary.empty:
        html_parts.append(f"<h2>{html.escape(t['run_compare_summary'])}</h2>")
        html_parts.append(sanitize_export_df(summary).to_html(index=False, escape=True))
    if not best_models.empty:
        display_cols = get_compact_result_columns(best_models)
        html_parts.append(f"<h2>{html.escape(t['best_model_flag'])}</h2>")
        html_parts.append(
            sanitize_export_df(best_models[display_cols]).to_html(index=False, escape=True)
            if display_cols
            else sanitize_export_df(best_models).to_html(index=False, escape=True)
        )
    html_parts.append("</body></html>")
    return "\n".join(html_parts).encode("utf-8")


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


@st.cache_resource
def get_job_manager():
    return {"jobs": {}, "lock": Lock()}


JOB_MANAGER = get_job_manager()


def create_job(job_id, payload):
    with JOB_MANAGER["lock"]:
        JOB_MANAGER["jobs"][job_id] = {
            "status": "queued",
            "cancel_requested": False,
            "progress": 0.0,
            "completed_steps": 0,
            "total_steps": payload["total_steps"],
            "current_station": None,
            "current_configuration": None,
            "run_label": payload["run_label"],
            "result_df": pd.DataFrame(),
            "models": {},
            "extra_match": {},
            "coord_match": {},
            "error_message": None,
        }


def update_job(job_id, **kwargs):
    with JOB_MANAGER["lock"]:
        if job_id in JOB_MANAGER["jobs"]:
            JOB_MANAGER["jobs"][job_id].update(kwargs)


def get_job(job_id):
    if not job_id:
        return None
    with JOB_MANAGER["lock"]:
        if job_id not in JOB_MANAGER["jobs"]:
            return None
        job = JOB_MANAGER["jobs"][job_id]
        return {
            key: value
            for key, value in job.items()
        }


def request_cancel(job_id):
    with JOB_MANAGER["lock"]:
        if job_id in JOB_MANAGER["jobs"]:
            JOB_MANAGER["jobs"][job_id]["cancel_requested"] = True


def is_cancel_requested(job_id):
    with JOB_MANAGER["lock"]:
        return bool(JOB_MANAGER["jobs"].get(job_id, {}).get("cancel_requested"))


def consume_finished_job(job_id):
    with JOB_MANAGER["lock"]:
        job = JOB_MANAGER["jobs"].get(job_id)
        if not job:
            return None
        if job["status"] not in {"completed", "cancelled", "failed"}:
            return None
        finished_job = {
            key: value
            for key, value in job.items()
        }
        JOB_MANAGER["jobs"].pop(job_id, None)
        return finished_job


def finalize_results(result_df, extra_df, coords_df):
    result_df, extra_match_info = merge_extra_data(result_df, extra_df)
    result_df, coord_match_info = merge_coordinate_data(result_df, coords_df)
    return result_df, extra_match_info, coord_match_info


def run_station_configurations(
    station,
    mode_label,
    run_strategy_label,
    search_configurations,
    gw_df,
    rain,
    evap,
    run_label,
    t,
    early_stop_enabled=False,
    early_stop_r2=0.85,
):
    head = gw_df[station].dropna()
    station_results = []
    station_models = []
    processed_steps = 0

    for configuration in search_configurations:
        processed_steps += 1
        if head.shape[0] < 50:
            station_results.append(
                create_result_row(
                    station=station,
                    mode_label=mode_label,
                    run_strategy_label=run_strategy_label,
                    configuration_label=configuration["configuration_label"],
                    model_type=configuration["model_type"],
                    use_flex=configuration["use_flex"],
                    use_noise=configuration["use_noise"],
                    response_cutoff=configuration["response_cutoff"],
                    noise_norm=configuration["noise_norm"],
                    head=head,
                    parameter_values=configuration["parameter_values"],
                    vary_flags=configuration["vary_flags"],
                    run_label=run_label,
                    error_message=t["too_few_obs"],
                )
            )
            continue

        try:
            model = build_model(
                head=head,
                rain=rain,
                evap=evap,
                model_type=configuration["model_type"],
                use_flex=configuration["use_flex"],
                use_noise=configuration["use_noise"],
                parameter_values=configuration["parameter_values"],
                vary_flags=configuration["vary_flags"],
                response_cutoff=configuration["response_cutoff"],
                noise_norm=configuration["noise_norm"],
            )
            row = create_result_row(
                station=station,
                mode_label=mode_label,
                run_strategy_label=run_strategy_label,
                configuration_label=configuration["configuration_label"],
                model_type=configuration["model_type"],
                use_flex=configuration["use_flex"],
                use_noise=configuration["use_noise"],
                response_cutoff=configuration["response_cutoff"],
                noise_norm=configuration["noise_norm"],
                head=head,
                parameter_values=configuration["parameter_values"],
                vary_flags=configuration["vary_flags"],
                run_label=run_label,
                model=model,
            )
            station_results.append(row)
            station_models.append((row, model))
            if (
                early_stop_enabled
                and run_strategy_label == t["strategy_auto"]
                and row["R2"] is not None
                and float(row["R2"]) >= float(early_stop_r2)
            ):
                break
        except Exception as exc:
            station_results.append(
                create_result_row(
                    station=station,
                    mode_label=mode_label,
                    run_strategy_label=run_strategy_label,
                    configuration_label=configuration["configuration_label"],
                    model_type=configuration["model_type"],
                    use_flex=configuration["use_flex"],
                    use_noise=configuration["use_noise"],
                    response_cutoff=configuration["response_cutoff"],
                    noise_norm=configuration["noise_norm"],
                    head=head,
                    parameter_values=configuration["parameter_values"],
                    vary_flags=configuration["vary_flags"],
                    run_label=run_label,
                    error_message=str(exc),
                )
            )

    selected_model = None
    if run_strategy_label == t["strategy_auto"]:
        successful_rows = [
            row for row in station_results if row["Status"] == "ok" and row["R2"] is not None
        ]
        if successful_rows:
            best_row = max(successful_rows, key=lambda row: row["R2"])
            for row in station_results:
                row["BestStationModel"] = row["Konfiguration"] == best_row["Konfiguration"]
            for row, model in station_models:
                if row["Konfiguration"] == best_row["Konfiguration"]:
                    selected_model = model
                    break
    elif station_models:
        selected_model = station_models[0][1]
        station_results[0]["BestStationModel"] = True

    return station, station_results, selected_model, processed_steps


def run_batch_job(
    job_id,
    selected_stations,
    mode_label,
    run_strategy_label,
    search_configurations,
    gw_df,
    rain,
    evap,
    extra_df,
    coords_df,
    run_label,
    t,
    parallel_workers=1,
    early_stop_enabled=False,
    early_stop_r2=0.85,
):
    update_job(job_id, status="running")
    results = []
    model_store = {}
    total_steps = len(selected_stations) * len(search_configurations)
    completed_steps = 0

    try:
        use_parallel = (
            int(parallel_workers) > 1
            and run_strategy_label == t["strategy_auto"]
            and len(selected_stations) > 1
        )
        if use_parallel:
            with ThreadPoolExecutor(max_workers=int(parallel_workers)) as executor:
                futures = {
                    executor.submit(
                        run_station_configurations,
                        station,
                        mode_label,
                        run_strategy_label,
                        search_configurations,
                        gw_df,
                        rain,
                        evap,
                        run_label,
                        t,
                        early_stop_enabled,
                        early_stop_r2,
                    ): station
                    for station in selected_stations
                }
                for future in as_completed(futures):
                    if is_cancel_requested(job_id):
                        for pending in futures:
                            pending.cancel()
                        raise JobCancelled
                    station = futures[future]
                    update_job(
                        job_id,
                        current_station=station,
                        current_configuration="parallel",
                    )
                    station, station_results, selected_model, processed_steps = future.result()
                    completed_steps += processed_steps
                    results.extend(station_results)
                    if selected_model is not None:
                        model_store[station] = selected_model
                    update_job(
                        job_id,
                        completed_steps=completed_steps,
                        progress=min(completed_steps / max(total_steps, 1), 1.0),
                    )
        else:
            for station in selected_stations:
                if is_cancel_requested(job_id):
                    raise JobCancelled
                update_job(
                    job_id,
                    current_station=station,
                    current_configuration="station",
                )
                station, station_results, selected_model, processed_steps = run_station_configurations(
                    station,
                    mode_label,
                    run_strategy_label,
                    search_configurations,
                    gw_df,
                    rain,
                    evap,
                    run_label,
                    t,
                    early_stop_enabled,
                    early_stop_r2,
                )
                completed_steps += processed_steps
                results.extend(station_results)
                if selected_model is not None:
                    model_store[station] = selected_model
                update_job(
                    job_id,
                    completed_steps=completed_steps,
                    progress=min(completed_steps / max(total_steps, 1), 1.0),
                )

        result_df = pd.DataFrame(results)
        result_df, extra_match, coord_match = finalize_results(result_df, extra_df, coords_df)
        update_job(
            job_id,
            status="completed",
            result_df=result_df,
            models=model_store,
            extra_match=extra_match,
            coord_match=coord_match,
            progress=1.0,
        )
    except JobCancelled:
        result_df = pd.DataFrame(results)
        result_df, extra_match, coord_match = finalize_results(result_df, extra_df, coords_df)
        update_job(
            job_id,
            status="cancelled",
            result_df=result_df,
            models=model_store,
            extra_match=extra_match,
            coord_match=coord_match,
        )
    except Exception as exc:
        result_df = pd.DataFrame(results)
        if not result_df.empty:
            result_df, extra_match, coord_match = finalize_results(result_df, extra_df, coords_df)
        else:
            extra_match, coord_match = {}, {}
        update_job(
            job_id,
            status="failed",
            result_df=result_df,
            models=model_store,
            extra_match=extra_match,
            coord_match=coord_match,
            error_message=str(exc),
        )


def start_batch_thread(job_id, **kwargs):
    thread = Thread(target=run_batch_job, kwargs={"job_id": job_id, **kwargs}, daemon=True)
    thread.start()


@st.fragment(run_every="2s")
def render_active_job_status():
    current_job = get_job(st.session_state.active_job_id)
    if current_job is None:
        return
    if current_job["status"] not in {"queued", "running"}:
        st.rerun()
        return

    st.info(t["running_info"])
    progress_col1, progress_col2, progress_col3 = st.columns(3)
    progress_col1.metric(t["progress"], current_job.get("current_station") or "-")
    progress_col2.metric(
        t["progress_config"], current_job.get("current_configuration") or "-"
    )
    progress_col3.metric(
        t["progress_steps"],
        f"{current_job.get('completed_steps', 0)} / {current_job.get('total_steps', 0)}",
    )
    st.progress(float(current_job.get("progress", 0.0)))


t = TEXT[st.session_state.lang]


with st.sidebar:
    st.title(t["sidebar"])
    language = st.radio(
        t["lang"],
        ["Deutsch", "English"],
        index=0 if st.session_state.lang == "Deutsch" else 1,
    )
    if language != st.session_state.lang:
        next_t = TEXT[language]
        if persist_current_sources():
            st.session_state.upload_cache_notice = ("success", next_t["cache_lang_saved"])
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
    gw_file = st.file_uploader(
        t["upload_gw"],
        type=["csv", "xlsx", "ods"],
        key="gw_upload_file",
    )
    weather_file = st.file_uploader(
        t["upload_weather"],
        type=["csv", "xlsx", "ods"],
        key="weather_upload_file",
    )
    extra_file = st.file_uploader(
        t["upload_extra"],
        type=["csv", "xlsx", "ods"],
        key="extra_upload_file",
    )
    coords_file = st.file_uploader(
        t["upload_coords"],
        type=["csv", "xlsx", "ods"],
        key="coords_upload_file",
    )
    with st.expander(t["remote_heading"], expanded=False):
        st.caption(t["remote_help"])
        gw_url = st.text_input(t["remote_gw"], key="remote_gw_url")
        weather_url = st.text_input(t["remote_weather"], key="remote_weather_url")
        extra_url = st.text_input(t["remote_extra"], key="remote_extra_url")
        coords_url = st.text_input(t["remote_coords"], key="remote_coords_url")
        if st.button(t["reload_sources"], key="reload_sources_button"):
            load_data.clear()
            st.session_state.data_refresh_token += 1
            st.session_state.data_refresh_notice = ("success", t["reload_sources_done"])
            st.session_state.last_forecast_df = pd.DataFrame()
            st.session_state.last_forecast_info = {}
            st.rerun()
    with st.expander(t["cache_heading"], expanded=False):
        st.caption(t["cache_help"])
        cache_descriptions = describe_cached_sources(t)
        if cache_descriptions:
            st.caption(t["cache_active"])
            for description in cache_descriptions:
                st.caption(description)
        cache_col1, cache_col2 = st.columns(2)
        with cache_col1:
            if st.button(t["cache_save"], key="save_upload_cache_button"):
                manifest = persist_current_sources()
                st.session_state.upload_cache_notice = (
                    "success" if manifest else "warning",
                    t["cache_saved"] if manifest else t["cache_empty"],
                )
                st.rerun()
        with cache_col2:
            if st.button(t["cache_clear"], key="clear_upload_cache_button"):
                clear_upload_cache()
                st.session_state.upload_cache_notice = ("success", t["cache_cleared"])
                st.rerun()


st.markdown(
    get_accessibility_css(font_profile, text_scale, high_contrast, strong_focus, t),
    unsafe_allow_html=True,
)

st.title(t["title"])
st.caption(t["beta_note"])


active_job = get_job(st.session_state.active_job_id)
finished_job = consume_finished_job(st.session_state.active_job_id)
if finished_job is not None:
    result_df = finished_job["result_df"].copy()
    if not result_df.empty:
        st.session_state.history = append_history_entries(
            st.session_state.history,
            result_df,
            st.session_state.history_run_limit,
        )
        st.session_state.last_run_results = result_df
    else:
        st.session_state.last_run_results = pd.DataFrame()

    st.session_state.last_run_models = finished_job["models"]
    st.session_state.last_run_label = finished_job["run_label"]
    st.session_state.last_extra_match = finished_job["extra_match"]
    st.session_state.last_coord_match = finished_job["coord_match"]
    st.session_state.last_forecast_df = pd.DataFrame()
    st.session_state.last_forecast_info = {}
    st.session_state.active_job_id = None

    if finished_job["status"] == "completed":
        st.session_state.job_notice = ("success", t["success"])
    elif finished_job["status"] == "cancelled":
        st.session_state.job_notice = ("warning", t["cancelled"])
    else:
        message = finished_job.get("error_message") or t["job_failed"]
        st.session_state.job_notice = ("error", f"{t['job_failed']}: {message}")

active_job = get_job(st.session_state.active_job_id)


if st.session_state.job_notice is not None:
    notice_type, notice_text = st.session_state.job_notice
    if notice_type == "success":
        st.success(notice_text)
    elif notice_type == "warning":
        st.warning(notice_text)
    else:
        st.error(notice_text)
    st.session_state.job_notice = None

if st.session_state.upload_cache_notice is not None:
    notice_type, notice_text = st.session_state.upload_cache_notice
    if notice_type == "success":
        st.success(notice_text)
    else:
        st.warning(notice_text)
    st.session_state.upload_cache_notice = None

if st.session_state.data_refresh_notice is not None:
    notice_type, notice_text = st.session_state.data_refresh_notice
    if notice_type == "success":
        st.success(notice_text)
    else:
        st.info(notice_text)
    st.session_state.data_refresh_notice = None

if st.session_state.history_notice is not None:
    notice_type, notice_text = st.session_state.history_notice
    if notice_type == "success":
        st.success(notice_text)
    else:
        st.info(notice_text)
    st.session_state.history_notice = None


cached_sources, active_cached_source_keys = resolve_cached_sources()


gw_source = gw_file if gw_file is not None else (gw_url.strip() if gw_url.strip() else cached_sources.get("gw"))
weather_source = weather_file if weather_file is not None else (weather_url.strip() if weather_url.strip() else cached_sources.get("weather"))
extra_source = extra_file if extra_file is not None else (extra_url.strip() if extra_url.strip() else cached_sources.get("extra"))
coords_source = coords_file if coords_file is not None else (coords_url.strip() if coords_url.strip() else cached_sources.get("coords"))

source_mapping = {
    "gw": gw_source,
    "weather": weather_source,
    "extra": extra_source,
    "coords": coords_source,
}
source_descriptions = describe_active_sources(
    source_mapping,
    active_cached_source_keys,
    t,
)

with st.sidebar:
    if source_descriptions:
        st.caption(t["source_summary"])
        for description in source_descriptions:
            st.caption(description)


if not (gw_source and weather_source):
    st.info(t["waiting"])
    st.stop()


try:
    gw_df, rain, evap, extra_df, coords_df = load_data(
        gw_source,
        weather_source,
        extra_source,
        coords_source,
        st.session_state.data_refresh_token,
    )
except Exception as exc:
    st.error(f"{t['load_error']}: {exc}")
    st.stop()


stations = [column for column in gw_df.columns if gw_df[column].dropna().shape[0] > 50]
if not stations:
    st.error(t["no_valid_stations"])
    st.stop()


data_status = build_data_status(gw_df, rain, evap)
with st.expander(t["data_status"], expanded=True):
    st.caption(t["data_status_note"])
    status_cols = st.columns(4)
    status_cols[0].metric(t["data_gw_until"], format_date_value(data_status["gw_last"]))
    status_cols[1].metric(t["data_weather_until"], format_date_value(data_status["weather_last"]))
    max_gap = max(
        data_status["gw_gap_days"] or 0,
        data_status["weather_gap_days"] or 0,
    )
    status_cols[2].metric(t["data_gap_days"], f"{max_gap} d")
    status_cols[3].metric(t["data_station_count"], data_status["station_count"])
    if max_gap > 30:
        st.warning(t["data_update_hint"])
    else:
        st.success(t["data_fresh"])
    with st.expander(t["data_quality_table"], expanded=False):
        st.dataframe(
            build_data_quality_table(gw_df),
            use_container_width=True,
            hide_index=True,
        )


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
    run_strategy = st.selectbox(
        t["run_strategy"],
        [t["strategy_manual"], t["strategy_auto"]],
    )

with control_col4:
    if run_strategy == t["strategy_manual"]:
        model_type = st.selectbox(t["model"], ["Gamma", "Exponential"])
    else:
        st.metric(t["auto_combo_count"], 8)
        model_type = "Gamma"

with control_col5:
    if run_strategy == t["strategy_manual"]:
        use_flex = st.checkbox(t["use_flex"], value=True)
        use_noise = st.checkbox(t["use_noise"], value=True)
    else:
        st.caption(t["strategy_auto_note"])
        use_flex = True
        use_noise = True

selected_station_preview = get_selected_stations(
    mode, single_station, multi_stations, stations, t
)
st.caption(f"{t['selection_count']}: {len(selected_station_preview)}")


parameter_values = {}
vary_flags = {}

manual_specs = get_parameter_specs(model_type, use_flex) if run_strategy == t["strategy_manual"] else []

with st.expander(t["parameter_box"], expanded=False):
    st.caption(
        t["parameter_help_manual"]
        if run_strategy == t["strategy_manual"]
        else t["parameter_help_auto"]
    )
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
        noise_norm = st.checkbox(t["noise_norm"], value=True)

    if run_strategy == t["strategy_manual"]:
        for spec in manual_specs:
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

parallel_workers = 1
early_stop_enabled = False
early_stop_r2 = 0.85
with st.expander(t["performance_box"], expanded=False):
    perf_col1, perf_col2, perf_col3 = st.columns(3)
    with perf_col1:
        parallel_workers = st.slider(
            t["parallel_workers"],
            min_value=1,
            max_value=MAX_PARALLEL_WORKERS,
            value=1,
            step=1,
            disabled=run_strategy != t["strategy_auto"],
        )
    with perf_col2:
        early_stop_enabled = st.checkbox(
            t["early_stop"],
            value=False,
            disabled=run_strategy != t["strategy_auto"],
        )
    with perf_col3:
        early_stop_r2 = st.slider(
            t["early_stop_r2"],
            min_value=0.5,
            max_value=0.99,
            value=0.85,
            step=0.01,
            disabled=not early_stop_enabled or run_strategy != t["strategy_auto"],
        )
    st.caption(t["parallel_note"])

search_configurations = get_search_configurations(
    run_strategy_label=run_strategy,
    manual_model_type=model_type,
    manual_use_flex=use_flex,
    manual_use_noise=use_noise,
    manual_parameter_values=parameter_values,
    manual_vary_flags=vary_flags,
    response_cutoff=response_cutoff,
    noise_norm=noise_norm,
    t=t,
)


run_col1, run_col2 = st.columns([1, 1])
with run_col1:
    if st.button(
        t["run"],
        type="primary",
        disabled=active_job is not None and active_job["status"] in {"queued", "running"},
    ):
        selected_stations = get_selected_stations(
            mode, single_station, multi_stations, stations, t
        )
        if not selected_stations:
            st.warning(t["choose_station"])
        else:
            run_label = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            job_id = uuid4().hex
            payload = {
                "run_label": run_label,
                "total_steps": len(selected_stations) * len(search_configurations),
            }
            create_job(job_id, payload)
            st.session_state.active_job_id = job_id
            start_batch_thread(
                job_id,
                selected_stations=selected_stations,
                mode_label=mode,
                run_strategy_label=run_strategy,
                search_configurations=search_configurations,
                gw_df=gw_df.copy(),
                rain=rain.copy(),
                evap=evap.copy(),
                extra_df=None if extra_df is None else extra_df.copy(),
                coords_df=None if coords_df is None else coords_df.copy(),
                run_label=run_label,
                t=t,
                parallel_workers=parallel_workers,
                early_stop_enabled=early_stop_enabled,
                early_stop_r2=early_stop_r2,
            )
            st.rerun()

with run_col2:
    if active_job is not None and active_job["status"] in {"queued", "running"}:
        if st.button(t["cancel_run"], key="cancel_active_run"):
            request_cancel(st.session_state.active_job_id)
            st.warning(t["cancel_requested"])


active_job = get_job(st.session_state.active_job_id)
if active_job is not None and active_job["status"] in {"queued", "running"}:
    render_active_job_status()


last_run_results = st.session_state.last_run_results
last_run_models = st.session_state.last_run_models

if not last_run_results.empty:
    summary_base = get_summary_base(last_run_results)
    error_base = last_run_results[last_run_results["Status"] != "ok"].copy()

    summary_col1, summary_col2, summary_col3, summary_col4 = st.columns(4)
    summary_col1.metric(t["successful_runs"], int(summary_base.shape[0]))
    summary_col2.metric(t["failed_runs"], int(error_base.shape[0]))
    if not summary_base.empty and summary_base["R2"].notna().any():
        best_row = summary_base.sort_values("R2", ascending=False).iloc[0]
        summary_col3.metric(
            t["best_station"],
            f"{best_row['Messstelle']} ({best_row['R2']:.3f})",
        )
        summary_col4.metric(t["mean_r2"], f"{summary_base['R2'].mean():.3f}")
    else:
        summary_col3.metric(t["best_station"], "-")
        summary_col4.metric(t["mean_r2"], "-")

    if st.session_state.last_run_label is not None:
        st.caption(f"{t['run_label']}: {st.session_state.last_run_label}")

    if extra_source is not None and st.session_state.last_extra_match:
        extra_info = st.session_state.last_extra_match
        st.caption(
            f"{t['extra_match']}: {extra_info.get('matched', 0)} | "
            f"{t['extra_missing']}: {extra_info.get('missing', 0)} | "
            f"{t['extra_unused']}: {extra_info.get('unused', 0)}"
        )

    if coords_source is not None and st.session_state.last_coord_match:
        coord_info = st.session_state.last_coord_match
        st.caption(
            f"{t['coords_match']}: {coord_info.get('matched', 0)} | "
            f"{t['coords_missing']}: {coord_info.get('missing', 0)} | "
            f"{t['coords_unused']}: {coord_info.get('unused', 0)}"
        )


tab_plot, tab_diagnostics, tab_compare, tab_map, tab_forecast, tab_save = st.tabs(
    [
        t["tab_plot"],
        t["tab_diagnostics"],
        t["tab_compare"],
        t["tab_map"],
        t["tab_forecast"],
        t["tab_save"],
    ]
)


with tab_plot:
    if not last_run_models:
        st.info(t["no_run"])
    else:
        plot_station = st.selectbox(t["plot_station"], list(last_run_models.keys()))
        plot_type = st.selectbox(
            t["plot_type"],
            [
                t["plot_overview"],
                t["plot_obs_sim"],
                t["plot_residuals"],
                t["plot_res_hist"],
                t["plot_step"],
                t["plot_pulse"],
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


with tab_diagnostics:
    if not last_run_models:
        st.info(t["no_run"])
    else:
        diagnostic_station = st.selectbox(
            t["diag_station"],
            list(last_run_models.keys()),
            key="diagnostic_station_select",
        )
        diagnostic_model = last_run_models[diagnostic_station]
        diagnostic_row = get_station_result_row(last_run_results, diagnostic_station)
        residuals = diagnostic_model.residuals().dropna()
        bias = float(residuals.mean()) if not residuals.empty else 0.0
        resid_std = float(residuals.std()) if residuals.shape[0] > 1 else 0.0
        resid_acf = residual_lag1_autocorr(residuals)
        station_quality = build_data_quality_table(gw_df)
        missing_pct = 0.0
        if not station_quality.empty:
            quality_match = station_quality[station_quality["Messstelle"] == diagnostic_station]
            if not quality_match.empty:
                missing_pct = float(quality_match.iloc[0]["missing_pct"])
        quality_key = classify_model_quality(diagnostic_row, resid_acf)

        st.markdown(f"### {t['diag_heading']}")
        diag_cols = st.columns(5)
        diag_cols[0].metric(t["diag_bias"], f"{bias:.4f}")
        diag_cols[1].metric(t["diag_resid_std"], f"{resid_std:.4f}")
        diag_cols[2].metric(
            t["diag_resid_acf"],
            "-" if resid_acf is None or pd.isna(resid_acf) else f"{resid_acf:.3f}",
        )
        diag_cols[3].metric(t["diag_missing_pct"], f"{missing_pct:.1f}%")
        diag_cols[4].metric(t["diag_quality"], quality_label(quality_key, t))

        seasonal_error = build_seasonal_error_table(diagnostic_model)
        if not seasonal_error.empty:
            st.markdown(f"#### {t['diag_seasonal_error']}")
            figure, axis = plt.subplots(figsize=(9, 3.5))
            axis.bar(
                seasonal_error["month"],
                seasonal_error["mean_abs_residual"],
                color="#2f6db3",
                alpha=0.75,
            )
            axis.set_xlabel("month")
            axis.set_ylabel(t["residuals"])
            axis.grid(axis="y", alpha=0.25)
            figure.tight_layout()
            st.pyplot(figure)
            plt.close(figure)
            st.dataframe(seasonal_error, use_container_width=True, hide_index=True)

        cluster_df = build_response_cluster_table(last_run_models)
        if not cluster_df.empty:
            st.markdown(f"#### {t['diag_cluster']}")
            st.caption(t["diag_cluster_note"])
            st.dataframe(cluster_df.round(4), use_container_width=True, hide_index=True)


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
        compare_df = (
            last_run_results.copy()
            if scope == t["scope_last"] and not last_run_results.empty
            else st.session_state.history.copy()
        )

        available_runs = []
        if "Run" in compare_df.columns:
            available_runs = [run for run in compare_df["Run"].dropna().unique().tolist()]
            available_runs = sorted(available_runs)
        if scope == t["scope_all"] and available_runs:
            selected_runs = st.multiselect(
                t["run_filter"],
                available_runs,
                default=available_runs,
            )
            if selected_runs:
                compare_df = compare_df[compare_df["Run"].isin(selected_runs)].copy()

        filter_col1, filter_col2, filter_col3 = st.columns(3)
        with filter_col1:
            best_only = False
            if "BestStationModel" in compare_df.columns:
                best_only = st.checkbox(
                    t["best_only"],
                    value=False,
                )
        with filter_col2:
            r2_filter_enabled = st.checkbox(t["r2_filter_toggle"], value=False)
        with filter_col3:
            r2_threshold = st.slider(
                t["r2_threshold"],
                min_value=0.0,
                max_value=1.0,
                value=0.6,
                step=0.05,
            )

        if best_only:
            compare_df = get_best_only_rows(compare_df)
        if r2_filter_enabled and "R2" in compare_df.columns:
            compare_df = compare_df[
                compare_df["R2"].isna() | (compare_df["R2"] >= float(r2_threshold))
            ].copy()

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
                default_selection = [
                    column for column in ["R2", "RMSE", "EVP", "AIC"] if column in numeric_columns
                ]
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
                default_selection = [
                    column
                    for column in ["R2", "RMSE", "CPC", "PC1", "PC2", "PC3", "PC4", "PC5", "PC6", "PC7"]
                    if column in numeric_columns
                ]
                selected_corr_columns = st.multiselect(
                    t["corr_cols"],
                    numeric_columns,
                    default=default_selection[:10],
                )
                if len(selected_corr_columns) < 2:
                    st.info(t["no_numeric"])
                else:
                    corr = compare_df[selected_corr_columns].corr()
                    figure, axis = plt.subplots(figsize=(9, 7))
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

        if "Run" in st.session_state.history.columns and st.session_state.history["Run"].nunique() > 1:
            st.markdown(f"### {t['run_compare']}")
            st.caption(t["run_compare_note"])
            compare_history = st.session_state.history.copy()
            if "BestStationModel" in compare_history.columns:
                compare_history = get_best_only_rows(compare_history)
            run_compare_metrics = default_numeric_columns(get_numeric_columns(compare_history))
            if run_compare_metrics:
                run_compare_cols = st.columns(2)
                available_run_labels = sorted(compare_history["Run"].dropna().unique().tolist())
                with run_compare_cols[0]:
                    compare_runs = st.multiselect(
                        t["run_compare_runs"],
                        available_run_labels,
                        default=available_run_labels[-2:] if len(available_run_labels) >= 2 else available_run_labels,
                    )
                with run_compare_cols[1]:
                    compare_metric = st.selectbox(
                        t["run_compare_metric"],
                        run_compare_metrics,
                        key="run_compare_metric_select",
                    )

                if compare_runs:
                    run_compare_df = compare_history[compare_history["Run"].isin(compare_runs)].copy()
                    summary = (
                        run_compare_df.groupby("Run")[compare_metric]
                        .agg(["mean", "median", "min", "max", "count"])
                        .reset_index()
                        .round(3)
                    )
                    st.markdown(f"#### {t['run_compare_summary']}")
                    st.dataframe(summary, use_container_width=True, hide_index=True)

                    station_pivot = (
                        run_compare_df.pivot_table(
                            index="Messstelle",
                            columns="Run",
                            values=compare_metric,
                            aggfunc="max",
                        )
                        .reset_index()
                        .rename_axis(None, axis=1)
                    )
                    st.markdown(f"#### {t['run_compare_station']}")
                    st.dataframe(station_pivot, use_container_width=True, hide_index=True)

        if "Fehler" in compare_df.columns:
            error_df = compare_df[
                compare_df["Fehler"].notna() & (compare_df["Fehler"].astype(str) != "")
            ]
            if not error_df.empty:
                with st.expander(t["error_table"], expanded=False):
                    error_display = error_df[
                        ["Run", "Messstelle", "Konfiguration", "Status", "Fehler"]
                    ].rename(columns=get_display_column_labels(t))
                    st.dataframe(error_display, use_container_width=True, hide_index=True)


with tab_map:
    if st.session_state.history.empty:
        st.info(t["no_run"])
    else:
        map_scope = st.radio(
            t["map_scope"],
            [t["scope_last"], t["scope_all"]],
            horizontal=True,
            key="map_scope_radio",
        )
        map_df = (
            last_run_results.copy()
            if map_scope == t["scope_last"] and not last_run_results.empty
            else st.session_state.history.copy()
        )
        if "BestStationModel" in map_df.columns and map_df["BestStationModel"].astype(bool).any():
            map_df = get_best_only_rows(map_df)
        map_df = map_df.dropna(subset=["Latitude", "Longitude"]) if {"Latitude", "Longitude"}.issubset(map_df.columns) else pd.DataFrame()

        if map_df.empty:
            st.info(t["map_no_coords"])
            st.caption(t["map_note"])
        else:
            metric_options = default_numeric_columns(get_numeric_columns(map_df))
            if metric_options:
                map_control_cols = st.columns(2)
                with map_control_cols[0]:
                    map_metric = st.selectbox(
                        t["map_metric"],
                        metric_options,
                        index=metric_options.index("R2") if "R2" in metric_options else 0,
                    )
                with map_control_cols[1]:
                    size_metric = st.selectbox(
                        t["map_size_metric"],
                        ["n_obs"] + [metric for metric in metric_options if metric != "n_obs"],
                        index=0,
                    )
                map_plot_df = map_df.copy()
                map_plot_df[map_metric] = pd.to_numeric(map_plot_df[map_metric], errors="coerce")
                size_source = (
                    pd.to_numeric(map_plot_df[size_metric], errors="coerce")
                    if size_metric in map_plot_df.columns
                    else pd.Series(1, index=map_plot_df.index)
                )
                map_plot_df["color"] = normalize_metric_to_color(
                    map_plot_df[map_metric],
                    higher_is_better=not ranking_ascending(map_metric),
                )
                map_plot_df["radius"] = normalize_metric_to_radius(size_source)
                map_plot_df["tooltip"] = map_plot_df.apply(
                    lambda row: (
                        f"{row.get('Messstelle', '')}<br>"
                        f"{map_metric}: {row.get(map_metric, '')}<br>"
                        f"R2: {row.get('R2', '')}<br>"
                        f"RMSE: {row.get('RMSE', '')}"
                    ),
                    axis=1,
                )
                view_state = pdk.ViewState(
                    latitude=float(map_plot_df["Latitude"].mean()),
                    longitude=float(map_plot_df["Longitude"].mean()),
                    zoom=7,
                    pitch=0,
                )
                layer = pdk.Layer(
                    "ScatterplotLayer",
                    data=map_plot_df,
                    get_position="[Longitude, Latitude]",
                    get_fill_color="color",
                    get_radius="radius",
                    pickable=True,
                    auto_highlight=True,
                )
                st.pydeck_chart(
                    pdk.Deck(
                        layers=[layer],
                        initial_view_state=view_state,
                        tooltip={"html": "{tooltip}", "style": {"color": "white"}},
                    )
                )
                st.caption(t["map_color_note"])
                map_columns = [
                    column
                    for column in [
                        "Messstelle",
                        "Konfiguration",
                        "R2",
                        "RMSE",
                        "CPC",
                        "Latitude",
                        "Longitude",
                        map_metric,
                    ]
                    if column in map_df.columns
                ]
                map_table = sort_results_for_display(
                    map_df[map_columns].drop_duplicates()
                ).rename(columns=get_display_column_labels(t))
                st.markdown(f"### {t['map_table']}")
                st.dataframe(map_table, use_container_width=True, hide_index=True)


with tab_forecast:
    if not last_run_models or last_run_results.empty:
        st.info(t["forecast_missing"])
    else:
        st.caption(t["forecast_note"])
        forecast_presets = get_forecast_presets(t)
        default_forecast_state = {
            "forecast_years_control": 10,
            "forecast_pattern_years_control": 3,
            "forecast_weather_source_control": t["forecast_source_pattern"],
            "forecast_rain_factor_control": 0.8,
            "forecast_rain_offset_control": 0.0,
            "forecast_evap_factor_control": 1.0,
            "forecast_evap_offset_control": 0.0,
            "forecast_clip_control": True,
            "forecast_api_days_control": 7,
            "forecast_uncertainty_control": 0,
        }
        for state_key, state_value in default_forecast_state.items():
            if state_key not in st.session_state:
                st.session_state[state_key] = state_value
        weather_source_options = [t["forecast_source_pattern"], t["forecast_source_open_meteo"]]
        if st.session_state.get("forecast_weather_source_control") not in weather_source_options:
            st.session_state.forecast_weather_source_control = t["forecast_source_pattern"]
        if st.session_state.get("forecast_preset_select") not in forecast_presets:
            st.session_state.forecast_preset_select = t["preset_custom"]

        preset_cols = st.columns([2, 1])
        with preset_cols[0]:
            selected_preset = st.selectbox(
                t["forecast_preset"],
                list(forecast_presets.keys()),
                key="forecast_preset_select",
            )
        with preset_cols[1]:
            if st.button(t["forecast_apply_preset"], key="forecast_apply_preset_button"):
                apply_forecast_preset(forecast_presets[selected_preset])
                st.rerun()

        if (data_status.get("weather_gap_days") or 0) > 30:
            st.warning(t["forecast_stale_weather"])

        forecast_station_options = list(last_run_models.keys())
        forecast_controls = st.columns(4)
        with forecast_controls[0]:
            forecast_station = st.selectbox(
                t["forecast_station"],
                forecast_station_options,
                key="forecast_station_select",
            )
        forecast_station_row = get_station_metadata_row(last_run_results, forecast_station)
        station_latitude, station_longitude = get_station_coordinates(forecast_station_row)
        has_station_coords = station_latitude is not None and station_longitude is not None

        with forecast_controls[1]:
            forecast_years = st.slider(
                t["forecast_years"],
                min_value=1,
                max_value=20,
                step=1,
                key="forecast_years_control",
            )
        with forecast_controls[2]:
            pattern_years = st.slider(
                t["forecast_pattern_years"],
                min_value=1,
                max_value=10,
                step=1,
                key="forecast_pattern_years_control",
            )
        with forecast_controls[3]:
            weather_source = st.selectbox(
                t["forecast_weather_source"],
                weather_source_options,
                key="forecast_weather_source_control",
            )

        scenario_controls = st.columns(6)
        with scenario_controls[0]:
            rain_factor = st.slider(
                t["forecast_rain_factor"],
                min_value=0.5,
                max_value=1.5,
                step=0.05,
                key="forecast_rain_factor_control",
            )
        with scenario_controls[1]:
            rain_offset = st.number_input(
                t["forecast_rain_offset"],
                min_value=-20.0,
                max_value=20.0,
                step=0.1,
                key="forecast_rain_offset_control",
            )
        with scenario_controls[2]:
            evap_factor = st.slider(
                t["forecast_evap_factor"],
                min_value=0.5,
                max_value=1.5,
                step=0.05,
                key="forecast_evap_factor_control",
            )
        with scenario_controls[3]:
            evap_offset = st.number_input(
                t["forecast_evap_offset"],
                min_value=-20.0,
                max_value=20.0,
                step=0.1,
                key="forecast_evap_offset_control",
            )
        with scenario_controls[4]:
            clip_nonnegative = st.checkbox(
                t["forecast_clip_nonnegative"],
                key="forecast_clip_control",
            )
        with scenario_controls[5]:
            uncertainty_pct = st.slider(
                t["forecast_uncertainty"],
                min_value=0,
                max_value=30,
                step=5,
                key="forecast_uncertainty_control",
            )

        api_forecast_days = 7
        forecast_timezone = "Europe/Berlin"
        manual_coords = False
        forecast_latitude = station_latitude
        forecast_longitude = station_longitude

        if weather_source == t["forecast_source_open_meteo"]:
            api_controls = st.columns(3)
            with api_controls[0]:
                api_forecast_days = st.slider(
                    t["forecast_api_days"],
                    min_value=1,
                    max_value=16,
                    step=1,
                    key="forecast_api_days_control",
                )
            with api_controls[1]:
                forecast_timezone = st.selectbox(
                    t["forecast_timezone"],
                    ["Europe/Berlin", "auto", "UTC"],
                )
            with api_controls[2]:
                manual_coords = st.checkbox(
                    t["forecast_use_manual_coords"],
                    value=not has_station_coords,
                )

            if manual_coords or not has_station_coords:
                coord_controls = st.columns(2)
                with coord_controls[0]:
                    forecast_latitude = st.number_input(
                        t["forecast_latitude"],
                        min_value=-90.0,
                        max_value=90.0,
                        value=float(station_latitude) if has_station_coords else 52.4,
                        step=0.01,
                        format="%.5f",
                    )
                with coord_controls[1]:
                    forecast_longitude = st.number_input(
                        t["forecast_longitude"],
                        min_value=-180.0,
                        max_value=180.0,
                        value=float(station_longitude) if has_station_coords else 13.1,
                        step=0.01,
                        format="%.5f",
                    )

        if st.button(t["forecast_run"], key="run_forecast_button"):
            if forecast_station_row is None:
                st.warning(t["forecast_missing"])
            elif (
                weather_source == t["forecast_source_open_meteo"]
                and (forecast_latitude is None or forecast_longitude is None)
            ):
                st.warning(t["forecast_api_missing_coords"])
            else:
                try:
                    external_weather_df = None
                    weather_source_key = (
                        "open_meteo"
                        if weather_source == t["forecast_source_open_meteo"]
                        else "pattern"
                    )
                    forecast_info = {
                        "weather_source": weather_source,
                        "weather_source_key": weather_source_key,
                        "api_rows": 0,
                        "api_days": api_forecast_days,
                    }
                    if weather_source == t["forecast_source_open_meteo"]:
                        external_weather_df = fetch_open_meteo_daily_forecast(
                            forecast_latitude,
                            forecast_longitude,
                            api_forecast_days,
                            forecast_timezone,
                        )
                        forecast_info.update(
                            {
                                "api_rows": int(external_weather_df.shape[0]),
                                "latitude": float(forecast_latitude),
                                "longitude": float(forecast_longitude),
                                "timezone": forecast_timezone,
                            }
                        )

                    _, forecast_df, run_forecast_info = create_forecast_simulation(
                        station=forecast_station,
                        station_row=forecast_station_row,
                        rain=rain,
                        evap=evap,
                        gw_df=gw_df,
                        forecast_years=forecast_years,
                        pattern_years=pattern_years,
                        rain_factor=rain_factor,
                        evap_factor=evap_factor,
                        rain_offset=rain_offset,
                        evap_offset=evap_offset,
                        clip_nonnegative=clip_nonnegative,
                        external_weather_df=external_weather_df,
                    )
                    if uncertainty_pct > 0:
                        band = float(uncertainty_pct) / 100.0
                        _, dry_df, _ = create_forecast_simulation(
                            station=forecast_station,
                            station_row=forecast_station_row,
                            rain=rain,
                            evap=evap,
                            gw_df=gw_df,
                            forecast_years=forecast_years,
                            pattern_years=pattern_years,
                            rain_factor=rain_factor * (1.0 - band),
                            evap_factor=evap_factor * (1.0 + band),
                            rain_offset=rain_offset,
                            evap_offset=evap_offset,
                            clip_nonnegative=clip_nonnegative,
                            external_weather_df=external_weather_df,
                        )
                        _, wet_df, _ = create_forecast_simulation(
                            station=forecast_station,
                            station_row=forecast_station_row,
                            rain=rain,
                            evap=evap,
                            gw_df=gw_df,
                            forecast_years=forecast_years,
                            pattern_years=pattern_years,
                            rain_factor=rain_factor * (1.0 + band),
                            evap_factor=evap_factor * (1.0 - band),
                            rain_offset=rain_offset,
                            evap_offset=evap_offset,
                            clip_nonnegative=clip_nonnegative,
                            external_weather_df=external_weather_df,
                        )
                        forecast_df = forecast_df.merge(
                            dry_df[["date", "simulated_head"]].rename(
                                columns={"simulated_head": "simulated_head_dry"}
                            ),
                            on="date",
                            how="left",
                        )
                        forecast_df = forecast_df.merge(
                            wet_df[["date", "simulated_head"]].rename(
                                columns={"simulated_head": "simulated_head_wet"}
                            ),
                            on="date",
                            how="left",
                        )
                    forecast_df["station"] = forecast_station
                    forecast_df["rain_factor"] = rain_factor
                    forecast_df["evap_factor"] = evap_factor
                    forecast_df["rain_offset"] = rain_offset
                    forecast_df["evap_offset"] = evap_offset
                    forecast_df["forecast_years"] = forecast_years
                    forecast_df["pattern_years"] = pattern_years
                    forecast_df["uncertainty_pct"] = uncertainty_pct
                    forecast_df["weather_source_mode"] = weather_source
                    forecast_info.update(run_forecast_info)
                    st.session_state.last_forecast_df = forecast_df
                    st.session_state.last_forecast_info = forecast_info
                except Exception as exc:
                    st.error(f"{t['job_failed']}: {exc}")

        if not st.session_state.last_forecast_df.empty:
            forecast_df = st.session_state.last_forecast_df.copy()
            forecast_info = st.session_state.last_forecast_info
            if forecast_info.get("weather_source_key") == "open_meteo":
                inserted = int(forecast_info.get("inserted_external_rows", 0))
                api_rows = int(forecast_info.get("api_rows", forecast_info.get("external_rows", 0)))
                st.caption(f"{t['forecast_api_inserted']}: {inserted}/{api_rows}")
                if api_rows and inserted == 0:
                    st.warning(t["forecast_api_outside_horizon"])

            figure, (axis, weather_axis) = plt.subplots(
                2,
                1,
                figsize=(12, 7),
                gridspec_kw={"height_ratios": [2, 1]},
            )
            hist_df = forecast_df[~forecast_df["is_forecast"]]
            fut_df = forecast_df[forecast_df["is_forecast"]]
            axis.plot(hist_df["date"], hist_df["simulated_head"], label=t["simulated"], linewidth=1.0)
            if {"simulated_head_dry", "simulated_head_wet"}.issubset(fut_df.columns):
                lower_band = fut_df[["simulated_head_dry", "simulated_head_wet"]].min(axis=1)
                upper_band = fut_df[["simulated_head_dry", "simulated_head_wet"]].max(axis=1)
                axis.fill_between(
                    fut_df["date"],
                    lower_band,
                    upper_band,
                    color="#d99a2b",
                    alpha=0.18,
                    label=t["forecast_band_label"],
                )
            axis.plot(fut_df["date"], fut_df["simulated_head"], label=t["tab_forecast"], linewidth=1.2, color="#b24c3d")
            if not fut_df.empty:
                axis.axvline(fut_df["date"].min(), color="black", linestyle="--", linewidth=1.0)
            axis.set_title(t["forecast_results"])
            axis.set_ylabel(t["forecast_axis"])
            axis.grid(alpha=0.3)
            axis.legend()

            if not fut_df.empty and {"rain", "evap"}.issubset(fut_df.columns):
                weather_axis.bar(
                    fut_df["date"],
                    fut_df["rain"],
                    label=t["forecast_rain_axis"],
                    color="#2f6db3",
                    alpha=0.35,
                    width=1.0,
                )
                evap_axis = weather_axis.twinx()
                evap_axis.plot(
                    fut_df["date"],
                    fut_df["evap"],
                    label=t["forecast_evap_axis"],
                    color="#4b8f29",
                    linewidth=0.9,
                )
                weather_axis.set_title(t["forecast_weather_forcing"])
                weather_axis.set_ylabel(t["forecast_rain_axis"])
                evap_axis.set_ylabel(t["forecast_evap_axis"])
                weather_axis.grid(axis="y", alpha=0.25)
                handles, labels = weather_axis.get_legend_handles_labels()
                evap_handles, evap_labels = evap_axis.get_legend_handles_labels()
                weather_axis.legend(
                    handles + evap_handles,
                    labels + evap_labels,
                    loc="upper right",
                )
            figure.tight_layout()
            st.pyplot(figure)
            plt.close(figure)
            st.markdown(f"### {t['forecast_weather_table']}")
            st.dataframe(
                forecast_df.tail(30),
                use_container_width=True,
                hide_index=True,
            )
            forecast_csv = sanitize_export_df(forecast_df).to_csv(index=False, sep=";").encode("utf-8")
            st.download_button(
                t["forecast_download"],
                data=forecast_csv,
                file_name="forecast_results.csv",
                mime="text/csv",
            )


with tab_save:
    st.markdown(f"### {t['history_heading']}")
    history_col1, history_col2 = st.columns(2)
    with history_col1:
        new_history_limit = st.number_input(
            t["history_limit"],
            min_value=1,
            max_value=200,
            value=int(st.session_state.history_run_limit),
            step=1,
            key="history_run_limit_control",
        )
        st.caption(t["history_limit_note"])
        if int(new_history_limit) != int(st.session_state.history_run_limit):
            st.session_state.history_run_limit = int(new_history_limit)
            st.session_state.history = limit_history_runs(
                st.session_state.history.copy(),
                st.session_state.history_run_limit,
            )
    with history_col2:
        if st.button(t["history_clear"], key="clear_history_button"):
            st.session_state.history = pd.DataFrame()
            st.session_state.history_notice = ("success", t["history_cleared"])
            st.rerun()

    if not st.session_state.history.empty:
        export_csv = sanitize_export_df(st.session_state.history).to_csv(index=False, sep=";").encode("utf-8")
        st.download_button(
            t["export_csv"],
            data=export_csv,
            file_name="gw_results.csv",
            mime="text/csv",
        )

        excel_bytes = create_excel_export(
            st.session_state.history.copy(),
            last_run_results.copy(),
        )
        st.download_button(
            t["export_excel"],
            data=excel_bytes,
            file_name="gw_results.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )
        st.caption(t["save_note"])

        report_bytes = create_html_report(
            st.session_state.history.copy(),
            last_run_results.copy(),
            data_status,
            t,
        )
        st.download_button(
            t["export_report"],
            data=report_bytes,
            file_name="gw_report.html",
            mime="text/html",
        )

    imported_file = st.file_uploader(t["import_csv"], type=["csv"], key="import_results")
    if imported_file is not None and st.button(t["import_button"], key="import_results_button"):
        imported_df = pd.read_csv(imported_file, sep=";")
        st.session_state.history = append_history_entries(
            st.session_state.history,
            imported_df,
            st.session_state.history_run_limit,
        )
        st.success(t["import_success"])
