import io
import gc
import html
import ipaddress
import json
import os
import socket
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
from datetime import datetime
from itertools import product
from pathlib import Path, PurePosixPath
from threading import Lock, Thread
from urllib.parse import urlparse
from uuid import uuid4

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pastas as ps
import pydeck as pdk
import requests
import streamlit as st
from pandas.api.types import is_bool_dtype, is_numeric_dtype

from ml_export import (
    create_ml_run_package,
    load_ml_artifacts_bytes,
    load_ml_run_package_bytes,
    ml_results_to_csv_bytes,
    save_ml_artifacts_bytes,
)
from ml_features import build_hybrid_feature_frame, build_hybrid_forecast_feature_frame, make_supervised_sequences
from ml_models import configure_torch_cpu_budget, torch_available
from ml_training import (
    build_artifacts,
    metric_summary,
    predict_hybrid_residuals,
    predict_impulse_response,
    train_evaluate_hybrid,
)

# Project concept and application lead: Robin Carow / RCnet
# Modeling framework credit: Pastas by Collenteur et al. (2019), https://doi.org/10.1111/gwat.12925


st.set_page_config(page_title="GW Analyzer", layout="wide")

APP_DIR = Path(__file__).resolve().parent
UPLOAD_CACHE_DIR = APP_DIR / ".upload_cache"
UPLOAD_CACHE_MANIFEST = UPLOAD_CACHE_DIR / "manifest.json"
MAX_REMOTE_DOWNLOAD_BYTES = 25 * 1024 * 1024
MAX_ZIP_ENTRY_BYTES = 50 * 1024 * 1024
MAX_PROJECT_BUNDLE_BYTES = 250 * 1024 * 1024
MAX_ZIP_ENTRIES = 250
MAX_PLOT_POINTS = 2500
MAX_PARALLEL_WORKERS = 4
ML_MODEL_TYPES = ["CNN", "TCN", "LSTM"]
ML_BATCH_CACHE_DIR = APP_DIR / ".ml_run_cache"
ML_WINDOW_OPTIONS = [365, 365 * 3, 365 * 5, 365 * 10, 365 * 15, 365 * 20, 365 * 30]
ML_HORIZON_OPTIONS = [1, 7, 30, 90, 180, 365, 365 * 2, 365 * 5, 365 * 10, 365 * 20]
ML_MAX_EPOCHS = 1000
PARAMETER_SEARCH_MAX_COMBINATIONS = 81
SPREADSHEET_FORMULA_PREFIXES = ("=", "+", "-", "@")
RESPONSE_MODEL_TYPES = ["Gamma", "Exponential", "Hantush"]
AUTO_FLEX_OPTIONS = [False, True]
AUTO_NOISE_OPTIONS = [False, True]
PROJECT_BUNDLE_VERSION = 1
SOURCE_KEYS = ["gw", "weather", "extra", "coords"]


class JobCancelled(Exception):
    pass


TEXT = {
    "Deutsch": {
        "title": "Grundwasser Analyse Tool",
        "subtitle": "PASTAS-Modellierung mit CNN/TCN/LSTM-Forecasts, Hybridvergleich, Projektpaketen und Szenarioanalyse.",
        "sidebar": "Einstellungen",
        "lang": "Sprache / Language",
        "accessibility": "Barrierefreiheit",
        "font_profile": "Schriftprofil",
        "font_standard": "Standard",
        "font_readable": "Lesefreundlich",
        "font_dyslexia": "Dyslexia/LRS-freundlich",
        "font_note": "Die Dyslexia-Option nutzt OpenDyslexic mit Atkinson-Fallback, falls verfügbar.",
        "text_scale": "Schriftgröße (%)",
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
        "cache_lang_saved": "Aktuelle Quellen wurden vor dem Sprachwechsel temporär gesichert.",
        "source_summary": "Aktive Quellen",
        "source_uploaded": "Upload",
        "source_url_label": "URL",
        "source_cache_label": "Temp-Cache",
        "source_gw": "Grundwasser",
        "source_weather": "Wetter",
        "source_extra": "Zusatzdaten",
        "source_coords": "Koordinaten",
        "remote_heading": "Remote-Datenabruf",
        "remote_help": "Direkte öffentliche URLs zu CSV/XLSX/ODS oder ZIP-Dateien. Geeignet als Grundlage für DWD- oder andere API/Open-Data-Quellen.",
        "remote_gw": "Grundwasser-URL",
        "remote_weather": "Wetter-URL",
        "remote_extra": "Zusatzdaten-URL",
        "remote_coords": "Koordinaten-URL",
        "reload_sources": "Datenquellen neu laden",
        "reload_sources_done": "Datenquellen wurden neu eingelesen.",
        "waiting": "Bitte zuerst die Dateien links in der Sidebar hochladen.",
        "load_error": "Daten konnten nicht geladen werden",
        "no_valid_stations": "Es wurden keine gültigen Messstellen mit mehr als 50 Beobachtungen gefunden.",
        "project_bundle": "Alles importieren/exportieren",
        "project_bundle_note": "Ein Projektpaket enthält Originaldateien, Pastas-Ergebnisse, ML-/Hybrid-Ergebnisse, Forecasts, Modellzustände und App-Einstellungen. ML-Laufpakete (.gwml) können hier ebenfalls importiert werden.",
        "project_bundle_export": "Projektpaket exportieren",
        "project_bundle_import": "Projekt- oder ML-Paket importieren",
        "project_bundle_import_button": "Paket laden",
        "project_bundle_success": "Paket wurde importiert.",
        "project_bundle_empty": "Es gibt noch keine Quellen, Ergebnisse oder Einstellungen für ein Projektpaket.",
        "project_bundle_error": "Projektpaket konnte nicht importiert werden",
        "config_heading": "Konfiguration",
        "mode": "Auswertungsmodus",
        "mode_single": "Einzelne Messstelle",
        "mode_multi": "Mehrere Messstellen",
        "mode_all": "Alle Messstellen",
        "station": "Messstelle",
        "stations": "Messstellen",
        "available_stations": "Messstellen im Datensatz",
        "selection_count": "Ausgewählte Messstellen",
        "run_strategy": "Berechnungsstrategie",
        "strategy_manual": "Ausgewählte Konfiguration berechnen",
        "strategy_auto": "Bestes Modell automatisch suchen",
        "strategy_auto_note": "Die automatische Suche testet Gamma/Exponential/Hantush, Flex/ohne Flex und Noise/ohne Noise für jede Messstelle.",
        "auto_combo_count": "Modellkombinationen pro Messstelle",
        "auto_exclude_configs": "Modellkombinationen ausschließen",
        "auto_exclude_help": "Ausgeschlossene Kombinationen werden bei der Bestmodell-Suche übersprungen.",
        "auto_exclude_all_warning": "Bitte mindestens eine Modellkombination für die Suche aktiv lassen.",
        "performance_box": "Performance-Optionen",
        "parallel_workers": "Parallele Auto-Suche (Worker)",
        "parallel_note": "Parallelisierung läuft stationsweise und ist für größere Auto-Läufe gedacht.",
        "early_stop": "Frühstopp bei Ziel-R2",
        "early_stop_r2": "Ziel-R2 für Frühstopp",
        "model": "Reaktionsmodell",
        "use_flex": "FlexModel verwenden",
        "use_noise": "NoiseModel verwenden",
        "parameter_box": "Modellparameter anpassen",
        "parameter_help_manual": "Hier kannst du Startwerte und Optimierung einzelner Parameter für die ausgewählte Konfiguration steuern.",
        "parameter_help_auto": "Die automatische Modellsuche nutzt für jede Modellstruktur passende Standard-Startwerte.",
        "parameter_search": "Beste Parameter für Station suchen",
        "parameter_search_note": "Prüft mehrere Startwert-Kombinationen für die gewählte Modellstruktur und markiert je Station den besten Lauf.",
        "parameter_search_params": "Parameter für Startwert-Suche",
        "parameter_search_multipliers": "Startwert-Faktoren",
        "parameter_search_invalid_multipliers": "Bitte Startwert-Faktoren als Zahlenliste angeben, z. B. 0.5, 1, 2.",
        "parameter_search_combo_count": "Parameter-Kombinationen",
        "parameter_search_too_many": "Die Parameter-Suche ist auf 81 Kombinationen begrenzt. Reduziere Parameter oder Faktoren.",
        "vary": "optimieren",
        "response_cutoff": "Response cutoff",
        "noise_norm": "Noise normalisieren",
        "run": "Batch starten",
        "cancel_run": "Lauf abbrechen",
        "choose_station": "Bitte mindestens eine Messstelle auswählen.",
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
        "failed_runs": "Fehler oder Übersprungen",
        "best_station": "Beste Messstelle (R2)",
        "mean_r2": "Mittleres R2",
        "extra_match": "Zusatzdaten gematcht",
        "extra_missing": "Messstellen ohne Zusatzdaten",
        "extra_unused": "Zusatzdaten ohne passenden Lauf",
        "coords_match": "Koordinaten gematcht",
        "coords_missing": "Messstellen ohne Koordinaten",
        "coords_unused": "Koordinaten ohne passenden Lauf",
        "data_status": "Datenstand und Update-Assistent",
        "data_status_note": "Prüft Datenlücken, fehlende Werte und ob Forecasts mit altem Wetterantrieb laufen.",
        "data_gw_until": "Grundwasser bis",
        "data_weather_until": "Wetter bis",
        "data_gap_days": "Lücke bis heute",
        "data_station_count": "Messstellen",
        "data_quality_table": "Datenqualität je Messstelle",
        "data_update_hint": "Die lokalen Daten sind nicht aktuell. Für belastbare Forecasts sollten historische Wetter-/Grundwasserlücken vor dem Szenario geschlossen werden; Open-Meteo deckt hier nur die nächsten Forecast-Tage ab.",
        "data_fresh": "Datenstand wirkt aktuell genug für einen Kurzforecast.",
        "tab_diagnostics": "Diagnose",
        "tab_plot": "Visualisierung",
        "tab_compare": "Vergleich",
        "tab_map": "Karte",
        "tab_forecast": "Forecast",
        "tab_ml_forecast": "ML Forecast",
        "tab_save": "Speichern und Laden",
        "main_nav": "Ansicht",
        "no_run": "Führe zuerst eine Analyse aus, um Ergebnisse anzuzeigen.",
        "plot_station": "Messstelle für die Visualisierung",
        "plot_type": "Visualisierungsmethode",
        "plot_overview": "Pastas Übersicht",
        "plot_obs_sim": "Beobachtet vs. simuliert",
        "plot_residuals": "Residuale über Zeit",
        "plot_res_hist": "Residual-Histogramm",
        "plot_step": "Schrittantwort",
        "plot_pulse": "Impulse Response Function",
        "plot_downsampled": "Diagramm für schnelle Darstellung ausgedünnt",
        "observed": "Beobachtet",
        "simulated": "Simuliert",
        "residuals": "Residuale",
        "head_axis": "Grundwasserstand",
        "count_axis": "Anzahl",
        "days_axis": "Tage",
        "response_axis": "Antwort",
        "download_plot": "Diagramm herunterladen",
        "simulated_ts_heading": "Simulierte Zeitreihe",
        "simulated_ts_note": "Zeitreihe aus Beobachtung, Simulation und Residuum für das ausgewählte Modell.",
        "show_simulated_series": "Simulierte Zeitreihe anzeigen",
        "simulated_ts_download": "Simulierte Zeitreihe als CSV herunterladen",
        "simulated_all_heading": "Simulierte Zeitreihen",
        "simulated_all_download": "Alle simulierten Zeitreihen als CSV exportieren",
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
        "box_cols": "Spalten für den Boxplot",
        "corr_cols": "Spalten für die Korrelationsmatrix",
        "no_numeric": "Es sind nicht genug numerische Spalten für diese Visualisierung vorhanden.",
        "error_table": "Fehlermeldungen",
        "run_compare": "Läufe nebeneinander vergleichen",
        "run_compare_note": "Vergleich mehrerer Läufe für eine Kennzahl in einer Pivot-Tabelle.",
        "run_compare_runs": "Läufe für den Vergleich",
        "run_compare_metric": "Kennzahl für den Laufvergleich",
        "run_compare_summary": "Zusammenfassung pro Lauf",
        "run_compare_station": "Vergleich pro Messstelle",
        "map_scope": "Datenbasis für die Karte",
        "map_metric": "Kennzahl für die Kartentabelle",
        "map_no_coords": "Keine Koordinaten vorhanden. Lade eine optionale Stationsdatei mit Messstelle/Site und Lat/Lon hoch.",
        "map_note": "Die Kartenansicht ist vorbereitet. Sobald Koordinaten vorliegen, werden die Messstellen direkt angezeigt.",
        "map_table": "Messstellen mit Koordinaten",
        "map_size_metric": "Markergröße",
        "map_details": "Kartendetails",
        "map_color_note": "Markerfarbe folgt der ausgewählten Kennzahl, Markergröße folgt der Datenlänge oder einer zweiten Kennzahl.",
        "diag_station": "Messstelle für Diagnose",
        "diag_heading": "Modelldiagnose",
        "diag_bias": "Bias",
        "diag_resid_std": "Residuen-Std.",
        "diag_resid_acf": "Residuen-ACF Lag 1",
        "diag_missing_pct": "Fehlende Werte",
        "diag_quality": "Qualität",
        "diag_green": "Grün",
        "diag_yellow": "Gelb",
        "diag_red": "Rot",
        "diag_seasonal_error": "Saisonaler Fehler",
        "diag_cluster": "Response-Cluster",
        "diag_cluster_note": "Gruppiert Messstellen grob nach Peak-Zeit der Pulse Response.",
        "forecast_station": "Messstelle für Forecast",
        "forecast_preset": "Szenario-Preset",
        "forecast_apply_preset": "Preset anwenden",
        "preset_custom": "Benutzerdefiniert",
        "preset_normal": "Normaljahr",
        "preset_dry": "Trocken",
        "preset_wet": "Nass",
        "preset_hot": "Heißer Sommer",
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
        "forecast_note": "Das Szenario nutzt historische Wetterjahre als Muster und kann aktuelle Open-Meteo-Tageswerte in den Zukunftshorizont einfügen.",
        "forecast_download": "Forecast als CSV herunterladen",
        "forecast_missing": "Forecast braucht ein erfolgreiches Modell aus dem letzten Lauf.",
        "forecast_results": "Forecast-Ergebnisse",
        "forecast_axis": "Simulierter Grundwasserstand",
        "forecast_weather_forcing": "Wetterantrieb",
        "forecast_weather_table": "Forecast-Wetterwerte",
        "forecast_rain_axis": "Niederschlag (mm/Tag)",
        "forecast_evap_axis": "Verdunstung (mm/Tag)",
        "forecast_api_missing_coords": "Open-Meteo braucht Stationskoordinaten oder manuelle Koordinaten.",
        "forecast_api_inserted": "Open-Meteo-Werte eingefügt",
        "forecast_api_outside_horizon": "Die Open-Meteo-Tage liegen außerhalb des gewählten Modellhorizonts. Erhöhe den Forecast-Horizont oder aktualisiere die Eingangsdaten.",
        "forecast_stale_weather": "Der Wetterantrieb ist alt. Nutze Open-Meteo für die nächsten Tage oder aktualisiere die Wetterdatei.",
        "forecast_band_label": "Unsicherheitsband",
        "export_csv": "Ergebnisse als CSV exportieren",
        "export_excel": "Ergebnisse als Excel exportieren",
        "export_report": "Kurzreport als HTML exportieren",
        "import_csv": "Ergebnisse importieren",
        "import_button": "Import starten",
        "import_success": "Ergebnisse wurden importiert.",
        "save_note": "Excel-Datei enthält mehrere Sheets für Historie, letzten Lauf, Bestmodelle und Fehler.",
        "history_heading": "Historie",
        "history_limit": "Maximal gespeicherte Läufe",
        "history_limit_note": "Begrenzt die Session-Historie auf die zuletzt gespeicherten Läufe.",
        "history_clear": "Historie leeren",
        "history_cleared": "Historie wurde geleert.",
        "config_column": "Konfiguration",
        "search_mode": "Suchmodus",
        "best_model_flag": "Bestes Modell",
        "status": "Status",
        "running_info": "Ein Batch-Lauf ist aktiv. Die Ansicht aktualisiert sich automatisch.",
        "ml_requires_pastas": "ML-Hybrid braucht ein erfolgreiches Pastas-Modell aus dem letzten Lauf oder Import.",
        "ml_requires_no_flex": "Für den Vergleich wird ein erfolgreiches PASTAS-Modell ohne FlexModel benötigt.",
        "ml_torch_missing": "PyTorch ist nicht installiert. Bitte `pip install -r requirements.txt` im Projektordner ausführen.",
        "ml_model_type": "Neural-Modell",
        "ml_model_parallel": "CNN, TCN und LSTM werden ressourcenschonend trainiert, jeweils als Pastas+ML und als Nur-ML.",
        "ml_model_detail": "Detailmodell",
        "ml_window": "Trainingsfenster",
        "ml_window_help": "Länge der täglichen Sequenz, die das neuronale Modell als Wetterhistorie sieht. Lange Fenster wie 10, 15, 20 oder 30 Jahre brauchen entsprechend lange Zeitreihen.",
        "ml_horizon": "Vorhersagehorizont",
        "ml_horizon_help": "Abstand zwischen Ende des Eingabefensters und Zielwert. Längere Horizonte sind schwerer und reduzieren die Zahl nutzbarer Trainingssequenzen.",
        "ml_epochs": "Trainings-Epochen",
        "ml_epochs_help": f"Wie oft das Modell den Trainingsblock durchläuft. Mehr Epochen können helfen, erhöhen aber Rechenzeit und Overfitting-Risiko. Maximum: {ML_MAX_EPOCHS}.",
        "ml_learning_rate": "Lernrate",
        "ml_learning_rate_help": "Schrittweite der Optimierung. Kleinere Werte trainieren ruhiger, größere Werte schneller, aber instabiler.",
        "ml_hidden_size": "Hidden Size / Filter",
        "ml_hidden_size_help": "Größe des LSTM-Speichers bzw. Anzahl der CNN-Filter. Größer ist flexibler, braucht aber mehr Daten.",
        "ml_data_epoch_limit": "Datenbasiertes Epochen-Limit: {limit} (App-Maximum: {max_epochs}). Grundlage: {sequences} nutzbare Sequenzen, davon {train} Training.",
        "ml_data_epoch_limit_empty": "Für diese Station/Fenster/Horizont-Kombination gibt es aktuell keine nutzbaren Trainingssequenzen.",
        "ml_recommendation": "Empfehlung: {epochs} Epochen, Lernrate {learning_rate}, Hidden Size/Filter {hidden_size}. Grund: {reason}.",
        "ml_recommendation_note": "Das ist eine datenbasierte Startempfehlung; die genaueste Variante ist danach die mit dem besten Validierungs-R² in der Tabelle.",
        "ml_target_hybrid": "Pastas + ML",
        "ml_target_direct": "Nur ML",
        "ml_split_note": "Fester zeitlicher Split: erste 60 % Training, nächste 20 % Test, letzte 20 % Validierung.",
        "ml_feature_restriction": "Vergleichssetup: keine vorherigen Grundwasserstände als Features und keine Jahreszeit-Sin/Cos-Features.",
        "ml_features": "Features",
        "ml_feature_weather": "Niederschlag und Verdunstung",
        "ml_feature_weather_help": "Tägliche Niederschlags- und Verdunstungswerte. Das ist der direkteste Vergleich zu den PASTAS-Wetterinputs.",
        "ml_feature_rollings": "rollierende Wetterfenster",
        "ml_feature_rollings_help": "Summen/Mittelwerte über 7, 30 und 90 Tage. Das gibt CNN/TCN/LSTM gröbere Feuchte- und Trockenheitsinformationen, ohne Grundwasserstände zu verwenden.",
        "ml_train": "CNN/TCN/LSTM trainieren (Hybrid und Nur-ML)",
        "ml_test": "Test",
        "ml_validation": "Validierung",
        "ml_results_table": "CNN/TCN/LSTM- und Zielmodus-Vergleich",
        "ml_add_history": "ML-Ergebnis in Historie übernehmen",
        "ml_added_history": "ML-Ergebnis wurde in die Historie übernommen.",
        "ml_download_validation": "Validierungsdaten exportieren",
        "ml_download_package": "ML-Laufpaket exportieren",
        "ml_import_package": "ML-Laufpaket importieren",
        "ml_import_package_button": "ML-Paket laden",
        "ml_import_package_success": "ML-Laufpaket wurde importiert.",
        "ml_scope": "ML-Laufumfang",
        "ml_scope_single": "Ausgewählte Station",
        "ml_scope_all": "Alle verfügbaren Stationen",
        "ml_batch_note": "All-Station läuft stationsweise und ressourcenschonend. Nach jeder erfolgreichen Station wird ein Paket in den lokalen ML-Cache geschrieben.",
        "ml_batch_cache": "ML-Zwischenspeicher",
        "ml_batch_cache_saved": "Zwischengespeichert",
        "ml_batch_cache_empty": "Noch keine ML-Zwischenspeicher-Pakete vorhanden.",
        "ml_resource_note": "Ressourcenschutz: Für All-Station wird sequenziell trainiert; PyTorch-Threads werden reduziert, damit CPU/RAM nicht voll belegt werden.",
        "ml_summary": "ML-Zusammenfassung",
        "ml_impulse_heading": "ML-Impulsantwort",
        "ml_impulse_amount": "Einmaliger Niederschlagsimpuls (mm)",
        "ml_impulse_amount_help": "Ein künstliches Einzelereignis. Die Verdunstung wird für den gesamten Antwortzeitraum auf 0 gesetzt.",
        "ml_impulse_days": "Antwortlänge (Tage)",
        "ml_impulse_days_help": "Zeitraum, über den das Abklingen der Reaktion nach dem Impuls beobachtet wird.",
        "ml_impulse_run": "Impulsantwort berechnen",
        "ml_download_impulse": "Impulsantwort als CSV exportieren",
        "ml_impulse_no_rows": "Für diese Fenster-/Horizont-Kombination konnte keine Impulsantwort berechnet werden.",
        "ml_future_heading": "ML-Forecast",
        "ml_future_needs_forecast": "Berechne im Forecast-Tab zuerst ein Pastas-Szenario für dieselbe Station. Danach kann der ML-Tab entweder das gelernte Residuum ergänzen oder eine Nur-ML-Vorhersage anzeigen.",
        "ml_future_no_rows": "Für das gewählte Trainingsfenster und den Horizont gibt es noch keine vorhersagbaren Zukunftszeilen.",
        "ml_download_future": "ML-Forecast als CSV exportieren",
    },
    "English": {
        "title": "Groundwater Analysis Tool",
        "subtitle": "PASTAS modelling with CNN/TCN/LSTM forecasts, hybrid comparison, project bundles and scenario analysis.",
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
        "remote_heading": "Remote data fetch",
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
        "project_bundle": "Import/export everything",
        "project_bundle_note": "A project bundle contains original files, Pastas results, ML/hybrid results, forecasts, model states and app settings. ML run packages (.gwml) can be imported here as well.",
        "project_bundle_export": "Export project bundle",
        "project_bundle_import": "Import project or ML package",
        "project_bundle_import_button": "Load package",
        "project_bundle_success": "Package was imported.",
        "project_bundle_empty": "There are no sources, results or settings for a project bundle yet.",
        "project_bundle_error": "Project bundle could not be imported",
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
        "strategy_auto_note": "Automatic search tests Gamma/Exponential/Hantush, Flex/no Flex and Noise/no Noise for each station.",
        "auto_combo_count": "Model combinations per station",
        "auto_exclude_configs": "Exclude model combinations",
        "auto_exclude_help": "Excluded combinations are skipped during the best-model search.",
        "auto_exclude_all_warning": "Please keep at least one model combination active for the search.",
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
        "parameter_search": "Find best parameters for station",
        "parameter_search_note": "Checks multiple initial-value combinations for the selected model structure and marks the best run per station.",
        "parameter_search_params": "Parameters for initial-value search",
        "parameter_search_multipliers": "Initial-value factors",
        "parameter_search_invalid_multipliers": "Please enter initial-value factors as a numeric list, e.g. 0.5, 1, 2.",
        "parameter_search_combo_count": "Parameter combinations",
        "parameter_search_too_many": "Parameter search is limited to 81 combinations. Reduce parameters or factors.",
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
        "tab_ml_forecast": "ML Forecast",
        "tab_save": "Save and Load",
        "main_nav": "View",
        "no_run": "Run an analysis first to show results.",
        "plot_station": "Station for visualization",
        "plot_type": "Visualization method",
        "plot_overview": "Pastas overview",
        "plot_obs_sim": "Observed vs simulated",
        "plot_residuals": "Residuals over time",
        "plot_res_hist": "Residual histogram",
        "plot_step": "Step response",
        "plot_pulse": "Impulse response function",
        "plot_downsampled": "Plot downsampled for faster rendering",
        "observed": "Observed",
        "simulated": "Simulated",
        "residuals": "Residuals",
        "head_axis": "Groundwater head",
        "count_axis": "Count",
        "days_axis": "Days",
        "response_axis": "Response",
        "download_plot": "Download plot",
        "simulated_ts_heading": "Simulated time series",
        "simulated_ts_note": "Time series with observation, simulation and residual for the selected model.",
        "show_simulated_series": "Show simulated time series",
        "simulated_ts_download": "Download simulated time series as CSV",
        "simulated_all_heading": "Simulated time series",
        "simulated_all_download": "Export all simulated time series as CSV",
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
        "ml_requires_pastas": "ML hybrid needs a successful Pastas model from the last run or import.",
        "ml_requires_no_flex": "The comparison requires a successful PASTAS model without FlexModel.",
        "ml_torch_missing": "PyTorch is not installed. Please run `pip install -r requirements.txt` in the project folder.",
        "ml_model_type": "Neural model",
        "ml_model_parallel": "CNN, TCN and LSTM are trained with resource protection, each as Pastas+ML and only-ML.",
        "ml_model_detail": "Detail model",
        "ml_window": "Training window",
        "ml_window_help": "Length of the daily input sequence seen by the neural model. Long windows such as 10, 15, 20 or 30 years require sufficiently long time series.",
        "ml_horizon": "Forecast horizon",
        "ml_horizon_help": "Lead time between the end of the input window and the target value. Longer horizons are harder and reduce the number of usable training sequences.",
        "ml_epochs": "Training epochs",
        "ml_epochs_help": f"Number of passes over the training block. More epochs may help but increase runtime and overfitting risk. Maximum: {ML_MAX_EPOCHS}.",
        "ml_learning_rate": "Learning rate",
        "ml_learning_rate_help": "Optimizer step size. Smaller values train more calmly, larger values faster but less stably.",
        "ml_hidden_size": "Hidden size / filters",
        "ml_hidden_size_help": "LSTM memory size or CNN filter count. Larger is more flexible but needs more data.",
        "ml_data_epoch_limit": "Data-based epoch limit: {limit} (app maximum: {max_epochs}). Based on {sequences} usable sequences, {train} for training.",
        "ml_data_epoch_limit_empty": "The current station/window/horizon combination does not yield usable training sequences.",
        "ml_recommendation": "Recommendation: {epochs} epochs, learning rate {learning_rate}, hidden size/filters {hidden_size}. Reason: {reason}.",
        "ml_recommendation_note": "This is a data-based starting point; the most accurate variant is the one with the best validation R² in the table afterwards.",
        "ml_target_hybrid": "Pastas + ML",
        "ml_target_direct": "Only ML",
        "ml_split_note": "Fixed temporal split: first 60% training, next 20% test, final 20% validation.",
        "ml_feature_restriction": "Comparison setup: no previous groundwater heads as features and no seasonal sin/cos features.",
        "ml_features": "Features",
        "ml_feature_weather": "Rainfall and evaporation",
        "ml_feature_weather_help": "Daily rainfall and evaporation values. This is the most direct comparison to the PASTAS weather inputs.",
        "ml_feature_rollings": "Rolling weather windows",
        "ml_feature_rollings_help": "7, 30 and 90 day sums/means. This gives CNN/TCN/LSTM coarse wetness and dryness information without groundwater heads.",
        "ml_train": "Train CNN/TCN/LSTM (hybrid and only ML)",
        "ml_test": "Test",
        "ml_validation": "Validation",
        "ml_results_table": "CNN/TCN/LSTM and target-mode comparison",
        "ml_add_history": "Add ML result to history",
        "ml_added_history": "ML result was added to history.",
        "ml_download_validation": "Export validation data",
        "ml_download_package": "Export ML run package",
        "ml_import_package": "Import ML run package",
        "ml_import_package_button": "Load ML package",
        "ml_import_package_success": "ML run package was imported.",
        "ml_scope": "ML run scope",
        "ml_scope_single": "Selected station",
        "ml_scope_all": "All available stations",
        "ml_batch_note": "All-station runs are processed station by station. After each successful station, a package is written to the local ML cache.",
        "ml_batch_cache": "ML checkpoint cache",
        "ml_batch_cache_saved": "Checkpoint saved",
        "ml_batch_cache_empty": "No ML checkpoint packages yet.",
        "ml_resource_note": "Resource protection: all-station runs train sequentially; PyTorch threads are reduced so CPU/RAM are not fully occupied.",
        "ml_summary": "ML summary",
        "ml_impulse_heading": "ML impulse response",
        "ml_impulse_amount": "One-time rainfall impulse (mm)",
        "ml_impulse_amount_help": "A synthetic single event. Evaporation is set to 0 for the full response period.",
        "ml_impulse_days": "Response length (days)",
        "ml_impulse_days_help": "Period over which the decay after the impulse is observed.",
        "ml_impulse_run": "Compute impulse response",
        "ml_download_impulse": "Export impulse response as CSV",
        "ml_impulse_no_rows": "No impulse response could be computed for this window/horizon setup.",
        "ml_future_heading": "ML forecast",
        "ml_future_needs_forecast": "Compute a Pastas scenario for the same station in the Forecast tab first. The ML tab can then add the learned residual or show a pure ML forecast.",
        "ml_future_no_rows": "The selected training window and horizon do not yield predictable future rows yet.",
        "ml_download_future": "Export ML forecast as CSV",
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
    ("Hantush", False): [
        {"name": "recharge_A", "label": "Recharge A", "initial": 0.220341, "vary": True},
        {"name": "recharge_a", "label": "Hantush a", "initial": 100.0, "vary": True},
        {"name": "recharge_b", "label": "Hantush b", "initial": 1.0, "vary": True},
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
    ("Hantush", True): [
        {"name": "recharge_A", "label": "Recharge A", "initial": 0.906149, "vary": True},
        {"name": "recharge_a", "label": "Hantush a", "initial": 100.0, "vary": True},
        {"name": "recharge_b", "label": "Hantush b", "initial": 1.0, "vary": True},
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
if "pending_project_model_rebuild" not in st.session_state:
    st.session_state.pending_project_model_rebuild = False
if "last_ml_validation_df" not in st.session_state:
    st.session_state.last_ml_validation_df = pd.DataFrame()
if "last_ml_forecast_df" not in st.session_state:
    st.session_state.last_ml_forecast_df = pd.DataFrame()
if "last_ml_impulse_df" not in st.session_state:
    st.session_state.last_ml_impulse_df = pd.DataFrame()
if "last_ml_summary_table" not in st.session_state:
    st.session_state.last_ml_summary_table = pd.DataFrame()
if "last_ml_summary" not in st.session_state:
    st.session_state.last_ml_summary = {}
if "last_ml_artifacts" not in st.session_state:
    st.session_state.last_ml_artifacts = {}


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
        elif isinstance(source_value, (str, Path)):
            source_path = Path(source_value)
            if source_path.exists() and source_path.is_file():
                suffix = source_path.suffix or ".bin"
                cache_filename = f"{source_key}{suffix}"
                cache_path = UPLOAD_CACHE_DIR / cache_filename
                cache_path.write_bytes(source_path.read_bytes())
                manifest[source_key] = {
                    "type": "file",
                    "path": str(cache_path),
                    "name": source_path.name,
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


def collect_project_source_entries():
    source_entries = collect_current_source_entries()
    cached_sources, _ = resolve_cached_sources()
    for source_key in SOURCE_KEYS:
        if not source_entries.get(source_key) and cached_sources.get(source_key) is not None:
            source_entries[source_key] = cached_sources[source_key]
    return source_entries


def project_setting_keys():
    exact_keys = {
        "lang",
        "font_profile",
        "text_scale",
        "high_contrast",
        "strong_focus",
        "history_run_limit",
        "active_main_view",
        "analysis_mode",
        "selected_station",
        "selected_stations",
        "run_strategy",
        "manual_response_model",
        "manual_use_flex",
        "manual_use_noise",
        "response_cutoff_control",
        "noise_norm_control",
        "parallel_workers_control",
        "early_stop_enabled",
        "early_stop_r2_control",
        "auto_excluded_configs",
        "parameter_search_enabled",
        "parameter_search_multipliers",
        "remote_gw_url",
        "remote_weather_url",
        "remote_extra_url",
        "remote_coords_url",
    }
    prefixes = (
        "forecast_",
        "ml_",
        "parameter_search_params_",
        "value_",
        "vary_",
    )
    keys = []
    for key in st.session_state.keys():
        if key in exact_keys or any(str(key).startswith(prefix) for prefix in prefixes):
            keys.append(key)
    return keys


def collect_project_settings():
    settings = {}
    for key in project_setting_keys():
        if key.endswith("_button") or key.endswith("_upload_file"):
            continue
        if key in {
            "project_bundle_import",
            "import_results",
            "ml_package_import",
            "active_job_id",
            "session_id",
        }:
            continue
        value = st.session_state.get(key)
        try:
            json.dumps(value)
        except TypeError:
            continue
        settings[key] = value
    return settings


def apply_project_settings(settings):
    for key, value in (settings or {}).items():
        if key.endswith("_upload_file") or key.endswith("_button") or key in {
            "project_bundle_import",
            "import_results",
            "ml_package_import",
            "active_job_id",
            "session_id",
        }:
            continue
        st.session_state[key] = value


def dataframe_to_project_csv(df):
    if df is None or df.empty:
        return None
    return sanitize_export_df(df).to_csv(index=False, sep=";").encode("utf-8")


def safe_bundle_name(name, fallback):
    cleaned = Path(str(name or "")).name.strip()
    if cleaned in {"", ".", ".."}:
        return fallback
    return cleaned or fallback


def validate_zip_archive(
    archive,
    max_total_bytes=MAX_PROJECT_BUNDLE_BYTES,
    max_entry_bytes=MAX_ZIP_ENTRY_BYTES,
    max_entries=MAX_ZIP_ENTRIES,
):
    infos = archive.infolist()
    if len(infos) > max_entries:
        raise ValueError("ZIP archive contains too many files.")

    total_size = 0
    for info in infos:
        member_path = PurePosixPath(info.filename)
        if member_path.is_absolute() or ".." in member_path.parts:
            raise ValueError("ZIP archive contains an unsafe path.")
        if info.file_size > max_entry_bytes:
            raise ValueError("ZIP archive contains a file that is too large.")
        total_size += int(info.file_size)
        if total_size > max_total_bytes:
            raise ValueError("ZIP archive is too large.")


def get_source_bundle_payload(source_key, source_value):
    if source_value is None:
        return None

    if is_remote_source(source_value):
        raw_bytes, extension_hint = download_remote_bytes(source_value)
        filename = safe_bundle_name(
            Path(urlparse(source_value).path).name,
            f"{source_key}{extension_hint or '.bin'}",
        )
        return {
            "raw_bytes": raw_bytes,
            "name": filename,
            "type": "file",
            "origin": "url",
            "url": source_value,
        }

    if hasattr(source_value, "getvalue") and hasattr(source_value, "name"):
        return {
            "raw_bytes": source_value.getvalue(),
            "name": safe_bundle_name(source_value.name, f"{source_key}.bin"),
            "type": "file",
            "origin": "upload",
        }

    if isinstance(source_value, (str, Path)):
        source_path = Path(source_value)
        if source_path.exists() and source_path.is_file():
            payload = {
                "raw_bytes": source_path.read_bytes(),
                "name": safe_bundle_name(source_path.name, f"{source_key}.bin"),
                "type": "file",
                "origin": "cache",
            }
            cache_entry = load_upload_cache_manifest().get(source_key, {})
            if cache_entry.get("url"):
                payload["url"] = cache_entry["url"]
            return payload

    return None


def create_project_bundle():
    buffer = io.BytesIO()
    manifest = {
        "version": PROJECT_BUNDLE_VERSION,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "settings": collect_project_settings(),
        "last_run_label": st.session_state.get("last_run_label"),
        "last_extra_match": st.session_state.get("last_extra_match", {}),
        "last_coord_match": st.session_state.get("last_coord_match", {}),
        "last_forecast_info": st.session_state.get("last_forecast_info", {}),
        "last_ml_summary": st.session_state.get("last_ml_summary", {}),
        "sources": {},
        "tables": {},
        "ml_checkpoints": [],
    }

    with zipfile.ZipFile(buffer, mode="w", compression=zipfile.ZIP_DEFLATED) as archive:
        table_sources = {
            "history": st.session_state.get("history", pd.DataFrame()),
            "last_run_results": st.session_state.get("last_run_results", pd.DataFrame()),
            "last_forecast_df": st.session_state.get("last_forecast_df", pd.DataFrame()),
            "last_ml_validation_df": st.session_state.get("last_ml_validation_df", pd.DataFrame()),
            "last_ml_forecast_df": st.session_state.get("last_ml_forecast_df", pd.DataFrame()),
            "last_ml_impulse_df": st.session_state.get("last_ml_impulse_df", pd.DataFrame()),
            "last_ml_summary_table": st.session_state.get("last_ml_summary_table", pd.DataFrame()),
        }
        for table_name, df in table_sources.items():
            csv_bytes = dataframe_to_project_csv(df)
            if csv_bytes is None:
                continue
            archive_path = f"tables/{table_name}.csv"
            archive.writestr(archive_path, csv_bytes)
            manifest["tables"][table_name] = archive_path

        for source_key, source_value in collect_project_source_entries().items():
            if source_value is None:
                continue
            try:
                payload = get_source_bundle_payload(source_key, source_value)
            except Exception as exc:
                if is_remote_source(source_value):
                    manifest["sources"][source_key] = {
                        "type": "url",
                        "value": source_value,
                        "download_error": str(exc),
                    }
                continue
            if payload is None:
                continue
            archive_path = f"sources/{source_key}_{safe_bundle_name(payload['name'], source_key + '.bin')}"
            archive.writestr(archive_path, payload["raw_bytes"])
            manifest["sources"][source_key] = {
                "type": "file",
                "path": archive_path,
                "name": payload["name"],
                "origin": payload.get("origin"),
            }
            if payload.get("url"):
                manifest["sources"][source_key]["url"] = payload["url"]

        ml_artifact_bytes = save_ml_artifacts_bytes(st.session_state.get("last_ml_artifacts", {}))
        if ml_artifact_bytes:
            archive_path = "ml/model_state.pt"
            archive.writestr(archive_path, ml_artifact_bytes)
            manifest["ml_artifacts"] = archive_path

        for checkpoint_path in list_ml_checkpoint_packages(limit=100):
            try:
                if checkpoint_path.stat().st_size > MAX_ZIP_ENTRY_BYTES:
                    continue
                checkpoint_bytes = checkpoint_path.read_bytes()
            except OSError:
                continue
            archive_path = f"ml_checkpoints/{safe_bundle_name(checkpoint_path.name, 'checkpoint.gwml')}"
            archive.writestr(archive_path, checkpoint_bytes)
            manifest["ml_checkpoints"].append(
                {
                    "path": archive_path,
                    "name": checkpoint_path.name,
                }
            )

        archive.writestr(
            "manifest.json",
            json.dumps(manifest, ensure_ascii=False, indent=2),
        )

    buffer.seek(0)
    return buffer.getvalue()


def read_bundle_csv(archive, manifest, table_name, normalize_results=False):
    archive_path = manifest.get("tables", {}).get(table_name)
    if not archive_path:
        return pd.DataFrame()
    with archive.open(archive_path) as handle:
        df = pd.read_csv(handle, sep=";")
    if table_name in {
        "last_forecast_df",
        "last_ml_validation_df",
        "last_ml_forecast_df",
        "last_ml_impulse_df",
    } and "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    return normalize_imported_results(df) if normalize_results else df


def install_project_sources(archive, manifest):
    ensure_upload_cache_dir()
    clear_upload_cache()
    cache_manifest = {}

    for source_key in SOURCE_KEYS:
        entry = manifest.get("sources", {}).get(source_key)
        if not entry:
            continue
        if entry.get("type") == "file" and entry.get("path"):
            raw_bytes = archive.read(entry["path"])
            suffix = Path(entry.get("name", "")).suffix or Path(entry["path"]).suffix or ".bin"
            cache_filename = f"{source_key}{suffix}"
            cache_path = UPLOAD_CACHE_DIR / cache_filename
            cache_path.write_bytes(raw_bytes)
            cache_entry = {
                "type": "file",
                "path": str(cache_path),
                "name": entry.get("name", cache_filename),
            }
            if entry.get("origin"):
                cache_entry["origin"] = entry["origin"]
            if entry.get("url"):
                cache_entry["url"] = entry["url"]
            cache_manifest[source_key] = cache_entry
        elif entry.get("type") == "url" and entry.get("value"):
            cache_manifest[source_key] = {"type": "url", "value": entry["value"]}

    UPLOAD_CACHE_MANIFEST.write_text(
        json.dumps(cache_manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return cache_manifest


def restore_project_ml_checkpoints(archive, manifest):
    checkpoints = manifest.get("ml_checkpoints", []) or []
    if not checkpoints:
        return

    ensure_ml_batch_cache_dir()
    restored_index = []
    for item in checkpoints:
        archive_path = item.get("path")
        if not archive_path:
            continue
        name = safe_bundle_name(item.get("name"), Path(archive_path).name or "checkpoint.gwml")
        checkpoint_path = ML_BATCH_CACHE_DIR / name
        checkpoint_path.write_bytes(archive.read(archive_path))
        restored_index.append(
            {
                "run_id": "project_import",
                "station": Path(name).stem,
                "path": str(checkpoint_path),
                "created_at": datetime.now().isoformat(timespec="seconds"),
            }
        )

    if restored_index:
        index_path = ML_BATCH_CACHE_DIR / "latest.json"
        current_index = []
        if index_path.exists():
            try:
                current_index = json.loads(index_path.read_text(encoding="utf-8"))
            except Exception:
                current_index = []
        index_path.write_text(
            json.dumps((current_index + restored_index)[-500:], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )


def import_project_bundle(uploaded_file):
    return import_project_bundle_bytes(uploaded_file.getvalue())


def import_project_bundle_bytes(raw_bytes):
    with zipfile.ZipFile(io.BytesIO(raw_bytes)) as archive:
        validate_zip_archive(archive)
        manifest = json.loads(archive.read("manifest.json").decode("utf-8"))
        apply_project_settings(manifest.get("settings", {}))
        cache_manifest = install_project_sources(archive, manifest)
        restore_project_ml_checkpoints(archive, manifest)

        history_df = read_bundle_csv(archive, manifest, "history", normalize_results=True)
        last_run_df = read_bundle_csv(
            archive,
            manifest,
            "last_run_results",
            normalize_results=True,
        )
        forecast_df = read_bundle_csv(archive, manifest, "last_forecast_df")
        ml_validation_df = read_bundle_csv(archive, manifest, "last_ml_validation_df")
        ml_forecast_df = read_bundle_csv(archive, manifest, "last_ml_forecast_df")
        ml_impulse_df = read_bundle_csv(archive, manifest, "last_ml_impulse_df")
        ml_summary_table = read_bundle_csv(archive, manifest, "last_ml_summary_table")
        ml_artifacts = {}
        if manifest.get("ml_artifacts"):
            try:
                ml_artifacts = load_ml_artifacts_bytes(archive.read(manifest["ml_artifacts"]))
            except Exception:
                ml_artifacts = {}

    if last_run_df.empty and not history_df.empty:
        last_run_df, imported_run_label = get_last_imported_run(history_df)
    else:
        imported_run_label = manifest.get("last_run_label")

    if history_df.empty and not last_run_df.empty:
        history_df = last_run_df.copy()

    st.session_state.history = limit_history_runs(
        history_df,
        st.session_state.get("history_run_limit", 15),
    )
    st.session_state.last_run_results = last_run_df
    st.session_state.last_run_label = imported_run_label
    st.session_state.last_run_models = {}
    st.session_state.last_extra_match = manifest.get("last_extra_match", {})
    st.session_state.last_coord_match = manifest.get("last_coord_match", {})
    st.session_state.last_forecast_df = forecast_df
    st.session_state.last_forecast_info = manifest.get("last_forecast_info", {})
    st.session_state.last_ml_validation_df = ml_validation_df
    st.session_state.last_ml_forecast_df = ml_forecast_df
    st.session_state.last_ml_impulse_df = ml_impulse_df
    st.session_state.last_ml_summary_table = ml_summary_table
    st.session_state.last_ml_summary = manifest.get("last_ml_summary", {})
    st.session_state.last_ml_artifacts = ml_artifacts
    st.session_state.pending_project_model_rebuild = not last_run_df.empty
    remote_key_map = {
        "gw": "remote_gw_url",
        "weather": "remote_weather_url",
        "extra": "remote_extra_url",
        "coords": "remote_coords_url",
    }
    for source_key, entry in cache_manifest.items():
        if entry.get("type") == "file":
            st.session_state[remote_key_map[source_key]] = ""
    st.session_state.data_refresh_token += 1
    load_data.clear()


def import_ml_run_package_bytes_to_session(raw_bytes):
    with zipfile.ZipFile(io.BytesIO(raw_bytes)) as archive:
        validate_zip_archive(archive)
    package = load_ml_run_package_bytes(raw_bytes)
    metadata = package.get("metadata", {}) or {}
    st.session_state.last_ml_validation_df = package.get("validation_df", pd.DataFrame())
    st.session_state.last_ml_forecast_df = package.get("forecast_df", pd.DataFrame())
    st.session_state.last_ml_impulse_df = package.get("impulse_df", pd.DataFrame())
    st.session_state.last_ml_summary_table = package.get("summary_df", pd.DataFrame())
    st.session_state.last_ml_summary = metadata
    st.session_state.last_ml_artifacts = package.get("artifacts", {})
    return package


def import_ml_run_package(uploaded_file):
    return import_ml_run_package_bytes_to_session(uploaded_file.getvalue())


def import_any_bundle(uploaded_file):
    raw_bytes = uploaded_file.getvalue()
    with zipfile.ZipFile(io.BytesIO(raw_bytes)) as archive:
        validate_zip_archive(archive)
        manifest = json.loads(archive.read("manifest.json").decode("utf-8"))
    if "version" in manifest and "tables" in manifest:
        import_project_bundle_bytes(raw_bytes)
        return "project"
    if "files" in manifest and "metadata" in manifest:
        import_ml_run_package_bytes_to_session(raw_bytes)
        return "ml"
    raise ValueError("Unbekannter Pakettyp.")


def ensure_ml_batch_cache_dir():
    ML_BATCH_CACHE_DIR.mkdir(parents=True, exist_ok=True)


def safe_cache_slug(value):
    cleaned = "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in str(value))
    return cleaned.strip("_") or "station"


def write_ml_checkpoint_package(run_id, station, validation_df, summary_df, artifacts, metadata):
    ensure_ml_batch_cache_dir()
    package_bytes = create_ml_run_package(
        validation_df=validation_df,
        metadata=metadata,
        forecast_df=pd.DataFrame(),
        impulse_df=pd.DataFrame(),
        artifacts=artifacts,
        summary_df=summary_df,
    )
    package_path = ML_BATCH_CACHE_DIR / f"{safe_cache_slug(run_id)}_{safe_cache_slug(station)}.gwml"
    package_path.write_bytes(package_bytes)
    index_path = ML_BATCH_CACHE_DIR / "latest.json"
    index_payload = []
    if index_path.exists():
        try:
            index_payload = json.loads(index_path.read_text(encoding="utf-8"))
        except Exception:
            index_payload = []
    index_payload.append(
        {
            "run_id": str(run_id),
            "station": str(station),
            "path": str(package_path),
            "created_at": datetime.now().isoformat(timespec="seconds"),
        }
    )
    index_path.write_text(json.dumps(index_payload[-500:], ensure_ascii=False, indent=2), encoding="utf-8")
    return package_path


def list_ml_checkpoint_packages(limit=20):
    if not ML_BATCH_CACHE_DIR.exists():
        return []
    packages = sorted(
        ML_BATCH_CACHE_DIR.glob("*.gwml"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    return packages[:limit]


def render_project_bundle_panel(t, expanded=False):
    with st.expander(t["project_bundle"], expanded=expanded):
        st.caption(t["project_bundle_note"])
        project_col1, project_col2 = st.columns(2)
        with project_col1:
            bundle_bytes = create_project_bundle()
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            st.download_button(
                t["project_bundle_export"],
                data=bundle_bytes,
                file_name=f"gw_project_{timestamp}.gwproject",
                mime="application/zip",
                key="project_bundle_export_button",
            )
        with project_col2:
            project_file = st.file_uploader(
                t["project_bundle_import"],
                type=["gwproject", "gwml", "zip"],
                key="project_bundle_import",
            )
            if st.button(
                t["project_bundle_import_button"],
                key="project_bundle_import_button",
                disabled=project_file is None,
            ):
                try:
                    import_any_bundle(project_file)
                    st.session_state.history_notice = ("success", t["project_bundle_success"])
                    st.rerun()
                except Exception as exc:
                    st.error(f"{t['project_bundle_error']}: {exc}")


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
            detail = entry.get("name", Path(entry.get("path", "")).name)
            if entry.get("url"):
                detail = f"{detail} | {t['source_url_label']}: {mask_url_for_display(entry['url'])}"
            descriptions.append(f"{label}: {detail}")
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
                if manifest_entry.get("url"):
                    detail = f"{detail} | {t['source_url_label']}: {mask_url_for_display(manifest_entry['url'])}"
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
        headers={"User-Agent": "GW-Analyzer/1.0"},
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
    lon_col = find_matching_column(coords_df.columns, ["longitude", "long", "lon", "lng", "länge", "x"])
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


def create_response_function(model_type, response_cutoff):
    response_functions = {
        "Gamma": ps.Gamma,
        "Exponential": ps.Exponential,
        "Hantush": ps.Hantush,
    }
    if model_type not in response_functions:
        raise ValueError(f"Unsupported response model: {model_type}")
    return response_functions[model_type](cutoff=float(response_cutoff))


def create_config_label(model_type, use_flex, use_noise):
    recharge_label = "Flex" if use_flex else "Linear"
    noise_label = "Noise" if use_noise else "NoNoise"
    return f"{model_type} | {recharge_label} | {noise_label}"


def get_auto_configuration_labels():
    return [
        create_config_label(model_type, use_flex, use_noise)
        for model_type in RESPONSE_MODEL_TYPES
        for use_flex in AUTO_FLEX_OPTIONS
        for use_noise in AUTO_NOISE_OPTIONS
    ]


def parse_float_list(value, default_values):
    if not isinstance(value, str) or not value.strip():
        return list(default_values)
    numbers = []
    for part in value.replace(";", ",").split(","):
        part = part.strip()
        if not part:
            continue
        numbers.append(float(part))
    return numbers or list(default_values)


def build_parameter_search_values(base_parameter_values, search_parameter_names, multipliers):
    if not search_parameter_names:
        return [deepcopy(base_parameter_values)]

    multiplier_sets = [multipliers for _ in search_parameter_names]
    configurations = []
    for multiplier_values in product(*multiplier_sets):
        values = deepcopy(base_parameter_values)
        for name, multiplier in zip(search_parameter_names, multiplier_values):
            values[name] = float(base_parameter_values[name]) * float(multiplier)
        configurations.append(values)
        if len(configurations) >= PARAMETER_SEARCH_MAX_COMBINATIONS:
            break
    return configurations


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
    response_function = create_response_function(model_type, response_cutoff)
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
    excluded_auto_config_labels=None,
    parameter_search_enabled=False,
    parameter_search_names=None,
    parameter_search_multipliers=None,
):
    if run_strategy_label == t["strategy_manual"]:
        base_label = create_config_label(manual_model_type, manual_use_flex, manual_use_noise)
        if parameter_search_enabled:
            parameter_values_list = build_parameter_search_values(
                manual_parameter_values,
                parameter_search_names or [],
                parameter_search_multipliers or [1.0],
            )
        else:
            parameter_values_list = [deepcopy(manual_parameter_values)]

        configurations = []
        for index, parameter_values in enumerate(parameter_values_list, start=1):
            label = base_label
            if parameter_search_enabled and len(parameter_values_list) > 1:
                label = f"{base_label} | Startset {index}"
            configurations.append(
                {
                    "model_type": manual_model_type,
                    "use_flex": manual_use_flex,
                    "use_noise": manual_use_noise,
                    "parameter_values": deepcopy(parameter_values),
                    "vary_flags": deepcopy(manual_vary_flags),
                    "response_cutoff": response_cutoff,
                    "noise_norm": noise_norm,
                    "configuration_label": label,
                }
            )
        return configurations

    configurations = []
    excluded_auto_config_labels = set(excluded_auto_config_labels or [])
    for model_type in RESPONSE_MODEL_TYPES:
        for use_flex in AUTO_FLEX_OPTIONS:
            for use_noise in AUTO_NOISE_OPTIONS:
                configuration_label = create_config_label(model_type, use_flex, use_noise)
                if configuration_label in excluded_auto_config_labels:
                    continue
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
                        "configuration_label": configuration_label,
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
    if "BestStationModel" in df.columns:
        best_flags = df["BestStationModel"].map(
            lambda value: (
                value.strip().lower() in {"1", "true", "yes", "ja", "y"}
                if isinstance(value, str)
                else bool(value)
                if pd.notna(value)
                else False
            )
        )
        if best_flags.any():
            return df[best_flags].copy()
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
        "R2": "R²",
        "Test_R2": "Test R²",
        "Test_RMSE": "Test RMSE",
        "Test_EVP": "Test EVP",
        "Validation_R2": "Validierung R²",
        "Validation_RMSE": "Validierung RMSE",
        "Validation_EVP": "Validierung EVP",
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


def format_duration_days(days, language="Deutsch"):
    days = int(days)
    if days >= 365 and days % 365 == 0:
        years = days // 365
        if language == "English":
            return f"{years} year" if years == 1 else f"{years} years"
        return f"{years} Jahr" if years == 1 else f"{years} Jahre"
    return f"{days} days" if language == "English" else f"{days} Tage"


def format_ml_run_label(model_type, target_mode, language="Deutsch"):
    model_type = str(model_type).upper()
    target_mode = str(target_mode or "hybrid").lower()
    if target_mode == "direct":
        return f"Only {model_type}" if language == "English" else f"Nur {model_type}"
    return f"Pastas + {model_type}"


def compute_ml_sequence_stats(frame, feature_columns, window_size, horizon, target_column="target_residual"):
    if frame is None or frame.empty or not feature_columns or target_column not in frame.columns:
        return {
            "available_targets": 0,
            "sequences": 0,
            "n_train": 0,
            "n_test": 0,
            "n_valid": 0,
            "required_days": int(window_size) + int(horizon),
            "first_target": None,
            "last_target": None,
        }
    _, _, target_dates = make_supervised_sequences(
        frame,
        feature_columns,
        target_column,
        int(window_size),
        int(horizon),
    )
    n_sequences = int(len(target_dates))
    train_end = int(n_sequences * 0.6)
    test_end = int(n_sequences * 0.8)
    available_targets = int(np.isfinite(frame[target_column].astype(float).to_numpy()).sum())
    return {
        "available_targets": available_targets,
        "sequences": n_sequences,
        "n_train": train_end,
        "n_test": max(0, test_end - train_end),
        "n_valid": max(0, n_sequences - test_end),
        "required_days": int(window_size) + int(horizon),
        "first_target": target_dates.min() if n_sequences else None,
        "last_target": target_dates.max() if n_sequences else None,
    }


def estimate_ml_epoch_limit(n_train, hard_limit=ML_MAX_EPOCHS):
    n_train = int(n_train or 0)
    if n_train < 5:
        return 5
    if n_train < 30:
        return 30
    if n_train < 80:
        return 80
    if n_train < 180:
        return 150
    if n_train < 450:
        return 250
    if n_train < 900:
        return 400
    if n_train < 1600:
        return 650
    return int(hard_limit)


def recommend_ml_hyperparameters(stats, window_size, horizon, n_features, hard_epoch_limit=ML_MAX_EPOCHS):
    n_train = int(stats.get("n_train", 0) or 0)
    window_years = float(window_size) / 365.0
    horizon_years = float(horizon) / 365.0
    epoch_limit = estimate_ml_epoch_limit(n_train, hard_epoch_limit)

    if n_train < 80:
        epochs = min(epoch_limit, 60)
        learning_rate = 0.0005
        hidden_size = 16
        reason = "wenige Trainingssequenzen"
    elif n_train < 250:
        epochs = min(epoch_limit, 100)
        learning_rate = 0.0005 if horizon_years >= 5 else 0.001
        hidden_size = 16 if n_features <= 2 else 32
        reason = "kleiner bis mittlerer Trainingsblock"
    elif n_train < 700:
        epochs = min(epoch_limit, 160)
        learning_rate = 0.001
        hidden_size = 32 if horizon_years >= 5 else 64
        reason = "mittlerer Trainingsblock"
    elif n_train < 1400:
        epochs = min(epoch_limit, 240)
        learning_rate = 0.001
        hidden_size = 64
        reason = "größerer Trainingsblock"
    else:
        epochs = min(epoch_limit, 320)
        learning_rate = 0.001 if horizon_years >= 2 else 0.002
        hidden_size = 128 if n_features >= 4 and window_years <= 10 else 64
        reason = "viele Trainingssequenzen"

    if window_years >= 20 or horizon_years >= 10:
        learning_rate = min(learning_rate, 0.001)
        hidden_size = min(hidden_size, 64)
        epochs = min(epoch_limit, max(epochs, 120))
        reason += ", sehr langes Fenster/Horizont"

    return {
        "epochs": int(max(5, min(epoch_limit, epochs))),
        "learning_rate": float(learning_rate),
        "hidden_size": int(hidden_size),
        "epoch_limit": int(epoch_limit),
        "reason": reason,
    }


def get_ml_run_specs(language):
    return [
        {
            "model_type": model_type,
            "target_mode": target_mode,
            "display_name": format_ml_run_label(model_type, target_mode, language),
        }
        for model_type in ML_MODEL_TYPES
        for target_mode in ("hybrid", "direct")
    ]


def get_ml_train_worker_count(batch_mode, spec_count):
    if batch_mode:
        return 1
    cpu_count = os.cpu_count() or 1
    usable = max(1, int(cpu_count * 0.90))
    return max(1, min(2, int(spec_count), usable))


def train_ml_station(
    station,
    last_run_results,
    gw_df,
    rain,
    evap,
    window_size,
    horizon,
    epochs,
    learning_rate,
    hidden_size,
    include_weather,
    include_rollings,
    language,
    batch_mode=False,
):
    station_row = get_station_no_flex_metadata_row(last_run_results, station)
    if station_row is None:
        raise ValueError(f"Kein erfolgreiches Pastas-Modell ohne FlexModel für {station}.")

    pastas_model = build_model_from_result_row(station_row, gw_df, rain, evap)
    feature_frame = build_hybrid_feature_frame(
        station=station,
        gw_df=gw_df,
        rain=rain,
        evap=evap,
        pastas_model=pastas_model,
        include_head=False,
        include_weather=include_weather,
        include_rollings=include_rollings,
        include_season=False,
    )
    feature_columns = [
        column
        for column in feature_frame.columns
        if column not in {"observed", "pastas_sim", "target_residual", "head_filled"}
    ]
    specs = get_ml_run_specs(language)
    worker_count = get_ml_train_worker_count(batch_mode, len(specs))
    torch_threads = configure_torch_cpu_budget(worker_count=worker_count, reserve_fraction=0.10)

    training_results = {}
    training_errors = {}

    def run_spec(spec):
        return train_evaluate_hybrid(
            feature_frame,
            feature_columns,
            model_type=spec["model_type"],
            window_size=window_size,
            horizon=horizon,
            train_fraction=0.6,
            epochs=int(epochs),
            learning_rate=float(learning_rate),
            hidden_size=int(hidden_size),
            target_mode=spec["target_mode"],
        )

    if worker_count == 1:
        for spec in specs:
            try:
                training_results[spec["display_name"]] = {"result": run_spec(spec), "spec": spec}
            except Exception as exc:
                training_errors[spec["display_name"]] = str(exc)
    else:
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            futures = {executor.submit(run_spec, spec): spec for spec in specs}
            for future in as_completed(futures):
                spec = futures[future]
                try:
                    training_results[spec["display_name"]] = {"result": future.result(), "spec": spec}
                except Exception as exc:
                    training_errors[spec["display_name"]] = str(exc)

    if not training_results:
        raise RuntimeError("CNN/TCN/LSTM konnten in keinem Zielmodus trainiert werden.")

    evaluation_frames = []
    summary_rows = []
    artifact_store = {}
    for spec in specs:
        display_name = spec["display_name"]
        if display_name not in training_results:
            continue
        result = training_results[display_name]["result"]
        run_key = f"{station} | {display_name}"
        evaluation_df = result["evaluation"].copy()
        evaluation_df["Messstelle"] = station
        evaluation_df["NeuralModel"] = display_name
        evaluation_df["MLRunKey"] = run_key
        evaluation_df["Architektur"] = spec["model_type"]
        evaluation_df["MLModus"] = "Nur ML" if spec["target_mode"] == "direct" else "Pastas + ML"
        evaluation_frames.append(evaluation_df)

        test_metrics = result["test_metrics"]
        validation_metrics = result["validation_metrics"]
        summary_rows.append(
            {
                "Messstelle": station,
                "MLRunKey": run_key,
                "NeuralModel": display_name,
                "Architektur": spec["model_type"],
                "MLModus": "Nur ML" if spec["target_mode"] == "direct" else "Pastas + ML",
                "R² Test": test_metrics["R2"],
                "RMSE Test": test_metrics["RMSE"],
                "EVP Test": test_metrics["EVP"],
                "R² Validierung": validation_metrics["R2"],
                "RMSE Validierung": validation_metrics["RMSE"],
                "EVP Validierung": validation_metrics["EVP"],
                "Train": result["n_train"],
                "Test": result["n_test"],
                "Validierung": result["n_valid"],
                "Epochen": int(epochs),
                "Lernrate": float(learning_rate),
                "Hidden Size/Filter": int(hidden_size),
                "Torch Threads": int(torch_threads),
            }
        )
        artifact = build_artifacts(
            result,
            spec["model_type"],
            window_size,
            horizon,
            hidden_size,
            target_mode=spec["target_mode"],
        )
        artifact["station"] = station
        artifact["display_name"] = display_name
        artifact["run_key"] = run_key
        artifact_store[run_key] = artifact

    validation_df = pd.concat(evaluation_frames, ignore_index=True)
    summary_table = pd.DataFrame(summary_rows)
    selected_summary_row = summary_table.sort_values(
        "R² Validierung",
        ascending=False,
        na_position="last",
    ).iloc[0]
    metadata = {
        "Run": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Messstelle": station,
        "Modell": "Pastas + ML und Nur-ML parallel",
        "BestesNeuralModell": selected_summary_row["NeuralModel"],
        "BestesMLRunKey": selected_summary_row["MLRunKey"],
        "Architektur": selected_summary_row.get("Architektur"),
        "MLModus": selected_summary_row.get("MLModus"),
        "Fenster": int(window_size),
        "Horizont": int(horizon),
        "Split": "60/20/20",
        "Features": ", ".join(feature_columns),
        "Epochen": int(epochs),
        "Lernrate": float(learning_rate),
        "HiddenSize": int(hidden_size),
        "TorchThreads": int(torch_threads),
        "R2": selected_summary_row["R² Validierung"],
        "RMSE": selected_summary_row["RMSE Validierung"],
        "EVP": selected_summary_row["EVP Validierung"],
        "Test_R2": selected_summary_row["R² Test"],
        "Test_RMSE": selected_summary_row["RMSE Test"],
        "Test_EVP": selected_summary_row["EVP Test"],
        "n_train": int(selected_summary_row["Train"]),
        "n_test": int(selected_summary_row["Test"]),
        "n_valid": int(selected_summary_row["Validierung"]),
    }
    return {
        "station": station,
        "validation_df": validation_df,
        "summary_table": summary_table,
        "summary": metadata,
        "artifacts": artifact_store,
        "errors": training_errors,
    }


def build_ml_history_rows(summary_table, run_label, window_size, horizon):
    if summary_table is None or summary_table.empty:
        return pd.DataFrame()
    rows = []
    for _, row in summary_table.iterrows():
        rows.append(
            {
                "Run": run_label,
                "Messstelle": row.get("Messstelle"),
                "Modus": "ML",
                "Suchmodus": "Pastas + ML und Nur-ML",
                "Konfiguration": f"{row.get('NeuralModel')} | {int(window_size)}d -> {int(horizon)}d",
                "Modell": row.get("NeuralModel"),
                "Flex": False,
                "Noise": None,
                "Cutoff": None,
                "NoiseNorm": None,
                "n_obs": row.get("Validierung"),
                "Status": "ok",
                "BestStationModel": False,
                "R2": row.get("R² Validierung"),
                "RMSE": row.get("RMSE Validierung"),
                "EVP": row.get("EVP Validierung"),
                "AIC": None,
                "Fehler": None,
            }
        )
    history_df = pd.DataFrame(rows)
    if not history_df.empty and "R2" in history_df.columns:
        best_indices = history_df.groupby("Messstelle")["R2"].idxmax()
        history_df.loc[best_indices, "BestStationModel"] = True
    return history_df


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


def create_pastas_impulse_response_df(model, impulse_mm):
    response = model.get_block_response("recharge")
    response_index = pd.Index(response.index)
    if isinstance(response_index, pd.TimedeltaIndex):
        lag_days = response_index / pd.Timedelta(days=1)
    else:
        lag_days = pd.to_numeric(pd.Series(response_index), errors="coerce").to_numpy(dtype=float)
    response_values = pd.to_numeric(pd.Series(response.values), errors="coerce").to_numpy(dtype=float)
    response_df = pd.DataFrame(
        {
            "lag_days": lag_days,
            "pastas_irf_response": response_values * float(impulse_mm),
        }
    )
    return response_df.dropna().sort_values("lag_days")


def figure_to_png_bytes(figure):
    buffer = io.BytesIO()
    figure.savefig(buffer, format="png", dpi=200, bbox_inches="tight")
    buffer.seek(0)
    return buffer.getvalue()


def build_simulated_timeseries_df(model, station=None):
    observations = model.observations()
    simulation = model.simulate(
        tmin=observations.index.min(),
        tmax=observations.index.max(),
    )
    result = pd.concat(
        [
            observations.rename("observed"),
            simulation.rename("simulated"),
        ],
        axis=1,
    )
    result["residual"] = result["observed"] - result["simulated"]
    result = result.reset_index().rename(columns={result.index.name or "index": "date"})
    if "date" not in result.columns:
        result = result.rename(columns={result.columns[0]: "date"})
    result.insert(0, "station", station or model.name)
    return result


def build_all_simulated_timeseries_df(models):
    frames = []
    for station, model in models.items():
        frames.append(build_simulated_timeseries_df(model, station))
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True, sort=False)


def get_station_metadata_row(last_run_df, station):
    if last_run_df.empty:
        return None
    station_df = last_run_df[last_run_df["Messstelle"] == station].copy()
    if station_df.empty:
        return None
    station_df = get_best_only_rows(station_df)
    if "R2" in station_df.columns:
        station_df = station_df.sort_values("R2", ascending=False, na_position="last")
    return station_df.iloc[0]


def get_station_no_flex_metadata_row(last_run_df, station):
    if last_run_df.empty:
        return None
    station_df = last_run_df[last_run_df["Messstelle"] == station].copy()
    if station_df.empty or "Flex" not in station_df.columns:
        return None
    if "Status" in station_df.columns:
        station_df = station_df[station_df["Status"] == "ok"]
    station_df = station_df[~station_df["Flex"].map(coerce_bool)]
    if station_df.empty:
        return None
    if "R2" in station_df.columns:
        station_df = station_df.sort_values("R2", ascending=False, na_position="last")
    return station_df.iloc[0]


def get_no_flex_station_options(last_run_df):
    if last_run_df.empty or "Messstelle" not in last_run_df.columns or "Flex" not in last_run_df.columns:
        return []
    result_df = last_run_df.copy()
    if "Status" in result_df.columns:
        result_df = result_df[result_df["Status"] == "ok"]
    result_df = result_df[~result_df["Flex"].map(coerce_bool)]
    return sorted(result_df["Messstelle"].dropna().astype(str).unique().tolist())


def normalize_ml_artifacts(artifacts):
    if not isinstance(artifacts, dict) or not artifacts:
        return {}
    if "model" in artifacts:
        return {
            str(
                artifacts.get("run_key")
                or artifacts.get("display_name")
                or artifacts.get("model_type")
                or "ML"
            ): artifacts
        }
    return {
        str(artifact.get("run_key") or name): artifact
        for name, artifact in artifacts.items()
        if isinstance(artifact, dict) and "model" in artifact
    }


def coerce_bool(value):
    if pd.isna(value):
        return False
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


def build_model_from_result_row(station_row, gw_df, rain, evap):
    station = station_row["Messstelle"]
    if station not in gw_df.columns:
        raise ValueError(f"Station not found in groundwater data: {station}")

    head = gw_df[station].dropna()
    model = ps.Model(head, name=station)
    cutoff_value = station_row.get("Cutoff", 0.999)
    if pd.isna(cutoff_value):
        cutoff_value = 0.999
    response_function = create_response_function(str(station_row["Modell"]), float(cutoff_value))
    recharge_model = ps.rch.FlexModel() if coerce_bool(station_row["Flex"]) else ps.rch.Linear()
    stress_model = ps.RechargeModel(
        rain,
        evap,
        rfunc=response_function,
        recharge=recharge_model,
    )
    model.add_stressmodel(stress_model)

    if coerce_bool(station_row["Noise"]):
        noise_norm_value = True
        if "NoiseNorm" in station_row.index and pd.notna(station_row["NoiseNorm"]):
            noise_norm_value = coerce_bool(station_row["NoiseNorm"])
        model.add_noisemodel(ps.ArNoiseModel(norm=noise_norm_value))

    for name in model.parameters.index:
        optimal_col = f"param_opt_{name}"
        initial_col = f"param_init_{name}"
        requested_col = f"requested_{name}"
        if optimal_col in station_row.index and pd.notna(station_row[optimal_col]):
            value = float(station_row[optimal_col])
        elif initial_col in station_row.index and pd.notna(station_row[initial_col]):
            value = float(station_row[initial_col])
        elif requested_col in station_row.index and pd.notna(station_row[requested_col]):
            value = float(station_row[requested_col])
        else:
            value = float(model.parameters.loc[name, "initial"])
        model.set_parameter(name, initial=value, vary=False, optimal=value)

    return model


def rebuild_models_from_results(result_df, gw_df, rain, evap):
    if result_df.empty or "Messstelle" not in result_df.columns:
        return {}
    ok_df = result_df[result_df["Status"] == "ok"].copy() if "Status" in result_df.columns else result_df.copy()
    if ok_df.empty:
        return {}
    selected_rows = get_best_only_rows(ok_df)
    if selected_rows.empty:
        selected_rows = ok_df

    models = {}
    for station, station_df in selected_rows.groupby("Messstelle", sort=False):
        if "R2" in station_df.columns:
            station_df = station_df.sort_values("R2", ascending=False, na_position="last")
        try:
            models[station] = build_model_from_result_row(station_df.iloc[0], gw_df, rain, evap)
        except Exception:
            continue
    return models


def normalize_imported_results(imported_df):
    if imported_df is None or imported_df.empty:
        return pd.DataFrame()

    normalized = imported_df.copy()
    normalized = normalized.dropna(how="all")

    bool_columns = {"Flex", "Noise", "BestStationModel", "NoiseNorm"}
    bool_columns.update(
        column
        for column in normalized.columns
        if column.startswith("requested_vary_") or column.startswith("param_vary_")
    )
    for column in bool_columns & set(normalized.columns):
        normalized[column] = normalized[column].map(
            lambda value: pd.NA if pd.isna(value) else coerce_bool(value)
        )

    numeric_columns = {
        "Cutoff",
        "n_obs",
        "R2",
        "RMSE",
        "EVP",
        "AIC",
        "Latitude",
        "Longitude",
    }
    numeric_columns.update(
        column
        for column in normalized.columns
        if (
            (
                column.startswith("requested_")
                and not column.startswith("requested_vary_")
            )
            or column.startswith("param_init_")
            or column.startswith("param_opt_")
        )
    )
    for column in numeric_columns & set(normalized.columns):
        normalized[column] = pd.to_numeric(normalized[column], errors="coerce")

    if "Run" not in normalized.columns:
        normalized["Run"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    if "Status" not in normalized.columns:
        normalized["Status"] = "ok"
    if "BestStationModel" not in normalized.columns:
        normalized["BestStationModel"] = False

    return normalized.reset_index(drop=True)


def read_imported_results(uploaded_file):
    try:
        imported_df = pd.read_csv(uploaded_file, sep=";")
        if len(imported_df.columns) == 1 and "," in str(imported_df.columns[0]):
            uploaded_file.seek(0)
            imported_df = pd.read_csv(uploaded_file, sep=",")
    except Exception:
        uploaded_file.seek(0)
        imported_df = pd.read_csv(uploaded_file, sep=None, engine="python")
    return normalize_imported_results(imported_df)


def get_last_imported_run(imported_df):
    if imported_df.empty:
        return pd.DataFrame(), None
    if "Run" not in imported_df.columns:
        return imported_df.copy(), None
    run_values = imported_df["Run"].dropna().astype(str).tolist()
    if not run_values:
        return imported_df.copy(), None
    run_label = run_values[-1]
    run_df = imported_df[imported_df["Run"].astype(str) == run_label].copy()
    return run_df.reset_index(drop=True), run_label


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
    model = build_model_from_result_row(station_row, gw_df, rain_extended, evap_extended)

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
    should_select_best = run_strategy_label == t["strategy_auto"] or len(search_configurations) > 1
    if should_select_best:
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

st.title(t["title"])
st.caption(t["subtitle"])
render_project_bundle_panel(t, expanded=True)


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
            st.session_state.last_ml_validation_df = pd.DataFrame()
            st.session_state.last_ml_forecast_df = pd.DataFrame()
            st.session_state.last_ml_impulse_df = pd.DataFrame()
            st.session_state.last_ml_summary_table = pd.DataFrame()
            st.session_state.last_ml_summary = {}
            st.session_state.last_ml_artifacts = {}
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
    st.session_state.last_ml_validation_df = pd.DataFrame()
    st.session_state.last_ml_forecast_df = pd.DataFrame()
    st.session_state.last_ml_impulse_df = pd.DataFrame()
    st.session_state.last_ml_summary_table = pd.DataFrame()
    st.session_state.last_ml_summary = {}
    st.session_state.last_ml_artifacts = {}
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

if st.session_state.get("pending_project_model_rebuild") and not st.session_state.last_run_results.empty:
    st.session_state.last_run_models = rebuild_models_from_results(
        st.session_state.last_run_results,
        gw_df,
        rain,
        evap,
    )
    st.session_state.pending_project_model_rebuild = False


main_view_labels = {
    "plot": t["tab_plot"],
    "diagnostics": t["tab_diagnostics"],
    "compare": t["tab_compare"],
    "map": t["tab_map"],
    "forecast": t["tab_forecast"],
    "ml_forecast": t["tab_ml_forecast"],
    "save": t["tab_save"],
}
if st.session_state.get("active_main_view") not in main_view_labels:
    st.session_state.active_main_view = "plot"
active_main_view = st.radio(
    t["main_nav"],
    options=list(main_view_labels.keys()),
    format_func=lambda key: main_view_labels[key],
    horizontal=True,
    key="active_main_view",
)

data_status = build_data_status(gw_df, rain, evap)
with st.expander(t["data_status"], expanded=False):
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
            width="stretch",
            hide_index=True,
        )


st.markdown(f"### {t['config_heading']}")

auto_config_options = get_auto_configuration_labels()
auto_excluded_config_labels = [
    label
    for label in st.session_state.get("auto_excluded_configs", [])
    if label in auto_config_options
]
active_auto_combo_count = len(auto_config_options) - len(auto_excluded_config_labels)
control_col1, control_col2, control_col3, control_col4, control_col5 = st.columns(5)

mode_labels = {
    "single": t["mode_single"],
    "multi": t["mode_multi"],
    "all": t["mode_all"],
}
if st.session_state.get("analysis_mode") not in mode_labels:
    st.session_state.analysis_mode = "single"

with control_col1:
    mode_code = st.selectbox(
        t["mode"],
        list(mode_labels.keys()),
        format_func=lambda key: mode_labels[key],
        key="analysis_mode",
    )
    mode = mode_labels[mode_code]

single_station = None
multi_stations = []

with control_col2:
    if mode_code == "single":
        if st.session_state.get("selected_station") not in stations:
            st.session_state.selected_station = stations[0]
        single_station = st.selectbox(t["station"], stations, key="selected_station")
    elif mode_code == "multi":
        saved_multi_stations = st.session_state.get("selected_stations", [])
        if not isinstance(saved_multi_stations, list):
            saved_multi_stations = []
        cleaned_multi_stations = [station for station in saved_multi_stations if station in stations]
        if cleaned_multi_stations != saved_multi_stations:
            st.session_state.selected_stations = cleaned_multi_stations
        multi_stations = st.multiselect(t["stations"], stations, key="selected_stations")
    else:
        st.markdown(f"**{len(stations)}** {t['available_stations'].lower()}")

run_strategy_labels = {
    "manual": t["strategy_manual"],
    "auto": t["strategy_auto"],
}
if st.session_state.get("run_strategy") not in run_strategy_labels:
    st.session_state.run_strategy = "manual"

with control_col3:
    run_strategy_code = st.selectbox(
        t["run_strategy"],
        list(run_strategy_labels.keys()),
        format_func=lambda key: run_strategy_labels[key],
        key="run_strategy",
    )
    run_strategy = run_strategy_labels[run_strategy_code]

with control_col4:
    if run_strategy_code == "manual":
        if st.session_state.get("manual_response_model") not in RESPONSE_MODEL_TYPES:
            st.session_state.manual_response_model = RESPONSE_MODEL_TYPES[0]
        model_type = st.selectbox(t["model"], RESPONSE_MODEL_TYPES, key="manual_response_model")
    else:
        st.metric(t["auto_combo_count"], active_auto_combo_count)
        model_type = "Gamma"

with control_col5:
    if run_strategy_code == "manual":
        use_flex = st.checkbox(t["use_flex"], value=True, key="manual_use_flex")
        use_noise = st.checkbox(t["use_noise"], value=True, key="manual_use_noise")
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
parameter_search_enabled = False
parameter_search_names = []
parameter_search_multipliers = [1.0]

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
            key="response_cutoff_control",
        )
    with advanced_col2:
        noise_norm = st.checkbox(t["noise_norm"], value=True, key="noise_norm_control")

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

        st.divider()
        parameter_search_enabled = st.checkbox(
            t["parameter_search"],
            value=False,
            key="parameter_search_enabled",
        )
        if parameter_search_enabled:
            st.caption(t["parameter_search_note"])
            searchable_parameters = [spec["name"] for spec in manual_specs if spec["vary"]]
            default_parameters = searchable_parameters[:2]
            parameter_search_names = st.multiselect(
                t["parameter_search_params"],
                searchable_parameters,
                default=default_parameters,
                key=f"parameter_search_params_{model_type}_{use_flex}",
            )
            multiplier_text = st.text_input(
                t["parameter_search_multipliers"],
                value="0.5, 1, 2",
                key="parameter_search_multipliers",
            )
            try:
                parameter_search_multipliers = parse_float_list(multiplier_text, [1.0])
            except ValueError:
                parameter_search_multipliers = [1.0]
                st.warning(t["parameter_search_invalid_multipliers"])
            parameter_search_combo_count = (
                len(parameter_search_multipliers) ** len(parameter_search_names)
                if parameter_search_names
                else 1
            )
            st.metric(t["parameter_search_combo_count"], parameter_search_combo_count)
            if parameter_search_combo_count > PARAMETER_SEARCH_MAX_COMBINATIONS:
                st.warning(t["parameter_search_too_many"])

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
            key="parallel_workers_control",
        )
    with perf_col2:
        early_stop_enabled = st.checkbox(
            t["early_stop"],
            value=False,
            disabled=run_strategy != t["strategy_auto"],
            key="early_stop_enabled",
        )
    with perf_col3:
        early_stop_r2 = st.slider(
            t["early_stop_r2"],
            min_value=0.5,
            max_value=0.99,
            value=0.85,
            step=0.01,
            disabled=not early_stop_enabled or run_strategy != t["strategy_auto"],
            key="early_stop_r2_control",
        )
    st.caption(t["parallel_note"])
    if run_strategy == t["strategy_auto"]:
        auto_excluded_config_labels = st.multiselect(
            t["auto_exclude_configs"],
            auto_config_options,
            default=auto_excluded_config_labels,
            help=t["auto_exclude_help"],
            key="auto_excluded_configs",
        )
        active_auto_combo_count = len(auto_config_options) - len(auto_excluded_config_labels)
        if active_auto_combo_count <= 0:
            st.warning(t["auto_exclude_all_warning"])

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
    excluded_auto_config_labels=auto_excluded_config_labels,
    parameter_search_enabled=parameter_search_enabled,
    parameter_search_names=parameter_search_names,
    parameter_search_multipliers=parameter_search_multipliers,
)
if not search_configurations:
    st.warning(t["auto_exclude_all_warning"])


run_col1, run_col2 = st.columns([1, 1])
with run_col1:
    if st.button(
        t["run"],
        type="primary",
        disabled=(
            active_job is not None
            and active_job["status"] in {"queued", "running"}
        )
        or not search_configurations,
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


if active_main_view == "plot":
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
        with st.expander(t["simulated_ts_heading"], expanded=False):
            st.caption(t["simulated_ts_note"])
            if st.checkbox(
                t["show_simulated_series"],
                value=True,
                key=f"show_simulated_series_{plot_station}",
            ):
                simulated_ts_df = build_simulated_timeseries_df(
                    selected_model,
                    plot_station,
                )
                st.dataframe(
                    simulated_ts_df,
                    width="stretch",
                    hide_index=True,
                )
                simulated_ts_csv = (
                    sanitize_export_df(simulated_ts_df)
                    .to_csv(index=False, sep=";")
                    .encode("utf-8")
                )
                st.download_button(
                    t["simulated_ts_download"],
                    data=simulated_ts_csv,
                    file_name=f"{plot_station}_simulated_timeseries.csv",
                    mime="text/csv",
                    key=f"download_simulated_timeseries_{plot_station}",
                )


if active_main_view == "diagnostics":
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
            st.dataframe(seasonal_error, width="stretch", hide_index=True)

        cluster_df = build_response_cluster_table(last_run_models)
        if not cluster_df.empty:
            st.markdown(f"#### {t['diag_cluster']}")
            st.caption(t["diag_cluster_note"])
            st.dataframe(cluster_df.round(4), width="stretch", hide_index=True)


if active_main_view == "compare":
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
        st.dataframe(compact_df, width="stretch", hide_index=True)
        st.caption(t["compact_table_note"])

        with st.expander(t["extended_table"], expanded=False):
            st.dataframe(compare_df, width="stretch", hide_index=True)

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
                    st.dataframe(summary, width="stretch", hide_index=True)

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
                    st.dataframe(station_pivot, width="stretch", hide_index=True)

        if "Fehler" in compare_df.columns:
            error_df = compare_df[
                compare_df["Fehler"].notna() & (compare_df["Fehler"].astype(str) != "")
            ]
            if not error_df.empty:
                with st.expander(t["error_table"], expanded=False):
                    error_display = error_df[
                        ["Run", "Messstelle", "Konfiguration", "Status", "Fehler"]
                    ].rename(columns=get_display_column_labels(t))
                    st.dataframe(error_display, width="stretch", hide_index=True)


if active_main_view == "map":
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
        if "BestStationModel" in map_df.columns:
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
                st.dataframe(map_table, width="stretch", hide_index=True)


if active_main_view == "forecast":
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
                    st.session_state.last_ml_forecast_df = pd.DataFrame()
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
                width="stretch",
                hide_index=True,
            )
            forecast_csv = sanitize_export_df(forecast_df).to_csv(index=False, sep=";").encode("utf-8")
            st.download_button(
                t["forecast_download"],
                data=forecast_csv,
                file_name="forecast_results.csv",
                mime="text/csv",
            )


if active_main_view == "ml_forecast":
    if not last_run_models or last_run_results.empty:
        st.info(t["ml_requires_pastas"])
    elif not torch_available():
        st.warning(t["ml_torch_missing"])
    else:
        no_flex_station_options = get_no_flex_station_options(last_run_results)
        if not no_flex_station_options:
            st.info(t["ml_requires_no_flex"])
        else:
            ml_station = st.selectbox(
                t["forecast_station"],
                no_flex_station_options,
                key="ml_station_select",
            )
            ml_station_row = get_station_no_flex_metadata_row(last_run_results, ml_station)
            try:
                ml_model = build_model_from_result_row(ml_station_row, gw_df, rain, evap)
            except Exception as exc:
                ml_model = None
                st.error(str(exc))

            if ml_model is not None:
                st.caption(t["ml_model_parallel"])
                ml_window_cols = st.columns(2)
                with ml_window_cols[0]:
                    ml_window = st.selectbox(
                        t["ml_window"],
                        ML_WINDOW_OPTIONS,
                        index=2,
                        key="ml_window",
                        format_func=lambda days: format_duration_days(days, st.session_state.lang),
                        help=t["ml_window_help"],
                    )
                with ml_window_cols[1]:
                    ml_horizon = st.selectbox(
                        t["ml_horizon"],
                        ML_HORIZON_OPTIONS,
                        index=2,
                        key="ml_horizon",
                        format_func=lambda days: format_duration_days(days, st.session_state.lang),
                        help=t["ml_horizon_help"],
                    )

                st.caption(t["ml_split_note"])
                st.caption(t["ml_feature_restriction"])
                st.markdown(f"#### {t['ml_features']}")
                feature_cols = st.columns(2)
                with feature_cols[0]:
                    ml_include_weather = st.checkbox(
                        t["ml_feature_weather"],
                        value=True,
                        help=t["ml_feature_weather_help"],
                    )
                with feature_cols[1]:
                    ml_include_rollings = st.checkbox(
                        t["ml_feature_rollings"],
                        value=True,
                        help=t["ml_feature_rollings_help"],
                    )

                feature_frame = build_hybrid_feature_frame(
                    station=ml_station,
                    gw_df=gw_df,
                    rain=rain,
                    evap=evap,
                    pastas_model=ml_model,
                    include_head=False,
                    include_weather=ml_include_weather,
                    include_rollings=ml_include_rollings,
                    include_season=False,
                )
                feature_columns = [
                    column
                    for column in feature_frame.columns
                    if column not in {"observed", "pastas_sim", "target_residual", "head_filled"}
                ]
                setup_stats = compute_ml_sequence_stats(
                    feature_frame,
                    feature_columns,
                    ml_window,
                    ml_horizon,
                    target_column="target_residual",
                )
                recommendation = recommend_ml_hyperparameters(
                    setup_stats,
                    ml_window,
                    ml_horizon,
                    len(feature_columns),
                    ML_MAX_EPOCHS,
                )
                epoch_widget_max = max(5, int(recommendation["epoch_limit"]))
                current_ml_epochs = st.session_state.get("ml_epochs", 60) or 60
                if int(current_ml_epochs) > epoch_widget_max:
                    st.session_state.ml_epochs = epoch_widget_max

                if setup_stats["sequences"]:
                    st.caption(
                        t["ml_data_epoch_limit"].format(
                            limit=epoch_widget_max,
                            max_epochs=ML_MAX_EPOCHS,
                            sequences=setup_stats["sequences"],
                            train=setup_stats["n_train"],
                        )
                    )
                else:
                    st.caption(t["ml_data_epoch_limit_empty"])
                st.caption(
                    t["ml_recommendation"].format(
                        epochs=recommendation["epochs"],
                        learning_rate=f"{recommendation['learning_rate']:g}",
                        hidden_size=recommendation["hidden_size"],
                        reason=recommendation["reason"],
                    )
                )
                st.caption(t["ml_recommendation_note"])

                ml_param_cols = st.columns(3)
                with ml_param_cols[0]:
                    ml_epochs = st.number_input(
                        t["ml_epochs"],
                        min_value=5,
                        max_value=epoch_widget_max,
                        value=min(60, epoch_widget_max),
                        step=5,
                        help=t["ml_epochs_help"],
                        key="ml_epochs",
                    )
                with ml_param_cols[1]:
                    ml_learning_rate_options = [0.0005, 0.001, 0.002, 0.005]
                    recommended_lr_index = (
                        ml_learning_rate_options.index(recommendation["learning_rate"])
                        if recommendation["learning_rate"] in ml_learning_rate_options
                        else 1
                    )
                    ml_learning_rate = st.selectbox(
                        t["ml_learning_rate"],
                        ml_learning_rate_options,
                        index=recommended_lr_index,
                        format_func=lambda value: f"{value:g}",
                        help=t["ml_learning_rate_help"],
                        key="ml_learning_rate",
                    )
                with ml_param_cols[2]:
                    ml_hidden_size_options = [16, 32, 64, 128]
                    recommended_hidden_index = (
                        ml_hidden_size_options.index(recommendation["hidden_size"])
                        if recommendation["hidden_size"] in ml_hidden_size_options
                        else 1
                    )
                    ml_hidden_size = st.selectbox(
                        t["ml_hidden_size"],
                        ml_hidden_size_options,
                        index=recommended_hidden_index,
                        help=t["ml_hidden_size_help"],
                        key="ml_hidden_size",
                    )

                ml_scope = st.radio(
                    t["ml_scope"],
                    ["single", "all"],
                    index=0,
                    horizontal=True,
                    format_func=lambda value: t["ml_scope_all"] if value == "all" else t["ml_scope_single"],
                    key="ml_scope",
                )
                if ml_scope == "all":
                    st.caption(t["ml_batch_note"])
                    st.caption(t["ml_resource_note"])

                import_col, cache_col = st.columns(2)
                with import_col:
                    ml_import_file = st.file_uploader(
                        t["ml_import_package"],
                        type=["gwml", "zip"],
                        key="ml_package_import",
                    )
                    if st.button(
                        t["ml_import_package_button"],
                        key="ml_package_import_button",
                        disabled=ml_import_file is None,
                    ):
                        try:
                            import_ml_run_package(ml_import_file)
                            st.session_state.history_notice = ("success", t["ml_import_package_success"])
                            st.rerun()
                        except Exception as exc:
                            st.error(str(exc))
                with cache_col:
                    cached_packages = list_ml_checkpoint_packages()
                    if cached_packages:
                        selected_cache = st.selectbox(
                            t["ml_batch_cache"],
                            cached_packages,
                            format_func=lambda path: path.name,
                            key="ml_cache_package_select",
                        )
                        if st.button(t["ml_import_package_button"], key="ml_cache_package_load"):
                            try:
                                import_ml_run_package_bytes_to_session(selected_cache.read_bytes())
                                st.session_state.history_notice = ("success", t["ml_import_package_success"])
                                st.rerun()
                            except Exception as exc:
                                st.error(str(exc))
                    else:
                        st.caption(t["ml_batch_cache_empty"])

                if st.button(t["ml_train"], type="primary", key="train_ml_hybrid"):
                    try:
                        station_batch = no_flex_station_options if ml_scope == "all" else [ml_station]
                        progress_bar = st.progress(0.0)
                        progress_text = st.empty()
                        run_label = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
                        validation_frames = []
                        summary_frames = []
                        artifact_store = {}
                        all_errors = {}
                        with st.spinner(t["computing"]):
                            for station_index, station_name in enumerate(station_batch, start=1):
                                progress_text.caption(
                                    f"{t['progress']}: {station_name} ({station_index}/{len(station_batch)})"
                                )
                                try:
                                    station_output = train_ml_station(
                                        station_name,
                                        last_run_results,
                                        gw_df,
                                        rain,
                                        evap,
                                        ml_window,
                                        ml_horizon,
                                        int(ml_epochs),
                                        float(ml_learning_rate),
                                        int(ml_hidden_size),
                                        ml_include_weather,
                                        ml_include_rollings,
                                        st.session_state.lang,
                                        batch_mode=(ml_scope == "all"),
                                    )
                                    station_validation = station_output["validation_df"]
                                    station_summary = station_output["summary_table"]
                                    station_summary["Run"] = run_label
                                    station_metadata = station_output["summary"]
                                    station_metadata["Run"] = run_label
                                    for key in (
                                        "Empfohlene_Epochen",
                                        "Empfohlene_Lernrate",
                                        "Empfohlene_HiddenSize",
                                        "Datenbasiertes_Epochenlimit",
                                    ):
                                        station_metadata[key] = {
                                            "Empfohlene_Epochen": recommendation["epochs"],
                                            "Empfohlene_Lernrate": recommendation["learning_rate"],
                                            "Empfohlene_HiddenSize": recommendation["hidden_size"],
                                            "Datenbasiertes_Epochenlimit": epoch_widget_max,
                                        }[key]
                                    checkpoint_path = write_ml_checkpoint_package(
                                        run_id,
                                        station_name,
                                        station_validation,
                                        station_summary,
                                        station_output["artifacts"],
                                        station_metadata,
                                    )
                                    station_summary["Checkpoint"] = str(checkpoint_path)
                                    validation_frames.append(station_validation)
                                    summary_frames.append(station_summary)
                                    artifact_store.update(station_output["artifacts"])
                                    for model_type, error_message in station_output["errors"].items():
                                        all_errors[f"{station_name} | {model_type}"] = error_message
                                    del station_output
                                    gc.collect()
                                except Exception as exc:
                                    all_errors[str(station_name)] = str(exc)
                                    st.warning(f"{station_name}: {exc}")
                                progress_bar.progress(station_index / max(len(station_batch), 1))

                        for model_type, error_message in all_errors.items():
                            st.warning(f"{model_type}: {error_message}")
                        if not validation_frames or not summary_frames:
                            raise RuntimeError("CNN/TCN/LSTM konnten in keinem Zielmodus trainiert werden.")

                        validation_df = pd.concat(validation_frames, ignore_index=True)
                        summary_table = pd.concat(summary_frames, ignore_index=True)
                        selected_summary_row = summary_table.sort_values(
                            "R² Validierung",
                            ascending=False,
                            na_position="last",
                        ).iloc[0]
                        st.session_state.last_ml_validation_df = validation_df
                        st.session_state.last_ml_summary_table = summary_table
                        st.session_state.last_ml_summary = {
                            "Run": run_label,
                            "Messstelle": selected_summary_row.get("Messstelle", ml_station),
                            "Modell": "Pastas + ML und Nur-ML parallel",
                            "BestesNeuralModell": selected_summary_row["NeuralModel"],
                            "BestesMLRunKey": selected_summary_row.get("MLRunKey"),
                            "Architektur": selected_summary_row.get("Architektur"),
                            "MLModus": selected_summary_row.get("MLModus"),
                            "Fenster": int(ml_window),
                            "Horizont": int(ml_horizon),
                            "Split": "60/20/20",
                            "Features": ", ".join(feature_columns),
                            "Epochen": int(ml_epochs),
                            "Lernrate": float(ml_learning_rate),
                            "HiddenSize": int(ml_hidden_size),
                            "Empfohlene_Epochen": recommendation["epochs"],
                            "Empfohlene_Lernrate": recommendation["learning_rate"],
                            "Empfohlene_HiddenSize": recommendation["hidden_size"],
                            "Datenbasiertes_Epochenlimit": epoch_widget_max,
                            "R2": selected_summary_row["R² Validierung"],
                            "RMSE": selected_summary_row["RMSE Validierung"],
                            "EVP": selected_summary_row["EVP Validierung"],
                            "Test_R2": selected_summary_row["R² Test"],
                            "Test_RMSE": selected_summary_row["RMSE Test"],
                            "Test_EVP": selected_summary_row["EVP Test"],
                            "n_train": int(selected_summary_row["Train"]),
                            "n_test": int(selected_summary_row["Test"]),
                            "n_valid": int(selected_summary_row["Validierung"]),
                        }
                        st.session_state.last_ml_artifacts = artifact_store
                        st.session_state.last_ml_forecast_df = pd.DataFrame()
                        st.session_state.last_ml_impulse_df = pd.DataFrame()
                        if ml_scope == "all":
                            history_rows = build_ml_history_rows(summary_table, run_label, ml_window, ml_horizon)
                            st.session_state.history = append_history_entries(
                                st.session_state.history,
                                history_rows,
                                st.session_state.history_run_limit,
                            )
                        st.rerun()
                    except Exception as exc:
                        st.error(str(exc))

                ml_validation_df = st.session_state.last_ml_validation_df.copy()
                ml_summary = st.session_state.last_ml_summary
                if ml_summary and not ml_validation_df.empty:
                    st.markdown(f"### {t['ml_summary']}")
                    summary_table = st.session_state.last_ml_summary_table.copy()
                    if summary_table.empty and "NeuralModel" in ml_validation_df.columns:
                        summary_rows = []
                        for model_type, model_df in ml_validation_df.groupby("NeuralModel", sort=False):
                            test_df = model_df[model_df["split"].astype(str) == "test"]
                            valid_df = model_df[model_df["split"].astype(str) == "validation"]
                            summary_rows.append(
                                {
                                    "NeuralModel": model_type,
                                    "R² Test": metric_summary(
                                        test_df["observed"].values,
                                        test_df["hybrid_prediction"].values,
                                    )["R2"],
                                    "R² Validierung": metric_summary(
                                        valid_df["observed"].values,
                                        valid_df["hybrid_prediction"].values,
                                    )["R2"],
                                }
                            )
                        summary_table = pd.DataFrame(summary_rows)
                    if not summary_table.empty:
                        st.markdown(f"#### {t['ml_results_table']}")
                        st.dataframe(summary_table, width="stretch", hide_index=True)
                    detail_key_column = "MLRunKey" if "MLRunKey" in summary_table.columns else "NeuralModel"
                    available_ml_keys = (
                        summary_table[detail_key_column].astype(str).tolist()
                        if not summary_table.empty and detail_key_column in summary_table.columns
                        else sorted(ml_validation_df.get("NeuralModel", pd.Series(["ML"])).astype(str).unique())
                    )
                    preferred_key = str(
                        ml_summary.get("BestesMLRunKey")
                        or ml_summary.get("BestesNeuralModell")
                        or available_ml_keys[0]
                    )
                    detail_index = available_ml_keys.index(preferred_key) if preferred_key in available_ml_keys else 0
                    selected_ml_key = st.selectbox(
                        t["ml_model_detail"],
                        available_ml_keys,
                        index=detail_index,
                        key="ml_detail_model",
                    )
                    selected_summary = (
                        summary_table[summary_table[detail_key_column].astype(str) == str(selected_ml_key)]
                        if not summary_table.empty and detail_key_column in summary_table.columns
                        else pd.DataFrame()
                    )
                    selected_summary_row = selected_summary.iloc[0] if not selected_summary.empty else pd.Series(dtype=object)
                    selected_ml_model = str(selected_summary_row.get("NeuralModel", selected_ml_key))
                    if not selected_summary_row.empty:
                        metric_cols = st.columns(6)
                        metric_cols[0].metric(f"R² {t['ml_test']}", f"{float(selected_summary_row.get('R² Test', float('nan'))):.3f}")
                        metric_cols[1].metric(f"RMSE {t['ml_test']}", f"{float(selected_summary_row.get('RMSE Test', float('nan'))):.3f}")
                        metric_cols[2].metric(f"EVP {t['ml_test']}", f"{float(selected_summary_row.get('EVP Test', float('nan'))):.1f}")
                        metric_cols[3].metric(f"R² {t['ml_validation']}", f"{float(selected_summary_row.get('R² Validierung', float('nan'))):.3f}")
                        metric_cols[4].metric(f"RMSE {t['ml_validation']}", f"{float(selected_summary_row.get('RMSE Validierung', float('nan'))):.3f}")
                        metric_cols[5].metric(f"EVP {t['ml_validation']}", f"{float(selected_summary_row.get('EVP Validierung', float('nan'))):.1f}")
                    count_cols = st.columns(3)
                    count_cols[0].metric("Train", int(selected_summary_row.get("Train", ml_summary.get("n_train", 0))))
                    count_cols[1].metric(t["ml_test"], int(selected_summary_row.get("Test", ml_summary.get("n_test", 0))))
                    count_cols[2].metric(t["ml_validation"], int(selected_summary_row.get("Validierung", ml_summary.get("n_valid", 0))))

                    plot_df = ml_validation_df.copy()
                    if "MLRunKey" in plot_df.columns:
                        plot_df = plot_df[plot_df["MLRunKey"].astype(str) == str(selected_ml_key)]
                    elif "NeuralModel" in plot_df.columns:
                        plot_df = plot_df[plot_df["NeuralModel"].astype(str) == str(selected_ml_model)]
                    plot_df["date"] = pd.to_datetime(plot_df["date"], errors="coerce")
                    figure, axis = plt.subplots(figsize=(12, 5))
                    axis.plot(plot_df["date"], plot_df["observed"], label=t["observed"], linewidth=1.2)
                    if "pastas_sim" in plot_df.columns:
                        axis.plot(plot_df["date"], plot_df["pastas_sim"], label="Pastas ohne Flex", linewidth=1.0)
                    prediction_column = "final_prediction" if "final_prediction" in plot_df.columns else "hybrid_prediction"
                    axis.plot(
                        plot_df["date"],
                        plot_df[prediction_column],
                        label=selected_ml_model,
                        linewidth=1.1,
                    )
                    if "split" in plot_df.columns:
                        validation_start = plot_df.loc[
                            plot_df["split"].astype(str) == "validation",
                            "date",
                        ].min()
                        if pd.notna(validation_start):
                            axis.axvline(validation_start, color="black", linewidth=0.9, linestyle="--", alpha=0.6)
                    axis.set_title(f"{ml_summary.get('Messstelle')}: {t['ml_validation']}")
                    axis.set_ylabel(t["head_axis"])
                    axis.grid(alpha=0.3)
                    axis.legend()
                    figure.tight_layout()
                    st.pyplot(figure)
                    plt.close(figure)

                    st.dataframe(plot_df, width="stretch", hide_index=True)

                    summary_station = selected_summary_row.get("Messstelle", ml_summary.get("Messstelle"))
                    ml_summary_station_row = get_station_no_flex_metadata_row(last_run_results, summary_station)
                    ml_artifacts_by_model = normalize_ml_artifacts(st.session_state.get("last_ml_artifacts", {}))
                    ml_artifacts = ml_artifacts_by_model.get(str(selected_ml_key), {})
                    if not ml_artifacts:
                        ml_artifacts = ml_artifacts_by_model.get(selected_ml_model, {})
                    detail_pastas_model = None
                    if ml_summary_station_row is not None:
                        try:
                            detail_pastas_model = build_model_from_result_row(
                                ml_summary_station_row,
                                gw_df,
                                rain,
                                evap,
                            )
                        except Exception:
                            detail_pastas_model = ml_model

                    st.markdown(f"### {t['ml_impulse_heading']}")
                    impulse_cols = st.columns(3)
                    with impulse_cols[0]:
                        impulse_mm = st.number_input(
                            t["ml_impulse_amount"],
                            min_value=0.1,
                            max_value=500.0,
                            value=25.0,
                            step=1.0,
                            help=t["ml_impulse_amount_help"],
                        )
                    with impulse_cols[1]:
                        impulse_days = st.number_input(
                            t["ml_impulse_days"],
                            min_value=30,
                            max_value=3650,
                            value=730,
                            step=30,
                            help=t["ml_impulse_days_help"],
                        )
                    with impulse_cols[2]:
                        compute_impulse = st.button(t["ml_impulse_run"], key="ml_impulse_button")

                    if compute_impulse:
                        impulse_df = predict_impulse_response(
                            ml_artifacts,
                            impulse_mm=float(impulse_mm),
                            response_days=int(impulse_days),
                        )
                        if impulse_df.empty:
                            st.info(t["ml_impulse_no_rows"])
                        else:
                            target_mode = str(ml_artifacts.get("target_mode", "hybrid") or "hybrid").lower()
                            pastas_irf = (
                                create_pastas_impulse_response_df(detail_pastas_model, impulse_mm)
                                if target_mode == "hybrid" and detail_pastas_model is not None
                                else pd.DataFrame()
                            )
                            if target_mode != "hybrid":
                                impulse_df["pastas_irf_response"] = 0.0
                            elif pastas_irf.empty:
                                impulse_df["pastas_irf_response"] = np.nan
                            else:
                                impulse_df["pastas_irf_response"] = np.interp(
                                    impulse_df["lag_days"].astype(float),
                                    pastas_irf["lag_days"].astype(float),
                                    pastas_irf["pastas_irf_response"].astype(float),
                                    left=0.0,
                                    right=0.0,
                                )
                            impulse_df["hybrid_response"] = (
                                impulse_df["pastas_irf_response"].fillna(0.0)
                                + impulse_df["ml_residual_response"]
                            )
                            impulse_df["station"] = summary_station
                            impulse_df["model"] = selected_ml_model
                            impulse_df["target_mode"] = target_mode
                            impulse_df["evap"] = 0.0
                            st.session_state.last_ml_impulse_df = impulse_df
                            st.rerun()

                    ml_impulse_df = st.session_state.last_ml_impulse_df.copy()
                    if not ml_impulse_df.empty and "station" in ml_impulse_df.columns:
                        ml_impulse_df = ml_impulse_df[
                            ml_impulse_df["station"].astype(str) == str(summary_station)
                        ]
                    if not ml_impulse_df.empty and "model" in ml_impulse_df.columns:
                        ml_impulse_df = ml_impulse_df[
                            ml_impulse_df["model"].astype(str) == str(selected_ml_model)
                        ]
                    if not ml_impulse_df.empty:
                        figure, axis = plt.subplots(figsize=(12, 5))
                        if (
                            "target_mode" not in ml_impulse_df.columns
                            or (ml_impulse_df["target_mode"].astype(str) == "hybrid").any()
                        ):
                            axis.plot(
                                ml_impulse_df["lag_days"],
                                ml_impulse_df["pastas_irf_response"],
                                label="Pastas IRF ohne Flex",
                                linewidth=1.2,
                            )
                        axis.plot(
                            ml_impulse_df["lag_days"],
                            ml_impulse_df["ml_residual_response"],
                            label="ML-Residuum" if str(ml_artifacts.get("target_mode", "hybrid")) == "hybrid" else "ML-Antwort",
                            linewidth=1.1,
                        )
                        axis.plot(
                            ml_impulse_df["lag_days"],
                            ml_impulse_df["hybrid_response"],
                            label=selected_ml_model,
                            linewidth=1.1,
                        )
                        axis.axhline(0.0, color="black", linewidth=0.8, alpha=0.7)
                        axis.set_xlabel(t["days_axis"])
                        axis.set_ylabel(t["response_axis"])
                        axis.grid(alpha=0.3)
                        axis.legend()
                        figure.tight_layout()
                        st.pyplot(figure)
                        plt.close(figure)
                        st.dataframe(ml_impulse_df, width="stretch", hide_index=True)
                        st.download_button(
                            t["ml_download_impulse"],
                            data=ml_results_to_csv_bytes(sanitize_export_df(ml_impulse_df)),
                            file_name=f"{summary_station}_ml_impulse_response.csv",
                            mime="text/csv",
                        )

                    ml_future_df = st.session_state.last_ml_forecast_df.copy()
                    if not ml_future_df.empty and "MLRunKey" in ml_future_df.columns:
                        ml_future_df = ml_future_df[
                            ml_future_df["MLRunKey"].astype(str) == str(selected_ml_key)
                        ]
                    elif not ml_future_df.empty and "NeuralModel" in ml_future_df.columns:
                        ml_future_df = ml_future_df[
                            ml_future_df["NeuralModel"].astype(str) == str(selected_ml_model)
                        ]
                    forecast_source_df = st.session_state.last_forecast_df.copy()
                    if (
                        ml_artifacts
                        and summary_station
                        and ml_artifacts.get("station") == summary_station
                        and not forecast_source_df.empty
                    ):
                        forecast_source_df["date"] = pd.to_datetime(forecast_source_df["date"], errors="coerce")
                        forecast_source_df = forecast_source_df.dropna(subset=["date"])
                        if "station" in forecast_source_df.columns:
                            forecast_source_df = forecast_source_df[
                                forecast_source_df["station"].astype(str) == str(summary_station)
                            ]
                        if not forecast_source_df.empty:
                            forecast_frame = build_hybrid_forecast_feature_frame(
                                summary_station,
                                gw_df,
                                forecast_source_df,
                            )
                            residual_forecast_df = predict_hybrid_residuals(forecast_frame, ml_artifacts)
                            if not residual_forecast_df.empty:
                                ml_future_df = forecast_source_df.merge(
                                    residual_forecast_df,
                                    on="date",
                                    how="left",
                                )
                                target_mode = str(ml_artifacts.get("target_mode", "hybrid") or "hybrid").lower()
                                if target_mode == "hybrid":
                                    ml_future_df["final_prediction"] = (
                                        ml_future_df["simulated_head"] + ml_future_df["ml_residual_pred"]
                                    )
                                else:
                                    direct_column = (
                                        "ml_direct_pred"
                                        if "ml_direct_pred" in ml_future_df.columns
                                        else "ml_prediction"
                                    )
                                    ml_future_df["final_prediction"] = ml_future_df[direct_column]
                                ml_future_df["hybrid_prediction"] = ml_future_df["final_prediction"]
                                if "is_forecast" in ml_future_df.columns:
                                    ml_future_df = ml_future_df[
                                        ml_future_df["is_forecast"].apply(coerce_bool)
                                    ]
                                ml_future_df = ml_future_df.dropna(subset=["final_prediction"])
                                ml_future_df["NeuralModel"] = selected_ml_model
                                ml_future_df["MLRunKey"] = selected_ml_key
                                ml_future_df["MLModus"] = (
                                    t["ml_target_direct"] if target_mode == "direct" else t["ml_target_hybrid"]
                                )
                                st.session_state.last_ml_forecast_df = ml_future_df

                    if not ml_future_df.empty:
                        st.markdown(f"### {t['ml_future_heading']}")
                        future_plot_df = ml_future_df.copy()
                        future_plot_df["date"] = pd.to_datetime(future_plot_df["date"], errors="coerce")
                        if "final_prediction" not in future_plot_df.columns and "hybrid_prediction" in future_plot_df.columns:
                            future_plot_df["final_prediction"] = future_plot_df["hybrid_prediction"]
                        figure, axis = plt.subplots(figsize=(12, 5))
                        axis.plot(
                            future_plot_df["date"],
                            future_plot_df["simulated_head"],
                            label="Pastas ohne Flex",
                            linewidth=1.0,
                        )
                        axis.plot(
                            future_plot_df["date"],
                            future_plot_df["final_prediction"],
                            label=selected_ml_model,
                            linewidth=1.1,
                        )
                        axis.set_title(f"{summary_station}: {t['ml_future_heading']}")
                        axis.set_ylabel(t["head_axis"])
                        axis.grid(alpha=0.3)
                        axis.legend()
                        figure.tight_layout()
                        st.pyplot(figure)
                        plt.close(figure)
                        st.dataframe(future_plot_df, width="stretch", hide_index=True)
                        st.download_button(
                            t["ml_download_future"],
                            data=ml_results_to_csv_bytes(sanitize_export_df(future_plot_df)),
                            file_name=f"{summary_station}_ml_forecast.csv",
                            mime="text/csv",
                        )
                    elif forecast_source_df.empty:
                        st.info(t["ml_future_needs_forecast"])
                    elif ml_artifacts and ml_artifacts.get("station") == summary_station:
                        st.info(t["ml_future_no_rows"])

                    st.download_button(
                        t["ml_download_validation"],
                        data=ml_results_to_csv_bytes(plot_df),
                        file_name=f"{ml_summary.get('Messstelle')}_ml_validation.csv",
                        mime="text/csv",
                    )
                    st.download_button(
                        t["ml_download_package"],
                        data=create_ml_run_package(
                            validation_df=ml_validation_df,
                            metadata=ml_summary,
                            forecast_df=ml_future_df,
                            impulse_df=st.session_state.get("last_ml_impulse_df", pd.DataFrame()),
                            artifacts=ml_artifacts_by_model,
                            summary_df=summary_table,
                        ),
                        file_name=f"{ml_summary.get('Messstelle')}_ml_run.gwml",
                        mime="application/zip",
                    )

                    if st.button(t["ml_add_history"], key="ml_add_history_button"):
                        ml_history_row = {
                            "Run": ml_summary.get("Run"),
                            "Messstelle": ml_summary.get("Messstelle"),
                            "Modus": "ML",
                            "Suchmodus": "Pastas + ML und Nur-ML",
                            "Konfiguration": (
                                f"{selected_ml_model} | "
                                f"{ml_summary.get('Fenster')}d -> {ml_summary.get('Horizont')}d"
                            ),
                            "Modell": selected_ml_model,
                            "Flex": False,
                            "Noise": None if ml_summary_station_row is None else ml_summary_station_row.get("Noise"),
                            "Cutoff": None if ml_summary_station_row is None else ml_summary_station_row.get("Cutoff"),
                            "NoiseNorm": None if ml_summary_station_row is None else ml_summary_station_row.get("NoiseNorm"),
                            "n_obs": len(plot_df),
                            "Status": "ok",
                            "BestStationModel": True,
                            "R2": selected_summary_row.get("R² Validierung", ml_summary.get("R2")),
                            "RMSE": selected_summary_row.get("RMSE Validierung", ml_summary.get("RMSE")),
                            "EVP": selected_summary_row.get("EVP Validierung", ml_summary.get("EVP")),
                            "AIC": None,
                            "Fehler": None,
                        }
                        st.session_state.history = append_history_entries(
                            st.session_state.history,
                            pd.DataFrame([ml_history_row]),
                            st.session_state.history_run_limit,
                        )
                        st.success(t["ml_added_history"])


if active_main_view == "save":
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

    if last_run_models:
        st.markdown(f"### {t['simulated_all_heading']}")
        all_simulated_df = build_all_simulated_timeseries_df(last_run_models)
        if not all_simulated_df.empty:
            all_simulated_csv = (
                sanitize_export_df(all_simulated_df)
                .to_csv(index=False, sep=";")
                .encode("utf-8")
            )
            st.download_button(
                t["simulated_all_download"],
                data=all_simulated_csv,
                file_name="simulated_timeseries_all_stations.csv",
                mime="text/csv",
            )

    imported_file = st.file_uploader(t["import_csv"], type=["csv"], key="import_results")
    if imported_file is not None and st.button(t["import_button"], key="import_results_button"):
        imported_df = read_imported_results(imported_file)
        imported_last_run, imported_run_label = get_last_imported_run(imported_df)
        st.session_state.history = append_history_entries(
            st.session_state.history,
            imported_df,
            st.session_state.history_run_limit,
        )
        st.session_state.last_run_results = imported_last_run
        st.session_state.last_run_label = imported_run_label
        st.session_state.last_run_models = rebuild_models_from_results(
            imported_last_run,
            gw_df,
            rain,
            evap,
        )
        st.session_state.last_extra_match = {}
        st.session_state.last_coord_match = {}
        st.session_state.last_forecast_df = pd.DataFrame()
        st.session_state.last_forecast_info = {}
        st.session_state.last_ml_validation_df = pd.DataFrame()
        st.session_state.last_ml_forecast_df = pd.DataFrame()
        st.session_state.last_ml_impulse_df = pd.DataFrame()
        st.session_state.last_ml_summary_table = pd.DataFrame()
        st.session_state.last_ml_summary = {}
        st.session_state.last_ml_artifacts = {}
        st.session_state.history_notice = ("success", t["import_success"])
        st.rerun()

