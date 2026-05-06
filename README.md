<!-- Project lead / concept credit: Robin Carow / RCnet -->
<!-- Scientific modeling framework credit: Pastas by Collenteur et al. (2019), https://doi.org/10.1111/gwat.12925 -->

# GW Analyzer CNN/LSTM Test Beta

Separate Arbeitskopie der Streamlit-Anwendung zur Analyse von Grundwasser-Zeitreihen mit [Pastas](https://pastas.readthedocs.io/latest/) und ersten neuronalen Hybridmodellen.

Diese Variante ist die neue GitHub-Beta `CNN/LSTM Test`. Die bestehende alte Beta bleibt davon getrennt und wird nicht gelöscht.

## Projekt und Credits

- Projektkonzept und App-Entwicklung: Robin Carow / RCnet
- Die hydrologische Basismodellierung basiert auf [Pastas](https://pastas.readthedocs.io/latest/), dem Open-Source-Pythonpaket für hydro(geo)logische Zeitreihenanalysen
- Wissenschaftliche Referenz für Pastas:
  [Collenteur, R.A., Calje, R., Klop, S.A., Schaars, F., & Bakker, M. (2019). *Pastas: open source software for the analysis of groundwater time series*. Groundwater.](https://doi.org/10.1111/gwat.12925)
- Quellcode und Community:
  [Pastas auf GitHub](https://github.com/pastas/pastas)

## Installation

```bash
python -m pip install -r requirements.txt
```

Diese CNN/LSTM-Test-Beta nutzt zusätzlich `torch>=2.3.0` für LSTM und CNN. Ohne PyTorch startet die App weiterhin, der Tab `ML Forecast (Beta)` zeigt dann aber einen Hinweis zur Installation.

Die App nutzt `st.fragment` für das leichtere Batch-Monitoring und sollte deshalb mit `Streamlit >= 1.50` betrieben werden.

## Start

```bash
python -m streamlit run app.py
```

## CNN/LSTM Test Beta

Der neue Tab `ML Forecast (Beta)` setzt auf einen Pastas + ML Hybrid:

1. Pastas berechnet das interpretierbare Grundmodell ohne FlexModel.
2. Die App berechnet das Residuum: `Beobachtung - Pastas-Simulation`.
3. Ein neuronales Modell lernt dieses Residuum nur aus Wetterfeatures.
4. Die Hybrid-Ausgabe wird als `Pastas + ML-Residuum` dargestellt.

Aktuell enthalten:

- paralleles Training von `CNN` und `LSTM` als zwei unabhängige Vergleichsmodelle
- Trainingsfenster: 1, 3, 5, 10, 15 oder 20 Jahre
- Vorhersagehorizont: 1, 7, 30, 90, 180 Tage sowie 1, 2, 5 oder 10 Jahre
- Feature-Auswahl nur für Niederschlag/Verdunstung und rollierende Wetterfenster
- keine vorherigen Grundwasserstände als Features
- keine Jahreszeit-Features (`Sin/Cos`)
- fester zeitbasierter Split: erste 60 % Training, nächste 20 % Test, letzte 20 % Validierung
- Vergleichstabelle mit `R²`, `RMSE` und `EVP` getrennt für Test und Validierung
- Plot `Beobachtet vs. Pastas vs. Hybrid`
- ML-Impulsantwort für ein einmaliges stärkeres Niederschlagsereignis mit Verdunstung auf 0
- Vergleich der ML-Reaktion mit der Pastas-Impulse-Response-Function ohne FlexModel
- CSV-Export der Test-/Validierungsdaten und der Impulsantwort
- `.gwml`-Export mit Manifest, Test-/Validierung, optionalem Hybrid-Forecast, Impulsantwort und trainiertem Modellzustand
- Übernahme des ML-Ergebnisses in die normale Laufhistorie

Für einen echten Zukunfts-Hybrid zuerst im normalen `Forecast`-Tab ein Pastas-Szenario für dieselbe Messstelle berechnen. Danach ergänzt der ML-Tab diesen Forecast um das gelernte Residuum und exportiert die Hybrid-Prognose separat.

## Feature-Dokumentation

Das Vergleichssetup ist bewusst restriktiv, damit CNN/LSTM fair mit dem PASTAS-Modell ohne FlexModel verglichen werden können.

- `rain`: täglicher Niederschlag aus der Wetterdatei.
- `evap`: tägliche Verdunstung bzw. Evapotranspiration aus der Wetterdatei.
- `rain_sum_7`, `rain_sum_30`, `rain_sum_90`: rollierende Niederschlagssummen über 7, 30 und 90 Tage. Diese Features geben dem neuronalen Modell Informationen über Regencluster, Feuchtephasen und längere Trockenperioden.
- `evap_mean_7`, `evap_mean_30`, `evap_mean_90`: rollierende Mittelwerte der Verdunstung über 7, 30 und 90 Tage. Diese Features beschreiben die Verdunstungsbelastung auf kurzen und längeren Zeitskalen.
- keine Grundwasser-Lags: frühere Grundwasserstände werden nicht als Eingabe genutzt, damit das ML-Modell nicht einfach die Persistenz der Zeitreihe lernt.
- keine Saison-Sin/Cos-Features: Jahreszeit wird nicht direkt eingespeist, damit der Vergleich stärker auf Wetterreaktionen fokussiert bleibt.

Die Modellgüte kann durch diese Einschränkungen sichtbar niedriger ausfallen. Das ist hier gewollt: Ziel ist nicht der maximale R²-Wert, sondern die Analyse, ob CNN/LSTM unter vergleichbaren Restriktionen eine plausible Response-Struktur lernen.

## ML-Modulstruktur

- `ml_features.py`: Feature-Erzeugung, Sequenzfenster und Forecast-Feature-Frames
- `ml_models.py`: LSTM/CNN-Modelle auf PyTorch-Basis
- `ml_training.py`: Training, Skalierung, zeitbasierte Splits, Validierung und Residuenprognose
- `ml_export.py`: Export/Import-Helfer für ML-Laufpakete und Modellzustände

## Bestehende Beta-Funktionen

- Auswahlmodus für `einzelne`, `mehrere` oder `alle` Messstellen
- Automatische Best-Model-Suche über `Gamma/Exponential/Hantush`, `Flex/ohne Flex` und `Noise/ohne Noise`
- Ausschluss einzelner Modellkombinationen aus der Best-Model-Suche
- Optionale Startparameter-Suche für eine Station bzw. je ausgewählte Station
- Hintergrund-Batchlauf mit Fortschrittsanzeige und Abbruchfunktion
- R2-Filter, Bestmodell-Filter und Laufvergleich mehrerer Runs nebeneinander
- Projektpaket-Import/-Export direkt auf der Startansicht, auch ohne vorher geladene Daten
- Projektpakete enthalten Originaldateien, Ergebnisse, Forecasts, ML-Ergebnisse und App-Einstellungen
- CSV-Reimport stellt Historie, letzten Lauf und rekonstruierbare Modellplots wieder her
- Forecast-Tab für Zukunftsszenarien mit skaliertem Niederschlag/Verdunstung, additiven Offsets und nicht-negativem Clipping
- Optionaler Open-Meteo-Abruf für aktuelle Tageswerte mit `precipitation_sum` und `et0_fao_evapotranspiration`
- Diagnose-Tab mit Bias, Residuenstreuung, Lag-1-Autokorrelation, saisonalem Fehler und Response-Clustern
- Visualisierung der Impulse Response Function zusätzlich zur Pastas-Step-Function
- Kartenansicht auf Basis einer hochladbaren Koordinatendatei mit `Messstelle/Site`, `Lat`, `Lon`
- Export-Schutz gegen Spreadsheet-Formel-Injection und HTML-Kurzreport
- Remote-Datenabruf per direkter öffentlicher URL für CSV/XLSX/ODS/ZIP
- Accessibility-Optionen für lesefreundlichere Darstellung, größere Schrift, hohen Kontrast und Fokusmarkierung

## Erwartete Daten

- Grundwasserdatei: braucht eine Datums-Spalte (`Datum` oder `Date`) und Stationsspalten
- Wetterdatei: braucht eine Datums-Spalte sowie Spalten für Niederschlag und Verdunstung
- Zusatzdatei: wird stationsweise per `Site` oder `Messstelle` mit den Modell-Ergebnissen verknüpft
- Koordinatendatei für die Karte: braucht `Messstelle` oder `Site` sowie `Lat` und `Lon`
- Remote-Dateiquellen: direkte öffentliche URLs zu CSV/XLSX/ODS oder ZIP-Dateien mit einer passenden Tabelle

## Hinweise

- Der temporäre Upload-Cache wird lokal im App-Ordner unter `.upload_cache` abgelegt und kann über die Sidebar wieder geleert werden.
- Nach einem Reload oder Sprachwechsel können Datei-Inputs optisch leer wirken, obwohl die App die Quellen bereits aus dem Temp-Cache weiterverwendet. Die aktive Nutzung wird in der Sidebar angezeigt.
- Vollständige Projektstände sollten als `.gwproject` exportiert werden. Der reine CSV-Export ist weiterhin für Tabellenvergleiche gedacht und enthält keine Originaldateien oder UI-Einstellungen.
- ML-Training ist bewusst als Beta markiert. Besonders LSTM reagiert empfindlich auf kurze Zeitreihen, fehlende Werte und den festen 60/20/20-Split.
- Das CNN ist als schnelle Conv1D-Baseline gedacht und eignet sich zuerst für lokale Muster wie Regencluster, Trockenphasen und verzögerte Reaktionen.
- Der Hybridansatz bleibt hydrologisch besser erklärbar als ein reines Black-Box-Modell, weil Pastas das Grundsignal liefert und ML nur die Reststruktur lernt.

## Docker

Die Beta-Kopie kann direkt als Container gestartet werden:

```bash
docker build -t gw-analyzer-nn-beta .
docker run --rm -p 8501:8501 gw-analyzer-nn-beta
```

Danach ist die App im Browser unter `http://localhost:8501` erreichbar.

## Verwendete Bibliotheken

- [Streamlit](https://streamlit.io/)
- [Pandas](https://pandas.pydata.org/)
- [NumPy](https://numpy.org/)
- [Matplotlib](https://matplotlib.org/)
- [Pastas](https://pastas.readthedocs.io/latest/)
- [PyTorch](https://pytorch.org/)
- [OpenPyXL](https://openpyxl.readthedocs.io/)
- [odfpy](https://pypi.org/project/odfpy/)
- [Requests](https://requests.readthedocs.io/)
- [Open-Meteo Forecast API](https://open-meteo.com/en/docs) für optionale aktuelle Wettervorhersagewerte
- [PyDeck](https://deckgl.readthedocs.io/) für die erweiterte Kartenansicht

## Nächste ML-Schritte

- Wiederladen einzelner `.gwml`-Pakete direkt im ML-Tab ergänzen.
- TCN als robustere CNN-Variante testen, weil dilatierte Faltungen längere Reaktionszeiten besser abbilden können.
- Für die Impulsantwort zusätzlich mehrere Impulsstärken testen, z. B. 10, 25, 50 und 100 mm, um Nichtlinearität sichtbar zu machen.
- Response-Vergleich als Kennzahlen ergänzen: Peak-Zeit, Halbwertszeit, Gesamtfläche und Vorzeichenwechsel.
- Stationen nach Response-Typ clustern und prüfen, ob CNN/LSTM ähnliche Gruppen wie PASTAS erzeugen.
- Globales Modell über mehrere Messstellen erst später vorbereiten, aber weiterhin ohne Grundwasser-Lags und ohne Saison-Sin/Cos, falls der Vergleich so bleiben soll.
- Zusatzdaten wie CPC/PC-Spalten, Koordinaten oder Standortinformationen nur als separaten Experimentmodus einbinden, nicht in den strengen PASTAS-Vergleich.
- Saubere Testdaten und feste Benchmarks definieren, damit CNN/LSTM fair gegen PASTAS und den Hybrid verglichen werden.
- Optional eine Ergebnis-Notiz je Lauf speichern, damit Hypothese, Einschränkungen und Interpretation direkt mit dem Export archiviert werden.
