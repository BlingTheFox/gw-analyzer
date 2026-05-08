<!-- Project lead / concept credit: Robin Carow / RCnet -->
<!-- Scientific modeling framework credit: Pastas by Collenteur et al. (2019), https://doi.org/10.1111/gwat.12925 -->

# GW Analyzer

Streamlit-Anwendung zur Analyse von Grundwasser-Zeitreihen mit [Pastas](https://pastas.readthedocs.io/latest/), Forecast-Szenarien, Projektpaketen und neuronalen CNN/TCN/LSTM-Vergleichsmodellen.

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

Für CNN, TCN und LSTM wird `torch>=2.3.0` genutzt. Ohne PyTorch startet die App weiterhin, der Bereich `ML Forecast` zeigt dann einen Installationshinweis.

Die App nutzt `st.fragment` für das leichtere Batch-Monitoring und sollte deshalb mit `Streamlit >= 1.50` betrieben werden.

## Start

```bash
python -m streamlit run app.py
```

## Kernfunktionen

- Auswahlmodus für einzelne, mehrere oder alle Messstellen
- Automatische Best-Model-Suche über `Gamma`, `Exponential`, `Hantush`, FlexModel und NoiseModel
- Ausschluss einzelner Modellkombinationen aus der automatischen Suche
- Optionale Startparameter-Suche für eine Station bzw. je ausgewählte Station
- Hintergrund-Batchlauf mit Fortschrittsanzeige und Abbruchfunktion
- R2-Filter, Bestmodell-Filter und Laufvergleich mehrerer Runs nebeneinander
- Forecast-Tab für Zukunftsszenarien mit skaliertem Niederschlag/Verdunstung, additiven Offsets und nicht-negativem Clipping
- Optionaler Open-Meteo-Abruf für aktuelle Tageswerte mit `precipitation_sum` und `et0_fao_evapotranspiration`
- Diagnose-Tab mit Bias, Residuenstreuung, Lag-1-Autokorrelation, saisonalem Fehler und Response-Clustern
- Visualisierung der Impulse Response Function zusätzlich zur Pastas-Step-Function
- Kartenansicht auf Basis einer Koordinatendatei mit `Messstelle`/`Site`, `Lat`, `Lon`
- Remote-Datenabruf per direkter öffentlicher URL für CSV/XLSX/ODS/ZIP
- Accessibility-Optionen für lesefreundlichere Darstellung, größere Schrift, hohen Kontrast und Fokusmarkierung

## Projektpakete

Der Import/Export oben auf der Startansicht ist der vollständige Projekttransport. Er funktioniert auch ohne vorher geladene Daten.

Ein `.gwproject` enthält:

- Originaldateien bzw. gecachte Uploads für Grundwasser, Wetter, Zusatzdaten und Koordinaten
- Remote-Quellen-Metadaten inklusive ursprünglicher API-/Open-Data-Links, soweit vorhanden
- Pastas-Historie, letzten Lauf, rekonstruierbare Modellplots und Modellzustände
- Forecast-Ergebnisse und Forecast-Einstellungen
- ML-/Hybrid-Ergebnisse, direkte Nur-ML-Ergebnisse, Impulsantworten und Modellzustände
- App-Einstellungen wie Sprache, Ansicht, Modellwahl, Parameter, ML-Fenster, Horizonte, Epochen, Lernrate und Ressourcenoptionen
- zuletzt gespeicherte ML-Stationspakete aus `.ml_run_cache`, damit lange All-Station-Läufe nach Abstürzen weiter nutzbar bleiben

Der CSV-Export im Bereich `Speichern und Laden` bleibt für Tabellenvergleiche erhalten. Für vollständige Sicherungen und Umzüge immer `.gwproject` verwenden.

## CNN/TCN/LSTM

Der Bereich `ML Forecast` vergleicht Pastas + ML-Hybridmodelle mit direkten Nur-ML-Modellen:

1. Pastas berechnet das interpretierbare Grundmodell ohne FlexModel.
2. Die App berechnet das Residuum: `Beobachtung - Pastas-Simulation`.
3. CNN, TCN oder LSTM lernen dieses Residuum nur aus Wetterfeatures.
4. Die Hybrid-Ausgabe wird als `Pastas + ML-Residuum` dargestellt.
5. Zusätzlich trainiert die App direkte Varianten `Nur CNN`, `Nur TCN` und `Nur LSTM`, die den Grundwasserstand direkt aus denselben Wetterfeatures vorhersagen.

Enthalten sind:

- unabhängiges Training von `CNN`, `TCN` und `LSTM`
- pro Architektur zwei Zielmodi: `Pastas + ML` und `Nur ML`
- TCN mit dilatierten Faltungen für längere Reaktionsmuster
- Trainingsfenster: 1, 3, 5, 10, 15, 20 oder 30 Jahre
- Vorhersagehorizont: 1, 7, 30, 90, 180 Tage sowie 1, 2, 5, 10 oder 20 Jahre
- Trainings-Epochen bis maximal 1000
- datenbasiertes Epochen-Limit und Startempfehlungen für Epochen, Lernrate und Hidden Size/Filter
- Feature-Auswahl für Niederschlag/Verdunstung und rollierende Wetterfenster
- keine vorherigen Grundwasserstände als Features
- keine Jahreszeit-Features (`Sin/Cos`)
- fester zeitbasierter 60/20/20-Split für Training, Zwischenprüfung und Validierung
- Vergleichstabelle mit `R²`, `RMSE` und `EVP`
- Plot `Beobachtet vs. Pastas vs. ausgewählte ML-Variante`
- ML-Impulsantwort für ein einmaliges stärkeres Niederschlagsereignis mit Verdunstung auf 0
- Vergleich der ML-Reaktion mit der Pastas-Impulse-Response-Function ohne FlexModel
- CSV-Export, `.gwml`-Export und `.gwml`-Import für ML-Läufe
- All-Station-ML-Lauf mit stationsweisem Zwischenspeichern in `.ml_run_cache`
- Übernahme des ML-Ergebnisses in die normale Laufhistorie

Für einen echten Zukunftsforecast zuerst im normalen `Forecast`-Bereich ein Pastas-Szenario für dieselbe Messstelle berechnen. Danach ergänzt der ML-Bereich diesen Forecast entweder um das gelernte Residuum oder zeigt die direkte Nur-ML-Prognose.

Ausführliche Begriffserklärungen und Bedienhinweise stehen in [ML_BEGRIFFE_IN_DEPTH.md](ML_BEGRIFFE_IN_DEPTH.md).

## Feature-Dokumentation

Das Vergleichssetup ist bewusst restriktiv, damit CNN/TCN/LSTM fair mit dem PASTAS-Modell ohne FlexModel verglichen werden können.

- `rain`: täglicher Niederschlag aus der Wetterdatei
- `evap`: tägliche Verdunstung bzw. Evapotranspiration aus der Wetterdatei
- `rain_sum_7`, `rain_sum_30`, `rain_sum_90`: rollierende Niederschlagssummen über 7, 30 und 90 Tage
- `evap_mean_7`, `evap_mean_30`, `evap_mean_90`: rollierende Mittelwerte der Verdunstung über 7, 30 und 90 Tage
- keine Grundwasser-Lags: frühere Grundwasserstände werden nicht als Eingabe genutzt
- keine Saison-Sin/Cos-Features: Jahreszeit wird nicht direkt eingespeist

Die Modellgüte kann durch diese Einschränkungen sichtbar niedriger ausfallen. Das ist hier gewollt: Ziel ist nicht der maximale R²-Wert, sondern die Analyse, ob CNN/TCN/LSTM unter vergleichbaren Restriktionen plausible Response-Strukturen lernen.

## Sicherheit und Performance

- Remote-Dateien sind auf direkte öffentliche HTTP/HTTPS-URLs begrenzt; lokale, private, reservierte und umleitende Ziele werden abgelehnt.
- Remote-Downloads und ZIP-Inhalte haben Größenlimits. Importpakete werden auf Pfadsicherheit, Einzeldateigröße, Gesamtgröße und Anzahl der Dateien geprüft.
- Tabellenexporte werden gegen Spreadsheet-Formel-Injection abgesichert.
- ML-Pakete laden Modellgewichte mit PyTorchs sichererem `weights_only`-Pfad, sofern die installierte PyTorch-Version ihn unterstützt.
- All-Station-ML läuft stationsweise und speichert nach jeder erfolgreichen Station ein `.gwml`-Paket im lokalen `.ml_run_cache`.
- CPU-Threads werden für ML-Läufe begrenzt, damit das System während langer Läufe reaktionsfähig bleibt.
- `.upload_cache`, `.ml_run_cache`, Streamlit-Logs und virtuelle Umgebungen sind in `.gitignore` und `.dockerignore` ausgeschlossen.

## Erwartete Daten

- Grundwasserdatei: braucht eine Datums-Spalte (`Datum` oder `Date`) und Stationsspalten
- Wetterdatei: braucht eine Datums-Spalte sowie Spalten für Niederschlag und Verdunstung
- Zusatzdatei: wird stationsweise per `Site` oder `Messstelle` mit den Modell-Ergebnissen verknüpft
- Koordinatendatei für die Karte: braucht `Messstelle` oder `Site` sowie `Lat` und `Lon`
- Remote-Dateiquellen: direkte öffentliche URLs zu CSV/XLSX/ODS oder ZIP-Dateien mit einer passenden Tabelle

## Modulstruktur

- `app.py`: Streamlit-Oberfläche, Pastas-Workflows, Import/Export, Forecasts und Visualisierung
- `ml_features.py`: Feature-Erzeugung, Sequenzfenster und Forecast-Feature-Frames
- `ml_models.py`: CNN/TCN/LSTM-Modelle auf PyTorch-Basis und CPU-Ressourcensteuerung
- `ml_training.py`: Training, Skalierung, zeitbasierte Splits, Validierung und Residuenprognose
- `ml_export.py`: Export/Import-Helfer für ML-Laufpakete und Modellzustände

## Docker

Die App kann direkt als Container gestartet werden:

```bash
docker build -t gw-analyzer .
docker run --rm -p 8501:8501 gw-analyzer
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
- [Open-Meteo Forecast API](https://open-meteo.com/en/docs)
- [PyDeck](https://deckgl.readthedocs.io/)

## Weitere sinnvolle Verbesserungen

- ML-Impulsantwort um mehrere Impulsstärken ergänzen, z. B. 10, 25, 50 und 100 mm
- Response-Vergleich als Kennzahlen ergänzen: Peak-Zeit, Halbwertszeit, Gesamtfläche und Vorzeichenwechsel
- Stationen nach Response-Typ clustern und mit Pastas-Response-Gruppen vergleichen
- Globales Modell über mehrere Messstellen vorbereiten, weiterhin getrennt vom strengen Pastas-Vergleich
- Zusatzdaten wie CPC/PC-Spalten, Koordinaten oder Standortinformationen als separaten Experimentmodus einbinden
- Ergebnis-Notiz je Lauf speichern, damit Hypothese, Einschränkungen und Interpretation direkt mit dem Export archiviert werden
