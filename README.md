<!-- Project lead / concept credit: Robin Carow / RCnet -->
<!-- Scientific modeling framework credit: Pastas by Collenteur et al. (2019), https://doi.org/10.1111/gwat.12925 -->

# GW Analyzer Beta

Beta-Arbeitskopie der Streamlit-Anwendung zur Analyse von Grundwasser-Zeitreihen mit [Pastas](https://pastas.readthedocs.io/latest/).

## Projekt und Credits

- Projektkonzept und App-Entwicklung: Robin Carow / RCnet
- Die Modellierung basiert auf [Pastas](https://pastas.readthedocs.io/latest/), dem Open-Source-Pythonpaket fuer hydro(geo)logische Zeitreihenanalysen
- Wissenschaftliche Referenz fuer Pastas:
  [Collenteur, R.A., Calje, R., Klop, S.A., Schaars, F., & Bakker, M. (2019). *Pastas: open source software for the analysis of groundwater time series*. Groundwater.](https://doi.org/10.1111/gwat.12925)
- Quellcode und Community:
  [Pastas auf GitHub](https://github.com/pastas/pastas)

## Installation

```bash
python -m pip install -r requirements.txt
```

Die aktuelle Beta nutzt `st.fragment` fuer das leichtere Batch-Monitoring und sollte deshalb mit `Streamlit >= 1.50` betrieben werden.

## Start

```bash
python -m streamlit run app.py
```

## Neue Beta-Funktionen

- Auswahlmodus fuer `einzelne`, `mehrere` oder `alle` Messstellen
- Automatische Best-Model-Suche ueber `Gamma/Exponential`, `Flex/ohne Flex` und `Noise/ohne Noise`
- Hintergrund-Batchlauf mit Fortschrittsanzeige und Abbruchfunktion
- Persistenter Batch-Status ueber Streamlit-Reruns, damit `Start batch` sichtbar anlaeuft
- Leichteres Batch-Monitoring ohne blockierende Sleep-Schleife waehrend eines aktiven Laufs
- Temporaerer Upload-Cache per Knopfdruck, damit Uploads und URL-Quellen nach Reloads weiter genutzt werden koennen
- Automatische Temp-Sicherung der aktuellen Quellen vor einem Sprachwechsel, damit der Datenstand nicht verloren geht
- Sichtbare Anzeige aktiver Cache-Quellen in der Sidebar, damit klar bleibt, welche Daten gerade verwendet werden
- Sichtbare Zusammenfassung der aktuell genutzten Quellen in der Sidebar, auch wenn File-Inputs nach einem Reload optisch leer wirken
- Button zum erzwungenen Neuladen der Datenquellen fuer Remote-/API-Workflows
- R2-Filter, Bestmodell-Filter und Laufvergleich mehrerer Runs nebeneinander
- Kompakte Standard-Ergebnistabelle mit optionaler erweiterter Ansicht
- Export der Historie als CSV und als Excel mit mehreren Sheets
- Historie-Steuerung mit Limit fuer gespeicherte Laeufe und eigenem Leeren-Knopf
- Optionale Kartenansicht auf Basis einer hochladbaren Koordinatendatei mit `Messstelle/Site`, `Lat`, `Lon`
- Pulse-Response-Visualisierung zusaetzlich zur Pastas-Schrittantwort
- Forecast-Tab fuer Zukunftsszenarien mit skaliertem Niederschlag/Verdunstung, additiven Offsets und nicht-negativem Clipping
- Optionaler Open-Meteo-Abruf fuer aktuelle Tageswerte mit `precipitation_sum` und `et0_fao_evapotranspiration`, parametrisierbar ueber Koordinaten, Zeitzone und Forecast-Tage
- Wetterantrieb im Forecast als eigenes Diagramm und als exportierbare Spalten im Forecast-CSV
- Szenario-Presets fuer Normaljahr, trocken, nass, heisser Sommer und Open-Meteo-Kurzforecast
- Unsicherheitsband im Forecast ueber trockene/nasse Randvarianten
- Datenstand- und Update-Assistent mit Lueckenanzeige bis zum aktuellen Datum
- Diagnose-Tab mit Bias, Residuenstreuung, Lag-1-Autokorrelation, saisonalem Fehler und Response-Clustern
- Erweiterte Kartenansicht mit farbigen und skalierten Markern per PyDeck
- Performance-Optionen fuer stationsweise parallele Auto-Suche und optionalen Fruehstopp bei Ziel-R2
- Export-Schutz gegen Spreadsheet-Formel-Injection und HTML-Kurzreport
- Remote-URL-Haertung mit Credential-Ablehnung, maskierter Anzeige und defensiver Peer-IP-Pruefung
- Remote-Datenabruf per direkter oeffentlicher URL fuer CSV/XLSX/ODS/ZIP als Grundlage fuer DWD- oder andere Open-Data-Quellen
- Accessibility-Optionen fuer lesefreundlichere Darstellung, groessere Schrift, hohen Kontrast und Fokusmarkierung

## Erwartete Daten

- Grundwasserdatei: braucht eine Datums-Spalte (`Datum` oder `Date`) und Stationsspalten
- Wetterdatei: braucht eine Datums-Spalte sowie Spalten fuer Niederschlag und Verdunstung
- Zusatzdatei: wird stationsweise per `Site` oder `Messstelle` mit den Modell-Ergebnissen verknuepft
- Koordinatendatei fuer die Karte: braucht `Messstelle` oder `Site` sowie `Lat` und `Lon`
- Remote-Dateiquellen: direkte oeffentliche URLs zu CSV/XLSX/ODS oder ZIP-Dateien mit einer passenden Tabelle

## Hinweise

- Der temporaere Upload-Cache wird lokal im App-Ordner unter `.upload_cache` abgelegt und kann ueber die Sidebar wieder geleert werden.
- Nach einem Reload oder Sprachwechsel koennen Datei-Inputs optisch leer wirken, obwohl die App die Quellen bereits aus dem Temp-Cache weiterverwendet. Die aktive Nutzung wird in der Sidebar angezeigt.
- Fuer Remote- oder API-Quellen gibt es in der Sidebar einen Button zum erzwungenen Neuladen, damit gecachte Daten gezielt aktualisiert werden koennen.
- Der Remote-Import ist bewusst auf direkte oeffentliche HTTP(S)-Quellen ohne Redirects sowie auf begrenzte Dateigroessen ausgelegt. Das reduziert Sicherheits- und Stabilitaetsrisiken bei API-/URL-Importen.
- Die Session-Historie kann im Save-Tab begrenzt oder komplett geleert werden, damit lange Arbeitsrunden nicht unnoetig wachsen.
- Forecast ist ein Szenario auf Basis wiederholter historischer Wetterjahre. Open-Meteo-Tageswerte werden optional in den Modellhorizont eingefuegt, ersetzen aber keine vollwertige Klimaprojektion.
- Remote-Quellen sind praktisch fuer Open-Data-Workflows, brauchen fuer produktive API-Nutzung aber noch sauberes Stationsmapping, Fehlerbehandlung und ein klares Refresh-Konzept.
- Remote-URLs mit Zugangsdaten werden abgelehnt. URLs mit Query-Parametern werden in der Sidebar maskiert und nicht in den temporaeren Cache geschrieben.
- Parallele Auto-Suche nutzt Threads innerhalb des Streamlit-Prozesses. Fuer produktive Mehrnutzer-Deployments sollte zusaetzlich ein globales Job-Limit konfiguriert werden.

## Datenstand vom 24.04.2026

- `Messdaten/Grundwasserdaten.csv`: Messwerte bis `2021-04-23`
- `Messdaten/Potsdam_3987_gesamt_ab-1893.xlsx`: Wetterdaten bis `2021-01-17`
- `Messdaten/CPC.PC.csv` und `Messdaten/alle_ergebnisse.csv`: Zusatz-/Ergebnisdaten ohne Datumsachse

Die lokalen Mess- und Wetterdaten sind damit nicht bis zum aktuellen Datum gepflegt. Fuer echte operative Forecasts sollten neuere Grundwasser- und Wetterdaten oder belastbare Remote-Quellen angebunden werden.

## Docker

Die Beta-Kopie kann direkt als Container gestartet werden:

```bash
docker build -t gw-analyzer-beta .
docker run --rm -p 8501:8501 gw-analyzer-beta
```

Danach ist die App im Browser unter `http://localhost:8501` erreichbar.

## Verwendete Bibliotheken

- [Streamlit](https://streamlit.io/)
- [Pandas](https://pandas.pydata.org/)
- [Matplotlib](https://matplotlib.org/)
- [Pastas](https://pastas.readthedocs.io/latest/)
- [OpenPyXL](https://openpyxl.readthedocs.io/)
- [odfpy](https://pypi.org/project/odfpy/)
- [Requests](https://requests.readthedocs.io/)
- [Open-Meteo Forecast API](https://open-meteo.com/en/docs) fuer optionale aktuelle Wettervorhersagewerte
- [PyDeck](https://deckgl.readthedocs.io/) fuer die erweiterte Kartenansicht

## Hinweis zu noch offenen Erweiterungen

Die App ist jetzt fuer Bestmodellsuche, Laufvergleich, Export, Remote-Datenabruf und einfache Forecast-Szenarien erweitert. Themen wie vollautomatische Nachtlaeufe ausserhalb von Streamlit oder eine produktive DWD-Anbindung mit festem Stationsmapping brauchen noch eine konkrete Betriebsumgebung und abgestimmte Datenquellen.
