<!-- Project lead / concept credit: Robin Carow / RCnet -->
<!-- Scientific modeling framework credit: Pastas by Collenteur et al. (2019), https://doi.org/10.1111/gwat.12925 -->

# GW Analyzer Beta

Beta-Arbeitskopie der Streamlit-Anwendung zur Analyse von Grundwasser-Zeitreihen mit [Pastas](https://pastas.readthedocs.io/latest/).

## Projekt und Credits

- Projektkonzept und App-Entwicklung: Robin Carow / RCnet
- Die Modellierung basiert auf [Pastas](https://pastas.readthedocs.io/latest/), dem Open-Source-Pythonpaket für hydro(geo)logische Zeitreihenanalysen
- Wissenschaftliche Referenz für Pastas:
  [Collenteur, R.A., Calje, R., Klop, S.A., Schaars, F., & Bakker, M. (2019). *Pastas: open source software for the analysis of groundwater time series*. Groundwater.](https://doi.org/10.1111/gwat.12925)
- Quellcode und Community:
  [Pastas auf GitHub](https://github.com/pastas/pastas)

## Installation

```bash
python -m pip install -r requirements.txt
```

Die aktuelle Beta nutzt `st.fragment` für das leichtere Batch-Monitoring und sollte deshalb mit `Streamlit >= 1.50` betrieben werden.

## Start

```bash
python -m streamlit run app.py
```

## Neue Beta-Funktionen

- Auswahlmodus für `einzelne`, `mehrere` oder `alle` Messstellen
- Automatische Best-Model-Suche über `Gamma/Exponential`, `Flex/ohne Flex` und `Noise/ohne Noise`
- Hintergrund-Batchlauf mit Fortschrittsanzeige und Abbruchfunktion
- Persistenter Batch-Status über Streamlit-Reruns, damit `Start batch` sichtbar anläuft
- Leichteres Batch-Monitoring ohne blockierende Sleep-Schleife während eines aktiven Laufs
- Temporärer Upload-Cache per Knopfdruck, damit Uploads und URL-Quellen nach Reloads weiter genutzt werden können
- Automatische Temp-Sicherung der aktuellen Quellen vor einem Sprachwechsel, damit der Datenstand nicht verloren geht
- Sichtbare Anzeige aktiver Cache-Quellen in der Sidebar, damit klar bleibt, welche Daten gerade verwendet werden
- Sichtbare Zusammenfassung der aktuell genutzten Quellen in der Sidebar, auch wenn File-Inputs nach einem Reload optisch leer wirken
- Button zum erzwungenen Neuladen der Datenquellen für Remote-/API-Workflows
- R2-Filter, Bestmodell-Filter und Laufvergleich mehrerer Runs nebeneinander
- Kompakte Standard-Ergebnistabelle mit optionaler erweiterter Ansicht
- Export der Historie als CSV und als Excel mit mehreren Sheets
- Historie-Steuerung mit Limit für gespeicherte Läufe und eigenem Leeren-Knopf
- Optionale Kartenansicht auf Basis einer hochladbaren Koordinatendatei mit `Messstelle/Site`, `Lat`, `Lon`
- Visualisierung der Impulse Response Function zusätzlich zur Pastas-Step-Function
- Ausgabe simulierter Zeitreihen je Messstelle mit Beobachtung, Simulation und Residuum
- Gesamt-CSV-Export aller simulierten Zeitreihen aus dem letzten Modelllauf
- Forecast-Tab für Zukunftsszenarien mit skaliertem Niederschlag/Verdunstung, additiven Offsets und nicht-negativem Clipping
- Optionaler Open-Meteo-Abruf für aktuelle Tageswerte mit `precipitation_sum` und `et0_fao_evapotranspiration`, parametrisierbar über Koordinaten, Zeitzone und Forecast-Tage
- Wetterantrieb im Forecast als eigenes Diagramm und als exportierbare Spalten im Forecast-CSV
- Szenario-Presets für Normaljahr, trocken, nass, heißer Sommer und Open-Meteo-Kurzforecast
- Unsicherheitsband im Forecast über trockene/nasse Randvarianten
- Datenstand- und Update-Assistent mit Lückenanzeige bis zum aktuellen Datum
- Diagnose-Tab mit Bias, Residuenstreuung, Lag-1-Autokorrelation, saisonalem Fehler und Response-Clustern
- Erweiterte Kartenansicht mit farbigen und skalierten Markern per PyDeck
- Performance-Optionen für stationsweise parallele Auto-Suche und optionalen Frühstopp bei Ziel-R2
- Export-Schutz gegen Spreadsheet-Formel-Injection und HTML-Kurzreport
- Remote-URL-Härtung mit Credential-Ablehnung, maskierter Anzeige und defensiver Peer-IP-Prüfung
- Remote-Datenabruf per direkter öffentlicher URL für CSV/XLSX/ODS/ZIP als Grundlage für DWD- oder andere Open-Data-Quellen
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
- Für Remote- oder API-Quellen gibt es in der Sidebar einen Button zum erzwungenen Neuladen, damit gecachte Daten gezielt aktualisiert werden können.
- Der Remote-Import ist bewusst auf direkte öffentliche HTTP(S)-Quellen ohne Redirects sowie auf begrenzte Dateigrößen ausgelegt. Das reduziert Sicherheits- und Stabilitätsrisiken bei API-/URL-Importen.
- Die Session-Historie kann im Save-Tab begrenzt oder komplett geleert werden, damit lange Arbeitsrunden nicht unnötig wachsen.
- Simulierte Zeitreihen werden aus den gespeicherten Bestmodellen des letzten Laufs erzeugt und können stationsweise im Visualisierungs-Tab oder gesammelt im Save-Tab exportiert werden.
- Forecast ist ein Szenario auf Basis wiederholter historischer Wetterjahre. Open-Meteo-Tageswerte werden optional in den Modellhorizont eingefügt, ersetzen aber keine vollwertige Klimaprojektion.
- Remote-Quellen sind praktisch für Open-Data-Workflows, brauchen für produktive API-Nutzung aber noch sauberes Stationsmapping, Fehlerbehandlung und ein klares Refresh-Konzept.
- Remote-URLs mit Zugangsdaten werden abgelehnt. URLs mit Query-Parametern werden in der Sidebar maskiert und nicht in den temporären Cache geschrieben.
- Parallele Auto-Suche nutzt Threads innerhalb des Streamlit-Prozesses. Für produktive Mehrnutzer-Deployments sollte zusätzlich ein globales Job-Limit konfiguriert werden.

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
- [Open-Meteo Forecast API](https://open-meteo.com/en/docs) für optionale aktuelle Wettervorhersagewerte
- [PyDeck](https://deckgl.readthedocs.io/) für die erweiterte Kartenansicht

## Hinweis zu noch offenen Erweiterungen

Die App ist jetzt für Bestmodellsuche, Laufvergleich, Export, Remote-Datenabruf und einfache Forecast-Szenarien erweitert. Themen wie vollautomatische Nachtläufe außerhalb von Streamlit oder eine produktive DWD-Anbindung mit festem Stationsmapping brauchen noch eine konkrete Betriebsumgebung und abgestimmte Datenquellen.
