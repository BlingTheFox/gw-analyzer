<!-- Project lead / concept credit: Robin Carow / RCnet -->
<!-- Scientific modeling framework credit: Pastas by Collenteur et al. (2019), https://doi.org/10.1111/gwat.12925 -->

# GW Analyzer

Zweisprachige Streamlit-Anwendung zur Analyse von Grundwasser-Zeitreihen mit [Pastas](https://pastas.readthedocs.io/latest/).
Die Anwendung wurde fuer den Einsatzkontext am ZALF entwickelt.

## Projekt und Credits

- Projektkonzept und App-Entwicklung: Robin Carow / RCnet
- Entwickelt fuer den Einsatzkontext am ZALF
- Die Modellierung basiert auf [Pastas](https://pastas.readthedocs.io/latest/), dem Open-Source-Pythonpaket fuer hydro(geo)logische Zeitreihenanalysen
- Wissenschaftliche Referenz fuer Pastas:
  [Collenteur, R.A., Calje, R., Klop, S.A., Schaars, F., & Bakker, M. (2019). *Pastas: open source software for the analysis of groundwater time series*. Groundwater.](https://doi.org/10.1111/gwat.12925)
- Quellcode und Community:
  [Pastas auf GitHub](https://github.com/pastas/pastas)

## Installation

```bash
python -m pip install -r requirements.txt
```

## Start

```bash
python -m streamlit run app.py
```

## Docker

```bash
docker build -t gw-analyzer .
docker run --rm -p 8501:8501 gw-analyzer
```

Danach ist die App unter `http://localhost:8501` erreichbar.

## Funktionen

- Upload von Grundwasserdaten, Wetterdaten und optionalen stationsbezogenen Zusatzdaten wie `CPC.PC.csv`
- Auswahlmodus fuer `einzelne`, `mehrere` oder `alle` Messstellen
- Modelle mit `Gamma` oder `Exponential`, optional `FlexModel` und `NoiseModel`
- Anpassbare Modellparameter inklusive Startwerten und Optimierungs-Flags
- Kompakte Ergebnisansicht mit optionaler erweiterter Tabelle fuer Parameterdetails
- Visualisierung als Pastas-Uebersicht, Beobachtet-vs-Simuliert, Residualplot, Residual-Histogramm und Schrittantwort
- Vergleichsansichten als Ranking, Scatterplot, Histogramm, Boxplot und Korrelationsmatrix
- Accessibility-Optionen fuer lesefreundlichere Darstellung, groessere Schrift, hohen Kontrast und Fokusmarkierung
- Export und Import der Ergebnis-Historie als CSV

## Hinweise zu den Daten

- Grundwasserdatei: braucht eine Datums-Spalte (`Datum` oder `Date`) und Stationsspalten
- Wetterdatei: braucht eine Datums-Spalte sowie Spalten fuer Niederschlag und Verdunstung
- Zusatzdatei: wird stationsweise per `Site` oder `Messstelle` mit den Modell-Ergebnissen verknuepft

## Verwendete Bibliotheken

- [Streamlit](https://streamlit.io/)
- [Pandas](https://pandas.pydata.org/)
- [Matplotlib](https://matplotlib.org/)
- [Pastas](https://pastas.readthedocs.io/latest/)
- [OpenPyXL](https://openpyxl.readthedocs.io/)
- [odfpy](https://pypi.org/project/odfpy/)

## Hinweis zur Zitation

Wenn die App oder die Modellierungsergebnisse in einer wissenschaftlichen Auswertung genutzt werden, sollte die verwendete Pastas-Methodik entsprechend zitiert werden:
[https://doi.org/10.1111/gwat.12925](https://doi.org/10.1111/gwat.12925)

## Lizenz

Dieses Projekt steht unter der [MIT-Lizenz](./LICENSE).
