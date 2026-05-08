# ML- und Modellbegriffe: In-Depth-Info

Dieses Dokument erklärt die wichtigsten Begriffe der CNN/TCN/LSTM-Modelle: was sie technisch machen, warum sie im Grundwasser-Kontext relevant sind, wann man sie nutzt und wann man vorsichtig sein sollte.

## Grundidee der ML-Modelle

Der ML-Bereich vergleicht sechs neuronale Varianten:

- `Pastas + CNN`: Pastas simuliert den Grundwasserstand, CNN lernt nur den Restfehler.
- `Pastas + TCN`: Pastas simuliert den Grundwasserstand, TCN lernt den Restfehler mit dilatierten Faltungen.
- `Pastas + LSTM`: Pastas simuliert den Grundwasserstand, LSTM lernt nur den Restfehler.
- `Nur CNN`: CNN sagt den Grundwasserstand direkt aus Wetterfeatures voraus.
- `Nur TCN`: TCN sagt den Grundwasserstand direkt aus Wetterfeatures voraus.
- `Nur LSTM`: LSTM sagt den Grundwasserstand direkt aus Wetterfeatures voraus.

Der Hybridansatz ist meist hydrologisch besser interpretierbar, weil Pastas das physikalisch plausiblere Grundsignal liefert. Das neuronale Modell muss dann nur lernen, wo Pastas systematisch danebenliegt. Die Nur-ML-Varianten sind als direkter Vergleich wichtig: Sie zeigen, ob CNN/LSTM ohne Pastas-Basis überhaupt eine ähnliche oder bessere Güte erreichen.

## Trainingsfenster

Das Trainingsfenster ist die Länge der täglichen Eingangsequenz, die CNN/LSTM sehen.

Beispiel:

- Trainingsfenster `365 Tage`: Das Modell sieht für jede Vorhersage das Wetter der letzten 365 Tage.
- Trainingsfenster `20 Jahre`: Das Modell sieht für jede Vorhersage 20 Jahre tägliche Wettergeschichte.

Wichtig: Das Trainingsfenster ist nicht gleichbedeutend mit "so viele Jahre Trainingsdaten insgesamt". Es beschreibt die Länge einer einzelnen Probe.

Wann kleine Fenster sinnvoll sind:

- schnelle Reaktion der Messstelle
- kurze oder lückenhafte Zeitreihe
- kurze Vorhersagehorizonte
- erste Prüfung, ob ML überhaupt ein Signal findet

Wann lange Fenster sinnvoll sind:

- träge Grundwasserleiter
- langer Speicher- oder Verzögerungseffekt
- lange und stabile Messreihen
- Vergleich mit langsamen Pastas-Response-Funktionen

Risiko langer Fenster:

- braucht sehr lange Zeitreihen
- reduziert die Zahl nutzbarer Trainingssequenzen
- kann Training langsam und instabil machen
- kann irrelevante alte Wetterinformation mitschleppen

## Vorhersagehorizont

Der Vorhersagehorizont ist der Abstand zwischen Ende des Eingabefensters und Zielwert.

Beispiel:

- Fenster `20 Jahre`, Horizont `10 Jahre`: Das Modell sieht 20 Jahre Wetterhistorie und soll den Zielwert 10 Jahre später treffen.
- Dafür braucht eine einzelne Probe mindestens 30 Jahre Zeitreihe bis zum Zielwert.

Faustregel:

`benötigte Mindestzeit = Trainingsfenster + Vorhersagehorizont`

Bei `30 Jahre Fenster + 20 Jahre Horizont` werden also mindestens 50 Jahre nutzbarer Zeitverlauf gebraucht. Danach müssen zusätzlich noch genug Zielwerte übrig bleiben, damit 60/20/20-Training, Test und Validierung funktionieren.

Kurze Horizonte sind eher für direkte Reaktionsmuster geeignet. Lange Horizonte prüfen, ob das Modell langfristige Speicherwirkung und langsame Systemdynamik erkennt.

## 60/20/20-Split

Der ML-Bereich nutzt einen festen zeitlichen Split:

- erste 60 Prozent: Training
- nächste 20 Prozent: Test
- letzte 20 Prozent: Validierung

Das ist bewusst zeitlich sortiert. Es wird nicht zufällig gemischt, weil Zeitreihen sonst Zukunftsinformation in die Vergangenheit lecken könnten.

Training:

Das Modell lernt hier seine Gewichte.

Test:

Zwischenkontrolle nach der Trainingsphase.

Validierung:

Wichtigste Vergleichsfläche in der App. Die beste Variante wird nach Validierungs-R2 ausgewählt, weil sie am ehesten zeigt, ob das Modell auf späteren Zeiträumen generalisiert.

## Trainings-Epochen

Eine Epoche bedeutet: Das Modell läuft einmal komplett durch den Trainingsblock und passt seine Gewichte an.

Wenn `60 Epochen` eingestellt sind, sieht das Modell denselben Trainingsblock 60-mal. Es bekommt dadurch nicht mehr Daten, sondern mehr Lernzeit auf denselben Daten.

Zu wenige Epochen:

- Modell lernt nur grobe Muster
- R2 bleibt oft niedrig
- CNN/LSTM wirken "untertrainiert"
- Trainingsfehler bleibt hoch

Passende Epochen:

- Fehler sinkt, ohne dass Validierung deutlich schlechter wird
- Test und Validierung bewegen sich ähnlich
- Modell erkennt wiederkehrende Wetter-/Grundwasser-Muster

Zu viele Epochen:

- Modell merkt sich Trainingsdaten zu stark
- Trainingsgüte steigt, Validierung wird schlechter
- besonders kritisch bei wenigen Sequenzen
- LSTM ist davon oft stärker betroffen als CNN

Die App erlaubt maximal `1000` Epochen. Zusätzlich zeigt sie ein datenbasiertes Epochen-Limit an. Dieses Limit hängt von Station, Trainingsfenster und Vorhersagehorizont ab, weil diese drei Dinge bestimmen, wie viele nutzbare Sequenzen wirklich entstehen.

Wenn nur wenige Trainingssequenzen vorhanden sind, ist ein hohes Epochenlimit fachlich riskant. Dann kann das Modell denselben kleinen Trainingsblock zu oft wiederholen und überanpassen.

## Lernrate

Die Lernrate bestimmt, wie groß die Optimierungsschritte beim Training sind.

Kleine Lernrate, z. B. `0.0005`:

- ruhigeres Training
- besser bei langen Horizonten oder wenigen Daten
- braucht oft mehr Epochen
- weniger Gefahr, am Optimum vorbeizuspringen

Mittlere Lernrate, z. B. `0.001`:

- guter Standardwert
- meist stabil für CNN und LSTM
- sinnvoll für die meisten Stationsvergleiche

Höhere Lernrate, z. B. `0.002` oder `0.005`:

- schnelleres Lernen
- kann bei vielen Daten und kurzen Horizonten helfen
- kann instabil werden
- bei LSTM und langen Fenstern vorsichtig verwenden

Wenn der Loss stark schwankt oder die Validierung schlecht wird, obwohl Training scheinbar gut läuft, ist die Lernrate oft zu hoch.

## Hidden Size / Filter

Dieser Wert steuert die Größe des neuronalen Modells.

Bei LSTM bedeutet Hidden Size:

- Größe des internen Speichers
- wie viel langfristige Information das Modell intern halten kann
- größere Werte können komplexere Dynamik lernen

Bei CNN bedeutet Filter:

- Anzahl der Faltungskanäle
- wie viele Muster parallel erkannt werden können
- größere Werte erkennen mehr Varianten von Regen-/Trockenheitsmustern

Kleine Werte, z. B. `16`:

- stabiler bei wenig Daten
- schneller
- geringeres Overfitting-Risiko
- kann komplexe Dynamik verfehlen

Mittlere Werte, z. B. `32` oder `64`:

- meist guter Start
- genug Flexibilität für viele Stationen
- noch halbwegs robust

Große Werte, z. B. `128`:

- nur sinnvoll bei vielen Sequenzen
- langsamer
- höheres Overfitting-Risiko
- eher für kurze/mittlere Fenster mit vielen Daten

## Wetterfeatures

Die aktuelle Vergleichsvorgabe nutzt keine vorherigen Grundwasserstände als Features und keine Jahreszeit-Sin/Cos-Features. Dadurch bleibt der Vergleich mit Pastas strenger: CNN/LSTM sollen vor allem aus Wetterinformationen lernen.

Direkte Wetterfeatures:

- `rain`: täglicher Niederschlag
- `evap`: tägliche Verdunstung

Diese Features sind nah an den klassischen Pastas-Stressdaten. Sie sind fast immer sinnvoll, wenn Wetterdaten zuverlässig sind.

## Rollierende Wetterfenster

Rollierende Wetterfenster erzeugen zusammengefasste Wetterinformationen über die letzten Tage.

Aktuell:

- `rain_sum_7`: Niederschlagssumme der letzten 7 Tage
- `rain_sum_30`: Niederschlagssumme der letzten 30 Tage
- `rain_sum_90`: Niederschlagssumme der letzten 90 Tage
- `evap_mean_7`: mittlere Verdunstung der letzten 7 Tage
- `evap_mean_30`: mittlere Verdunstung der letzten 30 Tage
- `evap_mean_90`: mittlere Verdunstung der letzten 90 Tage

Was das bringt:

- CNN/LSTM bekommen direkt Informationen über nasse und trockene Phasen.
- Das Modell muss kurzfristige Summen nicht vollständig selbst lernen.
- Besonders CNN profitiert oft, weil es lokale Muster und Cluster erkennt.

Wann aktivieren:

- fast immer als Startwert sinnvoll
- wenn Niederschlag verzögert wirkt
- wenn Trockenperioden relevant sind
- wenn der direkte Tageswert zu sprunghaft ist

Wann vorsichtig sein:

- bei sehr kurzen Datenreihen
- bei stark fehlerhaften Wetterdaten
- wenn zu viele Features im Verhältnis zu wenigen Sequenzen entstehen

## Pastas + ML

Bei `Pastas + ML` wird zuerst ein Pastas-Modell ohne FlexModel berechnet. Danach wird das Residuum gebildet:

`Residuum = Beobachtung - Pastas-Simulation`

CNN oder LSTM lernt dann dieses Residuum.

Finale Vorhersage:

`Hybrid = Pastas-Simulation + ML-Residuum`

Vorteile:

- hydrologisch interpretierbarer
- ML muss weniger lernen
- gut geeignet, um systematische Pastas-Abweichungen zu finden
- Impulsantwort kann mit Pastas-IRF verglichen werden

Nachteile:

- abhängig von der Qualität des Pastas-Basismodells
- wenn Pastas falsche Struktur liefert, lernt ML nur Korrekturen darauf
- komplexer zu erklären als Nur-ML

## Nur CNN/LSTM

Bei `Nur ML` wird der Grundwasserstand direkt aus den Wetterfeatures vorhergesagt.

Finale Vorhersage:

`ML = CNN/LSTM(Wetterfeatures)`

Vorteile:

- direkter Vergleich gegen Pastas
- zeigt, ob die Wetterfeatures allein genug Signal tragen
- keine additive Abhängigkeit vom Pastas-Fehler

Nachteile:

- weniger hydrologisch interpretierbar
- muss Grundniveau, Trends und Dynamik selbst lernen
- bei langen Horizonten und wenigen Daten oft schwächer
- kann leichter überfitten

## CNN

CNN bedeutet hier ein eindimensionales Faltungsnetz über Zeitfenster.

Was CNN gut kann:

- lokale Muster erkennen
- Regencluster erkennen
- kurze Trocken-/Nassphasen erkennen
- schnelle bis mittlere Reaktionen abbilden
- oft schneller und robuster trainieren als LSTM

Warum CNN für Grundwasser interessant ist:

Niederschlag wirkt selten nur an einem einzelnen Tag. Häufig zählt die Struktur: mehrere Regentage, Vorfeuchte, Trockenphase davor, Verdunstungsphase danach. CNNs können solche lokalen Zeitmuster aus dem Eingabefenster herausfiltern.

Wann CNN zuerst nutzen:

- wenn Training schnell sein soll
- wenn die Station eher kurzfristig bis mittelfristig reagiert
- wenn LSTM instabil wird
- wenn viele Fenster/Horizonte verglichen werden sollen

## TCN

TCN bedeutet Temporal Convolutional Network. Es ist verwandt mit CNN, nutzt aber dilatierte Faltungen. Dilatiert heißt: Die Faltung schaut nicht nur auf direkt benachbarte Tage, sondern überspringt mit wachsendem Abstand einzelne Positionen. Dadurch kann das Modell längere Muster sehen, ohne dass das Netzwerk extrem tief oder langsam werden muss.

Was TCN gut kann:

- längere Reaktionszeiten robuster erfassen als ein einfaches CNN
- Regen- und Trockenheitsmuster auf mehreren Zeitskalen erkennen
- schneller und stabiler trainieren als viele LSTM-Setups
- lokale Muster und weiter entfernte Abhängigkeiten kombinieren

Warum TCN für Grundwasser interessant ist:

Grundwasser reagiert oft nicht nur auf die letzten Tage, sondern auf länger aufgebaute Feuchte- oder Trockenheitszustände. Ein einfaches CNN erkennt lokale Muster gut, kann aber bei langen Verzögerungen zu kurz greifen. TCN erweitert diesen Blick durch Dilationen, ohne gleich die volle Empfindlichkeit eines LSTM zu haben.

Wann TCN zuerst nutzen:

- wenn CNN zu kurzsichtig wirkt
- wenn LSTM instabil oder langsam trainiert
- bei mittleren bis langen Trainingsfenstern
- bei Stationen mit verzögerter, aber noch klar wettergetriebener Reaktion

Worauf achten:

- TCN kann trotzdem überfitten, wenn wenig Sequenzen vorhanden sind.
- Bei sehr langen Fenstern bleibt die Datenmenge entscheidend.
- Wenn TCN besser als CNN und LSTM validiert, ist das oft ein Hinweis auf mehrskalige Verzögerungsmuster.

## LSTM

LSTM ist ein rekurrentes neuronales Netz mit internem Speicher.

Was LSTM gut kann:

- längere Abhängigkeiten lernen
- Reihenfolge über längere Zeit berücksichtigen
- trägere Dynamik abbilden
- saisonähnliche Langzeitmuster teilweise indirekt erkennen

Warum LSTM schwieriger ist:

- empfindlicher gegenüber Skalierung
- braucht oft mehr Daten
- trainiert langsamer
- kann bei langen Fenstern und wenigen Sequenzen überfitten
- reagiert stärker auf Lernrate und Epochen

Wann LSTM sinnvoll ist:

- lange, saubere Zeitreihen
- träge Grundwasserreaktion
- Verdacht auf lange Speicherwirkung
- wenn CNN lokale Muster nicht ausreichend erfasst

## Gamma Response Function

Die Gamma-Funktion ist eine flexible Pastas-Response-Funktion. Sie beschreibt, wie ein Niederschlags-/Recharge-Impuls über die Zeit im Grundwasser ankommt.

Stärken:

- flexible Form
- kann verzögerte Peaks abbilden
- oft guter Standard für Grundwasser
- geeignet für viele natürliche Speicherreaktionen

Typisches Verhalten:

- Reaktion steigt nach einem Impuls an
- erreicht einen Peak
- klingt danach langsam ab

Wann nutzen:

- wenn die Reaktion verzögert und nicht sofort maximal ist
- wenn man einen robusten Standard braucht
- wenn die Messstelle einen erkennbaren Speicher-/Verzögerungseffekt hat

## Exponential Response Function

Die Exponential-Funktion beschreibt eine einfache abklingende Reaktion.

Stärken:

- sehr einfach
- wenige Freiheitsgrade
- robust bei weniger Daten
- gut für Systeme mit schnellerem Abklingen

Typisches Verhalten:

- stärkste Reaktion früh
- danach monotoner Rückgang
- weniger flexibel als Gamma

Wann nutzen:

- wenn Daten knapp sind
- wenn eine einfache Reaktion reicht
- wenn komplexere Funktionen instabil werden
- als Vergleichsbaseline

## Hantush Response Function

Die Hantush-Funktion ist stärker hydrogeologisch motiviert und kann verzögerte und gedämpfte Reaktionen abbilden.

Stärken:

- oft gut bei langsamer oder gedämpfter Grundwasserreaktion
- kann komplexere Aquifer-Reaktionen darstellen
- fachlich interessant für Response-Vergleiche

Risiken:

- braucht genügend Daten
- kann schwieriger zu kalibrieren sein
- Parameter können instabil werden, wenn die Zeitreihe zu kurz ist

Wann nutzen:

- bei träger Reaktion
- bei deutlicher Verzögerung
- wenn Gamma/Exponential die Reaktion nicht gut beschreiben
- für wissenschaftlichen Vergleich der Impulsantwort

## Impulsantwort

Die Impulsantwort zeigt, wie ein Modell auf ein einmaliges Niederschlagsereignis reagiert, während Verdunstung auf 0 gesetzt wird.

Bei Pastas:

- die Impulsantwort kommt direkt aus der Response-Funktion
- sie ist interpretierbar als hydrologische Reaktionsform

Bei ML:

- ein künstliches Wetterfenster wird erzeugt
- ein einmaliger Niederschlagsimpuls wird gesetzt
- das Modell wird mit Impuls und ohne Impuls verglichen
- Differenz = gelernte ML-Reaktion

Wichtig:

Die ML-Impulsantwort ist keine klassische analytische Response-Funktion. Sie ist eine empirische Reaktion des trainierten neuronalen Modells auf ein künstliches Szenario.

## R2

R2 misst, wie viel Varianz der Beobachtungen durch die Vorhersage erklärt wird.

Interpretation:

- `1.0`: perfekt
- `0.0`: nicht besser als Mittelwert
- negativ: schlechter als Mittelwert

R2 ist anschaulich, aber bei Zeitreihen nicht allein ausreichend. Ein Modell kann ein gutes R2 haben und trotzdem Peaks oder Trockenphasen schlecht treffen.

## RMSE

RMSE ist die Wurzel aus dem mittleren quadratischen Fehler.

Eigenschaften:

- gleiche Einheit wie Grundwasserstand
- bestraft große Fehler stark
- gut zum Vergleich absoluter Abweichungen

Kleiner RMSE ist besser.

## EVP

EVP steht für erklärte Varianz in Prozent.

Hohe EVP bedeutet:

- Restfehler schwanken wenig im Vergleich zur Beobachtung
- Modell trifft die Dynamik besser

EVP ist nützlich als Ergänzung zu R2 und RMSE.

## Datenbasiertes Epochen-Limit

Die App schätzt ein Epochen-Limit aus der Zahl der nutzbaren Trainingssequenzen.

Warum:

- wenige Sequenzen vertragen weniger Epochen
- viele Sequenzen erlauben längeres Training
- lange Fenster und Horizonte reduzieren die Sequenzzahl

Das Limit ist kein mathematisches Naturgesetz. Es ist ein Schutz gegen offensichtliches Overfitting und eine Orientierung für die Bedienung.

## Hyperparameter-Empfehlung

Die App zeigt automatisch eine Startempfehlung für:

- Trainings-Epochen
- Lernrate
- Hidden Size / Filter

Grundlage:

- Station
- Trainingsfenster
- Vorhersagehorizont
- Zahl nutzbarer Sequenzen
- Zahl aktiver Features

Wichtig:

Die Empfehlung ist kein garantierter globaler Bestwert. Der tatsächliche beste Wert wird erst sichtbar, wenn verschiedene Einstellungen trainiert und anhand der Validierung verglichen werden. Die Empfehlung soll einen sinnvollen Startpunkt liefern, damit man nicht blind mit zu großen oder zu kleinen Werten beginnt.

## All-Station-ML-Lauf

Die Option `Alle verfügbaren Stationen` geht alle Stationen durch, für die im letzten Pastas-Lauf ein erfolgreiches Modell ohne FlexModel vorhanden ist. Pro Station werden CNN, TCN und LSTM jeweils als Hybrid und als Nur-ML trainiert.

Warum stationsweise:

- reduziert RAM-Spitzen
- verhindert, dass alle Modelle gleichzeitig CPU belegen
- erlaubt Zwischenspeichern nach jeder erfolgreichen Station
- macht lange Läufe robuster gegen App- oder Browser-Abbrüche

Die App reduziert für diese Batchläufe die PyTorch-Threadzahl und trainiert Stationen sequenziell. Das ist absichtlich langsamer als maximale Parallelisierung, aber stabiler für normale Desktop-Arbeit.

## ML-Zwischenspeicher

Nach jeder erfolgreichen Station schreibt die App ein `.gwml`-Paket in `.ml_run_cache`. Dieses Paket enthält:

- Validierungs-/Prüfdaten
- Vergleichstabelle
- trainierte Modellzustände
- Metadaten zur Station und Konfiguration

Wenn die App später abstürzt, ist nicht automatisch alles verloren. Im ML-Tab kann das zuletzt gespeicherte Paket aus dem Zwischenspeicher wieder geladen werden. Für vollständige Projektstände bleibt das `.gwproject`-Paket der bessere Export, weil es zusätzlich Originaldateien, Pastas-Ergebnisse, Forecasts und Einstellungen enthält.

## Import und Export

Es gibt zwei Pakettypen:

- `.gwproject`: kompletter Projektstand mit Originaldateien, Pastas-Ergebnissen, ML-/Hybrid-Ergebnissen, Forecasts, Modellzuständen und Einstellungen.
- `.gwml`: einzelnes ML-Laufpaket mit ML-Ergebnissen und trainierten neuronalen Modellzuständen.

Das obere Import-/Export-Feld ist für den kompletten Arbeitsstand gedacht. Der ML-Tab kann zusätzlich `.gwml`-Pakete direkt importieren, damit einzelne ML-Läufe unabhängig wiederhergestellt oder verglichen werden können.

## Praktische Startstrategie

1. Mit `5 Jahre` Fenster und `30` oder `90 Tage` Horizont starten.
2. CNN und LSTM mit Standardwerten trainieren.
3. Prüfen, ob Hybrid oder Nur-ML besser validiert.
4. Fenster langsam erhöhen: `10`, `15`, `20 Jahre`.
5. Horizont nur erhöhen, wenn genug Sequenzen bleiben.
6. Wenn Validierung schlechter wird: weniger Epochen, kleinere Hidden Size, kleinere Lernrate.
7. Wenn Training und Validierung beide schlecht sind: mehr Features, anderes Fenster oder anderes Pastas-Basismodell prüfen.

## Typische Warnzeichen

Sehr gutes Training, schlechte Validierung:

- zu viele Epochen
- Hidden Size zu groß
- zu wenige Sequenzen
- Overfitting

Alle ML-Varianten schlecht:

- Wetterfeatures erklären die Station nicht ausreichend
- Zielhorizont zu lang
- Datenreihe zu kurz
- Pastas-Basismodell oder Eingangsdaten prüfen

Nur-ML schlechter als Hybrid:

- normaler Fall bei hydrologisch plausiblen Pastas-Modellen
- ML profitiert davon, nur Restfehler lernen zu müssen

Nur-ML besser als Hybrid:

- Pastas-Basismodell passt eventuell schlecht
- ML findet Muster, die Pastas nicht abbildet
- Ergebnisse fachlich prüfen, nicht nur R2 betrachten
