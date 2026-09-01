# Kickbase Tools

[![CI](https://github.com/Noltinho/kickbase-analytics/actions/workflows/ci.yml/badge.svg)](https://github.com/Noltinho/kickbase-analytics/actions/workflows/ci.yml)

**[▶ Live-Demo](https://noltinho.github.io/kickbase-analytics/)** — alle
Werkzeuge laufen direkt im Browser, ohne Installation und ohne Login.

Statische Analysewerkzeuge für Bundesliga und 2. Bundesliga: Ein Python-Fetcher
verdichtet Team-, Spieler- und Marktwertdaten, ein gegnerbereinigtes
Teamstärkemodell bewertet Matchups, und fünf frameworkfreie Werkzeuge machen
die Ergebnisse direkt im Browser nutzbar.

[![Marktwert gegen prognostizierten Punkteschnitt](screenshots/marktwert-prognose.png)](https://noltinho.github.io/kickbase-analytics/scatter.html)

<sub>Marktwert-Analyse: Marktwert gegen den prognostizierten Punkteschnitt je
Einsatz (Bundesliga 2026/27). Die Prognose stammt aus dem fallweisen
Spielermodell, R² bezieht sich auf die eingezeichnete Regression.</sub>

Das Projekt verbindet Datenbeschaffung, robuste Datenmodellierung,
Walk-forward-Evaluation und eine bewusst schlanke Visualisierung ohne
Frontend-Buildschritt.

> Inoffizielles, nicht mit Kickbase oder Transfermarkt verbundenes
> Forschungs- und Portfolio-Projekt. Hinweise zu Fremddaten und Marken stehen
> in [NOTICE.md](NOTICE.md).

## Was die Anwendung zeigt

- **Teampunkte:** erzielte Kickbase-Punkte nach Team und Mannschaftsteil
- **Matchup:** zugelassene Punkte des Gegners, nach Position und Spielort
- **Matchup-Score:** Spielplanraster von −3 bis +3, wahlweise als Paarungs- oder
  reine Gegnerstärke
- **Teamstärke-Editor:** lokale Abweichungen vom Modell, ohne die berechnete
  Basis zu überschreiben
- **Marktwert-Analyse:** Marktwert gegen erzielte beziehungsweise prognostizierte
  Durchschnittspunkte

Alle Seiten sind statische Dateien. Sie laden ausschließlich versionierte
JSON-Dateien aus `data/`; persönliche Logins oder Zugangsdaten gelangen nicht
ins Frontend.

## Architektur

```text
Kickbase v4 ── fetch.py ──► data/data_*.json + matchdays_*.json
                       └──► history.json ──► carryover.json

Archiv + Spielplan + Saisonquoten ──► Teamstärke-Ridge ──► ratings.json
historisches Panel + Kaderdaten ─────► Spielermodelle ───► projections*.json

data/*.json ──► statische HTML-/JavaScript-Dashboards
```

Die Produktionspipeline in `fetch.py` benötigt nur `requests`. Die Forschung
unter `kbxp/` nutzt zusätzlich pandas, NumPy, SciPy, PyArrow und scikit-learn.
Das Frontend verwendet kein Framework und keinen Bundler.

## Lokal ansehen

Direktes Öffnen per `file://` funktioniert wegen der Browser-Sicherheitsregeln
für `fetch()` nicht zuverlässig. Im Projektverzeichnis genügt ein lokaler
Webserver:

```bash
python -m http.server 8000
```

Danach `http://localhost:8000/` öffnen.

## Installation und Tests

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
python -m pip install -r requirements.lock

python -m pytest -m "not slow"
npm run test:frontend
```

Die schnelle Suite ist für CI gedacht. Rechenintensive Monte-Carlo- und
Forschungstests laufen zusätzlich mit:

```bash
python -m pytest
```

`requirements.lock` hält den geprüften Stand exakt fest;
`requirements.txt` enthält die bewusst breiteren direkten Anforderungen für
kontrollierte Aktualisierungen.

`kbxp/data/manual/tm_players.csv` wird bewusst nicht veröffentlicht. Tests, die
den vollständigen Transfermarkt-Crawl benötigen, werden in einem frischen Klon
mit sichtbarer Begründung übersprungen. Die übrigen Daten- und Modellverträge
bleiben prüfbar.

## Daten aktualisieren

Zugangsdaten gehören ausschließlich in Umgebungsvariablen oder eine lokale
`.env`-Datei; beides wird von Git ignoriert.

```bash
python fetch.py                         # laufende Saison
python fetch.py mv                      # Marktwert-Historie verdichten
python fetch.py carryover               # ohne Login
python fetch.py ratings                 # ohne Login
python fetch.py archive --season 2526   # abgeschlossene Saison
```

Der Fetcher schreibt veröffentlichte JSON-Dateien atomar. Ein Abbruch während
des Schreibens beschädigt daher insbesondere die nicht nachholbare
`history.json` nicht.

Forschungs- und Modellkommandos sind in [AGENTS.md](AGENTS.md) dokumentiert;
der unabhängige Modellvergleich steht in
[kbxp/player-benchmark.md](kbxp/player-benchmark.md).

Das fallweise Spielermodell lässt sich getrennt für beide Ligen exportieren:

```bash
cd kbxp
python -m src.model.player_avg                  # Liga 1
python -m src.model.player_avg --liga 2         # Liga 2
python -m src.model.player_avg --liga 2 --backtest
```

## Modellierung in Kürze

Die Teamstärke wird als Ridge-Modell geschätzt:

```text
zugelassene Punkte = Ligamittel + eigene Abwehr + gegnerischer Angriff + Heimvorteil
```

Die Evaluation ist strikt walk-forward: Für einen Spieltag oder eine
Zielsaison werden nur Informationen verwendet, die zu diesem Zeitpunkt bekannt
waren. Marktquoten dienen als Vorsaison-Prior; ihre mehrminütige Monte-Carlo-
Inversion wird anhand aller fachlichen Eingänge gecacht.

Die Spielermodelle beantworten zwei getrennte Fragen: ein Rollenmodell zerlegt
den Ertrag in Qualität und erwartete Einsatzzeit, das fallweise Modell schätzt
den Punkteschnitt für gesetzte Spieler abhängig von verfügbarer Historie. Das
fallweise Modell verwendet in beiden Ligen dieselbe fachliche Struktur, fittet
Koeffizienten, Teamniveaus und Kalibrierung aber getrennt. In Liga 2 ersetzt
eine walk-forward geprüfte Starter-Logistik die dort fehlenden handgepflegten
Kategorien. Die
gemessenen Grenzen und Vergleichsmodelle sind nicht aus dem README ausgelagert,
sondern im Benchmark nachvollziehbar.

Als flächendeckende Vergleichslinie dient eine einfache
**Marktwert-/Positionsbaseline**. Sie wird je Zielsaison ausschließlich auf
früheren Saisons geschätzt und verwendet den Marktwert sowie den kategorialen
Positions- und Fallkontext. Ihr MAE beträgt 19,28, der des fallweisen Modells
16,64. Auf der gemeinsamen Maske von 1.571 Starter-Spielersaisons liegt das
fallweise Modell paarweise um 2,64 MAE-Punkte vorn
(saisongeclustertes 95-%-Intervall 2,23 bis 3,00) und gewinnt in allen sieben
Zielsaisons. Aufbau, weitere Baselines und Aufschlüsselungen stehen in
[kbxp/player-benchmark.md](kbxp/player-benchmark.md).

## Repository-Struktur

```text
fetch.py                 Produktions-Fetcher und JSON-Exporte
common.js                gemeinsame Saison-, Rating- und Score-Logik
frontend-utils.js        kleine, separat testbare Frontend-Datenfunktionen
*.html                   Startseite und fünf statische Werkzeuge
data/                    veröffentlichte Anwendungsdaten und Modell-Exporte
kbxp/src/                Ingest-, Feature- und Modellcode
kbxp/tests/              Leakage-, Güte- und Datenvertragstests
.github/workflows/ci.yml Syntax-, Frontend- und Python-CI
```

## Lizenz

Der selbst entwickelte Quellcode steht unter der MIT-Lizenz. Die Lizenz erfasst
nicht automatisch Daten, Datenbanken, Namen, Marken oder Inhalte Dritter; siehe
[LICENSE](LICENSE) und [NOTICE.md](NOTICE.md).
