# Kickbase Tools — Überblick

Statische HTML-Werkzeuge zur Analyse von Bundesliga und 2. Bundesliga in
Kickbase, gespeist aus einem Python-Fetcher gegen die Kickbase-**v4**-API.
Kein Build, kein Framework, keine Abhängigkeit im Frontend außer drei
CDN-Skripten — jede Seite ist eine Datei, die `data/*.json` per relativem
`fetch()` lädt.

Diese Datei ist die Landkarte. Zwei Themen stehen bewusst woanders:

- **API-Endpunkte, Feldbedeutungen, empirische Fallen** →
  [Kickbase-API.md](Kickbase-API.md)
- **Güte des Teamstärke-Modells, verworfene Ansätze, offene Punkte** →
  [modell-befunde.md](modell-befunde.md)

## Landkarte

| Datei / Ordner | Rolle |
|---|---|
| [fetch.py](fetch.py) | Der Fetcher. Holt die laufende Saison, verdichtet die Marktwert-Historie, rekonstruiert abgeschlossene Saisons und baut `carryover.json` + `ratings.json`. Fünf Subkommandos, sonst nichts. |
| [common.js](common.js) | Von allen Seiten geteilt: Saison-Manifest und -Auswahl, Achse über Vorsaison + laufende Saison, Zugriff auf `carryover.json`, Auswertung von `ratings.json` (`expectedPoints`, `matchupScore`, `teamSideScore`), Klassifizierung in −3…+3, Score-Store. |
| [index.html](index.html) | Einstieg, Liga-Umschalter (`localStorage.kickbase_liga`), verlinkt die vier Tools. |
| [teampunkte.html](teampunkte.html) | Erzielte Punkte je Team und Mannschaftsteil, mit Positions-Ranking. |
| [matchup.html](matchup.html) | *Zugelassene* Punkte je Team, nach Position filterbar — die Rohsicht hinter dem Score. |
| [score.html](score.html) | Matchup-Ranking: Spielplan als Farbraster in Klassen −3…+3, zwei Modi (*Paarung* / *Nur Spielplan*). |
| [scores-edit.html](scores-edit.html) | Editor für ebendiese Scores. Speichert **Abweichungen** vom Modellwert im `localStorage`; `score.html` legt sie über die frische Basis (`readScoreStore` / `applyScoreDeltas`). Steht nicht im Kachelraster von `index.html`, sondern nur in der Navigationsleiste der Tools („Teamstärke"). |
| [scatter.html](scatter.html) | Marktwert gegen Gesamt- und Durchschnittspunkte, alle Spieler. |
| [kbxp/](kbxp/) | Eigenständige Forschungspipeline: Spieler-ID-Crawl, historisches Panel ab 2013/14, Teamstärke-Modell, Quoten-Inversion, Tests. |

**Zwei getrennte Python-Welten.** `fetch.py` ist Produktion und kommt mit
`requests` aus. `kbxp/` ist Forschung und braucht pandas/numpy/scipy/pyarrow.
Sie berühren sich an einer Stelle: `fetch.py ratings` importiert
`kbxp/src/model/team_strength.export()` — weich, fehlen die Pakete, entfällt
nur `ratings.json` und die Seiten rechnen mit dem rohen Mittelwert weiter.

**Der Export braucht die Parquet-Dateien nicht.** `export()` liest
ausschließlich `data/`: `seasons.json`, `data_*.json`, `matchdays_*.json`,
`carryover.json`, `season_odds.json`. Das Panel unter `kbxp/data/processed/`
trägt nur die Forschung — `--backtest`, die Tests und die Gegenprobe der
Quoten-Inversion. Ein Klon ohne Crawl kann `ratings.json` trotzdem bauen.

## Datenfluss

```
Kickbase v4 ──fetch.py live──┬─► data/data_{1,2}[_2526].json       Kader, Spieler, Punkte je Spieltag
                             ├─► data/matchdays_bl{1,2}[_…].json   Spielplan, Ergebnisse, Aufstellungen
                             └─► data/history.json                 Marktwert + Verletzungsstatus je Tag

data_*.json + matchdays_*.json ─────carryover──► data/carryover.json ──► alle Seiten
   dieselben + carryover.json
   + data/season_odds.json     ─────ratings────► data/ratings.json   ──► common.js

Kickbase v4 ──kbxp: enumerate_ids ─► raw/player_index ──backfill_history──► processed/panel.parquet
                                                                            processed/matches.parquet
                                                                            (nur Forschung, s. u.)
```

Drei Dinge, die man dem Quelltext sonst nur mühsam ansieht:

- **`data/seasons.json` ist die einzige Stelle für den Saisonwechsel.**
  Sie nennt die laufende Saison (`current`) und je Saison den Datei-Suffix
  (`""` für die laufende, `"_2526"` fürs Archiv). `fetch.py` **und** die
  Seiten lesen sie.
- **`data/history.json` ist nicht nachholbar.** `/marketValue` beantwortet nur
  ein rollendes 365-Tage-Fenster; was herausläuft, ist endgültig weg. Deshalb
  schreibt jeder Lauf den Tageswert fort. Dasselbe gilt für den
  Verletzungsstatus, den die API nur als Momentaufnahme kennt.
- **`ratings.json` entsteht am Ende jedes `fetch.py`-Laufs**, lässt sich aber
  mit `python fetch.py ratings` jederzeit ohne Login neu bauen — ebenso
  `carryover.json` mit `python fetch.py carryover`.

Unter `kbxp/data/` sind `raw/` und `interim/` ignoriert (reproduzierbar), aber
`processed/` und `manual/` bewusst **nicht**: das Panel kostet Stunden Crawl,
`manual/fine_positions.csv` ist handgepflegt und damit unwiederbringlich.

## Kommandos

```bash
python fetch.py                        # laufende Saison — jeden Spieltag
python fetch.py mv                     # Marktwert-Historie verdichten — etwa monatlich
python fetch.py carryover              # lokal, ohne Login
python fetch.py ratings                # lokal, ohne Login
python fetch.py archive --season 2526  # abgeschlossene Saison — sehr selten
```

Global: `--workers` (Default 4) und `--delay` (Default 0,3 s je Thread).

```bash
# aus kbxp/, venv aktiv, pip install -r requirements.txt
python -m src.ingest.enumerate_ids --max-id 20000 --workers 3   # einmaliger ID-Crawl
python -m src.ingest.backfill_history --workers 3               # Panel aufbauen (resumierbar)
python -m src.model.team_strength                # ratings.json schreiben + Kennzahlen
python -m src.model.team_strength --backtest     # walk-forward, aufgeschlüsselt nach Herkunft
python -m src.model.season_odds [2026/2027]      # Quoten-Inversion gegenprüfen
pytest                                           # tests/
```

Die Seiten brauchen einen Webserver — `file://` scheitert an `fetch('data/…')`.
Statisches Ausliefern genügt; darauf ist der Zuschnitt ausgelegt.

```bash
python -m http.server 8000    # dann http://localhost:8000/index.html
```

## Konventionen

- **Deutsch** in Kommentaren, Docstrings und Doku. Bestehende Dateien
  begründen, warum etwas so ist, statt zu beschreiben, was der Code tut —
  dieser Ton ist gewollt.
- **Requests bleiben gedrosselt.** Kickbase kann Zugriffe außerhalb der
  offiziellen Apps sperren. `--delay` gilt pro Thread vor jeder Anfrage, bei
  `--workers` Threads also grob `workers / delay` Requests/s — die Defaults
  sind bewusst konservativ, kein Versehen.
- **Zugangsdaten** über `.env` bzw. `KICKBASE_EMAIL` / `KICKBASE_PASSWORD`.
  `.env` ist ignoriert und bleibt es.
- **Frontend-Abhängigkeiten** sind drei CDN-Skripte (Plotly in `scatter.html`,
  Chart.js in `matchup.html`, simple-statistics in
  `score.html`/`scores-edit.html`) plus Google Fonts. Sonst nichts.
- **Saison-Zustand liegt je Seite** im `localStorage`
  (`kickbase_season_<seite>`), weil die Tools mit „Saison" Verschiedenes
  meinen. Die Liga (`kickbase_liga`) ist dagegen global.
- Vor Änderungen an API-Aufrufen: die elf empirischen Befunde in
  [Kickbase-API.md](Kickbase-API.md#empirische-besonderheiten-per-live-test-ermittelt)
  lesen. Mehrere davon (competition-spezifische Marktwerte, `null` statt `0`,
  unvollständiges `teamprofile`) liefern falsche Zahlen statt Fehlermeldungen.

## Offene Punkte

Aus [modell-befunde.md](modell-befunde.md), dort jeweils begründet:

- **Vorsaison-Gewicht datenabhängig** statt fest 0,30 — der Einbruch liegt an
  Spieltag 2–4, nicht an Spieltag 1 (§5).
- **Spielerebene ist der eigentliche Hebel** — Einsatzwahrscheinlichkeit,
  positionsspezifische Gegnerbewertung, der ungenutzte Marktwert-Verlauf aus
  `history.json` (§7).
- **Heimvorteil wird unterschätzt**, weil `hfa` von der Ridge-Strafe
  mitgeschrumpft wird (§8).

Auf **Teamebene ist das Modell fertig** — rund 90 % der messbaren Decke sind
ausgeschöpft. Gemessen wird an ρ und paarweiser Treffsicherheit, nicht am R².
