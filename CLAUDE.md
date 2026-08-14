# Kickbase Tools — Überblick

Statische HTML-Werkzeuge zur Analyse von Bundesliga und 2. Bundesliga in
Kickbase, gespeist aus einem Python-Fetcher gegen die Kickbase-**v4**-API.
Kein Build, kein Framework, keine Abhängigkeit im Frontend außer drei
CDN-Skripten — jede Seite ist eine Datei, die `data/*.json` per relativem
`fetch()` lädt.

Diese Datei ist die Landkarte. Ein Thema steht bewusst woanders:

- **API-Endpunkte, Feldbedeutungen, empirische Fallen** →
  [Kickbase-API.md](Kickbase-API.md)

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
| [kbxp/](kbxp/) | Eigenständige Forschungspipeline: Spieler-ID-Crawl, historisches Panel ab 2013/14, Teamstärke-Modell, Quoten-Inversion, Transfermarkt-Kaderdaten, Tests. |

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
python -m src.ingest.transfermarkt               # TM-Kaderdaten, laufende Saison
python -m src.ingest.transfermarkt --von 2013 --bis 2026   # ganze Historie, ~380 Requests
python -m src.ingest.transfermarkt --profile      # Vollnamen der offenen Stammspieler
python -m src.ingest.transfermarkt --nur-zuordnen  # player_id neu herleiten, ohne Netz
python -m src.ingest.transfermarkt --vergleich   # gegen fine_positions.csv, ohne Netz
pytest                                           # tests/
```

**Die Zuordnung ist ohne Netz wiederholbar.** Die TM-Zeilen ändern sich nicht,
wenn man an der Namensheuristik schraubt — nur die Brücke `player_id` tut es.
`--nur-zuordnen` leitet sie allein aus `tm_players.csv` neu her, Vereine
inklusive. Das kostet Sekunden statt 380 Requests und ist der Weg, eine
Änderung an `namensformen` / `schwache_formen` zu messen.

**Stand der Brücke:** 3.245 von 3.278 Kickbase-Spielern haben eine TM-Zeile
(99,0 %). Nach Einsatzzeit der Spielersaison: **>900 Min 100,0 %**, 90–900 Min
99,7 %, 1–90 Min 99,6 %, ohne Minute 98,7 %. Kein Stammspieler ist offen.

**Eine `player_id` mehrfach je Saison ist normal**, nämlich bei Wechslern: sie
stehen in den Kadern beider Vereine, beide Zeilen tragen dieselbe `tm_id` (305
Fälle). Kritisch ist nur dieselbe `player_id` bei **verschiedenen** `tm_id` —
das war dreimal der Fall, alle über den Vornamen aus dem zweiten Durchgang
(„Philipp Schulze" griff nach Maximilian Philipp). `saison_bereinigen()` löst
das saisonweit über beide Ligen hinweg auf: stärkere Herkunft gewinnt
(Handarbeit > starke Form > schwache Form), bei Gleichstand bleibt alles leer.

**Handarbeit gehört in `vollname`, nicht in `player_id`.** Wer den vollen Namen
einträgt, macht die Brücke *herleitbar*: „Douglas Costa de Souza" verbindet sich
von selbst mit Kickbase' „de Souza" und übersteht jedes `--nur-zuordnen`. Nur wo
der Name das nicht leisten kann — echte Namensvettern wie Chris und Justin Löwe
bei Dresden —, gehört die `player_id` direkt gesetzt. `--nur-zuordnen` **löscht
keine vorhandenen Werte**, es ersetzt nur, was es selbst herleiten kann.

**`pt` wird nie geglaubt, ohne es gegen die Paarung zu prüfen.** Kickbase
liefert das Spielerteam `pt` zwar je Spieltag und führt es bei Transfers meist
korrekt mit — aber nicht immer: 2021/22 nannte es reihenweise Vereine, die am
Spiel gar nicht beteiligt waren (505 Zeilen, u. a. stand ganz Hansa Rostock
unter fremden IDs), und die vor Saisonstart gecrawlte laufende Saison hatte gar
keins. Deshalb validieren **beide** Fetcher jede Zeile gegen `t1`/`t2` und
leiten den Rest aus der Häufigkeit über die Paarungen des Spielers her:
`derive_teams()` in `backfill_history.py` (greift bei `--consolidate` und nach
jedem Crawl), dieselbe Regel in Miniatur in `extract_season()` in `fetch.py`,
wo jeder Performance-Eintrag seither ein Feld `team` je Spieltag trägt. Ein
strenger Paarungs*schnitt* stand zuerst und fiel durch: eine einzige korrupte
Zeile (Fröde, 21/22) machte ihn leer und ließ 26 auflösbare Zeilen offen —
die Häufigkeitsregel mit 2×-Abstand ist robust dagegen und lässt zugleich das
Direktduell zweier Klubs eines Winterwechslers ehrlich offen. Rest nach der
Reparatur: 38 Zeilen ohne Team (7 mit Minuten, 362 gesamt) — korrupte
Geisterzeilen, ehrlich leer statt falsch.

**Kickbase führt Vereine unter wechselnden IDs.** Rostock steht 2021/22 unter
`team_id` 49, ab 2022/23 unter 23; unter 49 zeigt der Spielerindex heute
„1. FC Heidenheim 1846". Vereinsnamen aus `player_index` sind deshalb nur eine
Lesehilfe und **kein Beleg** — wer wirklich wissen will, wer gespielt hat, zählt
die Partien in `matches.parquet` (jeder Verein genau 34 pro Saison).

**Rufnamen brauchen das Spielerprofil.** Kickbase führt „Aparecido Rodrigues",
Transfermarkt „Naldo" — kein gemeinsamer Bestandteil. Das Profil nennt unter
*Vollständiger Name* die Langform; `--profile` holt sie für Spieler, die sonst
unzugeordnet blieben. **Transfermarkt drosselt Profile mit HTTP 403**, das aber
nur vorübergehend: 403 steht deshalb in `Sitzung.NOCHMAL` und bekommt über
`GEDULDIG` einen eigenen Zeitplan (30/60/120/240 s statt 5/10/20/40 s). Der Lauf
ist **resumierbar** — jeder Abruf landet sofort in `data/raw/tm_vollnamen.jsonl`,
Abbruch mit Strg+C kostet nichts. Dort steht auch das leere Ergebnis, damit ein
Profil ohne das Feld nicht bei jedem Lauf erneut geholt wird; eine unbeantwortete
Anfrage (`None`) wird dagegen **nicht** festgehalten.

**Transfermarkt liefert Nominalpositionen, `fine_positions.csv` Rollen.** Beide
Dateien liegen in `kbxp/data/manual/` und meinen Verschiedenes: TM sagt, was ein
Spieler *ist*, die handgepflegte Datei, wo er in der Formation seines Vereins
*spielt*. Gemessen stimmen sie zu 79 % überein, bei Vereinen mit Dreierkette nur
zu 71 % — deshalb überschreibt `transfermarkt.py` die Handarbeit nie, sondern
schreibt `tm_players.csv` daneben. Je Verein und Saison genügt **ein** Request
(`/kader/.../plus/1`); Alter, Vertragsende, Marktwert und Vorverein fallen dabei
mit ab. Vorhandene `player_id` werden übernommen, `--neu-zuordnen` verwirft sie.

**Wie weit zurück reicht das.** Nicht Transfermarkt begrenzt, sondern Kickbase:
`/performance` reicht bis **2013/14** ([Kickbase-API.md](Kickbase-API.md#empirische-besonderheiten-per-live-test-ermittelt),
Befund 1), die **2. Bundesliga im Panel erst ab 2021/22**. 2012/13 ist bei
Kickbase nicht zu haben, obwohl TM es hätte. Ligen ohne Panel-Zeilen überspringt
der Lauf, statt Zeilen ohne `player_id` zu erzeugen. Die Zuordnung der Vereine
läuft über die Überschneidung der Nachnamen (`MINDESTGUETE`), nicht über eine
gepflegte Liste — das überlebt Auf- und Abstiege ohne Nacharbeit.

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

- **Vorsaison-Gewicht datenabhängig** statt fest 0,30 — der Einbruch liegt an
  Spieltag 2–4, nicht an Spieltag 1.
- **Spielerebene ist der eigentliche Hebel** — Einsatzwahrscheinlichkeit,
  positionsspezifische Gegnerbewertung, der ungenutzte Marktwert-Verlauf aus
  `history.json`.
- **Heimvorteil wird unterschätzt**, weil `hfa` von der Ridge-Strafe
  mitgeschrumpft wird.

Auf **Teamebene ist das Modell fertig** — rund 90 % der messbaren Decke sind
ausgeschöpft. Gemessen wird an ρ und paarweiser Treffsicherheit, nicht am R².
