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
| [teampunkte.html](teampunkte.html) | Erzielte Punkte je Team und Mannschaftsteil, mit Positions-Ranking. Der Saisonfilter zeigt die laufende Saison und 25/26 getrennt; der Spieltagsfilter umfasst jeweils 1–34. |
| [matchup.html](matchup.html) | *Zugelassene* Punkte je Team, nach Position filterbar — die Rohsicht hinter dem Score. Saison- und Spieltagsfilter entsprechen `teampunkte.html`. |
| [score.html](score.html) | Matchup-Ranking: Spielplan als Farbraster in Klassen −3…+3, zwei Modi (*Paarung* / *Nur Spielplan*). |
| [scores-edit.html](scores-edit.html) | Editor für ebendiese Scores. Speichert **Abweichungen** vom Modellwert im `localStorage`; `score.html` legt sie über die frische Basis (`readScoreStore` / `applyScoreDeltas`). Steht nicht im Kachelraster von `index.html`, sondern nur in der Navigationsleiste der Tools („Teamstärke"). |
| [scatter.html](scatter.html) | Marktwert gegen Punkte, alle Spieler. Drei x-Achsen: *Ø Punkte*, *Ø Bereinigt* und aus `player_projections_avg.json` (Liga 1) beziehungsweise `player_projections_avg_2.json` (Liga 2) *Prognose Ø* — jeweils gesund und gesetzt, Kategorie 1–2. Die Punktgröße trägt in der Prognose die Verlässlichkeit; in Liga 2 nennt der Tooltip zusätzlich die automatisch geschätzte Starter-Wahrscheinlichkeit. Der Prognose-Knopf bleibt aus, wenn die passende Datei fehlt. Vor dem ersten Spieltag startet die Seite von selbst auf *Prognose Ø*, weil `data_{liga}.json` zwar 34 Spieltage trägt, aber überall 0 Punkte und `"0'"` Minuten. **Verletzte fließen nicht in die Fair-Value-Regression**, bleiben aber mit offenem Symbol sichtbar: ihr Marktwert ist ausfallbedingt gedrückt, und die Prognose nennt bewusst den Ertrag *nach* der Rückkehr. Das Rollenmodell bleibt Forschungs- und Vergleichsmodell, wird hier aber nicht mehr angezeigt. |
| [kbxp/](kbxp/) | Eigenständige Forschungspipeline: Spieler-ID-Crawl, historisches Panel ab 2013/14, Teamstärke-Modell, Spielermodell, Quoten-Inversion, Transfermarkt-Kaderdaten, Tests. |
| [kbxp/src/model/player_features.py](kbxp/src/model/player_features.py) | Leakagefreier Unterbau des Spielermodells: Panel ohne Saisonendstand-Spalten und mit 90er-Minutendeckel, Kennzahlen je Spielersaison (p90, Minuten je Team-Spieltag, Start-/Joker-/Fehlquote), Teamniveaus je Saison, Kaderkonkurrenz. |
| [kbxp/src/model/player_avg.py](kbxp/src/model/player_avg.py) | Das **fallweise** Spielermodell für beide Ligen: Zielgröße Ø Punkte je Einsatz, gelernt nur auf gesetzten Spielern, unterschieden nach dem, was über einen Spieler bekannt ist (a eigene Historie · b neue Rolle · c neu in der Liga). Beide Ligen verwenden dieselbe gemessene Struktur, aber getrennte Trainingszeilen, Teamniveaus, Koeffizienten und Kalibrierungen. Liga 1 exportiert nach `data/player_projections_avg.json`; Liga 2 nach `data/player_projections_avg_2.json` und bestimmt mangels handgepflegter Kategorien die besten acht Kandidaten je Verein mit einer vorgeschalteten Starter-Logistik. Alle Konstanten samt Messaufbau im Moduldocstring. |
| [kbxp/src/model/player_role_model.py](kbxp/src/model/player_role_model.py) | Forschungs-Challenger des Rollenmodells: `Ø Punkte je Einsatz = p90 × Minuten je Einsatz(Rolle)`, dazu die Rollenwahrscheinlichkeit als eigener Kanal. Prior-Kette, Torhüter-Sonderweg und Walk-forward-Backtest samt Orakel-Lauf; kein Live-Export. `player_benchmark.py` ruft `backtest_p90()` für den gemeinsamen Vergleich auf. |
| [kbxp/player-benchmark.md](kbxp/player-benchmark.md) | Reproduzierbarer Vergleich der Spielermodelle und ihrer Benchmarks. Enthält Messaufbau, Ergebnisgrenzen und verworfene Ansätze. |
| [kbxp/data/processed/season_splits.parquet](kbxp/data/processed/season_splits.parquet) | Letzter Spieltag vor der **Winterpause**, je Saison und Liga — das Ende der Hinrunde ist keine feste Zahl (Bundesliga 13–17). Kickbase resettet zur Winterpause, deshalb bewertet `team_strength` nur die Spiele davor (`load_splits()` / `before_break()`). Aus den Terminlücken neu ableiten trifft 19 von 20 — 2019/20 fände man so die Corona-Pause nach Spieltag 25 statt der Winterpause nach 17. Deshalb eingecheckt statt hergeleitet. |

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

**Die Spielermodelle hängen dagegen fest am Panel.** Deshalb ist
`player_avg.export()` bewusst nicht in `fetch.py` eingehängt: Es braucht
`panel.parquet` und `tm_players.csv`. Das Panel liegt für reproduzierbare Tests
im Repo, die gecrawlte Transfermarkt-Datei bewusst nicht. Ein weicher Import
würde die Garantie oben still aushöhlen. `scatter.html` behandelt die fallweise
Ausgabe als optional (404 → der Knopf bleibt aus). Das Rollenmodell hat keinen
Live-Export mehr und läuft ausschließlich als Forschungs-Challenger im
gemeinsamen Benchmark.

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

panel.parquet + manual/tm_players.csv + manual/fine_positions.csv
  + data/ratings.json + data/history.json ─┬─player_role_model ─► player_benchmark (Forschung/Vergleich)
                                           └─player_avg ────────► data/player_projections_avg{,_2}.json ──► scatter.html
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
Einzige Ausnahme in `manual/` ist `tm_players.csv`: Transfermarkt-Daten werden
**nicht mitveröffentlicht** — im Repo steht nur der (gedrosselte) Fetcher, die
Daten holt sich jeder Klon selbst per `python -m src.ingest.transfermarkt`.

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
python -m src.ingest.backfill_history --refresh                 # laufende Saison der aktuellen Kader nachziehen
python -m src.model.team_strength                # ratings.json schreiben + Kennzahlen
python -m src.model.team_strength --backtest     # walk-forward, aufgeschlüsselt nach Herkunft
python -m src.model.player_role_model --backtest # Rollenmodell-Challenger, walk-forward
python -m src.model.player_benchmark             # gemeinsamer Modellvergleich
python -m src.model.player_avg                   # player_projections_avg.json schreiben
python -m src.model.player_avg --backtest        # walk-forward je Fall und Mannschaftsteil
python -m src.model.player_avg --gitter          # delta, fenster und rho gegeneinander
python -m src.model.player_avg --liga 2          # player_projections_avg_2.json schreiben
python -m src.model.player_avg --liga 2 --backtest  # Punkte + Starter-Auswahl walk-forward
python -m src.model.season_odds [2026/2027]      # Quoten-Inversion gegenprüfen
python -m src.ingest.transfermarkt               # TM-Kaderdaten, laufende Saison
python -m src.ingest.transfermarkt --von 2013 --bis 2026   # ganze Historie, ~380 Requests
python -m src.ingest.transfermarkt --profile      # Vollnamen der offenen Stammspieler
python -m src.ingest.transfermarkt --marktwerte  # datierter Marktwertverlauf, ~95 min
python -m src.ingest.transfermarkt --nur-zuordnen    # player_id neu herleiten, ohne Netz
python -m src.ingest.transfermarkt --nur-marktwerte  # Stichtagswerte neu, ohne Netz
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

**Der Marktwert der Kaderseite wird nicht mehr gelesen.** Er trifft eher das
**Saisonende** — Naldo steht dort für 2016/17 bei 1,5 Mio, sein datierter Stand
zum 1. August 2016 war 3,0 Mio. Als Merkmal derselben Saison wäre das Leakage,
und ableiten lässt er sich aus dem Verlauf ohnehin. `mw_vor_saison` kommt
deshalb aus dem datierten Verlauf
(`/ceapi/marketValueDevelopment/graph/{tm_id}`, JSON, 5 KB statt 91 KB HTML)
und ist der letzte bekannte Wert **vor** dem 1. August;
`mw_vor_saison_dt` nennt das Datum, weil ein Jahre alter Stand anders zu lesen
ist als ein frischer (gemessen: Median 50 Tage, nur 0,4 % älter als ein Jahr).
Abdeckung **96,3 %** der Spielersaisons mit Einsatzminuten, **99,4 %** der
Stammspieler. Der Rest sind Debütanten, die TM zum Stichtag noch nicht
bewertet hatte — dass dort nichts steht, ist selbst die Information.

**Der volle Verlauf bleibt in `data/raw/tm_marktwerte.jsonl`** (3.360 Spieler,
78.765 Stützstellen, im Schnitt 31 je Spieler, zurück bis 2004). In die CSV
geht nur der Stichtagswert; ein feineres Merkmal — etwa der Trend der letzten
Monate — kostet damit keinen zweiten Crawl. Der Endpunkt wird **nicht**
gedrosselt wie die Profilseiten: 3.361 Abrufe, kein einziges 403.

**Eine neue Saison braucht deshalb keinen neuen Crawl, nur einen neuen
Stichtag** — `--nur-marktwerte` trägt ihn aus dem gespeicherten Verlauf nach,
ohne eine Seite zu holen (dasselbe Prinzip wie `--nur-zuordnen`: die Rohdaten
ändern sich nicht, nur was man aus ihnen liest). Das ist keine Kür: die
Saison 2026/27 stand mit 5 von 1.061 Marktwerten in der CSV, und ohne sie
liefen im Spielermodell Kaderkonkurrenz und MW-Residuum still ins Leere —
Stammspieler bekamen dadurch rund zehn Minuten je Spieltag zu wenig. Nach dem
Nachtrag sind es 969 von 1.061, wie in jeder anderen Saison.

**TM und Kickbase werden nicht zu einer Datei vereint.** Der Join gehört in
den Modellcode: `panel.merge(tm.drop_duplicates(["player_id","season"]),
on=["player_id","season"], how="left")`. Eine materialisierte Join-Tabelle
veraltet, sobald eine Seite sich ändert, und das `drop_duplicates` ist nötig,
weil Vereinswechsler zwei Kaderzeilen je Saison haben (305 Fälle) — im
Modellcode ist das sichtbar, in einer stillen Datei nicht.

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
(`/kader/.../plus/1`); Geburtsdatum fällt dabei mit ab. Vertragsende und
Vorverein füllt Transfermarkt auf historischen Kaderseiten **nicht** (nur für
die laufende Saison) und sind als Modellmerkmal ohnehin ohne Wert — sie werden
nicht gelesen. Vorhandene `player_id` werden übernommen, `--neu-zuordnen`
verwirft sie.

**Wie weit zurück reicht das.** Nicht Transfermarkt begrenzt, sondern Kickbase:
`/performance` reicht bis **2013/14** ([Kickbase-API.md](Kickbase-API.md#empirische-besonderheiten-per-live-test-ermittelt),
Befund 1), die **2. Bundesliga im Panel erst ab 2021/22**. 2012/13 ist bei
Kickbase nicht zu haben, obwohl TM es hätte. Ligen ohne Panel-Zeilen überspringt
der Lauf, statt Zeilen ohne `player_id` zu erzeugen. Die Zuordnung der Vereine
läuft über die Überschneidung der Nachnamen (`MINDESTGUETE`), nicht über eine
gepflegte Liste — das überlebt Auf- und Abstiege ohne Nacharbeit.

## Zwei Spielermodelle

Seit August 2026 stehen zwei Modellansätze nebeneinander. Sie teilen die
Zielgröße (Ø Punkte je Einsatz) und die Datenschicht (`player_features.py`),
sonst nichts. Nur das auf der bedingten Zielgruppe bessere fallweise Modell
schreibt eine Live-Datei für `scatter.html`; das Rollenmodell bleibt ein
Forschungs-Challenger im gemeinsamen Benchmark.

| | **Rollenmodell** (`player_role_model.py`) | **fallweise** (`player_avg.py`) |
|---|---|---|
| Aufbau | ein Modell für alle: p90 × Minuten je Einsatz | drei Fälle, je eigene Koeffizienten |
| gelernt auf | allen bewertbaren Spielersaisons | nur gesetzten Spielern (≥ 15 Startelfeinsätze) |
| Umfang | alle bewertbaren historischen Spielersaisons | Kategorie 1–2 (142 Spieler) |
| Ausgabe | OOF-Zeilen für Benchmark und Diagnostik | Live-Prognose `xp` |
| Güte | r 0,618 · MAE 23,1 (alle bewertbaren) | r 0,684 · MAE 16,7 (Starter) |

Die beiden r-Werte der Tabelle sind **nicht** vergleichbar: sie stehen auf
verschiedenen Grundgesamtheiten. Das fallweise Modell bewertet nur Spieler, die
eine normale Saison als Gesetzte spielen — eine leichtere und zugleich die
einzige Frage, die beim Kaderbau gestellt wird.

**Auf denselben Zeilen gemessen sind die beiden Modelle nicht zu
unterscheiden.** Beide Walk-forward-Läufe auf die gemeinsamen 1.571
Starter-Spielersaisons 2019–2025 — die historische Entsprechung von
Kategorie 1–2 —, Ziel Ganzjahres-Ø je Einsatz:

| | r | MAE | Verzerrung |
|---|---|---|---|
| Rollenmodell, roh | 0,691 | 16,7 | −5,1 (t = −2,4) |
| Rollenmodell, mit `VERSATZ` | 0,684 | 16,6 | **+0,7** (t = +0,3) |
| fallweise, mit `versatz_tabelle` | 0,684 | 16,7 | **+1,2** |
| naiv: Vorjahr fortschreiben | 0,648 | 20,8 | −3,3 |
| **Mittel beider korrigierten** | **0,700** | **16,2** | +1,0 |

Der Abstand zwischen den beiden ist Rauschen: die paarweise MAE-Differenz
beträgt 0,26 bei einem Standardfehler von 0,27, das über Zielsaisons
gebootstrappte Band für Δr ist [−0,021, +0,012], und das fallweise Modell liegt
in 48,5 % der Einzelfälle näher dran.

**Beide waren auf dieser Menge verzerrt, und beide sind es nicht mehr.** Die
Korrektur ist in beiden Fällen dieselbe Idee — der Niveauversatz, der nach der
Anpassung übrig bleibt, geschätzt aus den Out-of-fold-Resten *aller* früheren
Saisons und je Fall abgezogen. Dass das Fenster **wächst** statt zu rollen, ist
kein Detail: die Saisonverzerrung ist nicht autokorreliert (r(t, t−1) = −0,016),
ein rollendes Fenster jagt also Rauschen und macht alles schlechter.

Beim Rollenmodell wird im Benchmark ausschließlich die bedingte Prognose für
tatsächlich gesetzte Spieler verglichen. Über alle Kandidaten gemittelt ist es
unverzerrt; bedingt auf diese Zielgruppe liegt es roh systematisch zu tief.
Dieselbe historische ex-ante-Gruppe nach Ausgang getrennt (Fall b):

| | n | Verzerrung |
|---|---|---|
| alle | 776 | +6,7 |
| die Starter wurden | 351 | −8,4 |
| die übrigen | 425 | +19,2 |

Eine pauschale Eichung über alle Spieler wäre also falsch herum. Der gemeinsame
Benchmark eicht und bewertet deshalb auf seiner erklärten Starter-Zielgruppe.

**Dass der Mittelwert beider Modelle beide schlägt, bleibt der eigentliche
Befund**: ihre Fehler sind nur teilweise korreliert, sie tragen also
verschiedene Information. Nach der Korrektur liegen ihre Prognosen für
2026/27 im Mittel 7,8 Punkte auseinander, und ihr Mittelwert trifft die vier
Referenzspieler auf höchstens vier Punkte genau (Olise 206, Gadou 100,
De Cat 88, Treu 66). Solange beide laufen, ist der Mittelwert die bessere Zahl
als jede einzelne. Die Saison entscheidet den Rest.

### Das fallweise Modell

**Zielgröße ist der Kickbase-Schnitt Ø Punkte je Einsatz**, modelliert unter
der Annahme *keine Verletzungen*: Ausfälle sind nicht vorhersehbar und
verändern den Schnitt kaum, weil man den Spieler ersetzen kann. Verletzte
bekommen deshalb die volle Zahl — den Ertrag *nach* der Rückkehr — plus eine
Markierung, damit `scatter.html` sie aus der Fair-Value-Regression heraushält.

**Liga 2 übernimmt die Struktur, nicht die Liga-1-Koeffizienten.** Historie,
Vorgängerzellen, Teamstärke, Positionssockel, Fit und Kalibrierung werden je
Liga getrennt gebildet. Wegen des Panel-Starts 2021/22 ist ihr Walk-forward
kürzer (Zielsaisons 2023–2025), aber stabil genug für den Produktionsweg:
gesamt n 647 · r 0,529 · MAE 16,2 · Verzerrung +0,1; Fall a allein n 317 ·
r 0,591 · MAE 16,2. Eine stumpfe Übernahme des Liga-1-Niveaus wäre trotz
gleicher Modellform falsch.

**Die Zweitliga-Zielgruppe ist reproduzierbar statt handgepflegt.** Für sie
existiert keine `fine_positions.csv`. Eine vorgeschaltete Logistik verwendet
nur vor Saisonbeginn bekannte Größen — Starts und Einsätze aus den letzten
zwei Saisons, Alter sowie Marktwert-Rang in Verein und Liga — und wählt je
Verein die besten acht, davon vier als Kategorie 1. Das ergibt wie in Liga 1
144 Kandidaten. Im Walk-forward 2022–2025 wurden 64,2 % tatsächlich Starter
(Recall 42,5 %); Verletzungen stecken dabei als nicht vorhersagbarer Teil in
den vermeintlichen Fehlern. Eine globale Wahrscheinlichkeitsschwelle wurde
verworfen, weil sie kleine Vereine fast leer und große übervoll ließ.

Drei Fälle, danach unterschieden, was über einen Spieler bekannt ist:

| Fall | wer | Merkmale | n | r | MAE | Verzerrung |
|---|---|---|---|---|---|---|
| a | Vorsaison in Liga 1 und dort Starter | eigene Historie, belegungsabhängig | 1.011 | 0,709 | 16,4 | +1,9 |
| b | Vorsaison da, aber kein Starter | Historie als Nebenmerkmal | 278 | 0,556 | 15,9 | +0,1 |
| c | keine verwertbare Vorsaison | Vorgängerzelle, relatives MW und Alter | 282 | 0,544 | 18,1 | +0,8 |

Walk-forward über die Zielsaisons 2019–2025, gesamt r 0,689 bei MAE 16,64. Die
Vergleichsmarke gilt nur für Fall a: der rohe Vorjahres-Startelfschnitt
schreibt mit r 0,711 fast gleich gut fort, aber bei MAE 18,5 und einer
Verzerrung von **+8,7**. Der Gewinn liegt im Niveau, nicht im Rang — und
darin, dass b und c überhaupt eine Zahl bekommen.

**Eingänge aus Startelfeinsätzen, Ziel über alle Einsätze.** Der
Vorjahres-Schnitt *aus Startelfeinsätzen* sagt den nächsten Gesamtschnitt
besser voraus (r 0,678) als der Vorjahres-Gesamtschnitt selbst (0,664).
Joker-Kurzeinsätze sind im Eingang Rauschen, im Ziel aber echter Ertrag.

**Die Vorgänger-Zelle trägt Fall c — und nur ihn.** Was die Startelfspieler
desselben Vereins auf derselben Position zuletzt geholt haben, ist die
wörtliche Antwort auf die Frage, die bei einem Neuzugang gestellt wird: Gadou
erbt von Anton, Schlotterbeck und Bensebaini, Karetsas von Brandt und Beier.
Die feinste Stufe deckt zwei Drittel der Fälle c ab, der Rest fällt über
Mannschaftsteil und Verein bis aufs Ligamittel zurück.

**Jeder Neuzugang wird relativ zu dieser Zelle bewertet.** Das Modell nutzt
`log2(MW neu / MW Vorgänger)` und die Altersdifferenz, nicht denselben absoluten
Marktwertaufschlag bei jedem Verein. Die relative Variante erreicht in Fall c
r 0,544 / MAE 18,12; absolute Werte liegen beim MAE minimal besser (17,98),
widersprechen aber dem erklärten Vergleichsmaßstab. Der zusätzliche Aufschlag
für das auf früheren OOF-Fehlern unterschätzte Elite-Segment beginnt bei
35 Mio. Euro; er ersetzt die allgemeine relative Kalibrierung nicht.

**Wie stark sie durchschlagen darf, ist gemessen.** Neuzugänge holen im Schnitt
83 % ihrer Zelle (93,7 → 77,7), und der Abschlag wächst mit ihr: im obersten
Fünftel stehen 128,4 gegen tatsächlich erzielte 92,2 — 36 Punkte darunter.
Bivariat `Ist = 0,41 × Zelle + 39,1` bei r 0,345; neben `z_hat` und der
relativen Qualität bleibt ein Koeffizient von 0,095, weil die Teamstärke denselben Inhalt schon trägt
(r 0,749 zwischen beiden). Mehr Gewicht kostet monoton — 0,3 → Fall c r 0,498
bei MAE 19,0, volles Gewicht → 0,397 / 24,0. Als **Anker** taugt sie deshalb
nicht, als **Merkmal** neben der Teamstärke und relativen Qualität schon.

**Der Mannschaftsteil trägt dagegen — und nur als Wechselwirkung.** Ohne ihn
ist das Modell systematisch schief (Torhüter −8,5, Zentrum +7,4), was im
Gesamt-r verschwindet, weil sich die Fehler gegeneinander aufheben; deshalb
weist `--backtest` die Verzerrung je Teil aus. Das Gesetz dahinter:
Teamstärke schlägt sich in der Offensive nieder (+6,2 Punkte je sd gegenüber
dem Torwart), im Tor gar nicht — wer dominiert, lässt seinen Keeper ohne
Paraden. Die Aufteilung **TW · IV · AV · ZDM+ZM · ZOM+FL+ST** ist gemessen;
feiner wird schlechter, weil die Zellen zu dünn werden.

**Der Teil wird gemeinsam über alle Fälle geschätzt**, die Grundmerkmale je
Fall — das ist der Grund für die *eine* Regression mit Fall-Dummies statt drei
getrennter. Je Fall geschätzt landete Fall c bei r 0,517 statt 0,527 und setzte
den Abschlag fürs Zentrum auf −22 Punkte, wo die Ligamittel sechs hergeben.

**Die Live-Teamstärke muss skaliert werden.** `z_hat` ist historisch eine
geschrumpfte Prognose (sd 0,84), `ratings.json` liefert dagegen die Skala der
*gemessenen* Stärke (sd 0,97). Ohne Korrektur stünde Bayern bei +3,04 statt
+2,64 und alle Bayern-Spieler zu hoch.

**Eine verlorene Saison löscht nicht, was bekannt ist.** Ob ein Spieler auf
der Bank saß oder verletzt war, steht im Panel nicht: beides ist
`player_md_status == 4` mit null Minuten, und `history.json` trägt den
Verletzungsstatus erst ab dem Tag, an dem der Fetcher ihn mitzuschreiben begann
(2026-08-10) — für keine vergangene Saison. Statt die Ursache zu raten, sucht
die Fallzuordnung über **zwei** Saisons (`RUECKBLICK`): wer in einer davon
Starter war, bleibt Fall a. Gemessen besser auf jeder Kennzahl (r 0,684 statt
0,678, MAE 16,7 statt 16,9), und Fall b gewinnt am meisten — Verzerrung +0,7
statt +2,7. Drei Saisons sind wieder schlechter (0,682), dann kommen die
echten Absteiger mit herein. Praktisch trifft das Spieler wie Kleindienst
(2024/25 31 Startelfeinsätze bei Ø 112,7, danach eine Saison ohne Einsatz):
86,0 statt 72,6.

**Der Starter-Filter braucht nur eine Bedingung.** Eine zweite über die
Einsatzlänge lag nahe, greift aber nie: schon ab acht Startelfeinsätzen liegt
die kleinste gemessene Einsatzlänge bei exakt 60,0 Minuten. Ein Test hält das
fest, damit die Annahme nicht still veraltet.

### Das Rollenmodell

**Zielgröße ist der Punkteschnitt je Einsatz** — dieselbe Skala, die Kickbase
als „Ø Punkte" zeigt. Nicht Punkte je Spieltag: vergangene Saisons spiegeln
Rotation und Verletzungen bereits wider, und Ausfälle sind vorhersehbar und
ersetzbar. Bewertet wird die Hinrunde (Kickbase resettet zur Winterpause).

```
Ø Punkte je Einsatz = p90 (Spielerqualität) × Minuten je Einsatz (Rolle) / 90
```

**Der größte Hebel ist die Rolle, nicht die Qualität.** Das ist gemessen und
war die Überraschung des Umbaus: mit der tatsächlichen Rolle der Zielsaison
steigt das Modell von r 0,618 auf **0,779**, und die schwächste Gruppe
(Spieler ohne jede Historie) von 0,40 auf 0,72. Genau diese Information steht
handgepflegt in `fine_positions.csv` — die Datei ist damit nicht Beiwerk,
sondern der wertvollste Eingang des Modells.

Drei Kanäle, jeder mit eigener Zielgröße:

| Kanal | schätzt | Güte |
|---|---|---|
| **Qualität** | p90, über die Prior-Kette | r 0,733 bei Stamm-Vorsaison |
| **Rolle → Einsatzlänge** | Tabelle Position × Rolle | Zellstreuung nur ±5 Min |
| **Durchsetzung** | setzt er sich durch? | AUC 0,794, Brier-Skill +0,215 |

**Der dritte Kanal speist den zweiten.** Die Übergangsmatrix kennt nur die
*alte* Rolle und gibt damit jedem gesetzten Spieler dieselben 59 % Verbleib —
dem Torjäger wie dem Wackelkandidaten. Genau diese Unterscheidung leistet aber
die Rollen-Logistik. Deshalb liefert die Matrix nur noch die **Form** der
Verteilung, ihre Masse auf „gesetzt" kommt aus `p_rolle`
(`verteilung_kalibriert`). Das war an der Spitze bares Geld: dort sagte die
rohe Matrix 76,2 Minuten je Einsatz voraus, tatsächlich waren es 80,1.

**Der Ertrag wird nicht mehr nachkalibriert.** Bis August 2026 stand hinter dem
Produkt eine additive Ridge. Sie war redundant — ihre wirksamen Merkmale
stecken schon in der p90-Ridge, die rollenbezogenen trägt `p_rolle` — und ihre
Niveauwirkung war schädlich: sie mittelte den Rest über alle früheren Saisons,
darunter 2020–2023 mit −11 Punkten, und trug das in eine Gegenwart, in der das
Produkt unverzerrt ist. Ohne sie: r 0,618 statt 0,609, MAE 23,1 statt 23,2,
Verzerrung der jüngsten drei Saisons −1,4 statt +4,7. **Das Niveau kommt jetzt
allein aus der Prior-Kette**, die es je Zielsaison aus frischen Daten hat.

**Warum eine Tabelle und nicht die eigene Historie:** Die eigene Einsatzlänge
trägt die *alte* Rolle. Für gesetzte Spieler misst die Tabelle IV 85,5 · AV
84,1 · ZDM 83,1 · ZM 81,1 · ST 80,3 · ZOM 79,9 · FL 78,9 Minuten, und die
Streuung *innerhalb* jeder Zelle beträgt nur 4–6 Minuten — sie ist präziser
als ein Einzelspieler-Mittelwert und trägt zugleich die neue Rolle. Beide
Quellen werden trotzdem gemischt (`K_MINJE`), weil die Historie sagt, wie ein
Trainer diesen Spieler *behandelt*: nur Rolle 0,593 · nur Historie 0,591 ·
Mischung 0,610.

**Rollen halten nicht — deshalb wird über sie gemittelt.** Nur 59 % der
Gesetzten bleiben gesetzt, 10 % der Randspieler steigen auf. Eine
festgeschriebene Rolle überschätzte Stammspieler um 7,1 Minuten je Einsatz und
unterschätzte Randspieler um 16,1. Die Übergangsmatrix ist die Antwort:
`minje = Σ_r P(Rolle = r) × Tabelle[Position, r]` — dieselbe Logik wie in der
Prior-Kette, nur auf Rollen angewandt.

**Gemessen wird auf der Zielgruppe, nicht auf dem Kader.** Ein Kader besteht
zu rund einem Drittel aus Spielern, die normalerweise nicht spielen. Ein r
über alle bewertbaren Spielersaisons misst damit zum guten Teil, wie gut man
Randfiguren sortiert — eine Frage, die beim Kaderbau nicht gestellt wird.
`--backtest` weist deshalb beides getrennt aus: alle bewertbaren (r 0,618),
die Zielgruppe *ex ante* (r 0,661) und *ex post* (0,622 bzw. 0,640 gegen 0,587
bzw. 0,602 naiv), dazu den oberen Rand als Top-N-Treffer auf der gemeinsamen
Maske (Top 5: 49 % gegen 44 %) und den Ist-Schnitt der gewählten Top-20
(127,0 gegen 125,3; perfekte Rückschau 147,2).

Historisch gibt es `kategorie` nicht, deshalb ein Stellvertreter aus vorab
bekannten Größen: Vorjahres-Startquote > 0,70 **und** Marktwert-Rang ≤ 2 auf
der Position. Gegen die echten Kategorien 2026/27 geprüft trifft er den Anteil
(28 % gegen 31 %) bei 56 % Präzision — mehr ist ohne die Handarbeit nicht zu
holen, und genau das macht die Handarbeit wertvoll.

**Torhüter sind ein Sonderweg, kein eigenes Modul.** Sie mischen gegen die
Team*abwehr* statt gegen den Angriff. Mehr als eine bessere Niveau-Eichung ist
dort nicht zu holen: die Rangfolge etablierter Torhüter schlägt auch das
Modell nicht (Reliabilität ihres Punkteschnitts 0,38).

**Was verworfen wurde, mit Zahl:** das Vorgängermodell `p90 × erwartete
Minuten je Spieltag` (r 0,548 — *unter* dem stumpfen Fortschreiben des
Vorjahres mit 0,596); die Ertrags-Ridge (0,609 gegen 0,618 ohne sie); ein
Knick in der Ertragskalibrierung, um die Spitze anzuheben (r praktisch
unverändert, Spitzen-Verzerrung schlechter); eine Potenz `n_eff**a` in der
Prior-Kette (die p90-Ridge nimmt sie über `belegt` wieder zurück); ein
direkter Pfad über `avg_hist` neben `xp_rolle` (Koeffizient 0,01–0,03);
namentliche Rollenvorgänger (0,682 mit wie ohne); BL2-Saisons als zusätzliche
Trainingszeilen (0,678 statt 0,682). Alle stehen als Messung im
Moduldocstring, nicht als Schalter im Code.

**Zwei Fragen sind gemessen und beantwortet, damit sie nicht wiederkehren.**
Erstens: die Schrumpfung der Prior-Kette ist richtig dosiert — die partielle
Regression `ist_p90 ~ p90_hist + baseline` liefert je `n_eff`-Band die
Gewichte 0,00 / 0,48 / 0,59 / 0,74 / 1,01, das Modell setzt 0,22 / 0,51 /
0,68 / 0,78 / 0,84, keine Abweichung erreicht zwei Standardfehler. Zweitens:
dass auf dem Ganzjahr trainiert und auf der Hinrunde bewertet wird, kostet
nichts — der Hinrunden-Aufschlag eines Spielers wiederholt sich nicht
(r = −0,017 über 4.082 Spielersaison-Paare). Kanes 30 Punkte Hinrunden-Vorsprung
sind Rauschen, kein Merkmal.

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

- **Der Rollenkanal ist der Hebel, und er ist erst zur Hälfte gehoben.** Das
  Orakel (Rolle bekannt) erreicht r 0,779, der Forschungs-Challenger aus der
  Vorsaison-Rolle 0,606. Vor einer erneuten Produktion müsste eine belastbare,
  vor Saisonbeginn bekannte Rollenquelle diesen Abstand im Walk-forward
  nachweislich schließen.
- **Ungenutzt für den Rollen-Challenger:** der Marktwert-*Verlauf* aus
  `history.json` und `tm_marktwerte.jsonl` — ein steigender Marktwert im Sommer
  könnte ein Rollen-Signal sein. Das gehört zuerst in den Benchmark, nicht in
  einen neuen Live-Export.
- **Das Vorsaison-Gewicht der Teamstärke datenabhängig** statt fest 0,30 —
  der Einbruch liegt an Spieltag 2–4, nicht an Spieltag 1.
- **Positionsspezifische Gegnerbewertung** — ein Spielplan ist für
  Innenverteidiger und Stürmer verschieden schwer; das Spielermodell nutzt
  bisher nur die Teamstärke des eigenen Vereins, nicht die der Gegner.
- **Heimvorteil wird unterschätzt**, weil `hfa` von der Ridge-Strafe
  mitgeschrumpft wird.
- **Die Liga-2-Zielgruppe bleibt der schwächste Kanal.** Mangels
  handgepflegter Kategorien trifft die Starter-Logistik 64,2 % ihrer acht
  Kandidaten je Verein. Das reicht für eine ehrliche bedingte Prognose, ist
  aber klar unter dem Informationsgehalt gepflegter Rollen; Sommertransfers,
  Verletzungen und Trainerentscheidungen setzen hier die Grenze.
- **Das fallweise Modell überschätzt die Fälle b und c um +4,7 / +3,7.** Die
  Rückgewichtung (ρ 0,6) hat den größeren Teil geholt, der Rest ist zum guten
  Teil Kohortenglück: der Ist-Schnitt dieser Gruppe schwankt von Saison zu
  Saison mit sd 8,3. Das Kohortenniveau der Vorsaison als Merkmal senkt zwar
  die mittlere Verzerrung, verdoppelt aber die Schwankung je Saison — verworfen.
- **Zwei Modelle laufen nebeneinander.** Nach der Hinrunde 2026/27 sollte eins
  davon verschwinden. Dafür müssen jetzt beide Prognosen gesichert werden,
  bevor der erste Spieltag sie überschreibt.
- **Kategorie 3–4 fehlt im fallweisen Modell.** Es braucht dafür eine
  Einsatzwahrscheinlichkeit — genau den Kanal, den das Rollenmodell hat.

Auf **Teamebene ist das Modell fertig** — rund 90 % der messbaren Decke sind
ausgeschöpft. Gemessen wird an ρ und paarweiser Treffsicherheit, nicht am R².

Auf **Spielerebene** trifft das Rollenmodell den Punkteschnitt je Einsatz mit
r 0,642 gegen 0,627 fürs stumpfe Fortschreiben, bei kleinerem Fehler
(MAE 22,5 gegen 23,8) und für rund 1.200 Spieler zusätzlich, über die das
Vorjahr gar nichts sagt. Auf den drei jüngsten Zielsaisons ist es praktisch
unverzerrt (−1,5 / −2,3 / −0,2). Auf der Produktebene sind es paarweise
77,5 % gegen 72,8 % und 71,5 % (Marktwert), die Trefferquote ±10 Punkte je
Einsatz 28,8 % gegen 22,6 %.

Eine Ehrlichkeit gehört dazu: beim reinen Pick-Ertrag im Band 2–10 Mio liegt
stumpfes Sortieren nach Vorjahrespunkten knapp vorn (51 % der Lücke zur
perfekten Rückschau gegen 49 %). Es greift die Ausreißer des Vorjahres ab, von
denen einige wiederholen, während das Modell sie zur Mitte zieht — zu Recht,
denn gemessen fällt die Gruppe mit Vorjahres-Ø über 170 im Schnitt von 188 auf
170 zurück.
