# Kickbase API – Projekt-Referenz

Kompakte, **projektbezogene** Referenz der Kickbase-**v4**-API: dokumentiert die
Endpunkte und Felder, die `fetch.py` tatsächlich nutzt – plus die Bedeutung der
kryptischen Kurzfelder, die in den offiziellen Docs weitgehend undokumentiert
sind.

Für den vollständigen Endpunkt-Katalog siehe die externen Quellen; hier steht
nur, was dieses Projekt braucht, plus empirisch (per Live-Test) ermittelte
Besonderheiten.

## Quellen

- Swagger/Doc-Portal: <https://kevinskyba.github.io/kickbase-api-doc/index.html>
- API-Doku (Apidog): <https://share.apidog.com/bca1f84a-99d7-4f8f-96a5-5e084ee24fe3>
- Repo (OpenAPI-Spec, Postman-Collection): <https://github.com/kevinskyba/kickbase-api-doc>

Stand der externen Doku: v4.5.0, „alle bekannten Endpunkte per 06.03.2026".
Response-**Felder** sind dort größtenteils **nicht** dokumentiert – die Tabellen
unten stammen aus Live-Tests dieses Projekts (Juli 2026).

## Authentifizierung

- **Basis-URL:** `https://api.kickbase.com`
- **Login:** `POST /v4/user/login`
  Body: `{"em": <email>, "pass": <passwort>, "loy": false, "rep": {}}`
  → Antwort enthält `tkn` (Bearer-Token).
- **Alle weiteren Requests:** Header `Authorization: Bearer <tkn>`.
- **Rate-Limit:** HTTP `429` → kurz warten (im Projekt: 10 s) und erneut versuchen.
- Zugangsdaten im Projekt via `.env` / Umgebungsvariablen `KICKBASE_EMAIL`,
  `KICKBASE_PASSWORD` – **nie** committen (siehe `.gitignore`).

## Basis-Konzepte

- **Competition-IDs:** `1` = Bundesliga, `2` = 2. Bundesliga.
- Eine Saison hat 34 Spieltage („day" 1–34), je 9 Spiele.
- **Datumsfelder:** `dt`/`md` in Match-/Performance-Daten sind ISO-8601-Strings
  (`2026-08-28T18:30:00Z`). In der **Marktwert-Historie** ist `dt` dagegen ein
  **Integer = Tage seit 1970-01-01** (z. B. `20645`).

## Genutzte Endpunkte

| Endpunkt | Nutzung im Projekt | Wichtigste Felder |
|---|---|---|
| `GET /v4/competitions/{cid}/matchdays` | Spielplan + Teams (Modus `live`) | `it[]` = Spieltage → je `it[]` = Spiele |
| `GET /v4/competitions/{cid}/teams/{tid}/teamprofile` | aktueller Kader je Team **inkl. Marktwert** | `it[]` = Spieler (`i`,`n`,`pos`,`st`,`mv`,`mvt`,`ap`,`sdmvt`) |
| `GET /v4/competitions/{cid}/players/{pid}` | Stammdaten einzelner Nachzügler | `mv`,`mvt`,`tfhmvt`,`cv`,`st`,`stxt` |
| `GET /v4/competitions/{cid}/players/{pid}/performance` | Punkte je Spieltag (**alle** Saisons!) | `it[]` = Saisons → `ph[]` = Spieltage |
| `GET /v4/competitions/{cid}/players/{pid}/marketValue/{tage}` | Marktwert-Historie (Archiv-Saisonende) | `it[]` = `{dt, mv}` |
| `GET /v4/matches/{mid}/details` | Spiel-/Aufstellungsdetails | `t1lp`/`t2lp` (Aufstellung), `bo` (Quoten) |

## Feld-Dekodierung

### `/matchdays` → Spiel (`it[].it[]`)
| Feld | Bedeutung |
|---|---|
| `mi` | Match-ID |
| `day` | Spieltag (1–34) |
| `dt` | Anstoß (ISO-8601) |
| `t1`, `t2` | Team-ID Heim / Auswärts |
| `t1sy`, `t2sy` | Team-Kürzel (z. B. `FCB`) |
| `t1g`, `t2g` | Tore Heim / Auswärts (**`null`** wenn nicht gespielt → im Projekt zu `0` normalisiert) |
| `st` | Spielstatus: `0` = nicht gestartet, `2` = beendet (`1` = live) |

### `/performance` → Saison (`it[]`) und Spieltag (`ph[]`)
| Feld | Ebene | Bedeutung |
|---|---|---|
| `ti` | Saison | Saison-Titel, z. B. `"2025/2026"` |
| `n` | Saison | Liga-Name, exakt `"Bundesliga"` oder `"2. Bundesliga"` |
| `ph` | Saison | Liste der Spieltags-Einträge |
| `day` | Spieltag | Spieltag-Nummer |
| `p` | Spieltag | **Punkte** (`null`, wenn Spieler nicht spielte → im Projekt `0`) |
| `mp` | Spieltag | Einsatzminuten, z. B. `"87'"` |
| `md` | Spieltag | Match-Datum (ISO-8601) |
| `pt` | Spieltag | Team-ID **des Spielers** (zur Heim/Auswärts-Ableitung) |
| `t1`,`t2`,`t1g`,`t2g` | Spieltag | Teams + Endergebnis des Spiels |
| `mdst` | Spieltag | Matchday-Status (`2` = beendet) |
| `k` | Spieltag | Event-Codes (Tore/Assists/Karten als Zahl-Array) |

> **Wichtig:** `"Bundesliga"` ist Teilstring von `"2. Bundesliga"` – bei der
> Liga-Filterung **exakt** vergleichen (`==`), nicht mit `in`.

### `/players/{pid}` und `teamprofile` → Marktwert & Stammdaten
| Feld | Bedeutung |
|---|---|
| `i` | Spieler-ID |
| `n` / `fn`,`ln` | Name / Vor-, Nachname |
| `tid`,`tn` | Team-ID, Teamname |
| `pos` | Position: `1` Tor, `2` Abwehr, `3` Mittelfeld, `4` Sturm |
| `mv` | aktueller **Marktwert** |
| `mvt` | Marktwert-Trend: `0` =, `1` ↓, `2` ↑ |
| `tfhmvt` | Marktwertänderung 24 h |
| `cv` | Vertragswert (contract value) |
| `g`,`a` | Tore, Assists (Saison) |
| `ap`,`tp` | Durchschnitts-/Gesamtpunkte |

### `/marketValue/{tage}` → Historie
| Feld | Bedeutung |
|---|---|
| `it[]` | Zeitreihe `{dt, mv}` – `dt` = Tage seit 1970-01-01, `mv` = Marktwert |
| `lmv`,`hmv` | niedrigster / höchster Marktwert im Zeitraum |

## Spieler-Status (`st` / `stl` / `stxt`)

Verfügbar auf `players/{pid}` (mit Klartext) und `teamprofile` (nur Code).
`st` ist eine **Verletzungs-/Fitness-Angabe**, `stxt` der Klartext,
`stl` eine Liste der aktiven Status-Codes.

| `st` | Bedeutung | Beispiel-`stxt` (live 26/27) |
|---|---|---|
| `0` | Fit / verfügbar | – |
| `1` | Verletzt / längerfristig raus | „Shoulder operation – out for the time being" |
| `2` | Angeschlagen / fraglich | „Thigh problems – will miss …", „Slightly stricken" |
| `4` | Aufbautraining / Reha | „After groin injury – individual ball training" |
| `3` | Gesperrt | (während der Saison üblich, im Preseason-Test nicht beobachtet) |

Verteilung Live-Test BL 26/27 (450 Spieler): `st=0`: 417, `st=1`: 9, `st=2`: 15, `st=4`: 9.

## Einsatzwahrscheinlichkeit „Sicher / Erwartet / Unsicher / Unwahrscheinlich / Ausgeschlossen"

**Abschließendes Ergebnis (verifiziert am Beispiel Bochum vs. Hertha BSC,
Saisonauftakt BL2, `matchId` `12832`):** Diese fünf Kategorien sind **kein**
strukturiertes JSON-Feld der v4-API, sondern stecken **als Bild-Grafik** in
zwei Feldern des Match-Detail-Endpunkts:

- `GET /v4/matches/{mid}/details` liefert `t1pli`/`t2pli` – Pfade zu
  **vorgerenderten PNG-Grafiken** der voraussichtlichen Aufstellung je Team
  (CDN: `https://kickbase.b-cdn.net/{pfad}`).
- Die Grafik zeigt die voraussichtliche Elf mit Spielerfotos. Unsichere
  Positionen sind mit einem **orangen Punkt** markiert; darunter listet die
  Grafik die Rotationsoptionen als Text, z. B. `Dárdai ⇌ Kolbe`.
- Es gibt **keine feinere Abstufung** in der Grafik selbst (kein separates
  „Erwartet" vs. „Unsicher" als Text) – die vom Nutzer beobachteten 5 Stufen
  in der App werden dort vermutlich zusätzlich aus Kontext (z. B. `st`/`stxt`
  eines Spielers, Anzahl Rotationsoptionen) abgeleitet, nicht direkt aus
  diesem Bild oder einem weiteren API-Feld übernommen.
- Die separaten Felder `t1lp`/`t2lp`/`t1nlp`/`t2nlp` (strukturierte
  Spielerliste `{i,n,pos,pim}`, siehe oben) sind erst **kurz vor Anpfiff**
  gefüllt; `t1pli`/`t2pli` (die Grafik) existieren dagegen **schon Wochen vor
  Saisonstart** (bestätigt am 20.07.2026 für ein Spiel am 07.08.2026).

### Der Prognose-Endpunkt

`GET /v4/base/predictions/teams/{competitionId}` liefert pro Team:
| Feld | Bedeutung |
|---|---|
| `tid`, `tn` | Team-ID, Name |
| `plpim` | **Prognose-Grafik** (PNG, voraussichtliche Aufstellung), CDN `kickbase.b-cdn.net` |
| `plpims` | Liste der Grafik-Pfade (i. d. R. eine) |
| `ts` | **Zeitstempel** der Prognose (UTC) – z. B. Bochum `2026-07-18T21:16:11Z` = 23:16 MESZ |

Dieselben Grafiken stecken auch in `/v4/matches/{mid}/details` als `t1pli`/`t2pli`.

### Warum es kein strukturiertes 5-Stufen-Feld gibt

Datenquelle ist **Ligainsider** (Feld `plpt: "Ligainsider"` im Spieler-Detail),
das die Stufen Sicher/Erwartet/Unsicher/Unwahrscheinlich/Ausgeschlossen definiert.
Die Config (`/v4/config`) enthält je Competition `pspt: 5` (= 5 Prognose-Stufen),
d. h. das Modell existiert – aber Kickbase liefert es **nur als gerendertes
Bild**, nicht als Datenfeld. Erschöpfend geprüft (Preseason, Juli 2026):

- **Alle 147 Endpunkte** der v4-Spec gesichtet; Namespace `base/predictions/*`
  brute-force durchprobiert → nur Bild-Endpunkt.
- **Per-Spieler-Felder gegen die Grafik korreliert** (Bochum + Leverkusen, je
  Punkt- vs. Nicht-Punkt-Spieler), in Competition- **und** League-Scope:
  `st`, `stl`, `iposl`, `opl`, `sl`, `smc`, `ismc`, `lst`, `mdst` unterscheiden
  Rotationskandidaten **nicht** von sicheren Startern.
- `teamcenter?dayNumber=<kommend>` und `playercenter` → keine Prognose (leer bzw.
  nur letztes Spiel). League-`squad`-Felder `lst`/`mdst` sind rein binär
  (`st=128` = nicht im Kader).

**Konsequenz / Bezugsquellen:**
1. **Binär (Punkt ja/nein)** aus der Kickbase-Grafik `plpim`/`t1pli` – nur per
   Bildanalyse/OCR.
2. **Volle 5 Stufen** nur direkt bei **Ligainsider** (ligainsider.de) – dort
   liegen die strukturierten Daten, die Kickbase als Bild einbettet.
3. **Geklärt (07.08.2026):** Bei **beendeten** Spielen (`st`/`mst` = 2) liefert
   `/matches/{mid}/details` die **tatsächliche Startelf** in `t1lp`/`t2lp`
   (exakt 11 Einträge) und die Bank in `t1nlp`/`t2nlp` (9 Einträge) – siehe
   Feldtabelle unten. Bei ungespielten Partien fehlen diese Felder komplett,
   dort gibt es nur die Prognose-Grafik `t1pli`/`t2pli`.

### `/matches/{mid}/details` → Aufstellung (nur bei beendeten Spielen)

| Feld | Bedeutung |
|---|---|
| `t1lp`, `t2lp` | **Startelf** je Team, 11 × `{i, n, pos, pim}` |
| `t1nlp`, `t2nlp` | Bank je Team, 9 × gleiche Struktur |
| `t1pli`, `t2pli` | Prognose-Grafik (nur **vor** dem Spiel vorhanden) |
| `events` | Spielereignisse, `{pi, tid, ke, mt}` (`mt` = Spielminute) |

> **Achtung:** `pos` in `t1lp`/`t2lp` ist die **Rolle in der Formation**, nicht die
> Kickbase-Position des Spielers. Beide weichen bei rund **20 %** der Starter
> voneinander ab (Beispiel: Luis Díaz ist Kickbase-`pos` 4 (STU), stand aber als
> `pos` 3 (MIT) in der Aufstellung). Für positionsbezogene Auswertungen, die auf
> `data_*.json` beruhen, ist die Kickbase-Position aus `/players/{pid}` maßgeblich.

## Empirische Besonderheiten (per Live-Test ermittelt)

1. **`/performance` liefert ALLE historischen Saisons** eines Spielers (Test:
   2013/14 bis aktuell), nicht nur die laufende. → Ermöglicht den Archiv-Fetcher,
   filtern über `ti` (Saison-Titel) + `n` (Liga).
2. **`/matchdays` hat keinen Saison-Parameter** und liefert immer die **laufende**
   Saison. → Ergebnisse abgeschlossener Saisons müssen aus `/performance`
   (`t1g`/`t2g`/`mdst`) rekonstruiert werden.
3. **`teamprofile` liefert immer den aktuellen Kader** – für vergangene Saisons
   unbrauchbar (Kader-Snapshot nötig). Zudem ist dieser Kader **unvollständig**:
   `npt` nennt die Zahl der gelieferten Spieler (z. B. 25 beim FCA), einzelne
   Spieler fehlen aber trotz passender `tid` (Essende `8254`, Saad `4604` →
   beide `tid: 13`, aber nicht in `it[]`). Wer in einer Saison **tatsächlich**
   gespielt hat, lässt sich nur über die Aufstellungen aller Spiele ermitteln
   (`t1lp`/`t2lp`/`t1nlp`/`t2nlp`) – deshalb zieht `fetch.py` die Aufstellungen
   laufend mit und baut den Archivkader ausschließlich daraus.
4. **`pos` kann `0`/`null` sein**: Für einzelne Randspieler führt Kickbase gar
   keine Position (Beispiel Lemke `13433`). Solche Spieler lassen sich keiner
   Positionsgruppe zuordnen.
5. **`p` und `t1g`/`t2g` sind `null`**, nicht `0`, wenn (noch) nicht gespielt →
   für Format-Stabilität normalisieren.
6. **`teamprofile` liefert den Marktwert schon mit** (`mv`, `mvt`, verifiziert
   identisch mit `/players/{pid}`). Ein eigener Durchlauf über alle Spieler nur
   für Marktwerte ist überflüssig – 18 Team-Abfragen ersetzen ~460 Einzelabrufe.
7. **`/marketValue/{tage}` beantwortet nur `365`.** `730`, `1825` und `3650`
   liefern `it: []`. Das Fenster rollt mit (Test am 10.08.2026: 10.08.2025 bis
   09.08.2026). Marktwerte älterer Zeiträume sind damit **nicht nachholbar** –
   sie müssen laufend gesichert werden (`data/history.json`).
8. **`/marketValue` und `/performance` sind competition-spezifisch, und zwar
   still.** Jede Competition führt einen Spieler nur, solange er in ihr steht,
   und füllt den Rest des Fensters mit einem konstanten Platzhalter, statt zu
   antworten, dass sie ihn nicht kennt. Verifiziert an Seguin (`1207`):
   Competition 1 liefert an allen 365 Tagen exakt `500000`, Competition 2 eine
   echte Kurve zwischen 2,1 und 10,9 Mio. **Wer die falsche Competition fragt,
   bekommt keine Fehlermeldung, sondern falsche Zahlen.** Bei `/performance`
   fehlt entsprechend die laufende Saison.
9. **Ein Spieler kann in beiden Ligen derselben Saison stehen** (Wintertransfer
   über die Ligengrenze; 25/26: 15 Fälle). Welche Competition dann die echte
   Marktwertreihe führt, verrät die Streuung: die führende liefert täglich
   wechselnde Werte, die andere einen eingefrorenen.
10. **`/matches/{mid}/details` enthält keine Punkte** – nur `{i, n, pos, pim}`
   je Spieler. Punkte gibt es team-weise über `teamcenter` (`it[] = {i,n,k,p}`,
   letzter Spieltag, verifiziert an Tah: `p=146` = Tag 34 der 25/26), dort aber
   **ohne Minuten**. Für Punkte *und* Minuten bleibt `/performance` mit einem
   Aufruf je Spieler der einzige Weg.
11. **`bo` (Quoten) ist nach Anpfiff `null`.** Quoten und Prognose-Grafiken sind
   nur *vor* dem Spiel abrufbar; ein Lauf nach dem Spieltag kommt zu spät.

## Nutzungshinweis (ToS)

Laut Doku-Repo: Die API verantwortungsvoll nutzen. Kickbase kann Anfragen
außerhalb der offiziellen Apps überwachen und den Zugriff sperren. Requests
im Projekt daher gedrosselt (`REQUEST_DELAY`, begrenzte Parallelität).
