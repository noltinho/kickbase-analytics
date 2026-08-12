# Teamstärke-Modell — Befunde

Notizen zur Güte des gegnerbereinigten Modells und zu den Grenzen der
Datengrundlage. Alle Zahlen sind **walk-forward** über 2013/14–2025/26
(~10.700 Team-Spiele aus `kbxp/data/processed/panel.parquet`): für jeden
Spieltag wird ausschließlich aus vorangegangenen Spieltagen geschätzt.

## Stand

`kbxp/src/model/team_strength.py` schätzt

```
zugelassene Punkte(Team i gegen Gegner j) = mu + def_i + att_j + hfa·heim
```

als gewichtete Ridge über Team-Dummies. Parameter: Zerfall 0,98 je Spieltag,
Vorsaison mit Gewicht 0,30, λ = 8. Export nach `data/ratings.json`
(`python fetch.py ratings`, läuft auch automatisch am Ende von `fetch.py`).

Güte gegenüber dem abgelösten Verfahren, gemessen an der Zahl, die in der Zelle
steht (beide auf −3…+3 diskretisiert, gleiche Information):

| | alt (Rohmittel + Jenks) | neu (Ridge) |
|---|---|---|
| Rangkorrelation ρ | 0,226 | **0,445** |
| Paarweise Treffsicherheit | 59,1 % | **68,4 %** |
| Spanne −3 → +3 | 468 Punkte | **1.132 Punkte** |

Paarweise Treffsicherheit = zwei Paarungen desselben Spieltags mit verschiedener
Klasse; wie oft liegt die höher bewertete wirklich höher. Zufall wäre 50 %.

### Worauf sich jede Zahl bezieht

Alle ρ- und R²-Werte in diesem Dokument messen dasselbe: die Vorhersage der
**zugelassenen Punkte eines Teams in einem konkreten Spiel** — also eine Zelle,
nicht ein Spieltag als Ganzes und nicht ein Team über die Saison. Gerechnet wird
immer im Modus *Paarung*.

`score.html` kennt zwei Lesarten derselben Ratings. Beide drehen sich um die
**Abwehr des Gegners**; sie unterscheiden sich nur darin, ob der eigene
Angriffswert eingesetzt wird oder der Ligadurchschnitt:

```
Paarung:        mu + def(Gegner) + att(eigenes Team) + hfa
Nur Spielplan:  mu + def(Gegner) + mittlerer Angriff der Liga + hfa
```

| Modus | ρ | R² |
|---|---|---|
| **Paarung** | **0,464** | **0,243** |
| Nur Spielplan | 0,332 | 0,110 |

*Nur Spielplan* verschenkt als Punktevorhersage bewusst die eigene Stärke,
beantwortet dafür aber eine andere Frage: wie schwer die Partie für ein
durchschnittliches Team wäre. Nur so lassen sich Spielpläne zwischen Teams
vergleichen — über die volle Saison sind sie allerdings per Konstruktion fast
gleich (Spanne 6 Klassenpunkte über 33 Spieltage), informativ wird der Modus im
kurzen Fenster (Spanne 4 über 5 ST, 9 über 10 ST).

Nebenwirkung der geringeren Spannweite: im Modus *Nur Spielplan* kommen **±3
nicht vor** (0 von 594 Zellen), weil die Klassengrenzen auf der Verteilung aller
Paarungen liegen, dieser Modus aber nur über `def` variiert. Im Modus *Paarung*
treten sie auf (29 mal +3, 49 mal −3).

## 1. Der Horizont ist fast egal

Abstand = wie viele Spieltage vor der Partie das Modell zuletzt Daten sah.

| Abstand | 1 | 2 | 3 | 5 | 8 | 13 | 17 |
|---|---|---|---|---|---|---|---|
| ρ | 0,464 | 0,462 | 0,456 | 0,448 | 0,437 | 0,424 | 0,416 |
| R² | 0,243 | 0,242 | 0,236 | 0,228 | 0,217 | 0,204 | 0,197 |

Über eine halbe Saison gehen 10 % der Güte verloren. Teamstärke ist beständig —
**frischere Daten sind nicht der Hebel.**

## 2. Kein besseres Verfahren auf denselben Daten

| | ρ | R² |
|---|---|---|
| **Ridge (aktuell)** | **0,464** | **0,243** |
| multiplikativ (Log-Link) | 0,459 | 0,204 |
| Abwehr getrennt Heim/Auswärts | 0,460 | 0,240 |
| Ridge + Gegentore als Zusatzsignal | 0,464 | 0,239 |

Früher ebenfalls geprüft und verworfen: Gradient Boosting über alle Reichweiten
(ρ 0,408 gegen 0,411 linear), kurze Fenster (letzte 5 ST: ρ 0,366; letzte 10:
0,407; letzte 17: 0,427 — die ganze Saison mit sanftem Zerfall gewinnt).

## 3. Die Decke der Merkmale

Ein Modell, das Team, Gegner und Heimrecht kennt und **im Nachhinein** auf allen
Daten gefittet wird:

| | R² |
|---|---|
| nur Heimrecht | 0,033 |
| nur eigene Abwehr | 0,062 |
| nur Gegner-Angriff | 0,138 |
| Team + Gegner + Heim | 0,261 |
| dito, je Saison eigene Werte | 0,350 ← *in-sample, siehe unten* |

**Korrektur (nachgerechnet).** Die 0,350 sind zu rund 80 % Überanpassung: 650
Parameter auf 10.784 Zeilen. Kreuzvalidiert — ein Orakel, das die Saisonstärke
aus allen *anderen* Spieltagen kennt, aber nie aus dem vorherzusagenden Spiel:

| zurückgehalten | ~6 ST | ~3 ST | ~2 ST | 1 ST |
|---|---|---|---|---|
| R² | 0,265 | 0,269 | 0,269 | **0,270** |

Die echte Decke liegt bei **0,270**. Die Ridge holt walk-forward 0,242, also
**90 %** davon. Der Schluss bleibt: auf Teamebene ist fast nichts mehr zu holen —
aber der Spielraum beträgt 0,028 R², nicht 0,107.

Wichtig für alles Weitere: 0,270 begrenzt nur Modelle mit über die Saison
**konstanter** Teamstärke. Ein Signal mit partienspezifischer Information
(Verletzungen, Aufstellung, Quoten je Spiel) ist davon nicht gedeckelt.

## 4. Warum die Decke so niedrig liegt

Anteil der zugelassenen Punkte, den allein das **Spielergebnis** erklärt:

| | R² |
|---|---|
| Gegentore | 0,582 |
| Tordifferenz | 0,841 |
| Ergebnis + beide Torzahlen | **0,892** |

Kickbase-Punkte auf Teamebene sind praktisch eine Funktion des Spielergebnisses
(Siegprämie, Gegentore, Zu-Null). Das Modell ist ein verkapptes
Ergebnisvorhersagemodell, und Fußballergebnisse sind kaum vorhersagbar. Die
restlichen 76 % Varianz sind zum großen Teil Zufall des einzelnen Spiels.

**Aufstellungswissen bringt nichts** auf Teamebene: die Formsumme genau der
Spieler, die wirklich aufliefen, dem Modell verraten → ΔR² = 0,002. Fällt ein
Star aus, spielt ein Ersatz und das Ergebnis entscheidet trotzdem.

## 5. Saisonanfang — der Schwachpunkt liegt nicht, wo erwartet

| Spieltage | ρ | R² | n |
|---|---|---|---|
| **1** (nur Vorsaison) | **0,503** | 0,231 | 288 |
| **2–4** | **0,419** | **0,198** | 969 |
| 5–8 | 0,463 | 0,229 | 1.292 |
| 9–17 | 0,449 | 0,242 | 2.907 |
| 18–34 | 0,479 | 0,255 | 5.491 |

Spieltag 1 ohne jeden aktuellen Datenpunkt ist der **beste** Wert der Tabelle.
Der Einbruch kommt an **ST 2–4**: dort gehen ein bis drei aktuelle Spieltage mit
Gewicht 1,0 ein, während die Vorsaison bei 0,30 liegt — neun verrauschte
Beobachtungen verdrängen ein stabiles Signal.

**→ Offener Punkt:** Vorsaison-Gewicht abhängig vom Datenstand statt fest 0,30
(früh hoch, mit wachsender Zahl eigener Spieltage fallend). Kostet keine neuen
Daten. Noch nicht getestet.

## 6. Saisonquoten würden helfen

Echte Quoten liegen nicht vor (odds.api hat nur Einzelspiele). Simuliert als
tatsächliche Saisonstärke plus Rauschen, kalibriert auf eine Zielkorrelation.
Die Quote geht als **Vorwissen in die Ridge**: die Strafe zieht die Stärken
gegen den quotenimplizierten Wert statt gegen null; sobald echte Spieltage
vorliegen, übernimmt die Datenseite von selbst.

R² nach Saisonabschnitt:

| Spieltage | ohne Quote | ρ=0,50 | ρ=0,70 | ρ=0,85 |
|---|---|---|---|---|
| 1 | 0,231 | 0,273 | 0,295 | **0,312** |
| 2–4 | 0,198 | 0,217 | 0,250 | **0,274** |
| 5–8 | 0,229 | 0,237 | 0,263 | **0,284** |
| 9–17 | 0,242 | 0,255 | 0,264 | **0,270** |
| 18–34 | 0,255 | 0,260 | 0,266 | **0,270** |
| **gesamt** | **0,243** | 0,253 | 0,265 | **0,273** |

Bei ρ = 0,85 (für einen Saison-Außenwettmarkt plausibel) steigt R² über die
Saison um 12 %, am Saisonanfang (ST 2–8) um **25–38 %**. Der Effekt verschwindet
zur Rückrunde hin fast — die Quote ersetzt genau die Information, die früh
fehlt.

**Diese Tabelle ist zu optimistisch.** Die simulierte Quote war eine verrauschte
Version der *eingetretenen* Saisonstärke — also aus dem Ergebnis rückwärts
gerechnet, keine Prognose. Das sieht man an der Arithmetik: 0,273 liegt **über**
den 0,270, die ein perfektes Saisonorakel erreicht (§3). Eine echte
Vorsaison-Quote ist strikt schwächer.

Der Nutzen ist trotzdem real — er liegt nur woanders, als der Durchschnitt
vermuten lässt. Siehe §8.

## 7. Der eigentliche Hebel liegt auf Spielerebene

Auf Teamebene ist das Modell fertig. Die Entscheidung, um die es geht, fällt
aber je Spieler. Dort gemessen:

| | R² |
|---|---|
| Form der letzten 5 Spiele | 0,104 |
| + Paarung und Heimvorteil | 0,129 |

Die Paarung trägt ΔR² = 0,025 — real, aber zweitrangig. Ungenutzte Hebel:

- **Einsatzwahrscheinlichkeit.** Ob ein Spieler startet, schlägt jede
  Gegnerbewertung. Aufstellungen liegen eine Stunde vor Anpfiff vor,
  Rotationsmuster lassen sich aus dem Panel lernen.
- **Positionsspezifische Gegnerbewertung.** `matchup.html` zeigt zugelassene
  Punkte bereits nach Position; im Score steckt bisher nur die Summe.
- **Marktwert-Verlauf** aus `data/history.json` (6,3 MB, ungenutzt) — Kickbase'
  eigene Einschätzung der Formkurve.

Auf Spielerebene wäre auch ein Baum-Ensemble wieder einen Versuch wert, weil die
Merkmale dort heterogen werden und echte Wechselwirkungen auftreten
(Position × Gegnerstärke, Einsatzminuten, Alter). Auf Teamebene war es
messbar wertlos.

## 8. Ligawechsler — dort ist das Modell blind, dort helfen Quoten

**Gemessen wird ab hier vor der Winterpause.** In den Kickbase-Ligen ist dort
Reset, die Rückrunde ist ein neuer Wettbewerb. Die Grenze steht in
`kbxp/data/processed/season_splits.parquet` und ist keine feste Zahl: Bundesliga
13–17, 2. Bundesliga 16–18; 2026/27 sind es **14 und 16**. Alle Zahlen dieses
Abschnitts beziehen sich auf dieses Fenster und sind deshalb nicht mit denen
oben vergleichbar. Neue Messbasis: n = 4.969, ρ 0,435, R² 0,223, paarweise 64,9 %.

Aufgeschlüsselt nach Herkunft des bewerteten Teams:

| Herkunft | Anteil | ρ | R² | paarweise | R² an ST 1–8 |
|---|---|---|---|---|---|
| gleiche Liga | 76,7 % | 0,454 | 0,239 | 65,4 % | 0,243 |
| aufgestiegen | 2,5 % | 0,411 | 0,165 | 67,8 % | **0,058** |
| **abgestiegen** | 3,1 % | **0,271** | **0,061** | **59,2 %** | **0,066** |
| aus Liga 3 | 13,9 % | 0,340 | 0,135 | 62,4 % | 0,124 |

An Spieltag 1–8 erklärt das Modell bei einem Absteiger 6,6 % statt 24,3 %.
**31,7 % aller Zellen** haben einen Ligawechsler als Team oder als Gegner.

Der Pauschalfaktor in `fetch.py` kann das nicht leisten: er vergibt **eine** Zahl
je Richtung, geschätzt auf 8 bzw. 10 Teamsaisons, während die Streuung innerhalb
der Gruppe (66–74 Punkte) so groß ist wie der mittlere Effekt selbst (−67/+145).
2026/27 bekämen Wolfsburg, Heidenheim und St. Pauli denselben Wert — der Markt
sieht Wolfsburg bei 2,6-facher Aufstiegswahrscheinlichkeit von Heidenheim.

**Umgesetzt** in `kbxp/src/model/season_odds.py` und `team_strength.py`:
Quoten aus `data/season_odds.json` → Marge per Potenzmethode entfernt →
Stärkevektor durch iterative Saisonsimulation (8.000 Saisons je Runde, trifft die
Marktquoten auf 0,5 Prozentpunkte) → über den gemessenen Faktor **508 Punkte je
Tor Überlegenheit** (r = 0,98 über 324 Teamsaisons) auf die Punkteskala → als
Straf-Zentrum `beta_prior` in die Ridge.

Der Prior gilt für **alle** Teams, ohne Schalter: wer keine eigenen Zeilen hat,
bekommt ihn unverändert; wer eine volle Vorsaison hat, wird von den Daten
überstimmt. Das regelt die Ridge von selbst.

Zwei Stellschrauben mussten dabei getrennt werden, beide über einen sichtbaren
Fehlgriff gefunden:

**`PRIOR_LAMBDA = 32` statt `LAMBDA = 8`.** Eine einzelne Zelle streut mit
**sd = 565 Punkten** um ein Mittel von 1.046. Mit λ = 8 schlägt davon ein
Neuntel durch — nach *einem* Spieltag verschob das Cottbus um +124 Punkte (ein
gewonnenes Heimspiel, 612 statt 1.046 zugelassen) und warf die Marktrangfolge
um. λ gegen null und λ gegen ein Vorwissen sind verschiedene Größen; die zweite
darf deutlich fester ziehen. Größte Bewegung durch einen Spieltag jetzt 57 statt
141 Punkte.

**`STRENGTH_SD` normiert die Spreizung.** Eine Platzierungsquote sagt vor allem,
wie ein Team zu den anderen steht, und legt den Abstand nur lose fest. Ungeregelt
lieferte die Simulation eine Stärke-sd von 92 statt der historisch gemessenen
256 — die halbe Spreizung. Auf einer Seite mit Klassen von −3 bis +3 fällt das
sofort auf:

Dazu kam eine zweite, stille Stauchung: die allgemeine Strafe λ‖β‖² zog
**zusätzlich gegen null**, auch wo ein Prior vorlag. Ohne Daten landet ein Team
damit nicht bei seinem Prior, sondern bei λ/(λ+λ_prior) = 80 % davon. Die Strafe
zeigt jetzt vollständig auf den Prior.

| | Stärke-sd | Klassenbreite | Heimvorteil |
|---|---|---|---|
| Liga 2 ohne Quoten | 192 | 93 | 1,4 Klassen |
| Liga 2, Quoten ungeregelt | 92 | 62 | **2,1 Klassen** |
| Liga 2, nur Spreizung normiert | 206 | 108 | 1,2 Klassen |
| **Liga 2, beides behoben** | **253** | **136** | **1,0 Klassen** |
| Liga 1 (keine Quoten) | 452 | 208 | 0,8 Klassen |

Bei 2,1 Klassen wäre das Heimrecht mehr wert gewesen als der Unterschied
zwischen zwei mittleren Gegnern. Rangfolge und relative Abstände des Marktes
bleiben von der Normierung unberührt.

### Maßstabsprobe an der abgelaufenen Saison

Verhältnis der zugelassenen Punkte, Top 3 gegen die letzten 3:

| | Faktor |
|---|---|
| Bundesliga 2025/26, **tatsächlich** | **1,98** |
| dieselbe Saison, Modell darauf gefittet | 1,66 |
| Bundesliga 2026/27 aus `ratings.json` | 1,73 |
| 2. Bundesliga 2025/26, **tatsächlich** | 1,58 |
| dieselbe Saison, Modell darauf gefittet | 1,43 |
| 2. Bundesliga 2026/27 aus `ratings.json` | 1,36 |

Das Modell bleibt durchweg unter dem beobachteten Faktor — gewollt: die Ridge
schrumpft, weil ein realisierter Saisonwert Rauschen enthält und eine Prognose
zur Mitte ziehen muss. Der Abstand zwischen gefittetem Modell (1,66/1,43) und
Export (1,73/1,36) ist klein, die Skala passt also. Die Aufteilung der Stärke auf
Abwehr und Angriff stimmt ebenfalls: historisch trägt `def` 45–48 % der
Spreizung, im Export 41 %.

**Der Heimvorteil ist real und wird eher unterschätzt.** Gemessen über 10.659
Team-Spiele holt die Heimelf 1.171 Punkte gegen 957 auswärts (Bundesliga,
+214) bzw. 1.092 gegen 907 (2. Bundesliga, +185) — rund 18 %. Das Modell setzt
mit −167 und −132 weniger an, weil `hfa` von der Ridge-Strafe mitgeschrumpft
wird. Es unbestraft zu lassen ist ein offener Punkt.

**Mehrere Märkte je Liga.** Die 2. Bundesliga liegt als Aufstiegsmarkt vor
(2 Plätze, 43,8 % Marge), die Bundesliga als Top-6-Markt (6 Plätze, nur 11,3 %
Marge). Dort steht Bayern bei Quote 1,00 — das heißt lediglich „sicher" und sagt
nichts darüber, *wie viel* stärker sie sind. Erst die Meisterquote (1,12) pinnt
ihr Niveau fest. `fit_strength` gleicht deshalb mehrere Märkte gleichzeitig ab;
ein Teilmarkt, der nicht alle Teams nennt, übernimmt den Margen-Exponenten des
vollständigen Marktes derselben Liga. Die Anpassung trifft den Top-6-Markt auf
0,6 Prozentpunkte, den Meistermarkt auf 2,9 — die beiden Märkte widersprechen
sich leicht, das Verfahren mittelt.

Nicht rückprüfbar: historische Saisonquoten gibt es nicht, der Backtest bleibt
also unverändert (`beta_prior=None` ist bitgleich). Belegt ist nur, dass die
Inversion ihre eigenen Quoten reproduziert und dass die Rangfolge in
`ratings.json` den Quoten folgt — Wolfsburg vor Heidenheim und St. Pauli,
Bayern deutlich vor Dortmund.

## 9. Verworfen: Tore als zweiter Kanal

Dieselbe Ridge ein zweites Mal auf der **Torskala**, beide Vorhersagen gemischt
(Mischgewichte je Saison nur aus früheren Saisons):

| Zeitraum | R² alt → neu | ρ alt → neu | paarweise alt → neu |
|---|---|---|---|
| alle Spieltage | 0,240 → **0,245** | 0,460 → 0,459 | 65,7 % → 65,7 % |
| Spieltag 2–4 | 0,204 → **0,218** | 0,420 → 0,420 | 64,8 % → 64,9 % |

Das R² steigt, **die Rangfolge nicht**. Der Kanal kalibriert die Skala, ordnet
aber nichts um — und die Seite zeigt Klassen −3…+3, also ausschließlich eine
Rangfolge. Kein Nutzen fürs Produkt.

**Lehre über den Einzelfall hinaus:** alle Zahlen der Abschnitte 1–6 sind R² und
ρ, aber nur ρ entspricht dem, was auf der Seite ankommt. `score_metrics` weist
deshalb jetzt zusätzlich die paarweise Treffsicherheit aus, und `--backtest`
schlüsselt nach Herkunft auf.

## 10. Erfüllt die Spielplan-Ansicht ihren Zweck?

Das Ziel von `score.html` ist eng: Teams finden, die über ein Fenster von etwa
fünf Spielen einen besonders leichten oder schweren Spielplan haben, und das
sichtbar machen. Geprüft an **4.771 echten 5er-Fenstern**, walk-forward, nur
Fenster vor der Winterpause.

**Modus Paarung.** Die Σ-Spalte ordnet verlässlich, ρ = 0,642 gegen die
tatsächlich erzielten Punkte, und bildet monoton ab:

| Σ | −11 | −5 | 0 | +5 | +10 | +14 |
|---|---|---|---|---|---|---|
| tatsächliche Punkte über 5 Spiele | 3.186 | 4.527 | 5.450 | 6.166 | 7.228 | 9.590 |

Spanne über die 18 Teams eines Fensters 20,9 Klassenpunkte, im Schnitt 12,2
verschiedene Werte; paarweise 75,9 % richtig geordnet.

**Modus Nur Spielplan.** Er findet die extremen Fenster, und darum geht es:

| | Punkte über 5 Spiele, um die eigene Stärke bereinigt |
|---|---|
| leichteste 5 % | **+476** |
| schwerste 5 % | **−445** |
| Unterschied | **921 Punkte**, 184 je Partie |

Die Rangkorrelation über alle Fenster ist mit ρ = 0,25 schwach — der
Spielplaneffekt ist gegenüber dem Zufall einer einzelnen Partie klein. Für die
gestellte Frage genügt das trotzdem, weil sie nicht nach einer Reihenfolge aller
18 Teams verlangt, sondern nach den auffälligen Fällen an den Rändern.

**Additivität.** Die Klassen sind nicht gleich breit — zwischen −1 und 0 liegen
in der Bundesliga 182 Punkte, zwischen +2 und +3 aber 419. Summieren ist also
streng genommen unzulässig, praktisch aber unkritisch: `−1 und +1` weicht von
`0 und 0` um 10 von 2.181 Punkten ab (0,5 %), und *Nur Spielplan* liegt zu 78 %
in den Klassen −1/0/+1, wo die Abstände fast gleich sind. Spürbar wird es erst
am Rand: `−3 und +3` bringt 193 Punkte mehr als zwei Nullen. Wer nach der Summe
sortiert, unterschätzt also extreme Spielpläne leicht.

**Behoben: tote Farben im Spielplan-Modus.** Dieser Modus benutzte die
Klassengrenzen der *Paarungs*verteilung, kann deren Extreme aber nie erreichen,
weil der eigene Angriff auf dem Ligaschnitt festliegt. Damit kamen +3 in 0,6 %
und −3 in 3,5 % der Zellen vor — zwei von sieben Farben waren praktisch tot.
`ratings.json` führt jetzt ein zweites Feld `breaks_fixture`, gebildet über die
36 möglichen Spielplan-Zellen (18 Teams × heim/auswärts), und `teamSideScore`
klassifiziert dagegen.

**Und zwar über Quantile, nicht über Jenks.** Jenks minimiert die Streuung
innerhalb der Klassen und setzt seine Grenzen deshalb dort, wo die Werte weit
auseinanderliegen — ein Ausreißer bindet gleich zwei davon. In der Bundesliga
2026/27 isoliert Jenks den FC Bayern und wirft dafür die neun schwächsten Teams
gemeinsam in die Klasse +3, obwohl zwischen ihnen 103 Punkte liegen. Genau diesen
Unterschied soll die Ansicht zeigen. Der Modus beantwortet eine Rangfrage, keine
Frage nach absoluten Punktgrenzen; gleich besetzte Klassen sind dafür das
passende Werkzeug.

| Klasse | −3 | −2 | −1 | ±0 | +1 | +2 | +3 |
|---|---|---|---|---|---|---|---|
| Paarungsgrenzen (vorher) | 3,5 % | 12,1 % | 24,9 % | 30,6 % | 22,4 % | 5,9 % | **0,6 %** |
| eigene Grenzen, Jenks | 6,5 % | 13,1 % | 18,6 % | 20,7 % | 19,3 % | 15,3 % | 6,5 % |
| eigene Grenzen, Quantile | 15,6 % | 16,2 % | 13,8 % | 14,6 % | 13,6 % | 14,0 % | 12,3 % |

| | ρ | Σ-Spanne über 18 Teams | versch. Werte | Spreizung leicht/schwer |
|---|---|---|---|---|
| Paarungsgrenzen (vorher) | +0,254 | 7,4 | 7,5 von 18 | 881 |
| eigene Grenzen, Jenks | +0,253 | 9,6 | 8,7 von 18 | 921 |
| **eigene Grenzen, Quantile** | +0,250 | **11,9** | **9,7 von 18** | **1.016** |

Die größte Klasse eines Spieltags fasst mit Jenks im Mittel 30,1 % der Zellen
(schlimmster Fall 50 %), mit Quantilen 23,8 % (38,9 %). Die Rangfolge ändert sich
nicht — der Gewinn liegt in der Auflösung der Darstellung, und genau daran hängt,
ob vier schwache Gegner in Folge auffallen.

## Zusammengefasst

- Teamebene: **fertig**, ~90 % der (korrigierten) Decke von 0,270 ausgeschöpft.
- Der verbleibende Spielraum liegt fast ganz bei den **Ligawechslern** — dort
  greifen jetzt Saisonquoten statt eines Pauschalfaktors.
- **An ρ und paarweiser Treffsicherheit messen, nicht am R².**
- Alles Weitere gehört auf die **Spielerebene**, nicht in einen besseren
  Teamscore.
