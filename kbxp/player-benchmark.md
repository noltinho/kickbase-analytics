# Bedingter Spielermodell-Benchmark (19. August 2026)

## Ziel und Messaufbau

Prognostiziert wird der Ganzjahres-Schnitt an Kickbase-Punkten je Einsatz
unter der Bedingung, dass ein Spieler in der Zielsaison gesetzt ist. Historisch
ist die Zielgruppe deshalb ex post definiert: mindestens 15 Startelfeinsätze in
Saison n. Alle Merkmale stammen strikt aus n-1, n-2 und älter oder sind vor
Saisonbeginn bekannt.

Nicht Teil der Zielgröße sind Startelfwahrscheinlichkeit, Verletzungsrisiko,
erwartete Einsätze und Gesamtpunkte. Beim Rollenmodell wird daher nur
`xp_gesetzt`, nicht die risikoadjustierte Ausgabe, verglichen.

Der Walk-forward läuft über die Zielsaisons 2019--2025 auf 1.571
Starter-Spielersaisons. Primär sind MAE und Bias; Korrelation,
saisoninterne Rangfolge und Kalibrierungssteigung sind sekundär. Unabhängige
Challenger verwenden keine Prognose eines Bestandsmodells als Merkmal.

Reproduzierbar aus `kbxp/`:

```bash
python -m src.model.player_benchmark
```

## Ergebnis auf der gemeinsamen Zielgruppe

| Modell | r | MAE | Bias | Kalibrierungssteigung |
|---|---:|---:|---:|---:|
| **fallweise** | **0,689** | **16,64** | +1,36 | 0,922 |
| empirisches Bayes | 0,686 | 16,72 | +1,74 | 0,915 |
| Ridge | 0,672 | 16,85 | +1,38 | 0,958 |
| Spline-Ridge | 0,668 | 16,96 | +1,28 | 0,976 |
| Extra Trees | 0,669 | 17,03 | +1,85 | 1,024 |
| robustes Gradient Boosting | 0,664 | 17,11 | +1,34 | 1,006 |
| Rollenmodell `xp_gesetzt` | 0,672 | 17,46 | +2,87 | 0,818 |
| eigene Historie roh | 0,651 | 18,41 | +3,34 | 0,665 |
| nur Marktwert und Position | 0,554 | 19,28 | +1,84 | 1,006 |

Die Zeile `nur Marktwert und Position` ist die Preisauskunft ohne jede
eigene Spielerhistorie: eine gewichtete Ridge auf dem logarithmierten
Marktwert plus Positions- und Fallkategorien, mit derselben
Niveaukorrektur wie alle Challenger. Sie deckt dieselbe Zielgruppe ab
wie das Produktionsmodell, der Vergleich ist deshalb direkt: 2,64
MAE-Punkte Abstand, 95-%-Intervall 2,23 bis 3,00, in 0 von 7
Zielsaisons vorn. Was der Markt vor der Saison einpreist, ist also
ein tragfähiges, aber deutlich gröberes Signal als die Historie.

Die naive Vorsaison-Linie fehlt in dieser Tabelle mit Absicht: Sie deckt
eine kleinere Spielermenge ab, und ein direkter Vergleich der MAE-Spalte
wäre irreführend. Sie steht unten in [Die naive
Vergleichslinie](#die-naive-vergleichslinie).

Das fallweise Modell ist damit der beste belegte Produktionsweg für die
erklärte Frage. Empirisches Bayes ist statistisch nicht sicher schlechter
(Delta-MAE +0,09; saisongeclustertes 95-%-Intervall -0,27 bis +0,42), bietet
aber keinen belegten Vorteil. Das Rollenmodell bei gehaltener Rolle liegt um
0,83 MAE-Punkte zurück; sein Intervall gegenüber dem fallweisen Modell liegt
mit +0,49 bis +1,19 vollständig über null.

## Die naive Vergleichslinie

Die Tabelle oben vergleicht Modelle miteinander. Die schlichtere Frage lautet,
ob das Ganze überhaupt gegen die Regel gewinnt, die jeder Mitspieler ohnehin im
Kopf hat: **Ein Spieler holt nächste Saison den Punkteschnitt je Einsatz, den
er letzte Saison geholt hat.** Ungewichtet, ein Jahr Rückblick, kein
Abklingfaktor, keine Niveaukorrektur — der Vorjahreswert wird unverändert als
Prognose eingesetzt. Sie ist damit bewusst schlechter gestellt als
`eigene Historie roh`, die drei Saisons einsatzgewichtet zusammenzieht und
zusätzlich dieselbe Fall-Niveaukorrektur wie alle Challenger erhält. Genau
darum steht sie als einzige Zeile roh in der Tabelle: Eine Vergleichslinie,
die erst zurechtgerückt werden muss, ist keine mehr.

Antworten kann die Regel nur, wenn der Spieler in der Vorsaison in derselben
Liga gespielt hat — 1.189 der 1.571 Starter-Spielersaisons. Die übrigen 382
sind Aufsteiger, Rückkehrer und Zugänge aus dem Ausland; für sie bleibt die
Zeile leer, statt eine Zahl zu erfinden.

| Modell | n | MAE | Bias | Kalibrierungssteigung |
|---|---:|---:|---:|---:|
| fallweise | 1.571 | 16,64 | +1,36 | 0,922 |
| eigene Historie roh (gewichtet, korrigiert) | 1.239 | 18,41 | +3,34 | 0,665 |
| **Vorsaison-Schnitt, ungewichtet und roh** | 1.189 | **20,84** | −3,32 | 0,583 |

Weil die drei Zeilen unterschiedlich viele Spieler abdecken, ist die Differenz
der MAE-Spalte nicht der Vorsprung. Maßgeblich ist der paarweise Vergleich auf
den Fällen, zu denen beide etwas sagen: Dort liegt das fallweise Modell **4,40
MAE-Punkte** vorn, saisongeclustertes 95-%-Intervall 3,17 bis 5,54, und die
naive Regel gewinnt in **0 von 7** Zielsaisons. Das sind rund 21 Prozent
weniger Fehler.

Der Vorsprung stammt auch nicht nur aus einer Randgruppe:

| Fall | naiv (n) | fallweise (n) |
|---|---:|---:|
| a — eigene Startelfhistorie vorhanden | 19,52 (991) | 16,45 (1011) |
| b — Historie vorhanden, aber dünn | 27,44 (198) | 15,85 (278) |

Im gut belegten Fall a, wo die naive Regel ihre besten Karten hat, bleiben
gut drei MAE-Punkte Abstand. Ihre eigentliche Schwäche steht in der
Bias-Spalte: −3,32 insgesamt und −21,06 im Fall b. Der Vorjahresschnitt je
Einsatz mischt Kurzeinsätze mit Startelfeinsätzen, und wer letzte Saison
überwiegend eingewechselt wurde, sieht dort billig aus, obwohl er in der
Zielsaison gesetzt spielt. Genau diese Verwechslung von Einsatzrolle und
Punkteertrag ist der Grund, warum das Produktionsmodell auf Startelfschnitte
statt auf Rohschnitte schaut — und warum Startelfwahrscheinlichkeit als
getrennte Größe geführt wird.

## Gemeinsame Informationen, getrennte Modellentscheidungen

Die Forschungsdaten stellen allen Modellen dieselben drei Informationen zur
Verfügung: Belegung der eigenen Historie, prognostizierte Änderung der
Teamstärke und Marktwert-/Altersrelation zum tatsächlichen Vorgänger derselben
Positionszelle. Sie werden nicht automatisch in jede Produktionsformel
gezwungen, sondern je Modell im Walk-forward geprüft.

Im fallweisen Modell verbessert die Wechselwirkung aus Historienniveau und
Belegung den Gesamt-MAE zunächst von 16,67 auf 16,62 und Fall a von 16,50 auf 16,45.
Damit ist der Historienanteil nicht mehr konstant: Bei wenigen Starts wird
stärker geschrumpft, bei vielen belegten Saisons bleibt mehr vom eigenen
Niveau erhalten. Das starke Historiensegment verbessert sich von MAE 23,35
auf 22,28.

Für Fall c ist die relative Vorgängerqualität Teil der Produktionsdefinition.
Der geringfügig bessere MAE der absoluten Variante rechtfertigt nicht, einen
10-Mio.-Neuzugang bei Augsburg und Leverkusen gleich zu behandeln:

| Variante | Messung | Ergebnis |
|---|---:|---|
| absolute Marktwert-/Alterswerte | Fall-c r 0,538 / MAE 17,98 | etwas besserer MAE, fachlich falscher Vergleichsmaßstab |
| relative Vorgängerqualität | Fall-c r 0,544 / MAE 18,12 | Produktion; Gesamt-MAE 16,64 |
| relativ plus Teamstärkedelta | Fall-c r 0,534 / MAE 18,29 | verworfen |
| Vorgängerzelle als fester Anker | Fall-c r 0,397 / MAE 24,0 | klar verworfen |

Das Rollenmodell enthält die fachlichen Ideen bereits anders: Seine
`n_eff`-Prior-Kette macht die eigene Historie belegungsabhängig, `team_delta`
bildet einen Vereinswechsel ab und `mw_resid` misst Marktwert relativ zum
Positions-/Teamkontext. Zusätzliche explizite Merkmale verschlechterten den
bedingten Backtest: Teamprognosedelta MAE 17,58, relative Vorgängerqualität
17,70 und beide zusammen 17,73, jeweils gegen 17,52 ohne den Zusatz. Deshalb
bleibt dort die bestehende Struktur. Im Live-Export wird nun allerdings die
handgepflegte aktuelle Rolle vor der TM-Nominalposition verwendet.

Die gemeinsamen Merkmale helfen auch unabhängigen Challengern: Empirical
Bayes verbessert sich von MAE 16,80 auf 16,72, Ridge von 16,93 auf 16,85.
Beide bleiben dennoch hinter dem fallweisen Produktionsmodell.

## Qualitätsausreißer und Vorgängerzelle

Eine stärkere Vorgängerzelle für alle Liganeulinge verschlechtert den
Walk-forward-Test deutlich. Teure Liganeulinge sind jedoch ein eigenes
Restfehlersegment: Bei mindestens 35 Mio. Euro TM-Marktwert lagen die
bisherigen fallweisen Prognosen in 13 Testfällen systematisch zu tief.

Die Produktion kombiniert deshalb Positionszelle und aktuelles Teamniveau mit
dem logarithmischen Marktwertverhältnis und der Altersdifferenz zum Vorgänger.
Diese relative Kalibrierung gilt für **jeden** Fall-c-Spieler; fehlt die feine
Positionszelle, fällt der Vergleich gemeinsam mit der Punktebasis über
Mannschaftsteil und Verein bis zum Positions-Ligaprior zurück. Nur im historisch
unterschätzten Qualitätsende
wird der frühere OOF-Fehler über Saisons gemittelt und mit drei Prior-Saisons
zu null geschrumpft. Dieser zusätzliche Restfehleraufschlag beginnt bei 35 Mio.
Euro und ist kein namensbezogener Override.

## Referenzspieler 2026/27

| Spieler | Fall | Historie | Positionszelle | fallweise | Rolle gesetzt |
|---|---|---:|---:|---:|---:|
| Kimmich | a | 193,4 | 182,0 | **169,7** | 172,1 |
| Burger | a | 114,2 | 113,7 | **96,7** | 108,3 |
| Moreira | c | -- | 116,6 | **98,6** | 129,0 |
| Karetsas | c | -- | 118,4 | **109,9** | 132,3 |
| Tah | a | 136,9 | 139,3 | **129,9** | 120,8 |
| Kleindienst | a | 102,5 | 81,8 | **85,5** | -- |

Kimmich erhält keinen harten Sonderbonus mehr; sein Historieneinfluss steigt
kontinuierlich mit den belegten Starts. Burger wird wegen der erwarteten
Hoffenheim-Abschwächung nicht pauschal stark abgewertet: Historisch ist der
Effekt einer Teamstärkeänderung im Zentrum klein und ein direktes Delta-Merkmal
verschlechtert den Gesamtbacktest.

Moreira liegt mit 20 Mio. Euro nahezu exakt auf dem Marktwert seiner beiden
Leverkusener ZOM-Vorgänger (Verhältnis 1,02) und ist praktisch gleich alt;
seine relative Korrektur ist deshalb klein. Karetsas liegt beim 1,40-Fachen
seiner Dortmunder Vorgänger und ist 7,3 Jahre jünger. Zusätzlich greift bei ihm
die Elite-Restfehlerkalibrierung. Moreira illustriert trotzdem die verbleibende Modellunsicherheit. Der harte Zellanker
ist historisch klar schlechter, während das Rollenmodell ihn nach der
Positionskorrektur als ZOM bei 129,0 sieht. Diese Differenz bleibt sichtbar
und wird nicht durch einen unbelegten Mittelwert verdeckt. Kleindienst bleibt
durch den Zwei-Saisons-Rückblick Fall a: Die verlorene Saison löscht seine
frühere Starterhistorie nicht. Auf 114 historischen Fällen dieses Typs erreicht
das fallweise Modell r 0,600, MAE 14,64 und Bias +3,52.

## Entscheidung

Das fallweise Modell ist die primäre Prognose für Kategorie 1--2. Es modelliert
ausschließlich die bedingte Leistung bei gesetzter Rolle. Rollen-, Verletzungs-
und Ausfallwahrscheinlichkeiten bleiben getrennte Informationen und dürfen den
Punkteschnitt nicht absenken. Das Rollenmodell bleibt als strukturell anderer
Vergleich erhalten; seine Zusatzsignale werden nur übernommen, wenn sie seinen
eigenen bedingten Backtest verbessern.
