"""Kaderdaten von Transfermarkt - feine Positionen und Stammdaten.

Warum ueberhaupt: Kickbase kennt nur vier Positionsklassen (1..4) und keine
Stammdaten. Die feine Position lag deshalb bisher handgepflegt in
``data/manual/fine_positions.csv``. Transfermarkt hat beides, und zwar
vollstaendig auf **einer** Seite je Verein::

    /<slug>/kader/verein/<id>/saison_id/<jahr>/plus/1

Eine Zeile dort traegt TM-ID, Name, feine Position, Geburtsdatum, Groesse,
Fuss, Vorverein samt Abloese, Vertragsende und Marktwert. Damit kostet ein
kompletter Lauf **36 Requests** (18 + 18 Vereine) statt eines Profilaufrufs
je Spieler. Der Aufwand liegt nicht im Holen, sondern im Zuordnen.

Drei Entscheidungen, die man dem Code sonst nur muehsam ansieht:

* **Kein Parser-Paket.** Die Kadertabelle ist maschinell erzeugtes,
  gleichfoermiges Markup; die paar Felder holen gezielte Ausdruecke
  zuverlaessig genug. lxml/bs4 waeren eine Abhaengigkeit fuer 36 Seiten im
  Jahr - das lohnt nicht. Bricht das Markup, faellt es als Zeilenzahl 0 auf,
  nicht als stille Falschzuordnung.
* **Kein handgepflegtes Vereins-Mapping.** Welcher TM-Verein welchem
  Kickbase-Kuerzel entspricht, wird ueber die **Ueberschneidung der
  Nachnamen** bestimmt. Das ueberlebt Auf- und Abstiege ohne Pflege und
  verraet sich selbst, wenn es danebengeht (die Guete steht im Protokoll).
* **Eine Datei.** ``data/manual/tm_players.csv`` traegt Rohdaten *und* die
  Bruecke ``player_id`` (Kickbase). Ein erneuter Lauf holt die TM-Spalten
  frisch, **uebernimmt aber vorhandene ``player_id`` unveraendert** - von
  Hand nachgetragene Zuordnungen gehen also nicht verloren.

``fine_positions.csv`` wird **nicht** angefasst. Sie ist handgepflegt und
laut CLAUDE.md unwiederbringlich; ``--vergleich`` stellt beide nur
gegenueber.

**Wie weit zurueck.** Transfermarkt gibt jede Saison her - ``saison_id=2012``
antwortet mit vollen Kadern. Die Grenze setzt Kickbase: ``/performance``
reicht bis **2013/14** (Kickbase-API.md, Befund 1), und die **2. Bundesliga
erst ab 2021/22**. Was kein Kickbase-Gegenstueck hat, laesst sich nicht
verbinden - Ligen ohne Panel-Zeilen werden deshalb stillschweigend
uebersprungen, statt Zeilen ohne ``player_id`` zu erzeugen.

Aufruf (aus kbxp/)::

    python -m src.ingest.transfermarkt                     # laufende Saison
    python -m src.ingest.transfermarkt --von 2013 --bis 2026   # alles, ~380 Requests
    python -m src.ingest.transfermarkt --nur-zuordnen      # player_id neu, ohne Netz
    python -m src.ingest.transfermarkt --vergleich         # Abgleich, kein Netz
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import re
import time
import unicodedata
from collections import Counter

import requests

from ..paths import LEGACY_DIR, MANUAL, PROCESSED, RAW, ensure_dirs
from .kickbase_client import enable_utf8_stdout

ZIEL = MANUAL / "tm_players.csv"

# Zwischenspeicher der Profilabrufe. Getrennt von ZIEL, weil er reproduzierbar
# ist (raw/ ist ignoriert) und weil er nach *jedem* Abruf waechst - siehe
# profile_nachholen.
VOLLNAMEN = RAW / "tm_vollnamen.jsonl"

BASIS = "https://www.transfermarkt.de"

# Ohne Browser-Kennung antwortet Transfermarkt mit 403. Die Drosselung unten
# ist derselbe Gedanke wie --delay im Fetcher: der Lauf ist selten und klein,
# also darf er langsam sein.
KOPFZEILEN = {
    "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                   "AppleWebKit/537.36 (KHTML, like Gecko) "
                   "Chrome/126.0.0.0 Safari/537.36"),
    "Accept-Language": "de-DE,de;q=0.9",
}

# Kickbase-Liga -> Wettbewerbsschluessel bei Transfermarkt, und die
# Schreibweise, unter der dieselbe Liga im Panel steht.
WETTBEWERB = {"1": "L1", "2": "L2"}
LIGA_NAME = {"1": "Bundesliga", "2": "2. Bundesliga"}

# Ab wie vielen Namenstreffern ein TM-Verein als zugeordnet gilt. Richtige
# Paarungen liegen gemessen bei 18-31 von rund 30 Spielern, falsche bei 0-2:
# ein einzelner Treffer ist ein Namensvetter, kein Verein.
MINDESTGUETE = 6

# TM-Rohlabel -> die acht Kuerzel, die fine_positions.csv benutzt.
# Bewusst ohne Sammelbegriffe ("Abwehr", "Mittelfeld"): die stehen bei
# Spielern ohne festgelegte Position und waeren geraten. Unbekanntes bleibt
# leer und wird am Ende gemeldet.
POSITION = {
    "Torwart": "TW",
    "Innenverteidiger": "IV",
    "Linker Verteidiger": "AV",
    "Rechter Verteidiger": "AV",
    "Defensives Mittelfeld": "ZDM",
    "Zentrales Mittelfeld": "ZM",
    "Offensives Mittelfeld": "ZOM",
    "Linksaussen": "FL",
    "Rechtsaussen": "FL",
    # fine_positions.csv kennt keine Aussenbahn im Mittelfeld; dort sitzen
    # diese Spieler unter FL.
    "Linkes Mittelfeld": "FL",
    "Rechtes Mittelfeld": "FL",
    "Haengende Spitze": "ST",
    "Mittelstuermer": "ST",
}

SPALTEN = [
    "player_id", "tm_id", "season", "liga", "verein", "spieler",
    "position_tm", "position_fine", "geburtsdatum", "vertrag_bis",
    "marktwert_eur", "zuvor", "vollname",
]


# --------------------------------------------------------------------------
# Textnormalisierung - traegt sowohl den Positionsschluessel als auch den
# Namensabgleich, deshalb an einer Stelle.
# --------------------------------------------------------------------------

# Zeichen, die NFKD *nicht* zerlegt, weil sie keine Grundform mit Akzent sind,
# sondern eigene Buchstaben. Ohne diese Tabelle fielen sie unten ersatzlos weg -
# aus 'Grønbæk' wuerde 'grnbk' und aus 'Dźwigała' 'dzwigaa'. Das ist nicht nur
# haesslich, es laesst verschiedene Namen aufeinander fallen.
# ø wird zu 'oe', nicht zu 'o': Kickbase schreibt denselben Spieler mit
# deutschem Umlaut ('Pierre Höjbjerg'), Transfermarkt mit daenischem Zeichen
# ('Pierre-Emile Højbjerg'). Da ö bereits zu 'oe' wird, muessen beide dorthin
# fallen - sonst stehen 'hoejbjerg' und 'hojbjerg' nebeneinander und finden
# sich nie. Welche der beiden Schreibweisen "richtiger" waere, ist egal;
# entscheidend ist, dass beide Seiten dieselbe waehlen.
EIGENE_BUCHSTABEN = {
    "ø": "oe", "Ø": "Oe", "æ": "ae", "Æ": "Ae", "œ": "oe", "Œ": "Oe",
    "ł": "l", "Ł": "L", "đ": "d", "Đ": "D", "ð": "d", "Ð": "D",
    "þ": "th", "Þ": "Th", "ı": "i", "ħ": "h", "ŧ": "t",
}


def entumlauten(text: str) -> str:
    """Deutsche Umlaute ausschreiben, eigene Buchstaben ersetzen, Akzente weg.

    Erst ersetzen, dann zerlegen: sonst wuerde aus 'ue' in 'Mueller' und aus
    'ü' in 'Müller' Verschiedenes, und der Namensabgleich liefe ins Leere.
    """
    for alt, neu in (("ä", "ae"), ("ö", "oe"), ("ü", "ue"), ("ß", "ss"),
                     ("Ä", "Ae"), ("Ö", "Oe"), ("Ü", "Ue")):
        text = text.replace(alt, neu)
    for alt, neu in EIGENE_BUCHSTABEN.items():
        text = text.replace(alt, neu)
    zerlegt = unicodedata.normalize("NFKD", text)
    return "".join(c for c in zerlegt if not unicodedata.combining(c))


def schluessel(name: str) -> str:
    """Vergleichsform eines Namens: nur Kleinbuchstaben, sonst nichts."""
    return re.sub(r"[^a-z]", "", entumlauten(name).lower())


def text_von(fragment: str) -> str:
    """Markup raus, Entities auf, Leerraum zusammen."""
    ohne_tags = re.sub(r"<[^>]*>", " ", fragment)
    return re.sub(r"\s+", " ", html.unescape(ohne_tags)).strip()


def marktwert_zu_euro(text: str) -> int | None:
    """'6,00 Mio. €' -> 6000000, '800 Tsd. €' -> 800000, '-' -> None."""
    treffer = re.search(r"([\d.,]+)\s*(Mio|Tsd)?", text)
    if not treffer:
        return None
    try:
        zahl = float(treffer.group(1).replace(".", "").replace(",", "."))
    except ValueError:
        return None
    faktor = {"Mio": 1_000_000, "Tsd": 1_000}.get(treffer.group(2) or "", 1)
    return int(round(zahl * faktor))


# --------------------------------------------------------------------------
# Netz
# --------------------------------------------------------------------------

class Sitzung:
    """Eine Session mit fester Pause vor jedem Request und Ruecknahme bei 503.

    Ueber wenige Dutzend Seiten laeuft Transfermarkt ohne Murren mit; ueber
    mehrere hundert riegelt es zwischendurch mit **503** ab. Das ist keine
    Sperre, sondern eine Bitte um Geduld - nach einigen Sekunden geht es
    weiter. Ohne die Wiederholung reisst ein Lauf ueber alle Saisons
    zuverlaessig irgendwo in der Mitte ab.
    """

    # 429/503 sind Drosselung, 5xx sonst ein Ausrutscher - beides lohnt einen
    # zweiten Versuch. 404 dagegen nie.
    #
    # **403 gehoert dazu**, auch wenn es "verboten" heisst: die Spielerprofile
    # antworten nach einigen hundert Abrufen damit, dieselbe URL aber wenige
    # Minuten spaeter wieder mit 200. Es ist eine Bitte um Ruhe, keine
    # Ablehnung - nur braucht sie deutlich mehr Geduld als ein 503, deshalb
    # der eigene Zeitplan unten.
    NOCHMAL = {403, 429, 500, 502, 503, 504}
    GEDULDIG = {403}

    def __init__(self, delay: float = 1.5, versuche: int = 5) -> None:
        self.s = requests.Session()
        self.s.headers.update(KOPFZEILEN)
        self.delay = delay
        self.versuche = versuche
        self.gebremst = 0

    def holen(self, pfad: str) -> str | None:
        """Seitentext, oder None wenn sie auch nach allen Versuchen nicht kommt."""
        for versuch in range(self.versuche):
            time.sleep(self.delay)
            basis = 5
            try:
                antwort = self.s.get(BASIS + pfad, timeout=30)
            except requests.RequestException as e:
                grund: object = e
            else:
                if antwort.status_code == 200:
                    return antwort.text
                if antwort.status_code not in self.NOCHMAL:
                    print(f"    ! HTTP {antwort.status_code} fuer {pfad}")
                    return None
                grund = f"HTTP {antwort.status_code}"
                basis = 30 if antwort.status_code in self.GEDULDIG else 5

            self.gebremst += 1
            pause = basis * 2 ** versuch      # 5..40 s, bei 403 30..240 s
            if versuch < self.versuche - 1:
                print(f"    {grund} - warte {pause}s "
                      f"(Versuch {versuch + 2}/{self.versuche})")
                time.sleep(pause)
        print(f"    ! aufgegeben: {pfad}")
        return None


_VOLLNAME = re.compile(
    r'info-table__content--regular">\s*Vollst[^<]*Name:\s*</span>\s*'
    r'<span class="info-table__content info-table__content--bold">(.*?)</span>', re.S)


def profil_vollname(sitzung: Sitzung, tm_id: str) -> str | None:
    """Der buergerliche Name aus dem TM-Spielerprofil.

    Drei Ausgaenge, und der Unterschied zwischen den letzten beiden ist
    wichtig: der Name, ``""`` wenn die Seite kam aber das Feld nicht fuehrt,
    und ``None`` wenn die Seite gar nicht kam. Nur die ersten beiden duerfen
    im Zwischenspeicher landen.

    Die Kaderseite zeigt den **Rufnamen** - 'Naldo', 'Diego', 'Thiago'. Bei
    brasilianischen und iberischen Spielern hat der mit dem Namen, den
    Kickbase fuehrt, keinen Bestandteil gemeinsam, und keine Heuristik der
    Welt verbindet 'Naldo' mit 'Aparecido Rodrigues'. Das Profil fuehrt
    unter *Vollstaendiger Name* 'Ronaldo Aparecido Rodrigues' - damit passt
    es wieder.

    Nur deshalb gibt es diesen Zusatzaufruf, und nur fuer die Faelle, die
    ohne ihn offen blieben.
    """
    seite = sitzung.holen(f"/x/profil/spieler/{tm_id}")
    if not seite:
        return None
    treffer = _VOLLNAME.search(seite)
    return text_von(treffer.group(1)) if treffer else ""


def vereine_der_liga(sitzung: Sitzung, wettbewerb: str, saison: int) -> list[tuple[str, str]]:
    """(tm_id, slug) aller Vereine einer Liga in einer Saison.

    Die Ligaseite verlinkt jeden Verein mit ``saison_id`` im Pfad - daran
    lassen sich die Kaderlinks von allem anderen trennen, was sonst noch auf
    der Seite steht.
    """
    seite = sitzung.holen(f"/x/startseite/wettbewerb/{wettbewerb}/plus/?saison_id={saison}")
    if not seite:
        return []
    muster = rf'href="/([a-z0-9-]+)/startseite/verein/(\d+)/saison_id/{saison}"'
    gesehen: list[tuple[str, str]] = []
    for slug, tm_id in re.findall(muster, seite):
        if (tm_id, slug) not in gesehen:
            gesehen.append((tm_id, slug))
    return gesehen


# --------------------------------------------------------------------------
# Kadertabelle zerlegen
# --------------------------------------------------------------------------

# Die Zeile enthaelt eine verschachtelte <table class="inline-table"> mit
# Bild, Namenslink und - in der zweiten inneren Zeile - der Position. Die
# wird zuerst herausgeloest, danach stehen die aeusseren <td> in fester
# Reihenfolge da.
#
# Zeilen werden NICHT ueber ein Muster '<tr>...</tr>' geschnitten: das
# schliessende Tag der inneren Tabelle beendet sonst die aeussere Zeile zu
# frueh. Stattdessen wird an den Zeilenanfaengen getrennt - was hinter dem
# eigentlichen </tr> noch mitlaeuft, stoert nicht, weil unten ohnehin
# gezielt gesucht wird.
_ZEILEN_START = re.compile(r'<tr class="(?:odd|even)">')
_INLINE = re.compile(r'<table class="inline-table">.*?</table>', re.S)
_SPIELER = re.compile(r'href="/[^"]*/profil/spieler/(\d+)"[^>]*>\s*(.*?)\s*</a>', re.S)
_POSITION = re.compile(r"<tr>\s*<td[^>]*>\s*([^<]+?)\s*</td>\s*</tr>\s*</table>", re.S)
_ZELLE = re.compile(r"<td[^>]*>(.*?)</td>", re.S)


_KLUBNAME = re.compile(r"<h1[^>]*>(.*?)</h1>", re.S)


def klubname(seite: str) -> str:
    """Der Klubname aus der Kaderseite selbst.

    Als Anzeigename fuer Teams ohne Kuerzel aus ``data_*.json`` - der
    Spielerindex kennt nur die *heutige* Zugehoerigkeit einer team_id, und
    die kann historisch falsch sein: 49 traegt dort „1. FC Heidenheim 1846",
    war 2021/22 aber Hansa Rostock. Die Kaderseite der Saison weiss es
    richtig.
    """
    treffer = _KLUBNAME.search(seite)
    return text_von(treffer.group(1)) if treffer else ""


def kadertabelle(seite: str) -> str:
    """Nur die Kadertabelle - die Seite traegt weitere <tr class="odd">.

    ``inline-table`` wird ohne ``<tbody>`` ausgeliefert, deshalb gehoert das
    erste ``</tbody>`` nach ``<table class="items">`` verlaesslich zur
    Kadertabelle selbst.
    """
    start = seite.find('<table class="items">')
    if start < 0:
        return ""
    ende = seite.find("</tbody>", start)
    return seite[start:ende if ende > 0 else None]


def kader_zerlegen(seite: str) -> list[dict]:
    """Eine Kaderseite -> je Spieler ein Rohdatensatz."""
    spieler = []
    for zeile in _ZEILEN_START.split(kadertabelle(seite))[1:]:
        treffer = _SPIELER.search(zeile)
        if not treffer:
            continue
        tm_id, name = treffer.group(1), text_von(treffer.group(2))

        inline = _INLINE.search(zeile)
        pos_treffer = _POSITION.search(inline.group(0)) if inline else None
        position_tm = text_von(pos_treffer.group(1)) if pos_treffer else ""

        # Ohne die innere Tabelle stehen die aeusseren Zellen in fester
        # Ordnung: Nummer, (leer), Geb./Alter, Nat., Groesse, Fuss,
        # im Team seit, Zuvor, Vertrag, Marktwert.
        rest = _INLINE.sub("", zeile)
        zellen = _ZELLE.findall(rest)
        if len(zellen) < 10:
            continue

        vorverein = re.search(r'title="([^":]+)', zellen[7])

        spieler.append({
            "tm_id": tm_id,
            "spieler": name,
            "position_tm": position_tm,
            "position_fine": POSITION.get(entumlauten(position_tm), ""),
            "geburtsdatum": text_von(zellen[2]).split(" ")[0],
            "vertrag_bis": text_von(zellen[8]),
            "marktwert_eur": marktwert_zu_euro(text_von(zellen[9])),
            "zuvor": html.unescape(vorverein.group(1)).strip() if vorverein else "",
            # Bleibt leer, bis --profile das Spielerprofil holt.
            "vollname": "",
        })
    return spieler


# --------------------------------------------------------------------------
# Kickbase-Seite: Kader je Kuerzel, und die Bruecke dorthin
# --------------------------------------------------------------------------

# Kickbase fuehrt unter team_id 1 keinen Verein, sondern einen Sammeltopf.
SAMMELTOPF = "1"


def kickbase_namen() -> dict[str, str]:
    """{player_id: voller Name} aus dem ID-Crawl.

    ``data_*.json`` traegt nur den Nachnamen ('Blaswich'), der Index aus
    ``enumerate_ids`` dagegen Vor- und Nachnamen. Fuer die Zuordnung ist das
    der bessere Ausgangspunkt, und er deckt gemessen 100 % der Panel-Spieler.
    """
    import pandas as pd

    pfad = RAW / "player_index.parquet"
    if not pfad.exists():
        return {}
    idx = pd.read_parquet(pfad)
    namen = {}
    for pid, name, vor, nach in zip(idx["id"], idx["name"],
                                    idx["first_name"], idx["last_name"]):
        voll = name if isinstance(name, str) and name else None
        if not voll and isinstance(nach, str):
            voll = f"{vor if isinstance(vor, str) else ''} {nach}".strip()
        if voll:
            namen[str(pid)] = voll
    return namen


def kickbase_kuerzel() -> dict[str, str]:
    """{team_id: Dreibuchstaben-Kuerzel} aus ``data_*.json`` - nur die."""
    kuerzel: dict[str, str] = {}
    for datei in LEGACY_DIR.glob("data_*.json"):
        with open(datei, encoding="utf-8") as f:
            kuerzel.update({str(k): v for k, v in json.load(f)["teams"].items()})
    return kuerzel


def kickbase_anzeigenamen() -> dict[str, str]:
    """{team_id: lesbarer Name}. Kuerzel wo bekannt, sonst der Klarname.

    Die drei Buchstaben stehen nur in ``data_*.json`` und decken damit die
    heutigen Ligen ab; historische Vereine tragen den Namen aus dem
    Spielerindex. Achtung: der kennt nur die *heutige* Zugehoerigkeit einer
    team_id - beim Holen ist deshalb ``klubname()`` von der Kaderseite die
    bessere Quelle. Die Spalte dient der Orientierung beim Nachtragen von
    Hand - verbunden wird ueber ``player_id``.
    """
    import pandas as pd

    namen: dict[str, str] = {}
    pfad = RAW / "player_index.parquet"
    if pfad.exists():
        idx = pd.read_parquet(pfad).dropna(subset=["team_id", "team_name"])
        for tid, tname in zip(idx["team_id"], idx["team_name"]):
            namen.setdefault(str(tid), str(tname))
    namen.update(kickbase_kuerzel())
    return namen


def kickbase_kader(season: str, suffix: str | None, liga: str,
                   namen: dict[str, str]) -> dict[str, dict[str, dict[str, str]]]:
    """{team_id: {Namensschluessel: player_id}} fuer eine Saison und Liga.

    Zwei Quellen, weil keine allein reicht: ``data_*.json`` gibt es nur fuer
    die laufende und die vorige Saison, fuehrt dort aber den **vollstaendigen
    Kader**; das Panel reicht bis 2013/14 zurueck, kennt aber nur Spieler mit
    Einsatz - zu Saisonbeginn ist es deshalb duenn. Die Vereinigung beider
    ist in jeder Saison die vollstaendigere Liste.

    Indiziert wird ueber **alle** Namensformen, nicht nur die eine aus der
    Quelle: der Index fuehrt 'Hanno Balitsch', ``data_*.json`` 'Balitsch',
    Transfermarkt wieder etwas anderes. Mehrdeutigkeiten, die dabei
    entstehen, raeumt ``kader_zuordnen`` hinterher weg.
    """
    import pandas as pd

    kader: dict[str, dict[str, dict[str, str]]] = {}

    def eintragen(team_id: str, pid: str, *rohnamen: str) -> None:
        if not team_id or team_id == SAMMELTOPF:
            return
        eintrag = kader.setdefault(team_id, {"stark": {}, "schwach": {}})
        for roh in rohnamen:
            if not roh:
                continue
            for form in namensformen(roh):
                eintrag["stark"].setdefault(form, pid)
            for form in schwache_formen(roh):
                eintrag["schwach"].setdefault(form, pid)

    # suffix is None heisst: fuer diese Saison gibt es keine data_*.json. Der
    # Leerstring ist dagegen der Suffix der *laufenden* Saison - wer beides
    # gleich behandelt, liest fuer 2013/14 den heutigen Kader ein.
    if suffix is not None:
        pfad = LEGACY_DIR / f"data_{liga}{suffix}.json"
        if pfad.exists():
            with open(pfad, encoding="utf-8") as f:
                for p in json.load(f)["players"]:
                    eintragen(str(p["team_id"]), p["id"], p["name"],
                              namen.get(p["id"], ""))

    panel = PROCESSED / "panel.parquet"
    if panel.exists():
        df = pd.read_parquet(panel, columns=["player_id", "season", "league", "team_id"])
        df = df[(df["season"] == season) & (df["league"] == LIGA_NAME[liga])]
        for pid, tid in df[["player_id", "team_id"]].drop_duplicates().itertuples(index=False):
            if pd.notna(tid):
                eintragen(str(tid), str(pid), namen.get(str(pid), ""))

    return kader


def namensformen(voller_name: str) -> list[str]:
    """Schreibweisen, unter denen Kickbase denselben Spieler fuehren kann.

    Kickbase kennt nur *ein* Namensfeld, Transfermarkt den vollen Namen.
    Drei Faelle decken gemessen 99 % ab:

    * einnamig - 'Arthur' steht bei beiden gleich,
    * mehrteiliger Nachname - 'Daniel Heuer Fernandes' bei TM,
      'Heuer Fernandes' bei Kickbase; deshalb ist **jede Endung** des Namens
      ein Kandidat,
    * abgekuerzter Vorname - 'Hiroki Ito' gegen 'H. Ito'.
    """
    teile = voller_name.split()
    formen = [schluessel(voller_name)]
    formen += [schluessel("".join(teile[i:])) for i in range(1, len(teile))]
    if len(teile) > 1:
        formen.append(schluessel(teile[0][:1] + teile[-1]))
        # Reihenfolge umgedreht: Transfermarkt fuehrt 'Min-jae Kim', Kickbase
        # 'Kim Minjae' - dieselben Bestandteile, keine gemeinsame Form. Die
        # sortierte Verkettung nutzt *alle* Teile und ist damit so spezifisch
        # wie der volle Name, nur reihenfolgeunabhaengig.
        formen.append("".join(sorted(schluessel(t) for t in teile)))
    return [f for f in formen if f]


# Kuerzer als das wird ein Namensbestandteil zum Zufallsgenerator ('dos',
# 'van', 'de').
MINDESTLAENGE = 4


def schwache_formen(voller_name: str) -> list[str]:
    """Einzelne Namensbestandteile - fuer den zweiten Durchgang.

    Kickbase fuehrt 'Alcántara do Nascimiento', Transfermarkt 'Thiago
    Alcántara': der gemeinsame Teil steht bei Kickbase **vorn**, und
    ``namensformen`` bildet nur Endungen. Einzelbestandteile schliessen das,
    kosten aber Genauigkeit - 'Santos', 'Silva' und 'Mueller' kommen in
    einem Kader mehrfach vor. Gemessen verschlechtert es die Zuordnung um
    3-5 Prozentpunkte, wenn man damit *zuerst* sucht. Deshalb laufen diese
    Formen ausschliesslich im zweiten Durchgang gegen die Reste.
    """
    stark = set(namensformen(voller_name))
    schwach = {schluessel(t) for t in voller_name.split()}
    return [f for f in schwach - stark if len(f) >= MINDESTLAENGE]


def verein_zuordnen(tm_kader: list[dict],
                    kb_kader: dict[str, dict[str, dict[str, str]]]) -> tuple[str, int]:
    """TM-Kader -> Kickbase-Kuerzel, ueber die groesste Namensueberschneidung.

    Ein Kader hat gut 25 Spieler; die richtige Paarung trifft zweistellig,
    jede falsche nahe null. Der Abstand ist so gross, dass es kein
    handgepflegtes Vereins-Mapping braucht - deshalb wird die Trefferzahl
    mit zurueckgegeben und protokolliert.
    """
    formen = [tm_formen(s) for s in tm_kader]
    beste, punkte = "", 0
    for team_id, eintrag in kb_kader.items():
        treffer = sum(1 for f in formen if any(n in eintrag["stark"] for n in f))
        if treffer > punkte:
            beste, punkte = team_id, treffer
    return beste, punkte


def spieler_zuordnen(formen: list[str], index: dict[str, str]) -> str:
    """Kickbase-ID zu einem Satz Namensformen, oder '' wenn nicht eindeutig.

    Mehrdeutiges bleibt bewusst leer - lieber eine Luecke, die im Protokoll
    steht, als eine falsche Bruecke.
    """
    kandidaten = {index[f] for f in formen if f in index}
    return kandidaten.pop() if len(kandidaten) == 1 else ""


def tm_formen(zeile: dict, schwach: bool = False) -> list[str]:
    """Namensformen einer TM-Zeile - Rufname und, wenn geholt, Vollname.

    ``vollname`` steht nur bei den Zeilen, fuer die ``--profile`` das Profil
    geholt hat. Er wird gleichberechtigt behandelt: fuer 'Naldo' ist er die
    einzige Bruecke, fuer alle anderen eine zusaetzliche.
    """
    bilden = schwache_formen if schwach else namensformen
    formen = bilden(zeile["spieler"])
    voll = zeile.get("vollname") or ""
    if voll and voll != zeile["spieler"]:
        formen = formen + bilden(voll)
    return formen


def _eindeutig(kandidaten: list[str]) -> list[str]:
    """Alles leeren, was mehr als einmal vorkommt.

    Je Spieler betrachtet ist die Zuordnung mehrdeutigkeitsfrei, ueber den
    Kader hinweg nicht: zwei TM-Spieler koennen auf dieselbe Kickbase-ID
    zeigen (Brueder, gleicher Nachname). Dann bleiben beide leer - eine
    Luecke ist von Hand zu schliessen, eine falsche Bruecke faellt nie
    wieder auf.
    """
    zaehler = Counter(k for k in kandidaten if k)
    return [k if k and zaehler[k] == 1 else "" for k in kandidaten]


def kader_zuordnen(tm_kader: list[dict], eintrag: dict[str, dict[str, str]],
                   bruecke: dict[str, str]) -> tuple[list[str], int]:
    """Kickbase-IDs fuer einen ganzen Kader, in zwei Durchgaengen.

    Erst die starken Namensformen (Endungen, voller Name, Initial plus
    Nachname). Was danach offen ist, bekommt einen zweiten Versuch mit
    einzelnen Namensbestandteilen - aber **nur gegen die Reste**: Spieler,
    die schon vergeben sind, nehmen nicht mehr teil. So kann der zweite
    Durchgang nur hinzufuegen, nie eine gute Zuordnung durch eine schlechte
    ersetzen.

    ``bruecke`` sind die IDs aus einem frueheren Lauf. Sie gewinnen immer:
    die Datei kann von Hand Nachgetragenes nicht von einer alten Vermutung
    unterscheiden, und Handarbeit zu ueberschreiben waere der teurere
    Fehler. Wer neu raten lassen will, nimmt ``--neu-zuordnen``.
    """
    stark = [tm_formen(s) for s in tm_kader]
    roh = [spieler_zuordnen(f, eintrag["stark"]) for f in stark]
    ids = _eindeutig(roh)
    doppelt = sum(1 for r, i in zip(roh, ids) if r and not i)

    # Zweiter Durchgang: schwache Formen, aber nur gegen die noch freien
    # Kickbase-Spieler - so kann er nichts ueberschreiben.
    #
    # Der Rest-Index traegt die schwachen Formen - und zusaetzlich die starken
    # jener Kickbase-Spieler, die **einteilig** gefuehrt werden ('Hernández',
    # 'Sotiris', 'Löwe'). Deren einzige Form ist eine starke, waehrend
    # derselbe Bestandteil bei Transfermarkt mitten im vollen Namen steht
    # ('Javier Hernández Balcázar') und dort nur als schwache Form entsteht.
    # Ohne sie faenden sich die beiden nie.
    #
    # Pauschal **alle** starken Formen aufzunehmen war messbar schlechter
    # (-10 Zuordnungen bei +1): der Durchgang wird dann so weit, dass er mehr
    # Mehrdeutigkeiten erzeugt als Treffer. Die Einschraenkung auf einteilige
    # Namen trifft genau die Faelle, um die es geht.
    vergeben = {i for i in ids if i}
    mehrteilig = set(eintrag["schwach"].values())
    reste = {form: pid for form, pid in eintrag["schwach"].items()
             if pid not in vergeben}
    for form, pid in eintrag["stark"].items():
        if pid not in vergeben and pid not in mehrteilig:
            reste.setdefault(form, pid)
    nachschlag = _eindeutig([
        "" if fest else spieler_zuordnen(f + tm_formen(s, schwach=True), reste)
        for s, f, fest in zip(tm_kader, stark, ids)])

    # Herkunft je Zuordnung mitgeben - saisonweit muss unterscheidbar sein,
    # welche Bruecke stark und welche nur geraten ist (siehe saison_bereinigen).
    quelle = ["stark" if fest else ("schwach" if nach else "")
              for fest, nach in zip(ids, nachschlag)]
    ids = [fest or nach for fest, nach in zip(ids, nachschlag)]
    ids = [bruecke.get(s["tm_id"]) or i for s, i in zip(tm_kader, ids)]
    quelle = ["bruecke" if bruecke.get(s["tm_id"]) else q
              for s, q in zip(tm_kader, quelle)]
    return ids, doppelt, quelle


# Wer gewinnt, wenn zwei TM-Spieler dieselbe Kickbase-ID beanspruchen.
RANG = {"bruecke": 3, "stark": 2, "schwach": 1, "": 0}


def saison_bereinigen(zeilen: list[dict], quellen: dict[int, str]) -> int:
    """Dieselbe Kickbase-ID an zwei *verschiedene* TM-Spieler? Aufloesen.

    Die Sperre in ``kader_zuordnen`` wirkt nur innerhalb eines Kaders. Ueber
    die Saison hinweg koennen zwei Spieler verschiedener Vereine dieselbe ID
    beanspruchen - gemessen dreimal, alle ueber den Vornamen aus dem zweiten
    Durchgang: 'Philipp Schulze' griff nach Maximilian Philipp, 'Luca
    Raimund' nach Luca Pfeiffer, 'Christian Viet' nach Christian Conteh.

    **Dieselbe** ``tm_id`` mehrfach ist dagegen richtig und haeufig (305
    Faelle): ein Wechsler steht in den Kadern beider Vereine, beide Zeilen
    meinen denselben Menschen.

    Es gewinnt die stärkere Herkunft; bei Gleichstand bleiben alle leer -
    eine Luecke ist zu schliessen, eine falsche Bruecke faellt nie auf.
    """
    nach_id: dict[str, list[int]] = {}
    for i, z in enumerate(zeilen):
        if z["player_id"]:
            nach_id.setdefault(z["player_id"], []).append(i)

    geleert = 0
    for idxs in nach_id.values():
        if len({zeilen[i]["tm_id"] for i in idxs}) < 2:
            continue                      # derselbe Spieler, nur zwei Vereine
        rang = {i: RANG[quellen.get(id(zeilen[i]), "")] for i in idxs}
        beste = max(rang.values())
        gewinner = [i for i in idxs if rang[i] == beste]
        verlierer = [i for i in idxs if i not in gewinner]
        if len({zeilen[i]["tm_id"] for i in gewinner}) > 1:
            verlierer = idxs              # Gleichstand: keiner gewinnt
        for i in verlierer:
            zeilen[i]["player_id"] = ""
            geleert += 1
    return geleert


# --------------------------------------------------------------------------
# Datei
# --------------------------------------------------------------------------

def alte_bruecke(season: str) -> dict[str, str]:
    """{tm_id: player_id} aus einem frueheren Lauf, inklusive Handarbeit."""
    if not ZIEL.exists():
        return {}
    with open(ZIEL, encoding="utf-8", newline="") as f:
        return {r["tm_id"]: r["player_id"] for r in csv.DictReader(f)
                if r.get("season") == season and r.get("player_id")}


def schreiben(zeilen: list[dict], season: str) -> None:
    """Die Saison ersetzen, alle anderen unveraendert stehen lassen."""
    andere = []
    if ZIEL.exists():
        with open(ZIEL, encoding="utf-8", newline="") as f:
            andere = [r for r in csv.DictReader(f) if r.get("season") != season]
    alle = andere + zeilen
    alle.sort(key=lambda r: (r["season"], r["verein"], r["spieler"]))
    with open(ZIEL, "w", encoding="utf-8", newline="") as f:
        schreiber = csv.DictWriter(f, fieldnames=SPALTEN)
        schreiber.writeheader()
        schreiber.writerows(alle)


def offene_stammspieler(alle: list[dict], min_minuten: int) -> dict[tuple[str, str, str], list[str]]:
    """{(season, liga, verein): [Kickbase-Namen]} der offenen Stammspieler.

    Offen heisst: im Panel mit mehr als ``min_minuten`` Einsatzminuten, aber
    in keiner TM-Zeile zugeordnet. Der Verein kommt aus dem Panel und wird
    ueber ``kickbase_anzeigenamen`` in dieselbe Schreibweise gebracht, die
    in der Datei steht - nur so laesst sich die Suche auf den richtigen
    Kader eingrenzen.
    """
    import pandas as pd

    panel = PROCESSED / "panel.parquet"
    if not panel.exists():
        return {}
    df = pd.read_parquet(panel, columns=["player_id", "season", "league",
                                         "team_id", "minutes"])
    df["player_id"] = df["player_id"].astype(str)
    df["liga"] = df["league"].map({v: k for k, v in LIGA_NAME.items()})

    vergeben = {z["player_id"] for z in alle if z["player_id"]}
    namen, anzeige = kickbase_namen(), kickbase_anzeigenamen()

    summe = df.groupby(["player_id", "season", "liga", "team_id"])["minutes"].sum()
    offen: dict[tuple[str, str, str], list[str]] = {}
    for (pid, season, liga, team_id), minuten in summe.items():
        if pid in vergeben or minuten <= min_minuten or not namen.get(pid):
            continue
        schluss = (season, liga, anzeige.get(str(team_id), str(team_id)))
        offen.setdefault(schluss, []).append(namen[pid])
    return offen


def gelesene_vollnamen() -> dict[str, str]:
    """{tm_id: vollname} aus frueheren Profilabrufen.

    Auch die **leeren** Ergebnisse stehen darin - ein Profil ohne das Feld
    *Vollstaendiger Name* soll nicht bei jedem Lauf erneut geholt werden.
    Nur was gar nicht beantwortet wurde, fehlt und wird nachgeholt.
    """
    if not VOLLNAMEN.exists():
        return {}
    bekannt: dict[str, str] = {}
    with open(VOLLNAMEN, encoding="utf-8") as f:
        for zeile in f:
            zeile = zeile.strip()
            if zeile:
                eintrag = json.loads(zeile)
                bekannt[str(eintrag["tm_id"])] = eintrag.get("vollname", "")
    return bekannt


def vollnamen_einpflegen(alle: list[dict], bekannt: dict[str, str]) -> int:
    """Vollnamen in die Zeilen schreiben. Gilt je Spieler, nicht je Saison."""
    getroffen = 0
    for z in alle:
        voll = bekannt.get(z["tm_id"])
        if voll and not z["vollname"]:
            z["vollname"] = voll
            getroffen += 1
    return getroffen


def profile_nachholen(sitzung: Sitzung, min_minuten: int) -> None:
    """Vollnamen holen, wo ein Stammspieler sonst unzugeordnet bliebe.

    Ein Profil je TM-Spieler, nicht je Saison - dieselbe ``tm_id`` taucht in
    vielen Saisons auf, der buergerliche Name aendert sich nicht. Geholt
    wird nur in den Kadern, in denen tatsaechlich ein Stammspieler fehlt.

    **Resumierbar.** Transfermarkt drosselt die Profile hart; ein Lauf ueber
    250 Seiten dauert deshalb eine halbe Stunde und kann jederzeit
    unterbrochen werden muessen. Jeder Abruf wird sofort als JSONL-Zeile
    angehaengt - wie in ``enumerate_ids`` und aus demselben Grund. Ein
    Abbruch mit Strg+C kostet nichts, ein erneuter Start setzt fort.
    """
    if not ZIEL.exists():
        print("[tm] keine tm_players.csv")
        return
    with open(ZIEL, encoding="utf-8", newline="") as f:
        alle = [{**{s: "" for s in SPALTEN}, **z} for z in csv.DictReader(f)]

    bekannt = gelesene_vollnamen()
    if bekannt:
        print(f"[tm] {len(bekannt)} Profile aus frueheren Laeufen bekannt")

    offen = offene_stammspieler(alle, min_minuten)
    if not offen:
        print(f"[tm] kein offener Spieler ueber {min_minuten} Minuten")
        return

    gesucht = set(offen)
    kandidaten = {z["tm_id"] for z in alle
                  if not z["player_id"] and not z["vollname"]
                  and z["tm_id"] not in bekannt
                  and (z["season"], z["liga"], z["verein"]) in gesucht}
    fehlend = sum(len(v) for v in offen.values())
    print(f"[tm] {fehlend} offene Stammspieler in {len(gesucht)} Kadern "
          f"-> {len(kandidaten)} Profile zu holen")
    print("[tm] resumierbar: Abbruch mit Strg+C ist unkritisch")

    ensure_dirs()
    geholt = 0
    try:
        with open(VOLLNAMEN, "a", encoding="utf-8") as sink:
            for nr, tm_id in enumerate(sorted(kandidaten, key=int), 1):
                voll = profil_vollname(sitzung, tm_id)
                # Nur festhalten, was Transfermarkt wirklich beantwortet hat.
                # Eine Absage nach allen Versuchen als "kein Vollname" zu
                # verbuchen, wuerde den Fall dauerhaft verbrennen.
                if voll is None:
                    continue
                sink.write(json.dumps({"tm_id": tm_id, "vollname": voll},
                                      ensure_ascii=False) + "\n")
                sink.flush()
                bekannt[tm_id] = voll
                geholt += bool(voll)
                if nr % 25 == 0:
                    print(f"  [{nr}/{len(kandidaten)}] {geholt} mit Vollname")
    except KeyboardInterrupt:
        print("\n[tm] abgebrochen - Fortschritt ist gesichert")

    getroffen = vollnamen_einpflegen(alle, bekannt)
    alle.sort(key=lambda r: (r["season"], r["verein"], r["spieler"]))
    with open(ZIEL, "w", encoding="utf-8", newline="") as f:
        schreiber = csv.DictWriter(f, fieldnames=SPALTEN)
        schreiber.writeheader()
        schreiber.writerows(alle)
    print(f"[tm] {geholt} Vollnamen neu geholt, {getroffen} Zeilen ergaenzt")


def neu_zuordnen_offline() -> None:
    """player_id fuer die ganze Datei neu herleiten, ohne eine Seite zu holen.

    Die TM-Zeilen aendern sich nicht, wenn man an der Zuordnung schraubt -
    nur die Bruecke tut es. Ein voller Lauf kostet aber gut 380 Requests,
    und das ist keine Grundlage, um eine Heuristik zu erproben. Deshalb
    laesst sich die Zuordnung allein aus der Datei wiederholen: Vereine
    werden erneut ueber die Namensueberschneidung bestimmt, genau wie beim
    Holen.

    **Loescht nichts.** Wo die Herleitung zu keinem Ergebnis kommt, bleibt ein
    vorhandener Wert stehen. Die Datei kann Handarbeit nicht von einer alten
    Vermutung unterscheiden - aber was die Heuristik nicht selbst findet, kann
    sie auch nicht selbst gesetzt haben, und genau das sind die Faelle, die
    von Hand kommen (echte Namensvettern, fehlerhafte Vereinszuordnung).
    """
    if not ZIEL.exists():
        print("[tm] keine tm_players.csv - nichts zuzuordnen")
        return

    with open(ZIEL, encoding="utf-8", newline="") as f:
        alle = [{**{s: "" for s in SPALTEN}, **z} for z in csv.DictReader(f)]

    namen = kickbase_namen()
    suffixe = saison_suffixe()
    gruppen: dict[tuple[str, str], list[dict]] = {}
    nach_saison: dict[str, list[dict]] = {}
    for zeile in alle:
        gruppen.setdefault((zeile["season"], zeile["liga"]), []).append(zeile)
        nach_saison.setdefault(zeile["season"], []).append(zeile)
    # Herkunft je Zeile, ueber beide Ligen einer Saison hinweg gesammelt:
    # eine Doppelvergabe kann die Ligagrenze ueberschreiten (Luca Pfeiffer
    # stand 24/25 beim KSC in Liga 2, 'Luca Raimund' beim VfB in Liga 1).
    quellen: dict[int, str] = {id(z): "" for z in alle}

    for (season, liga), zeilen in sorted(gruppen.items()):
        kb_kader = kickbase_kader(season, suffixe.get(season[:4]), liga, namen)
        if not kb_kader:
            print(f"  {season} Liga {liga}: keine Kickbase-Daten")
            continue
        vorher = sum(1 for z in zeilen if z["player_id"])
        for verein, kader in gruppen_nach_verein(zeilen):
            team_id, guete = verein_zuordnen(kader, kb_kader)
            if guete < MINDESTGUETE:
                for z in kader:
                    z["player_id"] = ""
                continue
            ids, _, q = kader_zuordnen(kader, kb_kader[team_id], {})
            for z, pid, qq in zip(kader, ids, q):
                # Ein vorhandener Wert wird nur ersetzt, nicht geloescht: was
                # die Heuristik nicht selbst herleiten kann, hat jemand von
                # Hand eingetragen. Genau die Faelle - echte Namensvettern wie
                # Chris und Justin Loewe - kann sie nie herleiten, und sie bei
                # jedem Lauf wegzuwerfen waere die teuerste Art zu verlieren.
                if pid:
                    z["player_id"] = pid
                    quellen[id(z)] = qq
                elif z["player_id"]:
                    quellen[id(z)] = "bruecke"      # Handarbeit, unantastbar
        nachher = sum(1 for z in zeilen if z["player_id"])
        pfeil = "->" if nachher != vorher else "  "
        print(f"  {season} Liga {liga}: {len(zeilen):>4} Zeilen, "
              f"{vorher:>4} {pfeil} {nachher:>4} zugeordnet")

    geleert = sum(saison_bereinigen(z, quellen) for z in nach_saison.values())
    if geleert:
        print(f"[tm] {geleert} Zeilen geleert: dieselbe Kickbase-ID von zwei "
              f"verschiedenen TM-Spielern beansprucht")

    alle.sort(key=lambda r: (r["season"], r["verein"], r["spieler"]))
    with open(ZIEL, "w", encoding="utf-8", newline="") as f:
        schreiber = csv.DictWriter(f, fieldnames=SPALTEN)
        schreiber.writeheader()
        schreiber.writerows(alle)
    print(f"[tm] {ZIEL}: {len(alle)} Zeilen, "
          f"{sum(1 for z in alle if z['player_id'])} zugeordnet")


def gruppen_nach_verein(zeilen: list[dict]) -> list[tuple[str, list[dict]]]:
    """Zeilen einer Saison und Liga nach Verein, Reihenfolge egal."""
    nach: dict[str, list[dict]] = {}
    for z in zeilen:
        nach.setdefault(z["verein"], []).append(z)
    return sorted(nach.items())


def saison_suffixe() -> dict[str, str]:
    """{'2026': '', '2025': '_2526'} aus dem Saison-Manifest."""
    with open(LEGACY_DIR / "seasons.json", encoding="utf-8") as f:
        return {s["title"][:4]: s["suffix"] for s in json.load(f)["seasons"]}


def vergleichen(season: str) -> None:
    """tm_players.csv gegen fine_positions.csv - nur Bericht, kein Schreiben."""
    quelle = MANUAL / "fine_positions.csv"
    if not (ZIEL.exists() and quelle.exists()):
        print("[tm] Vergleich braucht tm_players.csv und fine_positions.csv")
        return

    with open(ZIEL, encoding="utf-8", newline="") as f:
        tm = {r["player_id"]: r for r in csv.DictReader(f)
              if r["season"] == season and r["player_id"]}
    with open(quelle, encoding="utf-8", newline="") as f:
        hand = {r["player_id"]: r for r in csv.DictReader(f)
                if r["season"] == season}

    gemeinsam = set(tm) & set(hand)
    abweichung = [(pid, hand[pid]["position_fine"], tm[pid]["position_fine"])
                  for pid in sorted(gemeinsam, key=int)
                  if tm[pid]["position_fine"]
                  and hand[pid]["position_fine"] != tm[pid]["position_fine"]]

    print(f"\n[tm] Vergleich {season}: {len(hand)} handgepflegt, {len(tm)} von TM, "
          f"{len(gemeinsam)} gemeinsam")
    print(f"[tm] nur in fine_positions.csv: {len(set(hand) - set(tm))}")
    print(f"[tm] nur bei TM (Kandidaten zum Nachtragen): {len(set(tm) - set(hand))}")
    print(f"[tm] Abweichungen: {len(abweichung)}")
    for pid, h, t in abweichung:
        print(f"     {pid:>6}  {tm[pid]['verein']:<4} {tm[pid]['spieler']:<28} "
              f"hand={h:<4} tm={t}")


# --------------------------------------------------------------------------

def main() -> None:
    enable_utf8_stdout()
    ensure_dirs()

    with open(LEGACY_DIR / "seasons.json", encoding="utf-8") as f:
        manifest = json.load(f)
    laufend = next(s for s in manifest["seasons"] if s["key"] == manifest["current"])

    jetzt = int(laufend["title"][:4])
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--von", type=int, default=jetzt,
                    help="erste TM-saison_id, z. B. 2013 fuer 2013/14")
    ap.add_argument("--bis", type=int, default=None,
                    help="letzte TM-saison_id (Default: wie --von)")
    ap.add_argument("--delay", type=float, default=1.5,
                    help="Pause je Request in Sekunden (Default 1,5)")
    ap.add_argument("--vergleich", action="store_true",
                    help="nur gegen fine_positions.csv abgleichen, nichts holen")
    ap.add_argument("--neu-zuordnen", action="store_true",
                    help="player_id komplett neu raten statt aus der Datei "
                         "zu uebernehmen - verwirft Handarbeit")
    ap.add_argument("--nur-zuordnen", action="store_true",
                    help="player_id aus der vorhandenen Datei neu herleiten, "
                         "ohne eine Seite zu holen - verwirft Handarbeit")
    ap.add_argument("--profile", action="store_true",
                    help="Vollnamen aus den Spielerprofilen nachholen, wo ein "
                         "Stammspieler sonst unzugeordnet bleibt")
    ap.add_argument("--min-minuten", type=int, default=900,
                    help="ab wie vielen Saisonminuten ein Spieler als "
                         "Stammspieler gilt (Default 900)")
    args = ap.parse_args()

    if args.vergleich:
        vergleichen(laufend["title"])
        return
    if args.nur_zuordnen:
        neu_zuordnen_offline()
        return
    if args.profile:
        profile_nachholen(Sitzung(delay=args.delay), args.min_minuten)
        neu_zuordnen_offline()
        return

    # Kickbase kennt nur wenige Saisons als data_*.json; alles Aeltere kommt
    # aus dem Panel und heisst dort 'JJJJ/JJJJ'.
    suffixe = {s["title"][:4]: s["suffix"] for s in manifest["seasons"]}

    sitzung = Sitzung(delay=args.delay)
    namen = kickbase_namen()
    anzeige = kickbase_anzeigenamen()
    kuerzel = kickbase_kuerzel()
    print(f"[tm] {len(namen)} Kickbase-Namen, {len(anzeige)} Vereinsnamen geladen")
    if args.neu_zuordnen:
        print("[tm] --neu-zuordnen: player_id wird komplett neu geraten")

    unbekannte_positionen: set[str] = set()
    bilanz: list[tuple[str, int, int]] = []

    for jahr in range(args.von, (args.bis or args.von) + 1):
        season = f"{jahr}/{jahr + 1}"
        suffix = suffixe.get(str(jahr))
        bruecke = {} if args.neu_zuordnen else alte_bruecke(season)
        zeilen: list[dict] = []
        quellen: dict[int, str] = {}

        for liga, wettbewerb in WETTBEWERB.items():
            kb_kader = kickbase_kader(season, suffix, liga, namen)
            if not kb_kader:
                print(f"[tm] {wettbewerb} {season}: keine Kickbase-Daten - uebersprungen")
                continue

            vereine = vereine_der_liga(sitzung, wettbewerb, jahr)
            print(f"\n[tm] {wettbewerb} {season}: {len(vereine)} Vereine bei TM, "
                  f"{len(kb_kader)} bei Kickbase")

            for tm_id, slug in vereine:
                seite = sitzung.holen(
                    f"/{slug}/kader/verein/{tm_id}/saison_id/{jahr}/plus/1")
                kader = kader_zerlegen(seite) if seite else []
                if not kader:
                    print(f"  ! {slug}: kein Kader gelesen")
                    continue

                team_id, guete = verein_zuordnen(kader, kb_kader)
                if guete < MINDESTGUETE:
                    print(f"  - {slug}: kein Kickbase-Verein (Guete {guete}) "
                          f"- uebersprungen")
                    continue

                ids, doppelt, q = kader_zuordnen(kader, kb_kader[team_id], bruecke)
                # Kuerzel wo vorhanden; sonst der Klubname von der Kaderseite
                # selbst, denn der stimmt historisch (siehe klubname()).
                verein = (kuerzel.get(team_id) or klubname(seite)
                          or anzeige.get(team_id, team_id))
                for s, pid, qq in zip(kader, ids, q):
                    if s["position_tm"] and not s["position_fine"]:
                        unbekannte_positionen.add(s["position_tm"])
                    zeile = {**s, "player_id": pid, "season": season,
                             "liga": liga, "verein": verein}
                    quellen[id(zeile)] = qq
                    zeilen.append(zeile)

                hinweis = f"  ({doppelt} mehrdeutig)" if doppelt else ""
                print(f"  {verein:<16} {slug:<28} {len(kader):>2} Spieler, "
                      f"{sum(1 for i in ids if i):>2} zugeordnet  "
                      f"(Guete {guete}){hinweis}")

        if not zeilen:
            print(f"[tm] {season}: nichts geholt - nicht geschrieben")
            continue

        geleert = saison_bereinigen(zeilen, quellen)
        if geleert:
            print(f"  {geleert} Zeilen geleert: dieselbe Kickbase-ID von zwei "
                  f"verschiedenen TM-Spielern beansprucht")
        schreiben(zeilen, season)
        zugeordnet = sum(1 for z in zeilen if z["player_id"])
        bilanz.append((season, len(zeilen), zugeordnet))
        print(f"[tm] {season}: {len(zeilen)} Zeilen, {zugeordnet} mit Kickbase-ID "
              f"({100 * zugeordnet / len(zeilen):.0f} %)")

    print(f"\n[tm] {ZIEL}")
    for season, n, zug in bilanz:
        print(f"  {season}  {n:>5} Zeilen  {zug:>5} zugeordnet")
    if sitzung.gebremst:
        print(f"[tm] {sitzung.gebremst}x von Transfermarkt gebremst - "
              f"ggf. --delay erhoehen")
    if unbekannte_positionen:
        print(f"[tm] unbekannte TM-Positionen (POSITION erweitern): "
              f"{sorted(unbekannte_positionen)}")
    print("[tm] Luecken bei player_id von Hand ergaenzen - "
          "ein erneuter Lauf uebernimmt sie.")


if __name__ == "__main__":
    main()
