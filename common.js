/* ══════════════════════════════════════════════════════════════════════
   common.js — Saison-Auswahl, Vorsaison-Übertrag und Teamstärke

   Alle Auswertungsseiten teilen sich drei Dinge:
     · das Saison-Manifest data/seasons.json (die einzige Stelle, an der eine
       Saison eingetragen wird),
     · die Achse, auf der gefiltert wird — in der aggregierten Sicht laufen
       Vorsaison und laufende Saison lückenlos hintereinander,
     · die Berechnung der Teamstärke aus den zugelassenen Punkten.

   Der gespeicherte Saison-Schlüssel ist ein Saison-Key aus dem Manifest
   ('2627', '2526') oder SEASON_AGG. Der früher von index.html geschriebene
   Leerstring meinte "laufende Saison" und wird als veraltet behandelt — er
   landet auf der aggregierten Sicht.

   Gespeichert wird je Seite. Vorher teilten sich alle Tools einen Schlüssel,
   sodass ein Wechsel im Scores-Editor still auch die Punkte-pro-Team-Seite
   umstellte — die Seiten meinen mit "Saison" aber Verschiedenes: dort ein
   Fenster über beobachtete Spieltage, hier nur die Wahl zwischen laufender
   Saison und Archiv.
   ══════════════════════════════════════════════════════════════════════ */

const SEASON_AGG    = 'agg';
const SEASON_KEY    = 'kickbase_season';   // Basis; je Seite um den Dateinamen ergänzt
const DAYS_KEY      = 'kickbase_days';     // dito, zusätzlich um die Saisonsicht
const POS_ORDER     = [1, 2, 3, 4];

const SRC_LABEL = { same: '', up: 'Aufsteiger', down: 'Absteiger', new: 'Neuling' };

/* ─── Saison-Manifest ───────────────────────────────────── */

let SEASONS = null;

async function loadSeasonManifest() {
  SEASONS = await fetch('data/seasons.json').then(r => r.json());
  return SEASONS;
}

function currentSeason() {
  return SEASONS.seasons.find(s => s.key === SEASONS.current);
}

function prevSeason() {
  return SEASONS.seasons
    .filter(s => s.key !== SEASONS.current)
    .sort((a, b) => b.key.localeCompare(a.key))[0] || null;
}

/* Dateien der aggregierten Sicht sind die der laufenden Saison — die Vorsaison
   kommt aus data/carryover.json und nicht aus den 2,6-MB-Archivdateien. */
function seasonInfo(key) {
  return SEASONS.seasons.find(s => s.key === key) || currentSeason();
}

function isAgg(key) {
  return key === SEASON_AGG;
}

/* Unbekanntes oder Veraltetes ('' aus der alten Startseite) → aggregierte Sicht */
function normalizeSeason(key) {
  if (key === SEASON_AGG) return SEASON_AGG;
  return SEASONS.seasons.some(s => s.key === key) ? key : SEASON_AGG;
}

/* Ein Schlüssel je Seite, abgeleitet vom Dateinamen. */
function seasonStorageKey() {
  const page = (location.pathname.split('/').pop() || '').replace(/\.html$/, '');
  return `${SEASON_KEY}_${page || 'index'}`;
}

function storedSeason() {
  const q = new URLSearchParams(location.search).get('season');
  if (q !== null) return normalizeSeason(q);
  // Beim ersten Aufruf nach der Umstellung gibt es noch keinen eigenen Wert —
  // dann gilt einmalig der alte gemeinsame, damit niemand seine Auswahl verliert.
  const own = localStorage.getItem(seasonStorageKey());
  return normalizeSeason(own !== null ? own : localStorage.getItem(SEASON_KEY));
}

function saveSeason(key) {
  localStorage.setItem(seasonStorageKey(), key);
}

/* ─── Spieltagsfenster merken ────────────────────────────────
   Wie die Saison je Seite, zusätzlich aber je Saisonsicht: die Eingaben sind
   Achsen-Indizes, und in der Aggregat-Sicht ist die Achse doppelt so lang. Ein
   gemeinsamer Schlüssel würde das Fenster beim Umschalten auf die kürzere
   Achse stauchen — und beim Zurückschalten käme es nicht wieder. */
function dayStorageKey(season) {
  const page = (location.pathname.split('/').pop() || '').replace(/\.html$/, '');
  return `${DAYS_KEY}_${page || 'index'}_${season}`;
}

function storedDayRange(season) {
  try {
    const v = JSON.parse(localStorage.getItem(dayStorageKey(season)) || 'null');
    if (v && Number.isFinite(v.from) && Number.isFinite(v.to)) return v;
  } catch (e) { /* beschädigter Eintrag — dann eben das Standardfenster */ }
  return null;
}

function saveDayRange(season, from, to) {
  try {
    localStorage.setItem(dayStorageKey(season), JSON.stringify({ from, to }));
  } catch (e) { /* voller oder gesperrter Speicher; kein Grund, die Seite zu stören */ }
}

function seasonLabel(key) {
  if (isAgg(key)) {
    const p = prevSeason();
    return currentSeason().label + (p ? ' + ' + p.label : '');
  }
  return seasonInfo(key).label;
}

/* Auswahlfeld befüllen: aggregiert zuerst, danach die Saisons einzeln.

   ``withAgg === false`` lässt die aggregierte Sicht weg. Das ist für Seiten
   gedacht, die ihre Werte aus data/ratings.json beziehen: dort steckt die
   Vorsaison ohnehin schon im Modell (Gewicht 0,30, Carryover, Quoten-Prior),
   der Eintrag würde also dasselbe Ergebnis liefern wie die laufende Saison. */
function fillSeasonSelect(el, value, withAgg) {
  const cur = currentSeason(), prev = prevSeason();
  const opts = [];
  if (withAgg !== false && prev) {
    opts.push(`<option value="${SEASON_AGG}">`
            + `Saison ${cur.label} + Vorsaison ${prev.label}</option>`);
  }
  SEASONS.seasons.forEach(s => {
    opts.push(`<option value="${s.key}">`
            + `${s.key === SEASONS.current ? 'Saison' : 'Archiv'} ${s.label}</option>`);
  });
  el.innerHTML = opts.join('');
  let v = normalizeSeason(value);
  if (withAgg === false && isAgg(v)) v = SEASONS.current;
  el.value = v;
  return el.value;
}

/* ─── Vorsaison-Übertrag (data/carryover.json) ──────────── */

async function loadCarryover() {
  try {
    const r = await fetch('data/carryover.json');
    return r.ok ? await r.json() : null;
  } catch (_) { return null; }
}

/* Nachschlagewerk je Liga: Team → Herkunft, Faktoren und Spieltage.
   Die Faktoren sind bewusst nicht eingerechnet — so kann jede Seite anzeigen,
   womit skaliert wurde. */
function carryIndex(carry, liga) {
  const L = carry && carry.leagues && carry.leagues[liga];
  if (!L) return null;
  const teams = {};
  Object.entries(L.teams).forEach(([tid, t]) => {
    const byDay = {};
    t.days.forEach(d => byDay[d.day] = d);
    teams[tid] = {
      src: t.src,
      srcLeague: t.src_league,
      srcAbbr: t.src_abbr,
      factor: carry.factors[t.src] || carry.factors.same,
      byDay
    };
  });
  return { teams, maxDay: L.prev_max_day, factors: carry.factors, prev: carry.prev };
}

function carryDay(cidx, teamId, day) {
  const t = cidx && cidx.teams[teamId];
  return t ? (t.byDay[day] || null) : null;
}

/* Positionswert aus einem Vorsaison-Feld, skaliert mit dem passenden Faktor.
   kind: 'conceded' | 'scored' | 'start' (letzteres skaliert wie 'scored');
   'slots' zählt Startplätze und wird nie skaliert. */
function carryVal(cidx, teamId, entry, kind, pos) {
  const t = cidx.teams[teamId];
  const field = kind === 'start' ? 'start' : kind;
  const raw = entry[field][pos - 1] || 0;
  if (kind === 'slots') return raw;
  return raw * t.factor[kind === 'start' ? 'scored' : kind];
}

/* Über alle Ablagen, nicht nur die vier Positionsgruppen: an letzter Stelle
   stehen die Spieler, für die Kickbase keine Position führt. Sie gehören in
   jede Summe — nur nicht in einen Positionsbalken. */
function carrySum(cidx, teamId, entry, kind) {
  const n = (entry[kind === 'start' ? 'start' : kind] || []).length;
  let sum = 0;
  for (let p = 1; p <= n; p++) sum += carryVal(cidx, teamId, entry, kind, p);
  return sum;
}

/* Punkte der Spieler ohne Position — die fünfte Ablage, falls vorhanden. */
function carryUnpositioned(cidx, teamId, entry, kind) {
  const arr = entry[kind === 'start' ? 'start' : kind] || [];
  return arr.length > POS_ORDER.length
    ? carryVal(cidx, teamId, entry, kind, arr.length) : 0;
}

/* Kurzer Hinweis für Tooltips: "Aufsteiger aus 2. Bundesliga, Vorsaison ×1,59" */
function carryNote(cidx, teamId, kind) {
  const t = cidx && cidx.teams[teamId];
  if (!t || t.src === 'same') return '';
  const f = t.factor[kind].toLocaleString('de', { minimumFractionDigits: 2 });
  const woher = t.src === 'new' ? 'ohne Vorsaison, Ligamittel'
                                : `aus ${t.srcLeague === '1' ? 'Bundesliga' : '2. Bundesliga'}`
                                  + (t.srcAbbr ? ` (${t.srcAbbr})` : '');
  return `${SRC_LABEL[t.src]} ${woher} · Vorsaison ×${f}`;
}

/* ─── Achse ─────────────────────────────────────────────── */

/* Spieltage der laufenden Saison, die wirklich gespielt wurden.
   Ohne diesen Filter zählen die 33 noch leeren Spieltage mit points: 0 als
   echte Nullen mit — genau daran scheiterte die Teamstärke zu Saisonbeginn. */
function playedDays(matchdays) {
  const days = new Set();
  if (!matchdays) return days;
  matchdays.matchdays.forEach(md =>
    md.matches.forEach(m => { if (m.st === 2) days.add(m.day); }));
  return days;
}

/* Letzter Spieltag vor der Winterpause — das Ende der Hinrunde.
   Keine feste Zahl: Bundesliga 13–17, 2. Bundesliga 16–18 (2026/27: 14 und 16).
   Gepflegt wird die Grenze in kbxp/data/processed/season_splits.parquet; die
   liegt aber im Analyse-Teil und ist im Browser nicht lesbar. Hier wird sie
   deshalb so abgeleitet, wie die Datei selbst entstanden ist: über die größte
   Terminlücke zwischen zwei Spieltagen. Für die vier Spielpläne unter data/
   trifft das die gepflegten Werte exakt. Die eine bekannte Fehlerquelle der
   Ableitung — 2019/20, wo die Corona-Pause nach ST 25 länger war als die
   Winterpause — kann in einem vorab angesetzten Spielplan nicht auftreten. */
function hinrundeEnde(matchdays) {
  const span = {};   // Spieltag -> [frühester, spätester Anstoß]
  (matchdays?.matchdays || []).forEach(md => md.matches.forEach(m => {
    if (!m.dt) return;
    const t = Date.parse(m.dt);
    if (isNaN(t)) return;
    const s = span[m.day];
    if (!s) span[m.day] = [t, t];
    else { if (t < s[0]) s[0] = t; if (t > s[1]) s[1] = t; }
  }));
  const days = Object.keys(span).map(Number).sort((a, b) => a - b);
  let best = 0, at = 0;
  for (let i = 1; i < days.length; i++) {
    const gap = span[days[i]][0] - span[days[i - 1]][1];
    if (gap > best) { best = gap; at = days[i - 1]; }
  }
  return at;
}

/* Fällt der Spielplan aus, bleiben die Tage mit Punkten als Näherung. */
function playedDaysFallback(rawData) {
  const days = new Set();
  rawData.players.forEach(p => (p.performance || []).forEach(e => {
    if (e.points !== 0 || parseInt(e.minutes, 10) > 0) days.add(e.day);
  }));
  return days;
}

/* Durchgehende Achse: erst die Vorsaison (Label 'V12'), dann die laufende.
   In der reinen Saisonsicht ist der Index gleich dem Spieltag. */
function buildAxis(cidx, curDays, agg) {
  const axis = [];
  if (agg && cidx) {
    for (let d = 1; d <= cidx.maxDay; d++)
      axis.push({ season: 'prev', day: d, label: 'V' + d });
  }
  [...curDays].sort((a, b) => a - b).forEach(d =>
    axis.push({ season: 'cur', day: d, label: String(d) }));
  axis.forEach((a, i) => a.i = i + 1);
  return axis;
}

/* Standardfenster: immer die ganze Achse — eine Saison 1–34, die Aggregat-Sicht
   über Vor- und laufende Saison 1–68. */
function defaultRange(axis) {
  return { from: 1, to: axis.length || 1 };
}

function axisLabel(axis, i) {
  const a = axis[i - 1];
  return a ? a.label : String(i);
}

function rangeLabel(axis, from, to) {
  return `${axisLabel(axis, from)}–${axisLabel(axis, to)}`;
}

/* Von/Bis-Paar in die Achse zwingen; liefert den bereinigten Bereich. */
function clampRange(fromEl, toEl, axis, changed) {
  const min = 1, max = axis.length || 1;
  let from = Math.max(min, Math.min(max, +fromEl.value || min));
  let to   = Math.max(min, Math.min(max, +toEl.value   || max));
  if (changed === 'from' && from > to) to = from;
  if (changed === 'to'   && to < from) from = to;
  fromEl.value = from;
  toEl.value   = to;
  return { from, to };
}

/* ─── Teamstärke ────────────────────────────────────────── */

function safeJenks(values, k) {
  try { return ss.jenks(values, k); }
  catch (_) {
    const s = values.slice().sort((a, b) => a - b);
    const mn = s[0], mx = s[s.length - 1], step = (mx - mn) / k;
    return Array.from({ length: k + 1 }, (_, i) => mn + step * i);
  }
}

function classify(value, breaks, smin) {
  for (let i = breaks.length - 2; i >= 0; i--) {
    if (value >= breaks[i]) return i - 3;
  }
  return smin;
}

/* Zugelassene Punkte je Team und Spieltag, getrennt nach Heim und Auswärts.
   Rückgabe: { teamId: { home: {p, w}, away: {p, w} } } — p ist die gewichtete
   Summe, w die Summe der Gewichte. */
function concededSeries(rawData, cidx, axis, range, decay) {
  const ids = Object.keys(rawData.teams);
  const acc = {};
  ids.forEach(id => acc[id] = { home: { p: 0, w: 0 }, away: { p: 0, w: 0 } });

  const span = axis.slice(range.from - 1, range.to);
  if (!span.length) return acc;
  const last = span[span.length - 1].i;

  // Laufende Saison: Spielerpunkte auf den Gegner buchen
  const curDays = new Set(span.filter(a => a.season === 'cur').map(a => a.day));
  const cur = {};
  rawData.players.forEach(p => (p.performance || []).forEach(e => {
    if (!curDays.has(e.day)) return;
    const opp = String(e.opponent);
    if (!acc[opp]) return;
    const k = `${opp}|${e.day}`;
    // entry.home = der Spieler war zuhause ⇒ sein Gegner war auswärts
    if (!cur[k]) cur[k] = { pts: 0, home: !e.home };
    cur[k].pts += e.points;
  }));

  span.forEach(a => {
    const w = Math.pow(decay, last - a.i);
    ids.forEach(id => {
      let pts, home;
      if (a.season === 'cur') {
        const c = cur[`${id}|${a.day}`];
        if (!c) return;
        pts = c.pts; home = c.home;
      } else {
        const d = carryDay(cidx, id, a.day);
        if (!d) return;
        pts = carrySum(cidx, id, d, 'conceded'); home = d.home;
      }
      const slot = home ? acc[id].home : acc[id].away;
      slot.p += pts * w;
      slot.w += w;
    });
  });
  return acc;
}

/* Heim-/Auswärts-Score −3…+3 je Team, Jenks über die zugelassenen Punkte.
   Teams ohne Spiele im Fenster bekommen den Ligadurchschnitt statt einer 0 —
   sonst rutschen sie grundlos ans untere Ende. */
function computeBaseScores(rawData, cidx, axis, range, decay = 0.95, smin = -3) {
  const acc = concededSeries(rawData, cidx, axis, range, decay);
  const ids = Object.keys(rawData.teams);

  const pick = side => {
    const raw = ids.map(id => acc[id][side].w ? acc[id][side].p / acc[id][side].w : null);
    const have = raw.filter(v => v !== null);
    const mean = have.length ? have.reduce((a, b) => a + b, 0) / have.length : 0;
    return raw.map(v => v === null ? mean : v);
  };

  // Ohne gespielte Spieltage sind alle Werte gleich — dann ist 0 die ehrliche
  // Antwort, nicht der oberste Jenks-Korb für die ganze Liga.
  const rank = vals => {
    if (Math.max(...vals) === Math.min(...vals)) return vals.map(() => 0);
    const breaks = safeJenks(vals, 7);
    return vals.map(v => classify(v, breaks, smin));
  };

  const hs = rank(pick('home')), as = rank(pick('away'));
  const out = {};
  ids.forEach((id, i) => { out[id] = { home: hs[i], away: as[i] }; });
  return out;
}

/* ─── Gegnerbereinigte Ratings (data/ratings.json) ───────── */

/* Der rohe Mittelwert oben misst zwei Dinge auf einmal: die eigene
   Abwehrschwäche und die Angriffsstärke der zufällig gezogenen Gegner. Wer früh
   die drei besten Kader erwischt, sieht defensiv schlecht aus, ohne es zu sein.

   kbxp/src/model/team_strength.py trennt beides in einer Ridge-Regression:

       zugelassene Punkte(Team i gegen Gegner j) = mu + def_i + att_j + hfa·heim

   Im Walk-forward über 2013/14–2025/26 steigt die Rangkorrelation zum
   tatsächlichen Ergebnis dadurch von 0,21 auf 0,46. Hier wird nur gelesen —
   fällt die Datei aus, rechnet computeBaseScores() weiter wie bisher. */

async function loadRatings() {
  try {
    const r = await fetch('data/ratings.json');
    return r.ok ? await r.json() : null;
  } catch (_) { return null; }
}

function ratingLeague(ratings, liga) {
  const L = ratings && ratings.leagues && ratings.leagues[liga];
  return L && L.teams && L.breaks ? L : null;
}

/* Erwartete Punkte, die die Spieler von teamId gegen oppId holen — also die
   Punkte, die oppId zulässt. Der Heimvorteil hängt am zulassenden Team:
   spielt teamId zuhause, ist oppId auswärts. */
function expectedPoints(L, teamId, oppId, teamAtHome) {
  const t = L.teams[teamId], o = L.teams[oppId];
  return L.mu + (o ? o.def : 0) + (t ? t.att : 0) + (teamAtHome ? 0 : L.hfa);
}

/* Paarungs-Score −3…+3 aus Sicht von teamId. */
function matchupScore(L, teamId, oppId, teamAtHome, smin) {
  return classify(expectedPoints(L, teamId, oppId, teamAtHome), L.breaks,
                  smin === undefined ? -3 : smin);
}

/* Die Ridge-Strafe zieht beide Blöcke gegen null, zentriert sie aber nicht:
   in einer Liga mit starken Absteigern liegt der mittlere Angriffswert deutlich
   über null. Für „ein durchschnittlicher Gegner“ zählt dieser Mittelwert, nicht
   die Null — sonst fällt jede Teamkarte um eine halbe Klasse. */
function meanAttack(L) {
  const v = Object.values(L.teams);
  return v.length ? v.reduce((a, t) => a + t.att, 0) / v.length : 0;
}

/* Score eines Teams gegen einen durchschnittlichen Gegner — das, was die
   Teamstärke-Seite je Team zeigt. side: 'home' | 'away' meint, wo dieses Team
   spielt.

   Klassifiziert wird gegen eigene Grenzen: weil hier der Ligaschnitt statt des
   eigenen Angriffs steht, deckt dieser Modus nur einen Ausschnitt der
   Paarungsverteilung ab. An den Paarungsgrenzen gemessen kamen +3 in 0,6 % und
   −3 in 3,5 % der Zellen vor — zwei der sieben Farben waren praktisch tot,
   ausgerechnet in dem Modus, der leichte und schwere Spielpläne zeigen soll.
   Die Grenzen entstehen im Export über Quantile, nicht über Jenks; die
   Begründung steht dort bei quantile_breaks.
   Fällt das Feld weg (ältere ratings.json), gilt wieder L.breaks. */
function teamSideScore(L, teamId, side, smin) {
  const t = L.teams[teamId];
  const v = L.mu + (t ? t.def : 0) + meanAttack(L) + (side === 'home' ? L.hfa : 0);
  return classify(v, L.breaks_fixture || L.breaks, smin === undefined ? -3 : smin);
}

/* Spieltage, die beim letzten Export in die Ratings eingeflossen sind. Bleibt
   der Wert hinter den tatsächlich gespielten zurück, ist die Datei veraltet. */
function ratingsDays(L) {
  return Object.values(L.teams).reduce((m, t) => Math.max(m, t.n || 0), 0);
}

function ratingBaseScores(L, teamIds) {
  const out = {};
  teamIds.forEach(id => {
    out[id] = { home: teamSideScore(L, id, 'home'), away: teamSideScore(L, id, 'away') };
  });
  return out;
}

/* ─── Gespeicherte Scores ────────────────────────────────────
   Gespeichert wird beides: die Basis, auf der gerechnet wurde, und der Stand
   nach Handkorrektur. Nur so darf sich die Basis mit jedem neuen Spieltag
   erneuern, ohne die Korrektur zu verlieren — vorher fror der erste
   Seitenaufruf die Scores für den Rest der Saison ein. */

const SCORE_STORE_VERSION = 2;

function readScoreStore(key) {
  try {
    const s = JSON.parse(localStorage.getItem(key));
    if (!s || typeof s !== 'object') return null;
    if (s.version === SCORE_STORE_VERSION) return s;
    // Format 1 hielt nur absolute Werte einer inzwischen abgelösten Skala fest.
    // Ohne die zugehörige Basis lässt sich die Handkorrektur nicht herauslösen.
    return null;
  } catch (_) { return null; }
}

function writeScoreStore(key, base, scores) {
  try {
    localStorage.setItem(key, JSON.stringify({
      version: SCORE_STORE_VERSION, base: base, scores: scores
    }));
  } catch (_) {}
}

/* Handkorrekturen als Differenz zur damaligen Basis auf eine neue Basis
   übertragen und wieder auf −3…+3 begrenzen. */
function applyScoreDeltas(base, store, smin, smax) {
  const out = {};
  Object.keys(base).forEach(id => {
    out[id] = { home: base[id].home, away: base[id].away };
    if (!store || !store.base || !store.base[id] || !store.scores || !store.scores[id]) return;
    ['home', 'away'].forEach(side => {
      const d = store.scores[id][side] - store.base[id][side];
      if (d) out[id][side] = Math.max(smin, Math.min(smax, out[id][side] + d));
    });
  });
  return out;
}
