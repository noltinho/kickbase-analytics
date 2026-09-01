"""Unabhaengiger Modellvergleich fuer Oe Punkte je Einsatz.

Dieses Modul benutzt keine Prognose eines bestehenden Spielermodells als
Merkmal. Es uebernimmt nur deren gemeinsame, leakagefreie Datenschicht und
stellt mehrere Modellfamilien auf exakt dieselbe Walk-forward-Maske. Die
Hyperparameter werden je aeusserer Zielsaison auf der jeweils letzten noch
bekannten Saison gewaehlt; die aeussere Testsaison bleibt unangetastet.

Der Vergleich beantwortet die bedingte Leistungsfrage getrennt nach:

* absoluter Prognose: MAE, RMSE, Verzerrung und Kalibrierungsgerade;
* Rangfolge: Korrelation innerhalb der Saison und paarweise Konkordanz;
* Problemsegmenten: verlorene Saison, Historien-Spitze und Elite-Neuzugang.

Primaeres Ziel ist der Ganzjahres-Schnitt der Spieler, die in der Zielsaison
tatsaechlich gesetzt waren. Einsatz-/Verletzungswahrscheinlichkeit, erwartete
Gesamteinsaetze und Pick-Ertrag sind ausdruecklich nicht Teil der Frage.

Aufruf aus ``kbxp/``::

    python -m src.model.player_benchmark

Die maschinenlesbare Ausgabe landet unter ``data/interim/`` und ist damit ein
reproduzierbares Forschungsartefakt, kein Produktionsdatum.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (ExtraTreesRegressor, GradientBoostingRegressor,
                              HistGradientBoostingRegressor)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, SplineTransformer, StandardScaler

from src.model.player_avg import (Daten, ERSTE_ZIELSAISON, LIGA1,
                                  backtest as avg_backtest,
                                  zeilen_aller_saisons)
from src.model.player_role_model import backtest_p90
from src.paths import INTERIM, atomic_write_json


ERSTE_OOF_SAISON = ERSTE_ZIELSAISON + 1   # 2017: zwei Jahre Anlauf fuer 2019
ERSTE_TESTSAISON = ERSTE_ZIELSAISON + 3  # 2019, identisch zu player_avg
RNG = 20260819

NUM = [
    "a1", "z_hat", "zelle", "log2_mw", "alter", "log_gewicht",
    "hat_hist", "a1_fehlt", "zelle_fehlt", "alter_fehlt", "mw_fehlt",
    "hist_belegt", "a1_belegt", "hist_team_delta", "mw_delta",
    "team_delta", "rel_mw", "rel_age", "cell_n",
    "z_TW", "z_IV", "z_AV", "z_MIT", "z_OFF",
    "hd_TW", "hd_IV", "hd_AV", "hd_MIT", "hd_OFF",
]
CAT = ["fall", "pos", "teil", "stufe"]

BASE_NUM = [
    "z_hat", "zelle", "log2_mw", "alter", "zelle_fehlt", "alter_fehlt",
    "mw_fehlt", "rel_mw", "rel_age", "cell_n", "team_delta",
    "z_TW", "z_IV", "z_AV", "z_MIT", "z_OFF",
]


def merkmale(df: pd.DataFrame) -> pd.DataFrame:
    """Abgeleitete, weiterhin vor Saisonbeginn bekannte Merkmale."""
    x = df.copy()
    x["log_gewicht"] = np.log1p(x.gewicht.fillna(0.0).clip(lower=0.0))
    for c in ("a1", "zelle", "alter", "log2_mw"):
        x[f"{c.replace('log2_mw', 'mw')}_fehlt"] = x[c].isna().astype(float)
    # Das einzige vorab gesetzte Interaktionsgesetz: Teamstaerke wirkt je
    # Mannschaftsteil verschieden. Keine Zielinformation fliesst hier ein.
    for teil in ("TW", "IV", "AV", "MIT", "OFF"):
        x[f"z_{teil}"] = x.z_hat * (x.teil == teil).astype(float)
        x[f"hd_{teil}"] = x.hist_team_delta * (x.teil == teil).astype(float)
    x["stufe"] = x.stufe.fillna("keine")
    return x


def _vorverarbeitung(num: list[str], spline: bool = False) -> ColumnTransformer:
    if spline:
        num_pipe = Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("spline", SplineTransformer(n_knots=4, degree=2,
                                         include_bias=False)),
            ("scale", StandardScaler()),
        ])
    else:
        num_pipe = Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ])
    cat_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    return ColumnTransformer([("num", num_pipe, num), ("cat", cat_pipe, CAT)],
                             sparse_threshold=0.0)


def _pipe(model, *, num: list[str] = NUM, spline: bool = False) -> Pipeline:
    return Pipeline([("prep", _vorverarbeitung(num, spline=spline)),
                     ("model", model)])


@dataclass(frozen=True)
class Variante:
    name: str
    configs: tuple[dict, ...]
    bauen: Callable[[dict], Pipeline]


def _gitter() -> list[Variante]:
    rhos = (0.6, 0.8, 1.0)
    return [
        Variante(
            "ridge",
            tuple({"alpha": a, "rho": r} for a in (1.0, 10.0, 100.0, 500.0)
                  for r in rhos),
            lambda c: _pipe(Ridge(alpha=c["alpha"])),
        ),
        Variante(
            "spline_ridge",
            tuple({"alpha": a, "rho": r} for a in (10.0, 100.0, 500.0)
                  for r in rhos),
            lambda c: _pipe(Ridge(alpha=c["alpha"]), spline=True),
        ),
        Variante(
            "gradient_huber",
            tuple({"depth": d, "leaf": leaf, "rho": r}
                  for d in (1, 2) for leaf in (10, 25) for r in (0.6, 1.0)),
            lambda c: _pipe(GradientBoostingRegressor(
                loss="huber", n_estimators=250, learning_rate=0.03,
                max_depth=c["depth"], min_samples_leaf=c["leaf"],
                random_state=RNG)),
        ),
        Variante(
            "hist_gradient",
            tuple({"leaves": leaves, "l2": l2, "rho": r}
                  for leaves in (7, 15) for l2 in (5.0, 30.0)
                  for r in (0.6, 1.0)),
            lambda c: _pipe(HistGradientBoostingRegressor(
                loss="squared_error", max_iter=300, learning_rate=0.04,
                max_leaf_nodes=c["leaves"], min_samples_leaf=20,
                l2_regularization=c["l2"], early_stopping=False,
                random_state=RNG)),
        ),
        Variante(
            "extra_trees",
            tuple({"leaf": leaf, "features": feat, "rho": r}
                  for leaf in (5, 12, 25) for feat in (0.6, 1.0)
                  for r in (0.6, 1.0)),
            lambda c: _pipe(ExtraTreesRegressor(
                n_estimators=400, min_samples_leaf=c["leaf"],
                max_features=c["features"], n_jobs=-1, random_state=RNG)),
        ),
    ]


def _gewichte(train: pd.DataFrame, rho: float) -> np.ndarray:
    return rho ** (int(train.j.max()) - train.j.to_numpy(dtype=int))


def _fit_predict(pipe: Pipeline, train: pd.DataFrame, test: pd.DataFrame,
                 config: dict, ziel: np.ndarray | None = None) -> np.ndarray:
    y = train.avg_alle.to_numpy(float) if ziel is None else np.asarray(ziel, float)
    pipe.fit(train[NUM + CAT], y,
             model__sample_weight=_gewichte(train, float(config["rho"])))
    return pipe.predict(test[NUM + CAT])


def _config_key(c: dict) -> str:
    return ",".join(f"{k}={c[k]}" for k in sorted(c))


def _waehlen(variante: Variante, train: pd.DataFrame) -> dict:
    """Hyperparameter nur auf der letzten bekannten Trainingssaison waehlen."""
    j_val = int(train.j.max())
    innen, val = train[train.j < j_val], train[train.j == j_val]
    if len(innen) < 200:
        return variante.configs[0]
    beste, best_score = variante.configs[0], float("inf")
    for c in variante.configs:
        pred = _fit_predict(variante.bauen(c), innen, val, c)
        # MAE ist die Hauptfrage; ein kleiner Bias-Zuschlag verhindert, dass
        # zwei gleich gute Varianten nur wegen eines Niveaufehlers tauschen.
        err = pred - val.avg_alle.to_numpy(float)
        score = float(np.abs(err).mean() + 0.10 * abs(err.mean()))
        if score < best_score:
            beste, best_score = c, score
    return beste


def _basis_pipe(alpha: float = 30.0) -> Pipeline:
    return _pipe(Ridge(alpha=alpha), num=BASE_NUM)


def _bayes_vorhersage(train: pd.DataFrame, test: pd.DataFrame,
                      k: float, rho: float) -> np.ndarray:
    """Direkte Historie gegen eine unabhaengige Kontext-Baseline schrumpfen."""
    pipe = _basis_pipe()
    pipe.fit(train[BASE_NUM + CAT], train.avg_alle.to_numpy(float),
             model__sample_weight=_gewichte(train, rho))
    basis = pipe.predict(test[BASE_NUM + CAT])
    w = test.gewicht.fillna(0.0).to_numpy(float)
    hist = test.a1.to_numpy(float)
    anteil = np.where(np.isfinite(hist), w / (w + k), 0.0)
    hist = np.where(np.isfinite(hist), hist, basis)
    return basis + anteil * (hist - basis)


def _bayes_waehlen(train: pd.DataFrame) -> dict:
    j_val = int(train.j.max())
    innen, val = train[train.j < j_val], train[train.j == j_val]
    beste, best_score = {"k": 20.0, "rho": 0.8}, float("inf")
    if len(innen) < 200:
        return beste
    for k in (2.0, 5.0, 10.0, 20.0, 40.0, 80.0):
        for rho in (0.6, 0.8, 1.0):
            pred = _bayes_vorhersage(innen, val, k, rho)
            err = pred - val.avg_alle.to_numpy(float)
            score = float(np.abs(err).mean() + 0.10 * abs(err.mean()))
            if score < best_score:
                beste, best_score = {"k": k, "rho": rho}, score
    return beste


def neue_oof(daten: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    tests = sorted(int(j) for j in daten.j.unique() if j >= ERSTE_OOF_SAISON)
    aus, wahlen = [], {"empirical_bayes": []}
    varianten = _gitter()
    for v in varianten:
        wahlen[v.name] = []

    for j in tests:
        train, test = daten[daten.j < j], daten[daten.j == j].copy()
        if len(train) < 200:
            continue

        cb = _bayes_waehlen(train)
        test["empirical_bayes"] = _bayes_vorhersage(
            train, test, float(cb["k"]), float(cb["rho"]))
        wahlen["empirical_bayes"].append(_config_key(cb))

        for v in varianten:
            c = _waehlen(v, train)
            test[v.name] = _fit_predict(v.bauen(c), train, test, c)
            wahlen[v.name].append(_config_key(c))

        # Faire Marktwert-Baseline auf der Zielskala, nicht der rohe log-MW.
        c_mw = {"rho": 0.8}
        mw = _pipe(Ridge(alpha=30.0), num=["log2_mw"])
        mw.fit(train[["log2_mw"] + CAT], train.avg_alle.to_numpy(float),
               model__sample_weight=_gewichte(train, c_mw["rho"]))
        test["market_only"] = mw.predict(test[["log2_mw"] + CAT])
        aus.append(test)
    return pd.concat(aus, ignore_index=True), wahlen


def _corr(a, b) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    return float(np.corrcoef(a, b)[0, 1]) if len(a) > 2 else float("nan")


def _konkordanz(df: pd.DataFrame, pred: str, target: str) -> float:
    """Anteil richtig geordneter Spielerpaare, nur innerhalb einer Saison."""
    gut = gesamt = 0
    for _, s in df.groupby("j"):
        y, p = s[target].to_numpy(float), s[pred].to_numpy(float)
        dy = y[:, None] - y[None, :]
        dp = p[:, None] - p[None, :]
        tri = np.triu(np.ones_like(dy, dtype=bool), 1) & (np.abs(dy) > 1e-9)
        gesamt += int(tri.sum())
        gut += int(((dy * dp) > 0)[tri].sum())
    return gut / gesamt if gesamt else float("nan")


def kennzahlen(df: pd.DataFrame, pred: str, target: str) -> dict:
    s = df.dropna(subset=[pred, target])
    e = s[pred] - s[target]
    rs = [_corr(g[pred], g[target]) for _, g in s.groupby("j")]
    slope, intercept = (np.polyfit(s[pred], s[target], 1)
                        if len(s) > 2 else (float("nan"), float("nan")))
    return {
        "n": int(len(s)),
        "r": _corr(s[pred], s[target]),
        "r_saison": float(np.nanmean(rs)),
        "spearman": float(spearmanr(s[pred], s[target]).statistic),
        "mae": float(np.abs(e).mean()),
        "rmse": float(np.sqrt(np.mean(e ** 2))),
        "bias": float(e.mean()),
        "cal_slope": float(slope),
        "cal_intercept": float(intercept),
        "paarweise": _konkordanz(s, pred, target),
    }


def _bootstrap_delta(df: pd.DataFrame, pred: str, ref: str, target: str,
                     n: int = 20_000) -> dict:
    """Saison-Clusterbootstrap der MAE-Differenz; Spieler sind nicht unabhaengig."""
    d = df.dropna(subset=[pred, ref, target]).copy()
    je = []
    for _, s in d.groupby("j"):
        je.append(float(np.abs(s[pred] - s[target]).mean()
                        - np.abs(s[ref] - s[target]).mean()))
    a = np.asarray(je, float)
    rng = np.random.default_rng(RNG)
    boot = rng.choice(a, size=(n, len(a)), replace=True).mean(axis=1)
    return {"delta_mae": float(a.mean()),
            "ci95": [float(np.quantile(boot, 0.025)),
                     float(np.quantile(boot, 0.975))],
            "saisons_besser": int((a < 0).sum()), "saisons": int(len(a))}


def _bestandsmodelle(dat: Daten) -> tuple[pd.DataFrame, dict]:
    # Das fallweise Modell bringt seine strikt wachsende OOF-Kalibrierung mit;
    # sie ist Teil der nun getesteten Architektur. Beim Rollenmodell wird nur
    # die bedingte Ausgabe fuer eine gehaltene Rolle verwendet. ``xp_dach``
    # waere eine Durchsetzungsprognose und damit eine andere Zielgroesse.
    avg = avg_backtest(dat, ab=ERSTE_OOF_SAISON)[
        ["j", "player_id", "pred"]].rename(
        columns={"pred": "fallweise"})

    rolle = backtest_p90()
    rolle["j"] = rolle.saison.str.slice(0, 4).astype(int)
    rolle = rolle.rename(columns={"xp_gesetzt": "rollenmodell_gesetzt"})
    rolle = rolle[["j", "player_id", "rollenmodell_gesetzt"]]
    return avg.merge(rolle, on=["j", "player_id"], how="outer"), {
        "hinweis": ("Fallweise inklusive eigener OOF-Tail-Kalibrierung; "
                    "Rollenmodell nur xp_gesetzt, ohne Durchsetzungsrisiko.")
    }


def _kalibrieren(df: pd.DataFrame, pred: str, target: str) -> pd.Series:
    """Wachsender Niveauversatz je Fall, ohne Blick in die Zielsaison."""
    out = pd.Series(np.nan, index=df.index, dtype=float)
    for j in sorted(df.j.unique()):
        te = df.j == j
        out.loc[te] = df.loc[te, pred]
        past = df[(df.j < j)].dropna(subset=[pred, target])
        for fall, s in past.groupby("fall"):
            js = s.assign(res=s[pred] - s[target]).groupby("j").res.mean()
            if len(js) >= 3:
                out.loc[te & (df.fall == fall)] = (
                    df.loc[te & (df.fall == fall), pred] - float(js.mean()))
    return out


def _drucke(alles: dict, wahlen: dict) -> None:
    print("Walk-forward 2019--2025, gleiche Starter-Maske, Oe Punkte/Einsatz")
    ergebnisse, robust = alles["metrics"], alles["robustness"]
    print("Modell                    r      r/Jahr   MAE    Bias  CalSlope Paarw.")
    sortiert = sorted(ergebnisse, key=lambda k: ergebnisse[k]["mae"])
    for name in sortiert:
        m = ergebnisse[name]
        print(f"{name:24s} {m['r']:.3f}   {m['r_saison']:.3f}   "
              f"{m['mae']:5.2f}  {m['bias']:+5.2f}    {m['cal_slope']:.3f}  "
              f"{m['paarweise']:.3f}")
    print("  Robustheit gegen fallweise (Delta MAE):")
    for name in sortiert:
        if name == "fallweise":
            continue
        r = robust[name]
        print(f"    {name:22s} {r['delta_mae']:+5.2f}  "
              f"95%-KI [{r['ci95'][0]:+5.2f}, {r['ci95'][1]:+5.2f}]  "
              f"besser {r['saisons_besser']}/{r['saisons']}")
    print("\nInnere Hyperparameterwahl (Haeufigkeiten ueber OOF-Saisons):")
    for name, werte in wahlen.items():
        print(f"  {name:21s} {dict(Counter(werte))}")


def main(path: Path | None = None) -> dict:
    dat = Daten()
    d = merkmale(zeilen_aller_saisons(dat))
    starter = (dat.ss[(dat.ss.league == LIGA1) & dat.ss.starter]
               [["player_id", "j"]].drop_duplicates())
    for abstand in (1, 2):
        flag = starter.assign(j=starter.j + abstand,
                              **{f"starter_v{abstand}": 1.0})
        d = d.merge(flag, on=["player_id", "j"], how="left")
        d[f"starter_v{abstand}"] = d[f"starter_v{abstand}"].fillna(0.0)
    # ``dat.ss`` enthaelt beide Ligen. Ein Spieler kann im selben Jahr in
    # beiden auftauchen; ohne den Liga-Filter wuerde der Join einzelne
    # Bundesliga-Zeilen verdoppeln und die gemeinsame Maske still veraendern.
    meta = (dat.ss[dat.ss.league == LIGA1]
            [["player_id", "j", "punkte", "n_einsaetze"]]
            .drop_duplicates(["player_id", "j"]))
    d = d.merge(meta, on=["player_id", "j"], how="left")
    d = d.rename(columns={"avg_alle": "jahr_avg", "punkte": "jahr_punkte"})
    # Die neuen Modelle werden auf dem rauscharmen Ganzjahresziel trainiert.
    # ``neue_oof`` erwartet dafuer weiterhin den internen Namen ``avg_alle``.
    d["avg_alle"] = d.jahr_avg

    oof, wahlen = neue_oof(d)
    bestand, notiz = _bestandsmodelle(dat)
    oof = oof.merge(bestand, on=["j", "player_id"], how="left")

    # Starke modellfreie Vergleichslinie. Fehlende Historie bekommt bewusst
    # keine erfundene Zahl und faellt nur aus deren eigener Kennzahl heraus.
    oof["historie_roh"] = oof.a1

    # Die naivste denkbare Regel: der ungewichtete Punkteschnitt je Einsatz
    # aus der Vorsaison, unveraendert uebernommen. Sie bekommt bewusst auch
    # keine Fall-Niveaukorrektur -- eine Vergleichslinie, die erst
    # zurechtgerueckt werden muss, ist keine mehr. Wer in der Vorsaison nicht
    # in dieser Liga gespielt hat, hat keinen Wert: Zu diesen Spielern sagt
    # die Regel nichts, statt etwas zu erfinden.
    vorsaison = (dat.ss[dat.ss.league == LIGA1][["player_id", "j", "avg_alle"]]
                 .drop_duplicates(["player_id", "j"])
                 .rename(columns={"avg_alle": "vorsaison_roh"}))
    vorsaison["j"] = vorsaison.j + 1
    oof = oof.merge(vorsaison, on=["player_id", "j"], how="left")

    basis = ["fallweise", "rollenmodell_gesetzt", "empirical_bayes", "ridge",
             "spline_ridge", "gradient_huber", "hist_gradient",
             "extra_trees", "market_only", "historie_roh",
             "vorsaison_roh"]
    cols = {}
    for p in basis:
        c = f"{p}__Ganzjahr"
        # Die neue fallweise Kalibrierung ist bereits innerhalb jedes
        # Walk-forward-Schritts gebaut. Alle Challenger erhalten weiterhin
        # dieselbe wachsende Fall-Niveaukorrektur; nur die naive Vorsaison-
        # Linie bleibt roh, weil genau ihre Unbehandeltheit der Punkt ist.
        oof[c] = (oof[p] if p in ("fallweise", "vorsaison_roh")
                  else _kalibrieren(oof, p, "jahr_avg"))
        cols[p] = c

    ev = oof[oof.j >= ERSTE_TESTSAISON].copy()
    metrics = {p: kennzahlen(ev, c, "jahr_avg") for p, c in cols.items()}
    robust = {p: _bootstrap_delta(ev, c, cols["fallweise"], "jahr_avg")
              for p, c in cols.items() if p != "fallweise"}
    by_case = {p: {f: kennzahlen(s, c, "jahr_avg")
                   for f, s in ev.groupby("fall")
                   if s[c].notna().sum() >= 20}
               for p, c in cols.items()}
    by_season = {p: {str(int(j)): kennzahlen(s, c, "jahr_avg")
                     for j, s in ev.groupby("j")}
                 for p, c in cols.items()}
    segmente = {
        "verlorene_saison": ((ev.fall == "a") & (ev.starter_v1 == 0.0)
                              & (ev.starter_v2 == 1.0)),
        "historie_spitze": (ev.fall == "a") & ev.a1.ge(170.0),
        "neuzugang_elite": ((ev.fall == "c")
                            & (np.power(2.0, ev.log2_mw) >= 35_000_000.0)),
    }
    by_segment = {
        p: {name: kennzahlen(ev[mask], c, "jahr_avg")
            for name, mask in segmente.items() if ev.loc[mask, c].notna().sum() >= 10}
        for p, c in cols.items()
    }
    alles = {"metrics": metrics, "robustness": robust, "by_case": by_case,
             "by_season": by_season, "by_segment": by_segment}

    out = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "target": ("Ganzjahres-Oe Punkte je Einsatz, bedingt auf ex-post "
                   "gesetzte Spieler; nur Informationen vor der Zielsaison"),
        "eval_seasons": sorted(int(j) for j in oof.j.unique()
                               if j >= ERSTE_TESTSAISON),
        "n_eval": int(oof.loc[oof.j >= ERSTE_TESTSAISON,
                              "fallweise"].notna().sum()),
        "bestandsmodell_notiz": notiz, "result": alles,
        "inner_choices": {k: dict(Counter(v)) for k, v in wahlen.items()},
    }
    ziel = path or (INTERIM / "player_benchmark.json")
    ziel.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(ziel, out, indent=2)
    oof.to_csv(INTERIM / "player_benchmark_oof.csv", index=False)
    _drucke(alles, wahlen)
    print(f"\ngeschrieben: {ziel}")
    return out


if __name__ == "__main__":
    main()
