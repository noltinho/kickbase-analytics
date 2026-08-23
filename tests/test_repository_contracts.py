from __future__ import annotations

import json
import re
from html.parser import HTMLParser
from pathlib import Path

import pytest

import fetch
from kbxp.src.paths import atomic_write_json


ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
HTML_FILES = sorted(ROOT.glob("*.html"))


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_published_season_files_are_consistent() -> None:
    manifest = _json(DATA / "seasons.json")
    seasons = {s["key"]: s for s in manifest["seasons"]}
    assert manifest["current"] in seasons

    for liga in ("1", "2"):
        for season in seasons.values():
            suffix = season["suffix"]
            data = _json(DATA / f"data_{liga}{suffix}.json")
            plan = _json(DATA / f"matchdays_bl{liga}{suffix}.json")
            assert data["generated_at"] == plan["generated_at"]
            assert data["teams"] and data["players"]
            assert len(plan["matchdays"]) == 34

            known = set(map(str, data["teams"]))
            for md in plan["matchdays"]:
                assert md["day"] in range(1, 35)
                assert len(md["matches"]) == 9
                for match in md["matches"]:
                    assert {str(match["t1"]), str(match["t2"])} <= known

            for player in data["players"]:
                assert isinstance(player.get("performance"), list)
                assert all(1 <= int(e["day"]) <= 34 for e in player["performance"])


def test_projection_exports_match_current_manifest() -> None:
    manifest = _json(DATA / "seasons.json")
    current = next(s["title"] for s in manifest["seasons"]
                   if s["key"] == manifest["current"])
    for name in ("player_projections.json", "player_projections_avg.json",
                 "player_projections_avg_2.json"):
        doc = _json(DATA / name)
        assert doc["season"] == current
        assert doc["players"]
    assert _json(DATA / "player_projections_avg.json")["liga"] == "1"
    assert _json(DATA / "player_projections_avg_2.json")["liga"] == "2"


def test_ratings_export_has_current_model_contract() -> None:
    ratings = _json(DATA / "ratings.json")
    assert ratings["model"] == "ridge-v2"
    assert set(ratings["leagues"]) == {"1", "2"}


def test_atomic_json_preserves_previous_file_on_serialization_error(tmp_path: Path) -> None:
    target = tmp_path / "state.json"
    target.write_text('{"old": true}', encoding="utf-8")
    with pytest.raises(TypeError):
        atomic_write_json(target, {"bad": object()})
    assert target.read_text(encoding="utf-8") == '{"old": true}'
    assert not list(tmp_path.glob("*.tmp"))


def test_fetcher_atomic_text_preserves_previous_file_on_error(tmp_path: Path) -> None:
    target = tmp_path / "history.json"
    target.write_text("old", encoding="utf-8")
    with pytest.raises(RuntimeError):
        with fetch.atomic_text_file(str(target)) as stream:
            stream.write("partial")
            raise RuntimeError("simulierter Abbruch")
    assert target.read_text(encoding="utf-8") == "old"
    assert not list(tmp_path.glob("*.tmp"))


def test_fetcher_writes_data_and_plan_as_one_version(tmp_path: Path,
                                                    monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(fetch, "DATA_DIR", str(tmp_path))
    comp = {"id": "1", "league": "Bundesliga"}
    season = {"suffix": "", "title": "2026/2027"}
    fetch.write_data(comp, season, {"1": "AAA"}, [], [])
    data = _json(tmp_path / "data_1.json")
    plan = _json(tmp_path / "matchdays_bl1.json")
    assert data["generated_at"] == plan["generated_at"]


class _A11yParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.controls: list[dict[str, str | None]] = []
        self.labels: set[str] = set()
        self.external_scripts: list[dict[str, str | None]] = []

    def handle_starttag(self, tag: str, attrs) -> None:
        a = dict(attrs)
        if tag in {"input", "select", "textarea"} and a.get("type") != "hidden":
            self.controls.append(a)
        if tag == "label" and a.get("for"):
            self.labels.add(a["for"])
        if tag == "script" and (a.get("src") or "").startswith("https://"):
            self.external_scripts.append(a)


@pytest.mark.parametrize("path", HTML_FILES, ids=lambda p: p.name)
def test_static_form_controls_have_accessible_names(path: Path) -> None:
    parser = _A11yParser()
    parser.feed(path.read_text(encoding="utf-8"))
    for control in parser.controls:
        cid = control.get("id")
        assert control.get("aria-label") or (cid and cid in parser.labels), (
            f"{path.name}: Steuerelement {cid or control} hat keinen Namen")


@pytest.mark.parametrize("path", HTML_FILES, ids=lambda p: p.name)
def test_external_scripts_are_versioned_and_have_sri(path: Path) -> None:
    parser = _A11yParser()
    parser.feed(path.read_text(encoding="utf-8"))
    for script in parser.external_scripts:
        assert re.search(r"\d+\.\d+", script["src"] or ""), script["src"]
        assert (script.get("integrity") or "").startswith("sha384-")
        assert script.get("crossorigin") == "anonymous"


def test_relative_document_links_stay_inside_repository() -> None:
    docs = [ROOT / "README.md", ROOT / "AGENTS.md", ROOT / "NOTICE.md"]
    pattern = re.compile(r"\[[^]]*\]\(([^)]+)\)")
    for doc in docs:
        for raw in pattern.findall(doc.read_text(encoding="utf-8")):
            target = raw.split("#", 1)[0]
            if not target or "://" in target or target.startswith("mailto:"):
                continue
            resolved = (doc.parent / target).resolve()
            resolved.relative_to(ROOT)
            assert resolved.exists(), f"{doc.name}: {target} fehlt"
