"""Kickbase-v4-Client für die Modell-Pipeline.

Bewusst eigenständig gehalten (gemeinsames Muster mit ``fetch.py`` im
Projektwurzelverzeichnis), damit die bestehende Fetcher-Pipeline von diesem
Projekt nicht berührt wird.

Zwei Unterschiede zum Original, die für den ID-Crawl entscheidend sind:

1. ``get()`` unterscheidet **404 (ID existiert nicht)** von **Fehler (unklar)**.
   Beim Enumerieren des ID-Raums ist genau das die Information, um die es geht.
2. Ein 429 bremst **alle** Threads gemeinsam aus (globales ``threading.Event``),
   statt dass jeder Thread für sich weiterläuft und das Limit erneut trifft.
"""

from __future__ import annotations

import os
import sys
import threading
import time
from dataclasses import dataclass
from typing import Any

import requests

BASE_URL = "https://api.kickbase.com"

HEADERS = {
    "Content-Type": "application/json",
    "Accept": "application/json",
}

# Sentinel: Anfrage lief sauber durch, die Ressource gibt es aber nicht.
# Unterscheidet sich von None (= Fehler/unklar) und von {} (= leere Antwort).
NOT_FOUND = object()

_thread_local = threading.local()


def _session() -> requests.Session:
    if not hasattr(_thread_local, "session"):
        s = requests.Session()
        s.headers.update(HEADERS)
        _thread_local.session = s
    return _thread_local.session


def load_credentials(env_dir: str | None = None) -> tuple[str, str]:
    """KICKBASE_EMAIL / KICKBASE_PASSWORD aus Umgebung oder .env lesen.

    Sucht die .env im Projektwurzelverzeichnis (zwei Ebenen über kbxp/src/ingest),
    damit dieselbe Datei wie für fetch.py genutzt wird.
    """
    email = os.environ.get("KICKBASE_EMAIL")
    password = os.environ.get("KICKBASE_PASSWORD")

    if not (email and password):
        if env_dir is None:
            here = os.path.dirname(os.path.abspath(__file__))
            env_dir = os.path.abspath(os.path.join(here, "..", "..", ".."))
        env_path = os.path.join(env_dir, ".env")
        if os.path.exists(env_path):
            with open(env_path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#") or "=" not in line:
                        continue
                    key, value = line.split("=", 1)
                    key, value = key.strip(), value.strip().strip('"').strip("'")
                    if key == "KICKBASE_EMAIL" and not email:
                        email = value
                    elif key == "KICKBASE_PASSWORD" and not password:
                        password = value

    if not (email and password):
        raise RuntimeError(
            "Zugangsdaten fehlen: KICKBASE_EMAIL und KICKBASE_PASSWORD als "
            "Umgebungsvariablen setzen oder in .env im Projektwurzelverzeichnis ablegen."
        )
    return email, password


def _is_not_found_body(response: requests.Response) -> bool:
    """Erkennt Kickbases als HTTP 500 verkleidetes "NotFound"."""
    try:
        body = response.json()
    except ValueError:
        return False
    if not isinstance(body, dict):
        return False
    return body.get("errMsg") == "NotFound" or body.get("err") == 2


@dataclass
class Stats:
    ok: int = 0
    not_found: int = 0
    errors: int = 0
    rate_limited: int = 0


class KickbaseClient:
    """Gedrosselter Client mit gemeinsamer Rate-Limit-Bremse.

    ``delay`` gilt **pro Thread** vor jeder Anfrage. Bei ``workers`` Threads
    ergibt sich also grob ``workers / delay`` Requests/s. Für den einmaligen
    ID-Crawl bewusst konservativ halten (siehe ToS-Hinweis in Kickbase-API.md).
    """

    def __init__(self, delay: float = 0.3, timeout: int = 15, verbose: bool = True):
        self.delay = delay
        self.timeout = timeout
        self.verbose = verbose
        self.token: str | None = None
        self.stats = Stats()
        # gesetzt = freie Fahrt; gelöscht = alle Threads warten
        self._go = threading.Event()
        self._go.set()
        self._backoff_lock = threading.Lock()

    # -- Auth ---------------------------------------------------------------

    def login(self) -> str:
        email, password = load_credentials()
        r = requests.post(
            f"{BASE_URL}/v4/user/login",
            json={"em": email, "pass": password, "loy": False, "rep": {}},
            headers=HEADERS,
            timeout=self.timeout,
        )
        r.raise_for_status()
        token = r.json().get("tkn")
        if not token:
            raise RuntimeError("Login lieferte kein Token (Feld 'tkn').")
        self.token = token
        if self.verbose:
            print("[client] Login erfolgreich")
        return token

    # -- Requests -----------------------------------------------------------

    def _pause_all(self, seconds: float) -> None:
        """Alle Threads anhalten. Nur der erste Thread setzt die Pause an."""
        if not self._backoff_lock.acquire(blocking=False):
            # Ein anderer Thread pausiert bereits - hier nur mitwarten.
            self._go.wait()
            return
        try:
            self._go.clear()
            self.stats.rate_limited += 1
            if self.verbose:
                print(f"[client] Rate-Limit - alle Threads pausieren {seconds:.0f}s")
            time.sleep(seconds)
        finally:
            self._go.set()
            self._backoff_lock.release()

    def get(self, path: str, retries: int = 3, quiet: bool = True) -> Any:
        """GET auf die v4-API.

        Rückgabe:
            dict/list  - Erfolg (HTTP 200)
            NOT_FOUND  - Ressource existiert nicht (HTTP 404)
            None       - Fehler, Ergebnis unklar (Timeout, 5xx, dauerhaftes 429)
        """
        if self.token is None:
            raise RuntimeError("Kein Token - erst login() aufrufen.")

        attempt = 0
        rate_hits = 0

        while attempt < retries:
            self._go.wait()  # globale Bremse respektieren
            time.sleep(self.delay)
            try:
                r = _session().get(
                    f"{BASE_URL}{path}",
                    headers={"Authorization": f"Bearer {self.token}"},
                    timeout=self.timeout,
                )
            except requests.RequestException as e:
                attempt += 1
                if not quiet:
                    print(f"[client] Netzfehler {path} ({attempt}/{retries}): {e}")
                time.sleep(2 * attempt)
                continue

            if r.status_code == 200:
                self.stats.ok += 1
                try:
                    return r.json()
                except ValueError:
                    self.stats.errors += 1
                    return None

            if r.status_code == 404:
                self.stats.not_found += 1
                return NOT_FOUND

            # Kickbase meldet "gibt es nicht" als HTTP 500 mit
            # {"err":2,"errMsg":"NotFound"} statt als 404. Ohne Sonderbehandlung
            # wuerde jede tote ID im Crawl dreimal mit Backoff wiederholt.
            if r.status_code >= 500 and _is_not_found_body(r):
                self.stats.not_found += 1
                return NOT_FOUND

            if r.status_code == 429:
                rate_hits += 1
                if rate_hits > 5:
                    self.stats.errors += 1
                    if not quiet:
                        print(f"[client] Dauerhaftes Rate-Limit bei {path}")
                    return None
                self._pause_all(10 * rate_hits)
                continue  # zählt nicht als regulärer Versuch

            if r.status_code == 401:
                # Token abgelaufen - einmal erneuern, dann weiter
                if not quiet:
                    print("[client] 401 - Token wird erneuert")
                self.login()
                attempt += 1
                continue

            if r.status_code >= 500:
                attempt += 1
                time.sleep(2 * attempt)
                continue

            # 4xx außer 404/401/429: Ressource für uns nicht abrufbar
            self.stats.errors += 1
            if not quiet:
                print(f"[client] HTTP {r.status_code} bei {path}")
            return None

        self.stats.errors += 1
        return None

    # -- Bequeme Wrapper ----------------------------------------------------

    def player(self, competition_id: str | int, player_id: str | int) -> Any:
        return self.get(f"/v4/competitions/{competition_id}/players/{player_id}")

    def performance(self, competition_id: str | int, player_id: str | int) -> Any:
        return self.get(f"/v4/competitions/{competition_id}/players/{player_id}/performance")


def enable_utf8_stdout() -> None:
    """Windows-Konsole nutzt sonst cp1252 und scheitert an Sonderzeichen."""
    if sys.stdout and hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
