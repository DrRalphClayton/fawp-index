"""
fawp_index.watchlist_store — Saved named watchlists.

Create and manage named watchlists that persist between sessions,
then re-scan them with a single command.

Storage: ``~/.fawp/watchlists.json`` (configurable via ``FAWP_STORE`` env var
or the ``store_path`` argument).

Usage::

    from fawp_index.watchlist_store import WatchlistStore

    store = WatchlistStore()
    store.create("tech", ["AAPL", "MSFT", "NVDA", "AMD"])
    store.create("crypto", ["BTC-USD", "ETH-USD", "SOL-USD"])

    print(store.list())          # ["crypto", "tech"]
    print(store.show("tech"))    # {"tickers": [...], "created": "...", ...}

    result = store.scan("tech")  # returns WatchlistResult
    result.to_html("tech.html")

    store.delete("tech")

Or via the CLI::

    fawp-watchlist create tech AAPL MSFT NVDA AMD
    fawp-watchlist scan tech
    fawp-watchlist scan tech --rank-by gap --out tech.html
    fawp-watchlist list
    fawp-watchlist show tech
    fawp-watchlist delete tech

Ralph Clayton (2026) · doi:10.5281/zenodo.18673949
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Union


# ── Default store location ────────────────────────────────────────────────────

def _default_store_path() -> Path:
    env = os.environ.get("FAWP_STORE")
    if env:
        return Path(env)
    return Path.home() / ".fawp" / "watchlists.json"


# ── WatchlistStore ────────────────────────────────────────────────────────────

class WatchlistStore:
    """
    Persistent named watchlist manager.

    Parameters
    ----------
    store_path : str or Path, optional
        Path to the JSON file that stores watchlists.
        Defaults to ``~/.fawp/watchlists.json`` or ``$FAWP_STORE``.

    Examples
    --------
    ::

        store = WatchlistStore()
        store.create("mylist", ["SPY", "QQQ", "GLD"])
        result = store.scan("mylist")
        print(result.summary())
    """

    def __init__(self, store_path: Optional[Union[str, Path]] = None):
        self._path = Path(store_path) if store_path else _default_store_path()
        self._data: Dict[str, dict] = self._load()

    # ── CRUD ─────────────────────────────────────────────────────────────────

    def create(
        self,
        name: str,
        tickers: List[str],
        period:     str = "2y",
        timeframes: Optional[List[str]] = None,
        window:     Optional[int] = None,
        tau_max:    Optional[int] = None,
        overwrite:  bool = False,
    ) -> None:
        """
        Create a named watchlist.

        Parameters
        ----------
        name : str
            Name for the watchlist (e.g. "tech", "crypto-majors").
        tickers : list of str
            Ticker symbols.
        period : str
            yfinance period string, default "2y".
        timeframes : list of str, optional
            e.g. ["1d", "1wk"]. Defaults to ["1d"].
        window : int, optional
            Rolling window bars. Default: auto (252 for daily).
        tau_max : int, optional
            Max tau. Default: auto (40).
        overwrite : bool
            If False and name already exists, raises ValueError.
        """
        name = name.strip().lower()
        if name in self._data and not overwrite:
            raise ValueError(
                f"Watchlist '{name}' already exists. "
                "Use overwrite=True or delete it first."
            )
        tickers = [t.strip().upper() for t in tickers if t.strip()]
        if not tickers:
            raise ValueError("tickers must not be empty.")

        self._data[name] = {
            "tickers":    tickers,
            "period":     period,
            "timeframes": timeframes or ["1d"],
            "window":     window,
            "tau_max":    tau_max,
            "created":    datetime.now().isoformat(),
            "last_scanned": None,
        }
        self._save()

    def delete(self, name: str) -> None:
        """Delete a named watchlist."""
        name = name.strip().lower()
        if name not in self._data:
            raise KeyError(f"Watchlist '{name}' not found.")
        del self._data[name]
        self._save()

    def show(self, name: str) -> dict:
        """Return the metadata dict for a named watchlist."""
        name = name.strip().lower()
        if name not in self._data:
            raise KeyError(f"Watchlist '{name}' not found.")
        return dict(self._data[name])

    def list(self) -> List[str]:
        """Return sorted list of all saved watchlist names."""
        return sorted(self._data.keys())

    def exists(self, name: str) -> bool:
        return name.strip().lower() in self._data

    # ── Scan ─────────────────────────────────────────────────────────────────

    def scan(
        self,
        name: str,
        period:     Optional[str]       = None,
        timeframes: Optional[List[str]] = None,
        window:     Optional[int]       = None,
        tau_max:    Optional[int]       = None,
        n_null:     int                 = 0,
        max_workers: int                = 4,
        verbose:    bool                = True,
    ):
        """
        Scan a named watchlist and return a WatchlistResult.

        Parameters override saved defaults.

        Parameters
        ----------
        name : str
            Name of the saved watchlist.
        period, timeframes, window, tau_max : optional overrides.
        n_null : int
            Null permutations (0 = fast mode).
        max_workers : int
            Parallel workers for batch fetch.
        verbose : bool
            Print progress.

        Returns
        -------
        WatchlistResult
        """
        from fawp_index.watchlist import scan_watchlist

        name = name.strip().lower()
        cfg  = self.show(name)

        _period     = period     or cfg["period"]
        _timeframes = timeframes or cfg["timeframes"]
        _window     = window     or cfg.get("window")
        _tau_max    = tau_max    or cfg.get("tau_max")

        kwargs: dict = dict(
            period      = _period,
            timeframes  = _timeframes,
            n_null      = n_null,
            max_workers = max_workers,
            verbose     = verbose,
        )
        if _window:
            kwargs["window"]  = _window
        if _tau_max:
            kwargs["tau_max"] = _tau_max

        if verbose:
            print(f"fawp-watchlist: scanning '{name}' — {len(cfg['tickers'])} tickers")

        result = scan_watchlist(cfg["tickers"], **kwargs)

        # Update last_scanned timestamp
        self._data[name]["last_scanned"] = datetime.now().isoformat()
        self._save()

        return result

    # ── Persistence ──────────────────────────────────────────────────────────

    def _load(self) -> Dict[str, dict]:
        if self._path.exists():
            try:
                return json.loads(self._path.read_text())
            except Exception:
                pass
        return {}

    def _save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(json.dumps(self._data, indent=2))

    # ── Summary ──────────────────────────────────────────────────────────────

    def summary(self) -> str:
        names = self.list()
        if not names:
            return "No saved watchlists. Use WatchlistStore.create() or fawp-watchlist create."
        lines = [
            f"{'Name':<20} {'Tickers':>7} {'Period':<6} {'TF':<12} {'Last scanned'}",
            "-" * 65,
        ]
        for n in names:
            d = self._data[n]
            tfs = ",".join(d.get("timeframes", ["1d"]))
            last = d.get("last_scanned") or "never"
            if "T" in last:
                last = last.split("T")[0]
            lines.append(
                f"{n:<20} {len(d['tickers']):>7} {d['period']:<6} {tfs:<12} {last}"
            )
        return "\n".join(lines)
