"""
fawp_index.scan_history — Persistent scan history store.

Records snapshots of WatchlistResult over time so users can see:
  - When an asset first entered a FAWP regime
  - How long it stayed there
  - Whether the score is strengthening or fading
  - The last N scan states per asset

Storage: ~/.fawp/history/ (or $FAWP_HISTORY env var)
  One JSON file per scan: {timestamp}_{n_assets}.json

Usage::

    from fawp_index.scan_history import ScanHistory
    from fawp_index.watchlist import scan_watchlist

    history = ScanHistory()
    result  = scan_watchlist(["SPY", "QQQ", "GLD"], period="2y")

    # Save snapshot after every scan
    history.save(result)

    # Load timeline for one asset
    timeline = history.asset_timeline("SPY", "1d")
    print(timeline)        # DataFrame of date, score, gap, regime_active

    # Recent snapshots
    recent = history.recent(n=5)

    # When did SPY first enter FAWP?
    onset = history.first_onset("SPY", "1d")
    print(onset)           # "2026-03-02"

Ralph Clayton (2026) · doi:10.5281/zenodo.18673949
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Union

import pandas as pd

from fawp_index import __version__ as _VERSION


# ── Default store location ────────────────────────────────────────────────────

def _default_history_dir() -> Path:
    env = os.environ.get("FAWP_HISTORY")
    if env:
        return Path(env)
    return Path.home() / ".fawp" / "history"


# ── ScanSnapshot ──────────────────────────────────────────────────────────────

class ScanHistory:
    """
    Persistent scan history — stores WatchlistResult snapshots over time.

    Parameters
    ----------
    history_dir : str or Path, optional
        Directory to store scan snapshots. Defaults to ~/.fawp/history/
        or $FAWP_HISTORY.
    max_snapshots : int
        Maximum number of snapshots to keep. Oldest are pruned. Default 500.

    Examples
    --------
    ::

        history = ScanHistory()

        # Save after each scan
        result = scan_watchlist(["SPY", "QQQ"], period="2y")
        history.save(result)

        # Asset timeline
        tl = history.asset_timeline("SPY", "1d")
        print(tl.tail(10))

        # Summary
        print(history.summary())
    """

    def __init__(
        self,
        history_dir:   Optional[Union[str, Path]] = None,
        max_snapshots: int = 500,
    ):
        self._dir = Path(history_dir) if history_dir else _default_history_dir()
        self._dir.mkdir(parents=True, exist_ok=True)
        self.max_snapshots = max_snapshots

    # ── Save ──────────────────────────────────────────────────────────────────

    def save(self, watchlist_result, label: str = "") -> Path:
        """
        Save a WatchlistResult snapshot to disk.

        Parameters
        ----------
        watchlist_result : WatchlistResult
        label : str, optional
            Short tag to identify this snapshot (e.g. "morning_scan").

        Returns
        -------
        Path to the saved file.
        """
        now = datetime.now()
        ts  = now.strftime("%Y%m%d_%H%M%S")
        n   = watchlist_result.n_assets
        fname = f"{ts}_{n}assets.json"
        if label:
            fname = f"{ts}_{label}_{n}assets.json"

        snapshot = {
            "scanned_at":         now.isoformat(),
            "label":              label,
            "fawp_index_version": _VERSION,
            "n_assets":           n,
            "n_flagged":          watchlist_result.n_flagged,
            "assets": [
                {
                    "ticker":         a.ticker,
                    "timeframe":      a.timeframe,
                    "latest_score":   round(float(a.latest_score),   6),
                    "peak_gap_bits":  round(float(a.peak_gap_bits),  6),
                    "regime_active":  bool(a.regime_active),
                    "days_in_regime": int(a.days_in_regime),
                    "signal_age_days":int(a.signal_age_days),
                    "odw_start":      a.peak_odw_start,
                    "odw_end":        a.peak_odw_end,
                    "error":          a.error,
                }
                for a in watchlist_result.assets
            ],
        }

        p = self._dir / fname
        p.write_text(json.dumps(snapshot, indent=2))
        # Persist to Supabase for cross-session history
        try:
            import os
            _url   = os.environ.get("FAWP_SUPABASE_URL",   "")
            _token = os.environ.get("FAWP_SUPABASE_TOKEN", "")
            if _url and _token:
                from supabase import create_client as _sbc
                _db = _sbc(_url, _token)
                for _a in snapshot.get("assets", []):
                    _db.table("fawp_scan_history").upsert({
                        "ticker":        _a.get("ticker",""),
                        "timeframe":     _a.get("timeframe","1d"),
                        "scanned_at":    snapshot.get("scanned_at",""),
                        "regime_active": bool(_a.get("regime_active",False)),
                        "latest_score":  float(_a.get("latest_score",0) or 0),
                        "peak_gap_bits": float(_a.get("peak_gap_bits",0) or 0),
                    }).execute()
        except Exception:
            pass  # Never crash scanner due to persistence failure
        self._prune()
        return p

    # ── Load ──────────────────────────────────────────────────────────────────

    def _list_files(self) -> List[Path]:
        """Return snapshot files sorted oldest → newest."""
        return sorted(self._dir.glob("*.json"))

    def _load_file(self, p: Path) -> dict:
        try:
            return json.loads(p.read_text())
        except Exception:
            return {}

    def _prune(self):
        """Remove oldest snapshots if over the limit."""
        files = self._list_files()
        if len(files) > self.max_snapshots:
            for old in files[:len(files) - self.max_snapshots]:
                try:
                    old.unlink()
                except Exception:
                    pass

    # ── Query ─────────────────────────────────────────────────────────────────

    def recent(self, n: int = 10) -> List[dict]:
        """
        Return the n most recent snapshot summaries (newest first).

        Returns
        -------
        list of dict with keys: scanned_at, n_assets, n_flagged, label
        """
        files = self._list_files()[-n:][::-1]
        out = []
        for f in files:
            snap = self._load_file(f)
            if snap:
                out.append({
                    "scanned_at": snap.get("scanned_at", ""),
                    "n_assets":   snap.get("n_assets",   0),
                    "n_flagged":  snap.get("n_flagged",  0),
                    "label":      snap.get("label",      ""),
                    "file":       f.name,
                })
        return out

    def asset_timeline(
        self,
        ticker:    str,
        timeframe: str = "1d",
        last_n:    int = 0,
    ) -> pd.DataFrame:
        """
        Return a DataFrame of all historical scores for one asset.

        Columns: scanned_at, latest_score, peak_gap_bits, regime_active,
                 days_in_regime, signal_age_days, odw_start, odw_end

        Parameters
        ----------
        ticker : str
        timeframe : str
        last_n : int
            If > 0, return only the last N snapshots.

        Returns
        -------
        pd.DataFrame, sorted by scanned_at ascending.
        """
        rows = []
        for f in self._list_files():
            snap = self._load_file(f)
            if not snap:
                continue
            for a in snap.get("assets", []):
                if a.get("ticker") == ticker and a.get("timeframe") == timeframe:
                    rows.append({
                        "scanned_at":     snap.get("scanned_at", ""),
                        "latest_score":   a.get("latest_score",   0.0),
                        "peak_gap_bits":  a.get("peak_gap_bits",  0.0),
                        "regime_active":  a.get("regime_active",  False),
                        "days_in_regime": a.get("days_in_regime", 0),
                        "signal_age_days":a.get("signal_age_days",0),
                        "odw_start":      a.get("odw_start"),
                        "odw_end":        a.get("odw_end"),
                    })

        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame(rows)
        df["scanned_at"] = pd.to_datetime(df["scanned_at"])
        df = df.sort_values("scanned_at").reset_index(drop=True)

        if last_n > 0:
            df = df.tail(last_n).reset_index(drop=True)
        return df

    def first_onset(self, ticker: str, timeframe: str = "1d") -> Optional[str]:
        """
        Return the first date when ticker entered a FAWP regime, or None.
        """
        tl = self.asset_timeline(ticker, timeframe)
        if tl.empty:
            return None
        active = tl[tl["regime_active"]]
        if active.empty:
            return None
        return str(active.iloc[0]["scanned_at"].date())

    def last_seen_active(self, ticker: str, timeframe: str = "1d") -> Optional[str]:
        """
        Return the most recent date when ticker was in a FAWP regime, or None.
        """
        tl = self.asset_timeline(ticker, timeframe)
        if tl.empty:
            return None
        active = tl[tl["regime_active"]]
        if active.empty:
            return None
        return str(active.iloc[-1]["scanned_at"].date())

    def all_assets(self) -> List[dict]:
        """
        Return list of all (ticker, timeframe) pairs seen in history
        with their latest score and regime state.
        """
        latest: Dict[str, dict] = {}
        for f in self._list_files():
            snap = self._load_file(f)
            for a in snap.get("assets", []):
                key = f"{a.get('ticker')}|{a.get('timeframe')}"
                latest[key] = {
                    "ticker":        a.get("ticker"),
                    "timeframe":     a.get("timeframe"),
                    "latest_score":  a.get("latest_score",  0.0),
                    "regime_active": a.get("regime_active", False),
                    "last_seen":     snap.get("scanned_at", ""),
                }
        return sorted(latest.values(), key=lambda x: x["latest_score"], reverse=True)

    def n_snapshots(self) -> int:
        return len(self._list_files())

    def clear(self):
        """Delete all snapshots. Irreversible."""
        for f in self._list_files():
            try:
                f.unlink()
            except Exception:
                pass

    # ── Summary ───────────────────────────────────────────────────────────────

    def summary(self) -> str:
        files  = self._list_files()
        n      = len(files)
        assets = self.all_assets()
        active = [a for a in assets if a["regime_active"]]
        lines  = [
            f"ScanHistory — {self._dir}",
            f"  Snapshots : {n} (max {self.max_snapshots})",
            f"  Assets    : {len(assets)} unique (ticker, timeframe) pairs",
            f"  Active    : {len(active)} currently in FAWP",
        ]
        if files:
            first = self._load_file(files[0]).get("scanned_at", "?")
            last  = self._load_file(files[-1]).get("scanned_at", "?")
            lines.append(f"  Range     : {first[:10]} → {last[:10]}")
        return "\n".join(lines)
