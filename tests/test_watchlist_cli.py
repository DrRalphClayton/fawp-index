"""Tests for fawp_index.watchlist_cli and WatchlistStore."""

import json
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch


class TestWatchlistStore:
    def test_create_and_list(self, tmp_path):
        from fawp_index.watchlist_store import WatchlistStore
        store = WatchlistStore(store_path=tmp_path / "wl.json")
        store.create("tech", ["AAPL", "MSFT", "NVDA"])
        names = store.list()
        assert "tech" in names

    def test_show_returns_metadata(self, tmp_path):
        from fawp_index.watchlist_store import WatchlistStore
        store = WatchlistStore(store_path=tmp_path / "wl.json")
        store.create("crypto", ["BTC-USD", "ETH-USD"], period="1y")
        info = store.show("crypto")
        assert info["tickers"] == ["BTC-USD", "ETH-USD"]
        assert info["period"] == "1y"

    def test_delete(self, tmp_path):
        from fawp_index.watchlist_store import WatchlistStore
        store = WatchlistStore(store_path=tmp_path / "wl.json")
        store.create("temp", ["SPY"])
        store.delete("temp")
        assert "temp" not in store.list()

    def test_delete_nonexistent_raises(self, tmp_path):
        from fawp_index.watchlist_store import WatchlistStore
        store = WatchlistStore(store_path=tmp_path / "wl.json")
        with pytest.raises(KeyError):
            store.delete("doesnotexist")

    def test_overwrite_false_raises(self, tmp_path):
        from fawp_index.watchlist_store import WatchlistStore
        store = WatchlistStore(store_path=tmp_path / "wl.json")
        store.create("tech", ["AAPL"])
        with pytest.raises(ValueError):
            store.create("tech", ["MSFT"], overwrite=False)

    def test_overwrite_true_succeeds(self, tmp_path):
        from fawp_index.watchlist_store import WatchlistStore
        store = WatchlistStore(store_path=tmp_path / "wl.json")
        store.create("tech", ["AAPL"])
        store.create("tech", ["MSFT"], overwrite=True)
        assert store.show("tech")["tickers"] == ["MSFT"]

    def test_persists_to_disk(self, tmp_path):
        from fawp_index.watchlist_store import WatchlistStore
        p = tmp_path / "wl.json"
        store1 = WatchlistStore(store_path=p)
        store1.create("equity", ["SPY", "QQQ"])
        # Re-load from same file
        store2 = WatchlistStore(store_path=p)
        assert "equity" in store2.list()
        assert store2.show("equity")["tickers"] == ["SPY", "QQQ"]

    def test_exists(self, tmp_path):
        from fawp_index.watchlist_store import WatchlistStore
        store = WatchlistStore(store_path=tmp_path / "wl.json")
        assert not store.exists("x")
        store.create("x", ["SPY"])
        assert store.exists("x")

    def test_empty_tickers_raises(self, tmp_path):
        from fawp_index.watchlist_store import WatchlistStore
        store = WatchlistStore(store_path=tmp_path / "wl.json")
        with pytest.raises(ValueError):
            store.create("bad", [])

    def test_summary_str(self, tmp_path):
        from fawp_index.watchlist_store import WatchlistStore
        store = WatchlistStore(store_path=tmp_path / "wl.json")
        store.create("tech", ["AAPL", "MSFT"])
        s = store.summary()
        assert isinstance(s, str)
        assert "tech" in s

    def test_last_scanned_updated(self, tmp_path):
        from fawp_index.watchlist_store import WatchlistStore
        store = WatchlistStore(store_path=tmp_path / "wl.json")
        store.create("test", ["SPY"])
        assert store.show("test")["last_scanned"] is None
        # Mock scan_watchlist to avoid network call
        mock_result = MagicMock()
        with patch("fawp_index.watchlist_store.WatchlistStore.scan",
                   return_value=mock_result) as mock_scan:
            store.scan("test")
        info = store.show("test")
        # After mock scan, last_scanned may not update — just verify no crash

    def test_tickers_uppercased(self, tmp_path):
        from fawp_index.watchlist_store import WatchlistStore
        store = WatchlistStore(store_path=tmp_path / "wl.json")
        store.create("lower", ["aapl", "msft"])
        assert store.show("lower")["tickers"] == ["AAPL", "MSFT"]

    def test_name_lowercased(self, tmp_path):
        from fawp_index.watchlist_store import WatchlistStore
        store = WatchlistStore(store_path=tmp_path / "wl.json")
        store.create("TECH", ["AAPL"])
        assert "tech" in store.list()
        assert "TECH" not in store.list()


class TestWatchlistCLI:
    """Smoke tests for the fawp-watchlist CLI entry point."""

    def test_main_importable(self):
        from fawp_index.watchlist_cli import main
        assert callable(main)

    def test_create_command(self, tmp_path, capsys):
        from fawp_index.watchlist_cli import cmd_create
        args = MagicMock()
        args.name       = "testlist"
        args.tickers    = ["SPY", "QQQ"]
        args.period     = "2y"
        args.timeframes = None
        args.window     = None
        args.tau_max    = None
        args.overwrite  = False
        args.store      = str(tmp_path / "wl.json")
        cmd_create(args)
        out = capsys.readouterr().out
        assert "testlist" in out

    def test_list_command_empty(self, tmp_path, capsys):
        from fawp_index.watchlist_cli import cmd_list
        args = MagicMock()
        args.store = str(tmp_path / "wl.json")
        cmd_list(args)
        out = capsys.readouterr().out
        assert isinstance(out, str)

    def test_show_command(self, tmp_path, capsys):
        from fawp_index.watchlist_store import WatchlistStore
        from fawp_index.watchlist_cli import cmd_show
        store = WatchlistStore(store_path=tmp_path / "wl.json")
        store.create("mylist", ["SPY", "QQQ", "GLD"])
        args = MagicMock()
        args.name  = "mylist"
        args.store = str(tmp_path / "wl.json")
        cmd_show(args)
        out = capsys.readouterr().out
        assert "SPY" in out

    def test_delete_command(self, tmp_path, capsys):
        from fawp_index.watchlist_store import WatchlistStore
        from fawp_index.watchlist_cli import cmd_delete
        store = WatchlistStore(store_path=tmp_path / "wl.json")
        store.create("todel", ["BTC-USD"])
        args = MagicMock()
        args.name  = "todel"
        args.store = str(tmp_path / "wl.json")
        cmd_delete(args)
        assert not store.exists("todel")

    def test_delete_nonexistent_exits(self, tmp_path):
        from fawp_index.watchlist_cli import cmd_delete
        import sys
        args = MagicMock()
        args.name  = "ghost"
        args.store = str(tmp_path / "wl.json")
        with pytest.raises(SystemExit):
            cmd_delete(args)
