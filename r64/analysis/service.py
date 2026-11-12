"""The weakness map for the web server: the cached report plus a background "analyze new games" job."""

from __future__ import annotations

import threading
import traceback
from pathlib import Path

from r64 import config
from r64.analysis import store


class WeaknessService:
    def __init__(self, data_dir: Path = config.DATA_DIR):
        self.db = data_dir / "analysis.sqlite"
        self.cache = data_dir / "weakness.json"
        self.running = False
        self.error: str | None = None
        self.last_added: int | None = None
        self._report_lock = threading.Lock()   # one report computation at a time
        self._job_lock = threading.Lock()

    def report(self, recent: int) -> dict:
        from r64.analysis.weakness import cached_report

        with self._report_lock:
            return cached_report(recent, db=self.db, cache=self.cache)

    def status(self) -> dict:
        conn = store.connect(self.db)
        analyzed = conn.execute("SELECT COUNT(*) FROM games").fetchone()[0]
        conn.close()
        return {"running": self.running, "error": self.error, "analyzed": analyzed,
                "last_added": self.last_added}

    def update(self) -> bool:
        """Export my r64 games and run Stockfish over every game not analyzed yet (background)."""
        with self._job_lock:
            if self.running:
                return False
            self.running, self.error = True, None
        threading.Thread(target=self._run, daemon=True).start()
        return True

    def _run(self) -> None:
        from r64.analysis.analyze import analyze
        from r64.data.my_games import export_r64_games

        try:
            export_r64_games()
            self.last_added = analyze(db=self.db, progress=False)
        except Exception as e:  # noqa: BLE001 - shown in the UI
            traceback.print_exc()
            self.error = f"{type(e).__name__}: {e}"
        finally:
            self.running = False
