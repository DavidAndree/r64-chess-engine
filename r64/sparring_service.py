"""Sparring for the web server: adaptive rating, per-game focus, drills, and the focus summary."""

from __future__ import annotations

import json
import random
from pathlib import Path

import chess

from r64.data.my_games import current_rating
from r64.game import GameService
from r64.sparring import (
    adaptive_rating,
    focus_shares,
    my_score,
    next_focus,
    pick_drill,
    targets_from_report,
)

# For the focus summary: the weakness-map tag that measures the same thing in my chess.com games.
BASELINE_TAG = {"mistake type: hung material": "situation: a piece under attack"}


class SparringService:
    def __init__(self, games: GameService, report_path: Path, bot_name: str = "sparring",
                 drill_opponent: str = "bot"):
        self.games = games
        self.report_path = Path(report_path)
        self.bot_name = bot_name
        self.drill_opponent = drill_opponent
        self._report: dict | None = None
        self._mtime: float | None = None
        self._rng = random.Random()

    # The weakness report, as last computed (no recomputation here: a bot move must stay fast).
    def report(self) -> dict:
        try:
            mtime = self.report_path.stat().st_mtime
        except FileNotFoundError:
            return {}
        if mtime != self._mtime:
            self._report, self._mtime = json.loads(self.report_path.read_text()), mtime
        return self._report or {}

    def targets(self) -> list[tuple[str, float]]:
        return targets_from_report(self.report())

    def rating(self) -> tuple[int, list[int], int]:
        results = []
        for g in self.games.store.finished(mode="sparring"):
            s = my_score(g.result, g.player_color)
            if s is not None and g.opponent_elo:
                results.append((g.opponent_elo, s))
        rating, history = adaptive_rating(results, start=current_rating())
        return rating, history, len(results)

    def new_sparring_game(self, color: str, focus: str | None = None):
        rating, _, _ = self.rating()
        targets = self.targets()
        if focus is None or focus == "auto":
            past = [g.drill_tag for g in self.games.store.finished(mode="sparring")]
            focus = next_focus(targets, past)
        elif focus not in {t for t, _ in targets}:
            raise ValueError(f"not one of my steerable weak spots: {focus}")
        return self.games.new_game(color, self.bot_name, rating, mode="sparring", drill_tag=focus)

    def new_drill(self, tag: str):
        candidates = (self.report().get("drills") or {}).get(tag) or []
        candidates = [c for c in candidates if not chess.Board(c["fen"]).is_game_over()]
        if not candidates:
            raise ValueError(f"no drill positions for {tag}")
        used = {g.start_fen for g in self.games.store.finished(mode="drill")}
        drill = pick_drill(candidates, used, self._rng)
        rating, _, _ = self.rating()
        return self.games.new_game("white", self.drill_opponent, rating, start_fen=drill["fen"],
                                   mode="drill", drill_tag=tag, start_wp=drill["wp_before"]), drill

    def status(self) -> dict:
        rating, history, n = self.rating()
        report = self.report()
        targets = self.targets()
        past = [g.drill_tag for g in self.games.store.finished(mode="sparring")]
        drills = []
        finished_drills = self.games.store.finished(mode="drill")
        for spot in report.get("weak_spots") or []:
            tag = spot["tag"]
            played = [g for g in finished_drills if g.drill_tag == tag]
            scores = [my_score(g.result, g.player_color) for g in played]
            drills.append({
                "tag": tag,
                "positions": len((report.get("drills") or {}).get(tag) or []),
                "played": len(played),
                "score": sum(s for s in scores if s is not None),
                "expected": sum(g.start_wp or 0 for g in played),
            })
        return {
            "rating": rating, "history": history, "games": n,
            "targets": [{"tag": t, "weight": w} for t, w in targets],
            "next_focus": next_focus(targets, past),
            "drills": drills,
            "report_ready": bool(report),
        }

    def focus(self, game_id: int, value_fn) -> dict:
        """How much of a sparring game was spent in its focus position, vs my chess.com games."""
        g = self.games.get(game_id)
        if not g.drill_tag:
            return {}
        share = focus_shares(g.start_fen, g.moves, g.player_color, [g.drill_tag], value_fn)[g.drill_tag]
        report = self.report()
        base_tag = BASELINE_TAG.get(g.drill_tag, g.drill_tag)
        rows = {r["tag"]: r for r in report.get("tags") or []}
        my_moves = (report.get("me") or {}).get("moves")
        baseline = rows[base_tag]["moves"] / my_moves if base_tag in rows and my_moves else None
        return {"tag": g.drill_tag, "share": share, "baseline": baseline}
