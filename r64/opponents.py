"""Opponents the player can face, all behind one interface.

    stockfish   Stockfish at limited strength (1320-3190)
    bot         the human-like network (r64.model.bot, 600-2200)

Phase 5 adds the personal "me" model the same way.
"""

from __future__ import annotations

import shutil
import threading
from typing import Protocol

import chess
import chess.engine

STOCKFISH_MIN_ELO = 1320  # limits of Stockfish's UCI_Elo option
STOCKFISH_MAX_ELO = 3190


class Opponent(Protocol):
    name: str
    label: str      # shown in the UI
    min_elo: int
    max_elo: int

    def choose_move(self, board: chess.Board, elo: int | None) -> chess.Move: ...

    def close(self) -> None: ...


class StockfishNotFound(RuntimeError):
    pass


class StockfishOpponent:
    """Stockfish with UCI_LimitStrength. One process, shared; calls are serialized."""

    name = "stockfish"
    label = "Stockfish"
    min_elo = STOCKFISH_MIN_ELO
    max_elo = STOCKFISH_MAX_ELO

    def __init__(self, path: str | None = None, think_time: float = 0.3):
        path = path or shutil.which("stockfish")
        if not path:
            raise StockfishNotFound("Stockfish not found. Install it with: brew install stockfish")
        self._engine = chess.engine.SimpleEngine.popen_uci(path)
        self._lock = threading.Lock()
        self._think_time = think_time

    def choose_move(self, board: chess.Board, elo: int | None) -> chess.Move:
        with self._lock:
            if elo is None:
                self._engine.configure({"UCI_LimitStrength": False})
            else:
                elo = max(STOCKFISH_MIN_ELO, min(STOCKFISH_MAX_ELO, elo))
                self._engine.configure({"UCI_LimitStrength": True, "UCI_Elo": elo})
            result = self._engine.play(board, chess.engine.Limit(time=self._think_time))
        if result.move is None:
            raise RuntimeError("Stockfish returned no move")
        return result.move

    def close(self) -> None:
        with self._lock:
            self._engine.quit()
