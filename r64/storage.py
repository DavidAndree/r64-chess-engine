"""SQLite storage for played games.

Every game is stored from its first move, so an unfinished game survives a server restart.
Moves are stored in UCI notation; the PGN is generated when the game ends.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS games (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at   TEXT    NOT NULL,
    finished_at  TEXT,
    player_color TEXT    NOT NULL CHECK (player_color IN ('white', 'black')),
    opponent     TEXT    NOT NULL,      -- e.g. 'stockfish'; later: 'r64-base', 'r64-me'
    opponent_elo INTEGER,
    moves        TEXT    NOT NULL DEFAULT '',  -- space-separated UCI moves
    status       TEXT    NOT NULL DEFAULT 'ongoing' CHECK (status IN ('ongoing', 'finished')),
    result       TEXT,                  -- '1-0', '0-1', '1/2-1/2'
    termination  TEXT,                  -- 'checkmate', 'stalemate', 'resignation', ...
    pgn          TEXT
);
"""


@dataclass
class GameRecord:
    id: int
    created_at: str
    finished_at: str | None
    player_color: str
    opponent: str
    opponent_elo: int | None
    moves: list[str]
    status: str
    result: str | None
    termination: str | None
    pgn: str | None


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


class GameStore:
    def __init__(self, path: Path | str):
        if str(path) != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(SCHEMA)
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    def create(self, player_color: str, opponent: str, opponent_elo: int | None) -> int:
        cur = self._conn.execute(
            "INSERT INTO games (created_at, player_color, opponent, opponent_elo) VALUES (?, ?, ?, ?)",
            (_now(), player_color, opponent, opponent_elo),
        )
        self._conn.commit()
        return cur.lastrowid

    def get(self, game_id: int) -> GameRecord | None:
        row = self._conn.execute("SELECT * FROM games WHERE id = ?", (game_id,)).fetchone()
        return _to_record(row) if row else None

    def list(self, limit: int = 50) -> list[GameRecord]:
        rows = self._conn.execute(
            "SELECT * FROM games ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
        return [_to_record(r) for r in rows]

    def count_finished(self) -> int:
        return self._conn.execute("SELECT COUNT(*) FROM games WHERE status = 'finished'").fetchone()[0]

    def set_moves(self, game_id: int, moves: list[str]) -> None:
        self._conn.execute("UPDATE games SET moves = ? WHERE id = ?", (" ".join(moves), game_id))
        self._conn.commit()

    def finish(self, game_id: int, moves: list[str], result: str, termination: str, pgn: str) -> None:
        self._conn.execute(
            """UPDATE games
               SET moves = ?, status = 'finished', result = ?, termination = ?, pgn = ?,
                   finished_at = ?
               WHERE id = ?""",
            (" ".join(moves), result, termination, pgn, _now(), game_id),
        )
        self._conn.commit()


def _to_record(row: sqlite3.Row) -> GameRecord:
    return GameRecord(
        id=row["id"],
        created_at=row["created_at"],
        finished_at=row["finished_at"],
        player_color=row["player_color"],
        opponent=row["opponent"],
        opponent_elo=row["opponent_elo"],
        moves=row["moves"].split() if row["moves"] else [],
        status=row["status"],
        result=row["result"],
        termination=row["termination"],
        pgn=row["pgn"],
    )
