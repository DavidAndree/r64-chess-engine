"""SQLite store for Stockfish analysis of my games (data/analysis.sqlite).

Raw engine facts only: the evaluation and best move of every position. Everything derived from
them (mistakes, position tags, the weakness map) is recomputed from these rows, so the tagging
can change without re-running Stockfish.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

from r64 import config

ANALYSIS_DB = config.DATA_DIR / "analysis.sqlite"

SCHEMA = """
CREATE TABLE IF NOT EXISTS games (
    id          INTEGER PRIMARY KEY,
    key         TEXT UNIQUE NOT NULL,   -- game URL, or source + number for r64 games
    source      TEXT NOT NULL,          -- chesscom, lichess, r64
    played_at   TEXT,                   -- 'YYYY.MM.DD HH:MM:SS' (UTC when known)
    my_color    TEXT NOT NULL,          -- white / black
    my_elo      INTEGER,
    opp_elo     INTEGER,
    result      TEXT,                   -- win / draw / loss, from my side
    base_time   INTEGER,                -- seconds, -1 unknown
    increment   INTEGER,
    eco         TEXT,
    plies       INTEGER NOT NULL,
    nodes       INTEGER NOT NULL,       -- Stockfish nodes per position
    analyzed_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS positions (
    game_id INTEGER NOT NULL REFERENCES games(id),
    ply     INTEGER NOT NULL,           -- 0 = start position
    fen     TEXT NOT NULL,
    move    TEXT,                       -- move played from here (UCI); NULL after the last move
    best    TEXT,                       -- Stockfish's best move here
    cp      INTEGER NOT NULL,           -- eval from White's view; mate in n = ±(100000 - n)
    clock   REAL,                       -- mover's clock after `move` (seconds), if recorded
    PRIMARY KEY (game_id, ply)
);
"""


def connect(path: Path | str | None = None) -> sqlite3.Connection:
    path = Path(path or ANALYSIS_DB)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.executescript(SCHEMA)
    return conn


def analyzed_keys(conn: sqlite3.Connection) -> set[str]:
    return {r[0] for r in conn.execute("SELECT key FROM games")}


def save_game(conn: sqlite3.Connection, game: dict, positions: list[dict]) -> None:
    cols = list(game)
    cur = conn.execute(
        f"INSERT INTO games ({', '.join(cols)}) VALUES ({', '.join('?' * len(cols))})",
        [game[c] for c in cols],
    )
    conn.executemany(
        "INSERT INTO positions (game_id, ply, fen, move, best, cp, clock) VALUES (?, ?, ?, ?, ?, ?, ?)",
        [(cur.lastrowid, p["ply"], p["fen"], p["move"], p["best"], p["cp"], p["clock"]) for p in positions],
    )
    conn.commit()
