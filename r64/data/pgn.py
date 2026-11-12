"""Fast, streaming PGN reading.

`python-chess`'s full PGN parser is thorough but slow for millions of games. Lichess and
chess.com exports are regular, so we split games and tokens ourselves and only use
python-chess to apply the moves (which validates them).
"""

from __future__ import annotations

import io
import re
import urllib.request
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from pathlib import Path

import chess
import numpy as np
import zstandard

from r64.data.encoding import encode_position, move_to_index

USER_AGENT = "r64-chess-project (personal, non-commercial)"

_HEADER = re.compile(r'^\[(\w+)\s+"(.*)"\]\s*$')
_TOKEN = re.compile(r"\{([^}]*)\}|([KQRBNOa-h][^\s{}()]*)")
_CLOCK = re.compile(r"\[%clk\s+(\d+):(\d+):(\d+(?:\.\d+)?)\]")


@dataclass
class RawGame:
    headers: dict[str, str]
    movetext: str


# ---------- reading ----------

def open_pgn(source: str | Path) -> io.TextIOBase:
    """Open a local .pgn / .pgn.zst file or an http(s) URL as a text stream."""
    source = str(source)
    if source.startswith(("http://", "https://")):
        req = urllib.request.Request(source, headers={"User-Agent": USER_AGENT})
        raw = urllib.request.urlopen(req)  # noqa: S310 - URL comes from the CLI user
    else:
        raw = open(source, "rb")  # noqa: SIM115
    if source.endswith(".zst"):
        raw = zstandard.ZstdDecompressor().stream_reader(raw, read_across_frames=True)
    return io.TextIOWrapper(raw, encoding="utf-8", errors="replace")


def read_games(lines: Iterable[str]) -> Iterator[RawGame]:
    """Split a PGN text stream into games (headers + raw movetext)."""
    headers: dict[str, str] = {}
    moves: list[str] = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        if line.startswith("["):
            if moves:  # a new game starts
                yield RawGame(headers, " ".join(moves))
                headers, moves = {}, []
            m = _HEADER.match(line)
            if m:
                headers[m.group(1)] = m.group(2)
        else:
            moves.append(line)
    if headers or moves:
        yield RawGame(headers, " ".join(moves))


# ---------- filtering ----------

@dataclass
class GameFilter:
    """Which games to keep. Checked on headers only, before the (slow) move parsing."""

    speeds: tuple[str, ...] = ("blitz", "rapid", "classical")
    rated_only: bool = True
    min_elo: int = 0
    max_elo: int = 4000
    terminations: tuple[str, ...] = ("Normal", "Time forfeit")

    def accepts(self, h: dict[str, str]) -> bool:
        event = h.get("Event", "").lower()
        if self.rated_only and not event.startswith("rated"):
            return False
        if self.speeds and not any(s in event for s in self.speeds):
            return False
        if self.terminations and h.get("Termination") not in self.terminations:
            return False
        if parse_time_control(h.get("TimeControl", "-")) is None:
            return False
        for key in ("WhiteElo", "BlackElo"):
            elo = h.get(key, "")
            if not elo.isdigit() or not (self.min_elo <= int(elo) <= self.max_elo):
                return False
        return h.get("Result") in ("1-0", "0-1", "1/2-1/2")


def parse_time_control(tc: str) -> tuple[int, int] | None:
    """'300+2' -> (300, 2). None for correspondence / unknown."""
    base, _, inc = tc.partition("+")
    if not base.isdigit():
        return None
    return int(base), int(inc) if inc.isdigit() else 0


# ---------- encoding ----------

FIELDS = {
    "pieces": (np.uint64, (12,)),
    "castling": (np.uint8, ()),
    "ep": (np.int8, ()),
    "halfmove": (np.uint8, ()),
    "ply": (np.int16, ()),
    "move": (np.int16, ()),
    "elo": (np.int16, ()),       # rating of the player to move (-1 unknown)
    "opp_elo": (np.int16, ()),
    "clock": (np.int16, ()),     # seconds left on the mover's clock after the move (-1 unknown)
    "base_time": (np.int16, ()),
    "increment": (np.int16, ()),
    "result": (np.int8, ()),     # final result for the player to move: 1 win, 0 draw, -1 loss
}


@dataclass
class EncodedGame:
    columns: dict[str, list] = field(default_factory=lambda: {k: [] for k in FIELDS})

    def __len__(self) -> int:
        return len(self.columns["move"])


def encode_game(game: RawGame, player: set[str] | None = None) -> EncodedGame | None:
    """Encode every position + the move played. With `player`, keep only that player's moves.

    Returns None if the movetext is broken (illegal / unparseable move).
    """
    h = game.headers
    tc = parse_time_control(h.get("TimeControl", "-")) or (-1, -1)
    white_elo = int(h["WhiteElo"]) if h.get("WhiteElo", "").isdigit() else -1
    black_elo = int(h["BlackElo"]) if h.get("BlackElo", "").isdigit() else -1
    white_score = {"1-0": 1, "0-1": -1, "1/2-1/2": 0}.get(h.get("Result", ""), 0)
    keep_white = keep_black = True
    if player is not None:
        keep_white = h.get("White", "").lower() in player
        keep_black = h.get("Black", "").lower() in player
        if not (keep_white or keep_black):
            return None

    out = EncodedGame()
    cols = out.columns
    board = chess.Board(h["FEN"]) if h.get("FEN") else chess.Board()  # drills start mid-game
    pending = None  # index into cols of the last move, waiting for its clock comment
    for comment, san in _TOKEN.findall(game.movetext):
        if comment:
            m = _CLOCK.search(comment)
            if m and pending is not None:
                h_, m_, s_ = m.groups()
                cols["clock"][pending] = min(int(h_) * 3600 + int(m_) * 60 + int(float(s_)), 32767)
            continue
        try:
            move = board.parse_san(san.rstrip("!?"))
        except ValueError:
            return None
        white = board.turn == chess.WHITE
        pending = None
        if (keep_white and white) or (keep_black and not white):
            pieces, castling, ep = encode_position(board)
            cols["pieces"].append(pieces)
            cols["castling"].append(castling)
            cols["ep"].append(ep)
            cols["halfmove"].append(min(board.halfmove_clock, 255))
            cols["ply"].append(board.ply())
            cols["move"].append(move_to_index(move, board.turn))
            cols["elo"].append(white_elo if white else black_elo)
            cols["opp_elo"].append(black_elo if white else white_elo)
            cols["clock"].append(-1)
            cols["base_time"].append(min(tc[0], 32767))
            cols["increment"].append(min(tc[1], 32767))
            cols["result"].append(white_score if white else -white_score)
            pending = len(cols["move"]) - 1
        board.push(move)
    return out
