"""Run Stockfish over my games: an evaluation and best move for every position.

    uv run python -m r64.analysis analyze            # all games not analyzed yet, newest first
    uv run python -m r64.analysis analyze --games 300

Several Stockfish processes run in parallel (one thread each). At 10,000 nodes per position
(depth ~13) a game takes about 2 s per process; plenty to catch the mistakes that decide games
at club level. Already-analyzed games are skipped, so it can be stopped and resumed.
"""

from __future__ import annotations

import hashlib
import multiprocessing as mp
import multiprocessing.util
import shutil
import time
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import chess
import chess.engine
import chess.pgn

from r64.analysis import store
from r64.data import my_games
from r64.data.pgn import parse_time_control

DEFAULT_NODES = 10_000
MATE = 100_000


@dataclass
class MyGame:
    key: str
    source: str
    played_at: str
    my_color: str
    my_elo: int | None
    opp_elo: int | None
    result: str
    base_time: int
    increment: int
    eco: str | None
    moves: list[str]
    clocks: list[float | None]


def _source(path: Path) -> str:
    return path.name.split("_")[0].removesuffix(".pgn")  # chesscom_x.pgn → chesscom, r64.pgn → r64


def _elo(h, key) -> int | None:
    v = h.get(key, "")
    return int(v) if v.isdigit() else None


def read_my_games(paths: list[Path] | None = None, names: set[str] | None = None) -> Iterator[MyGame]:
    """My games from data/my_games/*.pgn (chess.com, lichess, r64), standard chess only."""
    paths = paths if paths is not None else sorted(my_games.MY_GAMES_DIR.glob("*.pgn"))
    names = {n.lower() for n in (names or my_games.my_names(my_games.load_profile()))}
    for path in paths:
        source = _source(path)
        with open(path, encoding="utf-8", errors="replace") as f:
            while (game := chess.pgn.read_game(f)) is not None:
                h = game.headers
                if h.get("Variant", "Standard") not in ("Standard", "") or "FEN" in h:
                    continue
                if h.get("White", "").lower() in names:
                    color = "white"
                elif h.get("Black", "").lower() in names:
                    color = "black"
                else:
                    continue
                moves, clocks = [], []
                for node in game.mainline():
                    moves.append(node.move.uci())
                    clocks.append(node.clock())
                if len(moves) < 2:
                    continue
                res = h.get("Result", "*")
                result = ("draw" if res == "1/2-1/2" else
                          "win" if (res == "1-0") == (color == "white") and res != "*" else
                          "loss" if res != "*" else "unknown")
                tc = parse_time_control(h.get("TimeControl", "-"))
                link = h.get("Link") or h.get("Site", "")
                key = link if link.startswith("http") else (
                    f"{source}:" + hashlib.sha1(str(game).encode()).hexdigest()[:16]
                )
                date = h.get("UTCDate", h.get("Date", "????.??.??"))
                hour = h.get("UTCTime", h.get("StartTime", "00:00:00"))
                yield MyGame(
                    key=key, source=source, played_at=f"{date} {hour}", my_color=color,
                    my_elo=_elo(h, "WhiteElo" if color == "white" else "BlackElo"),
                    opp_elo=_elo(h, "BlackElo" if color == "white" else "WhiteElo"),
                    result=result, base_time=tc[0] if tc else -1, increment=tc[1] if tc else 0,
                    eco=h.get("ECO") or None, moves=moves, clocks=clocks,
                )


# ---------- Stockfish workers ----------

_engine: chess.engine.SimpleEngine | None = None


def _init_worker(path: str) -> None:
    global _engine
    _engine = chess.engine.SimpleEngine.popen_uci(path)
    _engine.configure({"Threads": 1, "Hash": 16})
    mp.util.Finalize(None, _engine.quit, exitpriority=10)  # stop Stockfish when the worker exits


def _score(board: chess.Board, info: dict | None) -> int:
    """Eval from White's view in centipawns; mate in n = ±(MATE - n)."""
    if board.is_checkmate():
        return -MATE if board.turn == chess.WHITE else MATE
    if board.is_game_over(claim_draw=False):
        return 0
    return info["score"].white().score(mate_score=MATE)


def analyze_moves(moves: list[str], nodes: int, engine=None) -> list[dict]:
    """Eval + best move for the start position and after every move."""
    engine = engine or _engine
    board = chess.Board()
    out = []
    for ply in range(len(moves) + 1):
        info, best = None, None
        if not board.is_game_over(claim_draw=False):
            info = engine.analyse(board, chess.engine.Limit(nodes=nodes))
            pv = info.get("pv") or []
            best = pv[0].uci() if pv else None
        out.append({"ply": ply, "fen": board.fen(), "best": best, "cp": _score(board, info)})
        if ply < len(moves):
            board.push_uci(moves[ply])
    return out


def _work(args: tuple[MyGame, int]) -> tuple[MyGame, list[dict]]:
    game, nodes = args
    positions = analyze_moves(game.moves, nodes)
    for p in positions:
        i = p["ply"]
        p["move"] = game.moves[i] if i < len(game.moves) else None
        p["clock"] = game.clocks[i] if i < len(game.clocks) else None
    return game, positions


def analyze(
    max_games: int | None = None,
    nodes: int = DEFAULT_NODES,
    workers: int | None = None,
    db: Path | None = None,
    paths: list[Path] | None = None,
    progress: bool = True,
) -> int:
    """Analyze my games that aren't in the database yet, newest first. Returns games analyzed."""
    sf = shutil.which("stockfish")
    if not sf:
        raise RuntimeError("Stockfish not found. Install it with: brew install stockfish")
    conn = store.connect(db)
    done = store.analyzed_keys(conn)
    todo = [g for g in read_my_games(paths) if g.key not in done]
    todo.sort(key=lambda g: g.played_at, reverse=True)
    todo = todo[:max_games] if max_games else todo
    if progress:
        print(f"{len(done):,} games already analyzed · {len(todo):,} to go · {nodes:,} nodes/position")
    if not todo:
        return 0
    workers = workers or max(1, mp.cpu_count() - 1)
    start, n = time.time(), 0
    with mp.get_context("spawn").Pool(workers, _init_worker, (sf,)) as pool:
        for game, positions in pool.imap_unordered(_work, [(g, nodes) for g in todo]):
            store.save_game(conn, {
                "key": game.key, "source": game.source, "played_at": game.played_at,
                "my_color": game.my_color, "my_elo": game.my_elo, "opp_elo": game.opp_elo,
                "result": game.result, "base_time": game.base_time, "increment": game.increment,
                "eco": game.eco, "plies": len(game.moves), "nodes": nodes,
                "analyzed_at": datetime.now(UTC).isoformat(timespec="seconds"),
            }, positions)
            n += 1
            if progress and (n % 20 == 0 or n == len(todo)):
                rate = n / (time.time() - start)
                print(f"\r{n:,}/{len(todo):,} games · {rate * 60:,.0f} games/min · "
                      f"eta {(len(todo) - n) / rate / 60:,.0f} min", end="", flush=True)
        pool.close()
        pool.join()
    if progress:
        print()
    conn.close()
    return n
