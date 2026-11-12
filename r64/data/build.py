"""Build training shards from a PGN source (lichess database URL, local file, or my games).

Pipeline: read games → filter on headers → encode in parallel → write shards.
Games are split into train / val by a hash of the game id, so all positions of a game
land in the same split.
"""

from __future__ import annotations

import json
import multiprocessing as mp
import time
import zlib
from collections import deque
from collections.abc import Iterable, Iterator
from dataclasses import asdict
from pathlib import Path

import numpy as np

from r64.data.pgn import FIELDS, EncodedGame, GameFilter, RawGame, encode_game, open_pgn, read_games

SHARD_POSITIONS = 500_000
BATCH_GAMES = 400  # games per worker task


class ShardWriter:
    """Buffers encoded positions and writes them as numbered .npz shards."""

    def __init__(self, out_dir: Path, prefix: str, shard_positions: int = SHARD_POSITIONS):
        self.out_dir = out_dir
        self.prefix = prefix
        self.shard_positions = shard_positions
        self.buf: dict[str, list] = {k: [] for k in FIELDS}
        self.buffered = 0
        self.shards = 0
        self.positions = 0
        self.games = 0

    def add(self, game: EncodedGame) -> None:
        for k, v in game.columns.items():
            self.buf[k].extend(v)
        self.buffered += len(game)
        self.positions += len(game)
        self.games += 1
        if self.buffered >= self.shard_positions:
            self.flush()

    def flush(self) -> None:
        if not self.buffered:
            return
        arrays = {}
        for k, (dtype, shape) in FIELDS.items():
            arr = np.asarray(self.buf[k], dtype=dtype)
            arrays[k] = arr.reshape((-1, *shape))
        path = self.out_dir / f"{self.prefix}-{self.shards:05d}.npz"
        np.savez(path, **arrays)
        self.shards += 1
        self.buf = {k: [] for k in FIELDS}
        self.buffered = 0


def _game_key(game: RawGame) -> str:
    """A string unique per game. Lichess: Site is the game URL; chess.com: Link is."""
    for key in ("Link", "Site"):
        if game.headers.get(key, "").startswith("http"):
            return game.headers[key]
    return (
        game.headers.get("White", "") + game.headers.get("Black", "") + game.headers.get("UTCDate", "")
        + game.headers.get("UTCTime", "") + game.movetext[:60]
    )


def is_val(game: RawGame, val_percent: int) -> bool:
    return zlib.crc32(_game_key(game).encode()) % 100 < val_percent


def _encode_batch(args: tuple[list[RawGame], set[str] | None, int]):
    games, player, val_percent = args
    out = []
    for g in games:
        enc = encode_game(g, player)
        if enc is not None and len(enc) > 0:
            out.append((is_val(g, val_percent), enc))
    return out


def _batches(
    games: Iterable[RawGame], game_filter: GameFilter | None, max_games: int | None, stats: dict
) -> Iterator[list[RawGame]]:
    batch: list[RawGame] = []
    for g in games:
        stats["read"] += 1
        if game_filter is not None and not game_filter.accepts(g.headers):
            continue
        stats["accepted"] += 1
        batch.append(g)
        if len(batch) == BATCH_GAMES:
            yield batch
            batch = []
        if max_games is not None and stats["accepted"] >= max_games:
            break
    if batch:
        yield batch


def build(
    sources: list[str | Path],
    out_dir: Path,
    game_filter: GameFilter | None = GameFilter(),
    max_games: int | None = None,
    player: set[str] | None = None,
    val_percent: int = 2,
    workers: int | None = None,
    shard_positions: int = SHARD_POSITIONS,
    progress: bool = True,
) -> dict:
    """Encode games from `sources` into `out_dir/{train,val}-NNNNN.npz` + manifest.json."""
    out_dir.mkdir(parents=True, exist_ok=True)
    for old in out_dir.glob("*.npz"):
        old.unlink()
    train = ShardWriter(out_dir, "train", shard_positions)
    val = ShardWriter(out_dir, "val", shard_positions)
    stats = {"read": 0, "accepted": 0}
    player = {p.lower() for p in player} if player else None
    workers = workers or max(1, mp.cpu_count() - 1)
    start = time.time()

    def all_games() -> Iterator[RawGame]:
        for src in sources:
            with open_pgn(src) as f:
                yield from read_games(f)

    def collect(results) -> None:
        for in_val, enc in results:
            (val if in_val else train).add(enc)

    ctx = mp.get_context("spawn")
    with ctx.Pool(workers) as pool:
        # Bounded number of batches in flight, so a huge stream never piles up in memory.
        in_flight: deque = deque()
        for batch in _batches(all_games(), game_filter, max_games, stats):
            in_flight.append(pool.apply_async(_encode_batch, ((batch, player, val_percent),)))
            if len(in_flight) < workers * 3:
                continue
            collect(in_flight.popleft().get())
            if progress:
                elapsed = time.time() - start
                print(
                    f"\r{stats['accepted']:,} games kept / {stats['read']:,} read · "
                    f"{train.positions + val.positions:,} positions · "
                    f"{stats['accepted'] / max(elapsed, 1e-9):,.0f} games/s",
                    end="", flush=True,
                )
        while in_flight:
            collect(in_flight.popleft().get())
    train.flush()
    val.flush()
    if progress:
        print()

    manifest = {
        "sources": [str(s) for s in sources],
        "filter": asdict(game_filter) if game_filter else None,
        "player": sorted(player) if player else None,
        "games_read": stats["read"],
        "games_kept": stats["accepted"],
        "train": {"games": train.games, "positions": train.positions, "shards": train.shards},
        "val": {"games": val.games, "positions": val.positions, "shards": val.shards},
        "seconds": round(time.time() - start, 1),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest
