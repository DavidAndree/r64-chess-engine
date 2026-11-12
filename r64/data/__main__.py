"""Data commands.

    uv run python -m r64.data lichess-db --month 2026-10 --games 200000
    uv run python -m r64.data import-mine --lichess MyName --chesscom MyName
    uv run python -m r64.data encode-mine
    uv run python -m r64.data info data/positions/lichess-2026-10
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from r64 import config
from r64.data import my_games
from r64.data.build import build
from r64.data.pgn import GameFilter

POSITIONS_DIR = config.DATA_DIR / "positions"
LICHESS_URL = "https://database.lichess.org/standard/lichess_db_standard_rated_{month}.pgn.zst"


def cmd_lichess_db(args) -> None:
    source = args.file or LICHESS_URL.format(month=args.month)
    name = args.name or (f"lichess-{args.month}" if not args.file else Path(args.file).name.split(".")[0])
    out = POSITIONS_DIR / name
    filt = GameFilter(
        speeds=tuple(s.strip() for s in args.speeds.split(",") if s.strip()),
        min_elo=args.min_elo,
        max_elo=args.max_elo,
    )
    print(f"Source: {source}\nOutput: {out}")
    if source.startswith("http"):
        print("Streaming — only the part of the file needed for the requested games is downloaded.")
    manifest = build([source], out, filt, max_games=args.games, workers=args.workers)
    _print_manifest(manifest)


def cmd_import_mine(args) -> None:
    profile = my_games.load_profile()
    if args.lichess:
        profile["lichess"] = args.lichess
    if args.chesscom:
        profile["chesscom"] = args.chesscom
    if not (profile.get("lichess") or profile.get("chesscom")):
        raise SystemExit("Give --lichess and/or --chesscom usernames.")
    my_games.save_profile(profile)
    if profile.get("lichess"):
        my_games.fetch_lichess(profile["lichess"])
    if profile.get("chesscom"):
        my_games.fetch_chesscom(profile["chesscom"])


def cmd_encode_mine(args) -> None:
    try:
        manifest = my_games.encode_mine(POSITIONS_DIR / "me")
    except FileNotFoundError as e:
        raise SystemExit(str(e)) from e
    print(f"My rating now: {my_games.current_rating()}")
    _print_manifest(manifest)


def cmd_info(args) -> None:
    _print_manifest(json.loads((Path(args.dir) / "manifest.json").read_text()))


def _print_manifest(m: dict) -> None:
    print(
        f"Kept {m['games_kept']:,} of {m['games_read']:,} games in {m['seconds']}s\n"
        f"  train: {m['train']['positions']:,} positions ({m['train']['games']:,} games, {m['train']['shards']} shards)\n"
        f"  val:   {m['val']['positions']:,} positions ({m['val']['games']:,} games, {m['val']['shards']} shards)"
    )


def main() -> None:
    p = argparse.ArgumentParser(prog="python -m r64.data")
    sub = p.add_subparsers(required=True)

    s = sub.add_parser("lichess-db", help="encode games from the lichess open database")
    s.add_argument("--month", default="2026-10", help="database month, YYYY-MM")
    s.add_argument("--file", help="local .pgn or .pgn.zst instead of streaming the month")
    s.add_argument("--games", type=int, default=200_000, help="stop after this many kept games")
    s.add_argument("--speeds", default="blitz,rapid", help="comma-separated: bullet,blitz,rapid,classical")
    s.add_argument("--min-elo", type=int, default=0)
    s.add_argument("--max-elo", type=int, default=4000)
    s.add_argument("--name", help="output folder name under data/positions/")
    s.add_argument("--workers", type=int, default=None)
    s.set_defaults(fn=cmd_lichess_db)

    s = sub.add_parser("import-mine", help="download my games from lichess / chess.com")
    s.add_argument("--lichess")
    s.add_argument("--chesscom")
    s.set_defaults(fn=cmd_import_mine)

    s = sub.add_parser("encode-mine", help="encode my moves from all my games")
    s.set_defaults(fn=cmd_encode_mine)

    s = sub.add_parser("info", help="show what a positions folder contains")
    s.add_argument("dir")
    s.set_defaults(fn=cmd_info)

    args = p.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
