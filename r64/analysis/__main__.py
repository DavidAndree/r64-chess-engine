"""Analysis commands.

    uv run python -m r64.analysis analyze           # Stockfish over my games (resumable)
    uv run python -m r64.analysis report            # the weakness map, as text
"""

from __future__ import annotations

import argparse


def main() -> None:
    p = argparse.ArgumentParser(prog="python -m r64.analysis")
    sub = p.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("analyze", help="run Stockfish over my games not analyzed yet")
    a.add_argument("--games", type=int, help="at most this many (newest first)")
    a.add_argument("--nodes", type=int, default=10_000, help="Stockfish nodes per position")
    a.add_argument("--workers", type=int)
    r = sub.add_parser("report", help="print the weakness map")
    r.add_argument("--recent", type=int, default=1000, help="use my most recent N analyzed games")
    args = p.parse_args()

    if args.cmd == "analyze":
        from r64.analysis.analyze import analyze

        analyze(args.games, args.nodes, args.workers)
    else:
        from r64.analysis.weakness import print_report

        print_report(recent=args.recent)


if __name__ == "__main__":
    main()
