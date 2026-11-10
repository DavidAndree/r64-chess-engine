"""Model commands.

    uv run python -m r64.model train --data data/positions/lichess-base --name base
    uv run python -m r64.model evaluate --checkpoint checkpoints/base/best.pt
    uv run python -m r64.model match bot:1000 bot:1800 --games 20
    uv run python -m r64.model personal            # train the next version of my personal model
    uv run python -m r64.model learning-curve      # does it improve as it gets more of my games?
"""

from __future__ import annotations

import argparse
from pathlib import Path

from r64 import config
from r64.model.evaluate import evaluate
from r64.model.network import NetConfig
from r64.model.train import TrainConfig, train


def main() -> None:
    p = argparse.ArgumentParser(prog="python -m r64.model")
    sub = p.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("train", help="train the move-prediction network")
    t.add_argument("--data", nargs="+", required=True, help="positions folder(s)")
    t.add_argument("--name", default="base")
    t.add_argument("--blocks", type=int, default=6)
    t.add_argument("--channels", type=int, default=64)
    t.add_argument("--epochs", type=float, default=3.0)
    t.add_argument("--batch-size", type=int, default=1024)
    t.add_argument("--lr", type=float, default=2e-3)
    t.add_argument("--eval-every", type=int, default=2000)
    t.add_argument("--eval-positions", type=int, default=20_000)
    t.add_argument("--init-from", help="checkpoint to start from (fine-tuning)")

    e = sub.add_parser("evaluate", help="accuracy by rating band, rating sensitivity, my games")
    e.add_argument("--checkpoint", default=str(config.CHECKPOINT_DIR / "base" / "best.pt"))
    e.add_argument("--data", default=str(config.DATA_DIR / "positions" / "lichess-base"))
    e.add_argument("--me", default=str(config.DATA_DIR / "positions" / "me"))
    e.add_argument("--positions", type=int, default=30_000)

    m = sub.add_parser("match", help="play two opponents against each other (alternating colors)")
    m.add_argument("a", help="opponent:rating, e.g. bot:1200 or stockfish:1320")
    m.add_argument("b")
    m.add_argument("--games", type=int, default=20)
    m.add_argument("--checkpoint", default=str(config.CHECKPOINT_DIR / "base" / "best.pt"))
    m.add_argument("--temperature", type=float, help="default: follows the rating")
    m.add_argument("--top-p", type=float, default=0.95)

    pm = sub.add_parser("personal", help="fine-tune the next version of my personal model")
    pm.add_argument("--fetch", action="store_true", help="re-download my chess.com/lichess games first")
    sub.add_parser("learning-curve", help="personal model accuracy vs number of my games")

    args = p.parse_args()
    if args.cmd == "match":
        _run_match(args)
    elif args.cmd == "personal":
        from r64.model.personal import train_version

        train_version(fetch=args.fetch)
    elif args.cmd == "learning-curve":
        from r64.model.personal import learning_curve

        learning_curve()
    elif args.cmd == "train":
        train(TrainConfig(
            data=args.data, name=args.name,
            net=NetConfig(blocks=args.blocks, channels=args.channels),
            epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
            eval_every=args.eval_every, eval_positions=args.eval_positions,
            init_from=args.init_from,
        ))
    else:
        me = Path(args.me) if args.me else None
        evaluate(Path(args.checkpoint), Path(args.data), me if me and me.exists() else None, args.positions)


def _run_match(args) -> None:
    from r64.model.bot import HumanBot
    from r64.model.match import match
    from r64.opponents import StockfishOpponent

    made: dict = {}

    def make(spec: str):
        kind, elo = spec.split(":")
        if kind not in made:
            if kind.startswith("me-v"):
                from r64.model.bot import personal_bot
                from r64.model.personal import versions

                found = [v for v in versions() if v.key == kind]
                if not found:
                    raise SystemExit(f"no personal model {kind}")
                made[kind] = personal_bot(found[0])
                made[kind].think_time = (0, 0)
            elif kind == "bot":
                made[kind] = HumanBot(args.checkpoint, args.temperature, args.top_p, think_time=(0, 0))
            elif kind == "stockfish":
                made[kind] = StockfishOpponent(think_time=0.05)
            else:
                raise SystemExit(f"unknown opponent {kind!r}: use bot, stockfish or me-vN")
        return made[kind], int(elo)

    a, b = make(args.a), make(args.b)
    match(a, b, args.games, args.a, args.b)
    for opp in made.values():
        opp.close()


if __name__ == "__main__":
    main()
