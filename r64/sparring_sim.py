"""Does steering work? Simulate sparring games and measure how often "I" land in my weak spots.

    uv run python -m r64.sparring_sim --games 40

"Me" is played by the human-like bot at my rating; the sparring bot plays the other side, once with
steering and once without. For each target weak spot it reports the share of my turns where that
kind of position came up.
"""

from __future__ import annotations

import argparse
import statistics

import chess

from r64 import config
from r64.analysis.weakness import cached_report
from r64.data.my_games import current_rating
from r64.game import outcome
from r64.model.bot import HumanBot
from r64.sparring import SparringBot, focus_shares, my_score, targets_from_report


def play(me: HumanBot, partner: SparringBot, my_color: chess.Color, rating: int, max_plies: int = 160):
    board = chess.Board()
    moves = []
    while outcome(board) is None and len(moves) < max_plies:
        bot = me if board.turn == my_color else partner
        move = bot.choose_move(board, rating)
        board.push(move)
        moves.append(move.uci())
    return moves


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--games", type=int, default=40)
    p.add_argument("--rating", type=int, default=None)
    p.add_argument("--strength", type=float, default=2.5)
    args = p.parse_args()

    rating = args.rating or current_rating()
    targets = targets_from_report(cached_report(1000))
    print(f"Targets: {targets}  ·  rating {rating}")
    me = HumanBot(config.BOT_CHECKPOINT, think_time=(0, 0), seed=1)
    tags = [t for t, _ in targets]
    rows = []
    for focus in tags:
        result = {}
        for strength in (0.0, args.strength):
            partner = SparringBot(config.BOT_CHECKPOINT, lambda: targets, strength=strength,
                                  think_time=(0, 0), seed=2)
            partner.focus = focus
            shares, scores = [], []
            for g in range(args.games):
                color = chess.WHITE if g % 2 == 0 else chess.BLACK
                moves = play(me, partner, color, rating)
                name = "white" if color == chess.WHITE else "black"
                shares.append(focus_shares(None, moves, name, [focus],
                                           lambda bs: me.expected_scores(bs, rating))[focus])
                board = chess.Board()
                for m in moves:
                    board.push_uci(m)
                res = outcome(board)
                scores.append(my_score(res[0] if res else "1/2-1/2", name))
            result[strength] = (statistics.fmean(shares), statistics.fmean(scores))
        off, on = result[0.0], result[args.strength]
        rows.append((focus, off, on))
        print(f"{focus:<40} off {off[0]:6.1%} (score {off[1]:.0%})   on {on[0]:6.1%} (score {on[1]:.0%})", flush=True)

if __name__ == "__main__":
    main()
