"""Play opponents against each other to check that the bot's rating setting changes its strength.

    uv run python -m r64.model match bot:1000 bot:1800 --games 20
    uv run python -m r64.model match bot:1400 stockfish:1320 --games 20
"""

from __future__ import annotations

import time

import chess

from r64.game import outcome
from r64.opponents import Opponent

MAX_PLIES = 300


def play_game(white: tuple[Opponent, int], black: tuple[Opponent, int]) -> tuple[str, int]:
    board = chess.Board()
    while (over := outcome(board)) is None and board.ply() < MAX_PLIES:
        opp, elo = white if board.turn == chess.WHITE else black
        board.push(opp.choose_move(board, elo))
    return (over[0] if over else "1/2-1/2"), board.ply()


def match(a: tuple[Opponent, int], b: tuple[Opponent, int], games: int, label_a: str, label_b: str):
    """Alternate colors; print a running score from A's point of view."""
    score, plies = 0.0, []
    start = time.monotonic()
    for g in range(games):
        a_white = g % 2 == 0
        result, n = play_game(a, b) if a_white else play_game(b, a)
        plies.append(n)
        if result == "1/2-1/2":
            score += 0.5
        elif (result == "1-0") == a_white:
            score += 1
        print(f"  game {g + 1:3}: {label_a} as {'white' if a_white else 'black'}  {result:7}  "
              f"{n // 2} moves   score {score}/{g + 1}", flush=True)
    print(f"\n{label_a} vs {label_b}: {score}/{games} ({score / games:.0%}), "
          f"avg {sum(plies) / len(plies) / 2:.0f} moves, {time.monotonic() - start:.0f}s")
    return score
