"""Phase 7: targeted sparring.

Three tools that point the human-like bot at my weak spots (from the weakness map):

**Steering.** Among the moves a human at the bot's rating would plausibly play (its top-p moves),
the sparring bot prefers ones that leave me in my weak kinds of position: if I'm bad at converting
winning positions, it lets me get winning positions more often; if I hang pieces, it creates
positions where my pieces are attacked. Only plausible moves are reweighted, so it still plays
like a human — just one with an agenda.

**Adaptive level.** The sparring bot's rating moves after every sparring game (Elo-style), toward
the level where I score about 50%.

**Drills.** Games that start from real positions from my own games where I went wrong in a weak
spot, played against the human-like bot. Scored against Stockfish's winning chances at the start.

"Is this position one of my weak kinds?" is answered without Stockfish: material, phase, attacked
pieces and tactics come from the board; winning/losing from the network's value head
(thresholds calibrated against Stockfish on my games: ~85% agreement).
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass

import chess

from r64.analysis.tags import (
    capture_gain,
    endgame_type,
    hanging,
    is_fork,
    material,
    non_pawn_material,
)
from r64.model.bot import BOT_MAX_ELO, BOT_MIN_ELO, HumanBot, temperature_for

WINNING, LOSING = 0.62, 0.42   # value-head expected score ≈ Stockfish win probability 0.7 / 0.3
STEER_STRENGTH = 2.5            # how hard the bot leans toward targets (exp(strength × score))
MAX_CANDIDATES = 12
K_FACTOR = 40                   # adaptive rating: Elo K
MAX_TARGETS = 3


# ---------- which weak spots can the bot steer toward, and how to recognise them ----------

def _any_move(board: chess.Board, test) -> bool:
    return any(test(board, m) for m in board.legal_moves)


def _has_mate_in_one(board: chess.Board) -> bool:
    for m in board.legal_moves:
        board.push(m)
        mate = board.is_checkmate()
        board.pop()
        if mate:
            return True
    return False


def position_tags(board: chess.Board, value: float | None) -> set[str]:
    """Weak-spot tags that describe this position for the side to move (me).
    `value` is my expected score from the value head (None: skip winning/losing)."""
    me = board.turn
    tags: set[str] = set()
    if value is not None:
        tags.add("position: " + ("winning" if value >= WINNING else "losing" if value <= LOSING else "balanced"))
    # Material once the dust settles: count my best immediate capture, so the middle of an
    # exchange (they just took, I'm about to take back) doesn't look like "behind".
    best_take = max((capture_gain(board, m) for m in board.legal_moves if board.is_capture(m)), default=0)
    diff = material(board, me) - material(board, not me) + max(0, best_take)
    tags.add("material: " + ("ahead" if diff >= 2 else "behind" if diff <= -2 else "even"))
    if non_pawn_material(board) <= 26:
        tags.add("phase: endgame")
        tags.add(f"endgame type: {endgame_type(board)}")
    elif board.fullmove_number <= 10:
        tags.add("phase: opening")
    else:
        tags.add("phase: middlegame")
    if board.is_check():
        tags.add("situation: in check")
    if hanging(board, me):
        tags.add("situation: a piece under attack")
        tags.add("mistake type: hung material")  # practice: positions where my pieces are in danger
    king = board.king(me)
    if (king is not None and board.fullmove_number > 12 and board.pieces(chess.QUEEN, not me)
            and chess.square_rank(king) == (0 if me == chess.WHITE else 7) and chess.square_file(king) in (3, 4)):
        tags.add("situation: king still in the center")
    if _any_move(board, lambda b, m: capture_gain(b, m) >= 2):
        tags.add("mistake type: missed material")  # practice: there is material to win
    if _any_move(board, is_fork):
        tags.add("mistake type: missed fork")      # practice: there is a fork to find
    if _has_mate_in_one(board):
        tags.add("mistake type: missed mate")
    return tags


STEERABLE = {
    "position: winning", "position: losing", "position: balanced",
    "material: ahead", "material: behind", "material: even",
    "phase: endgame", "phase: middlegame",
    "endgame type: pawn endgame", "endgame type: rook endgame", "endgame type: minor-piece endgame",
    "endgame type: queen endgame",
    "situation: a piece under attack", "situation: in check", "situation: king still in the center",
    "mistake type: hung material", "mistake type: missed material", "mistake type: missed fork",
    "mistake type: missed mate",
}


def targets_from_report(report: dict, k: int = MAX_TARGETS) -> list[tuple[str, float]]:
    """The top steerable weak spots, weighted by what they cost me (strongest = 1)."""
    spots = [w for w in report.get("weak_spots") or [] if w["tag"] in STEERABLE]
    spots = spots[:k]
    if not spots:
        return []
    top = max(w["extra_per_game"] for w in spots) or 1.0
    return [(w["tag"], w["extra_per_game"] / top) for w in spots]


# ---------- the sparring bot ----------

@dataclass
class SteerInfo:
    candidates: int
    chosen_score: float
    best_plain: str          # the move it would most likely play without steering
    chosen: str


class SparringBot(HumanBot):
    """The human-like bot, steering toward my weak spots."""

    name = "sparring"
    label = "Sparring partner"

    def __init__(self, checkpoint, targets_fn, strength: float = STEER_STRENGTH, **kw):
        super().__init__(checkpoint, **kw)
        self.targets_fn = targets_fn   # () -> [(tag, weight)], re-read each move (report may update)
        self.strength = strength
        self.focus: str | None = None  # one weak spot per game (set by prepare); None = all targets
        self.last: SteerInfo | None = None

    def prepare(self, record) -> None:
        """Called by the game service before each move: the game's focus is in `drill_tag`."""
        self.focus = record.drill_tag

    def steer_scores(self, board: chess.Board, moves: list[chess.Move], elo: int | None,
                     targets: list[tuple[str, float]]) -> list[float]:
        afters = []
        for m in moves:
            b = board.copy(stack=False)
            b.push(m)
            afters.append(b)
        need_value = any(t.startswith("position: ") for t, _ in targets)
        values = self.expected_scores(afters, elo) if need_value else [None] * len(afters)
        scores = []
        for b, v in zip(afters, values, strict=True):
            tags = position_tags(b, v) if not b.is_game_over() else set()
            scores.append(sum(w for t, w in targets if t in tags))
        return scores

    def choose_move(self, board: chess.Board, elo: int | None) -> chess.Move:
        import time

        start = time.monotonic()
        targets = [(self.focus, 1.0)] if self.focus else (self.targets_fn() or [])
        temp = self.temperature if self.temperature is not None else temperature_for(elo or 1500)
        probs = self.move_probabilities(board, elo)
        moves, weights = self.candidates(probs, temp)
        moves, weights = moves[:MAX_CANDIDATES], weights[:MAX_CANDIDATES]
        if targets and len(moves) > 1:
            scores = self.steer_scores(board, moves, elo, targets)
            weights = [w * math.exp(self.strength * s) for w, s in zip(weights, scores, strict=True)]
        else:
            scores = [0.0] * len(moves)
        i = self._rng.choices(range(len(moves)), weights=weights)[0]
        self.last = SteerInfo(len(moves), scores[i], moves[0].uci(), moves[i].uci())
        lo, hi = self.think_time
        pause = self._rng.uniform(lo, hi) - (time.monotonic() - start)
        if pause > 0:
            time.sleep(pause)
        return moves[i]


def next_focus(targets: list[tuple[str, float]], past_focuses: list[str | None]) -> str | None:
    """Rotate through my targets, more often for the costlier ones: pick the target furthest
    behind its share (weight / total weight) of past sparring games."""
    if not targets:
        return None
    total_w = sum(w for _, w in targets)
    n = len(past_focuses) + 1
    deficit = {t: w / total_w * n - past_focuses.count(t) for t, w in targets}
    return max(targets, key=lambda tw: (deficit[tw[0]], tw[1]))[0]


# ---------- adaptive rating ----------

def adaptive_rating(results: list[tuple[int, float]], start: int, k: float = K_FACTOR) -> tuple[int, list[int]]:
    """Elo-style: `results` = [(bot rating played, my score 0/0.5/1)] oldest first. Returns the
    rating at which I'm expected to score 50% next, and its history."""
    r = float(start)
    history = [round(r)]
    for opp, score in results:
        expected = 1 / (1 + 10 ** ((opp - r) / 400))   # my expected score vs a bot at `opp`
        r += k * (score - expected)
        r = max(BOT_MIN_ELO, min(BOT_MAX_ELO, r))
        history.append(round(r))
    return round(r), history


def my_score(result: str | None, color: str) -> float | None:
    if result == "1/2-1/2":
        return 0.5
    if result in ("1-0", "0-1"):
        return 1.0 if (result == "1-0") == (color == "white") else 0.0
    return None


# ---------- focus: did a game actually put me in my weak spots? ----------

def focus_shares(start_fen: str | None, moves: list[str], my_color: str,
                 targets: list[str], value_fn) -> dict[str, float]:
    """Share of my turns in this game where each target tag described the position."""
    board = chess.Board(start_fen) if start_fen else chess.Board()
    mine = chess.WHITE if my_color == "white" else chess.BLACK
    positions = []
    for uci in moves:
        if board.turn == mine:
            positions.append(board.copy(stack=False))
        board.push_uci(uci)
    if not positions:
        return {t: 0.0 for t in targets}
    values = value_fn(positions) if any(t.startswith("position: ") for t in targets) else [None] * len(positions)
    counts = dict.fromkeys(targets, 0)
    for b, v in zip(positions, values, strict=True):
        tags = position_tags(b, v)
        for t in targets:
            counts[t] += t in tags
    return {t: counts[t] / len(positions) for t in targets}


# ---------- drills ----------

def pick_drill(candidates: list[dict], used: set[str], rng: random.Random | None = None) -> dict | None:
    """A drill position not played yet (or the least recently used ones once all are done)."""
    rng = rng or random.Random()
    fresh = [c for c in candidates if c["fen"] not in used]
    return rng.choice(fresh or candidates) if candidates else None
