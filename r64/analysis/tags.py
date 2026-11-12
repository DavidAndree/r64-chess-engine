"""Describe moves: how much they cost, what kind of position they were played in, and — for
mistakes — what went wrong. Pure functions of the stored analysis (no Stockfish needed).

Cost = drop in the mover's **winning chances** (0-1), from Stockfish's eval before and after the
move, using lichess's eval → win-probability curve. Unlike centipawns this ignores "losing" 300cp
in a position that stays completely won, and counts a blunder in a balanced position fully.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import chess

MATE_THRESHOLD = 50_000

# Lichess's thresholds (on winning chances in [-1, 1]: 0.1 / 0.2 / 0.3), as win-probability drops.
INACCURACY, MISTAKE, BLUNDER = 0.05, 0.10, 0.15

VALUES = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 100}
PIECE_NAMES = {chess.PAWN: "pawn", chess.KNIGHT: "knight", chess.BISHOP: "bishop",
               chess.ROOK: "rook", chess.QUEEN: "queen", chess.KING: "king"}

# Mistake causes, most specific first.
CAUSES = ["allowed mate", "missed mate", "hung material", "allowed fork", "missed material",
          "missed fork", "other"]


def win_prob(cp: int) -> float:
    """Winning chances (0-1) for the side the eval is from."""
    if cp >= MATE_THRESHOLD:
        return 1.0
    if cp <= -MATE_THRESHOLD:
        return 0.0
    return 1 / (1 + math.exp(-0.00368208 * cp))


def severity(loss: float) -> str | None:
    if loss >= BLUNDER:
        return "blunder"
    if loss >= MISTAKE:
        return "mistake"
    if loss >= INACCURACY:
        return "inaccuracy"
    return None


# ---------- board helpers ----------

def material(board: chess.Board, color: chess.Color) -> int:
    return sum(VALUES[pt] * len(board.pieces(pt, color)) for pt in VALUES if pt != chess.KING)


def non_pawn_material(board: chess.Board) -> int:
    return sum(
        VALUES[pt] * len(board.pieces(pt, c))
        for pt in (chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN) for c in chess.COLORS
    )


def capture_gain(board: chess.Board, move: chess.Move) -> int:
    """Rough material won by a capture: the victim, minus the capturer if the square is defended."""
    if not board.is_capture(move):
        return 0
    if board.is_en_passant(move):
        victim = 1
    else:
        victim = VALUES[board.piece_type_at(move.to_square)]
    attacker = VALUES[board.piece_type_at(move.from_square)]
    defended = board.is_attacked_by(not board.turn, move.to_square)
    return victim - (attacker if defended else 0)


def is_fork(board: chess.Board, move: chess.Move) -> bool:
    """Does `move` (legal in `board`) attack two or more targets worth attacking: the king, a
    piece worth more than the attacker, or an undefended piece worth 3+?"""
    piece = board.piece_at(move.from_square)
    if piece is None:
        return False
    after = board.copy(stack=False)
    after.push(move)
    attacker_value = VALUES[piece.piece_type if not move.promotion else move.promotion]
    # Not a fork if the forking piece can simply be taken for free.
    if (after.is_attacked_by(after.turn, move.to_square)
            and not after.is_attacked_by(not after.turn, move.to_square)):
        return False
    targets = 0
    for sq in after.attacks(move.to_square):
        victim = after.piece_at(sq)
        if victim is None or victim.color != after.turn:
            continue
        v = VALUES[victim.piece_type]
        if v == 100 or v > attacker_value or (v >= 3 and not after.is_attacked_by(after.turn, sq)):
            targets += 1
    return targets >= 2


def hanging(board: chess.Board, color: chess.Color) -> bool:
    """Is a piece (knight or better) of `color` attacked and undefended, or attacked by a cheaper piece?"""
    for sq, piece in board.piece_map().items():
        if piece.color != color or piece.piece_type in (chess.PAWN, chess.KING):
            continue
        attackers = board.attackers(not color, sq)
        if not attackers:
            continue
        if not board.is_attacked_by(color, sq):
            return True
        cheapest = min(VALUES[board.piece_type_at(a)] for a in attackers)
        if cheapest < VALUES[piece.piece_type]:
            return True
    return False


def endgame_type(board: chess.Board) -> str:
    kinds = {pt for pt in (chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN)
             for c in chess.COLORS if board.pieces(pt, c)}
    if not kinds:
        return "pawn endgame"
    if chess.QUEEN in kinds:
        return "queen endgame"
    if chess.ROOK in kinds:
        return "rook endgame"
    return "minor-piece endgame"


# ---------- describing one move ----------

@dataclass
class MoveFacts:
    loss: float                         # winning chances lost by this move (0-1)
    wp_before: float
    severity: str | None
    tags: list[str] = field(default_factory=list)   # "dimension: value"
    cause: str | None = None            # for mistakes and blunders


def describe(
    fen: str,
    move_uci: str,
    best_uci: str | None,
    cp_before: int,
    cp_after: int,
    reply_uci: str | None,
    clock_before: float | None,
    base_time: int,
) -> MoveFacts:
    """`cp_*` from White's view; `reply_uci` is Stockfish's best reply after the move."""
    board = chess.Board(fen)
    me = board.turn
    sign = 1 if me == chess.WHITE else -1
    wp_before, wp_after = win_prob(sign * cp_before), win_prob(sign * cp_after)
    loss = max(0.0, wp_before - wp_after)
    move = chess.Move.from_uci(move_uci)

    tags = []
    npm = non_pawn_material(board)
    if npm <= 26:
        phase = "endgame"
        tags.append(f"endgame type: {endgame_type(board)}")
    elif board.fullmove_number <= 10:
        phase = "opening"
    else:
        phase = "middlegame"
    tags.append(f"phase: {phase}")
    tags.append("position: " + ("winning" if wp_before >= 0.7 else "losing" if wp_before <= 0.3 else "balanced"))
    diff = material(board, me) - material(board, not me)
    tags.append("material: " + ("ahead" if diff >= 2 else "behind" if diff <= -2 else "even"))
    if clock_before is not None and base_time and base_time > 0:
        trouble = clock_before < 30 or clock_before < 0.1 * base_time
        tags.append("clock: " + ("time trouble (<10% or <30s)" if trouble else "enough time"))
    if board.is_check():
        tags.append("situation: in check")
    if hanging(board, me):
        tags.append("situation: a piece under attack")
    king = board.king(me)
    queens_on = bool(board.pieces(chess.QUEEN, not me))
    if (king is not None and board.fullmove_number > 12 and queens_on
            and chess.square_rank(king) == (0 if me == chess.WHITE else 7)
            and chess.square_file(king) in (3, 4)):
        tags.append("situation: king still in the center")
    moved = board.piece_type_at(move.from_square)
    if moved is not None:
        tags.append(f"piece moved: {PIECE_NAMES[moved]}")

    sev = severity(loss)
    cause = None
    if loss >= MISTAKE:
        best = chess.Move.from_uci(best_uci) if best_uci else None
        after = board.copy(stack=False)
        after.push(move)
        reply = chess.Move.from_uci(reply_uci) if reply_uci else None
        if reply is not None and reply not in after.legal_moves:
            reply = None
        if sign * cp_after <= -MATE_THRESHOLD and sign * cp_before > -MATE_THRESHOLD:
            cause = "allowed mate"
        elif sign * cp_before >= MATE_THRESHOLD:
            cause = "missed mate"
        elif reply is not None and capture_gain(after, reply) >= 2:
            cause = "hung material"
        elif reply is not None and is_fork(after, reply):
            cause = "allowed fork"
        elif best is not None and best != move and capture_gain(board, best) >= 2:
            cause = "missed material"
        elif best is not None and best != move and is_fork(board, best):
            cause = "missed fork"
        else:
            cause = "other"
    return MoveFacts(loss, wp_before, sev, tags, cause)
