"""Turn chess positions and moves into numbers the network can learn from.

Everything is seen from the **side to move** ("us"): when Black is to move, the board is
flipped vertically and colors are swapped, so the network always sees "my pieces moving up
the board". This halves what the network has to learn.

Position (compact storage, expanded to planes at training time):
    pieces    12 bitboards: us P N B R Q K, them P N B R Q K
    castling  4 bits: us king-side, us queen-side, them king-side, them queen-side
    ep        en-passant file (0-7) or -1

Move index (policy target), in the flipped frame:
    0..4095      from_square * 64 + to_square   (queen promotions included here)
    4096..4167   under-promotions: piece (N, B, R) x from_file (8) x direction (left, straight, right)
"""

from __future__ import annotations

import chess
import numpy as np

NUM_MOVES = 4096 + 3 * 8 * 3  # 4168
NUM_PLANES = 12 + 4 + 1       # pieces, castling, en passant

_PIECE_TYPES = [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING]
_UNDERPROMO = {chess.KNIGHT: 0, chess.BISHOP: 1, chess.ROOK: 2}
_UNDERPROMO_PIECE = [chess.KNIGHT, chess.BISHOP, chess.ROOK]


# ---------- positions ----------

def encode_position(board: chess.Board) -> tuple[np.ndarray, int, int]:
    """(pieces uint64[12], castling bits, ep file or -1), from the side to move's view."""
    us, them = board.turn, not board.turn
    flip = us == chess.BLACK
    pieces = np.empty(12, dtype=np.uint64)
    for i, color in enumerate((us, them)):
        for j, pt in enumerate(_PIECE_TYPES):
            bb = board.pieces_mask(pt, color)
            pieces[i * 6 + j] = chess.flip_vertical(bb) if flip else bb

    castling = (
        board.has_kingside_castling_rights(us) * 1
        | board.has_queenside_castling_rights(us) * 2
        | board.has_kingside_castling_rights(them) * 4
        | board.has_queenside_castling_rights(them) * 8
    )
    ep = chess.square_file(board.ep_square) if board.has_legal_en_passant() else -1
    return pieces, castling, ep


def planes_from_arrays(pieces: np.ndarray, castling: np.ndarray, ep: np.ndarray) -> np.ndarray:
    """Expand a batch of stored positions into float32 planes [B, 17, 8, 8].

    Plane layout: [rank][file], rank 0 = our back rank.
    """
    b = pieces.shape[0]
    bits = np.unpackbits(
        np.ascontiguousarray(pieces).view(np.uint8).reshape(b, 12, 8), axis=2, bitorder="little"
    )  # [B, 12, 64], bit k = square k (a1=0 ... h8=63) on little-endian machines
    out = np.zeros((b, NUM_PLANES, 8, 8), dtype=np.float32)
    out[:, :12] = bits.reshape(b, 12, 8, 8)
    for k in range(4):
        out[:, 12 + k] = ((castling >> k) & 1).astype(np.float32)[:, None, None]
    has_ep = ep >= 0
    out[np.nonzero(has_ep)[0], 16, 5, ep[has_ep]] = 1.0  # ep target is on rank 6 (index 5)
    return out


# ---------- moves ----------

def _flip_sq(sq: int, flip: bool) -> int:
    return chess.square_mirror(sq) if flip else sq


def move_to_index(move: chess.Move, turn: chess.Color) -> int:
    flip = turn == chess.BLACK
    frm, to = _flip_sq(move.from_square, flip), _flip_sq(move.to_square, flip)
    if move.promotion and move.promotion != chess.QUEEN:
        direction = chess.square_file(to) - chess.square_file(frm) + 1  # 0, 1, 2
        return 4096 + _UNDERPROMO[move.promotion] * 24 + chess.square_file(frm) * 3 + direction
    return frm * 64 + to


def index_to_move(index: int, board: chess.Board) -> chess.Move:
    """Inverse of move_to_index for a given position (adds queen promotion when needed)."""
    flip = board.turn == chess.BLACK
    if index >= 4096:
        k = index - 4096
        piece = _UNDERPROMO_PIECE[k // 24]
        from_file, direction = (k % 24) // 3, k % 3
        frm = chess.square(from_file, 6)
        to = chess.square(from_file + direction - 1, 7)
        return chess.Move(_flip_sq(frm, flip), _flip_sq(to, flip), promotion=piece)
    frm, to = _flip_sq(index // 64, flip), _flip_sq(index % 64, flip)
    move = chess.Move(frm, to)
    if board.piece_type_at(frm) == chess.PAWN and chess.square_rank(to) in (0, 7):
        move.promotion = chess.QUEEN
    return move


def board_from_arrays(pieces: np.ndarray, castling: int, ep: int) -> chess.Board:
    """Rebuild a position from stored arrays, in the flipped frame: "us" = White to move.

    Legal moves of this board map to the same indices as in the original position.
    """
    board = chess.Board(None)
    for i, color in enumerate((chess.WHITE, chess.BLACK)):
        for j, pt in enumerate(_PIECE_TYPES):
            for sq in chess.scan_forward(int(pieces[i * 6 + j])):
                board.set_piece_at(sq, chess.Piece(pt, color))
    board.turn = chess.WHITE
    fen_castling = "".join(c for bit, c in zip((1, 2, 4, 8), "KQkq", strict=True) if castling & bit)
    board.set_castling_fen(fen_castling or "-")
    board.ep_square = chess.square(ep, 5) if ep >= 0 else None
    return board


def legal_move_mask(board: chess.Board) -> np.ndarray:
    """Boolean mask over NUM_MOVES: True for legal moves. Used when choosing a move."""
    mask = np.zeros(NUM_MOVES, dtype=bool)
    for m in board.legal_moves:
        mask[move_to_index(m, board.turn)] = True
    return mask
