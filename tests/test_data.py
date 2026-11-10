import io
import random

import chess
import numpy as np
import pytest

from r64.data.build import build
from r64.data.dataset import PositionSet
from r64.data.encoding import (
    NUM_MOVES,
    encode_position,
    index_to_move,
    legal_move_mask,
    move_to_index,
    planes_from_arrays,
)
from r64.data.pgn import GameFilter, encode_game, parse_time_control, read_games

LICHESS_GAME = """[Event "Rated Blitz game"]
[Site "https://lichess.org/abcd1234"]
[White "alice"]
[Black "bob"]
[Result "0-1"]
[WhiteElo "1500"]
[BlackElo "1620"]
[TimeControl "180+2"]
[Termination "Normal"]

1. f3 { [%clk 0:03:00] } 1... e5 { [%eval 0.5] [%clk 0:02:58] } 2. g4 { [%clk 0:02:59] } 2... Qh4# { [%clk 0:02:55] } 0-1

[Event "Rated Bullet game"]
[Site "https://lichess.org/bullet01"]
[White "carol"]
[Black "dave"]
[Result "1-0"]
[WhiteElo "1500"]
[BlackElo "1500"]
[TimeControl "60+0"]
[Termination "Normal"]

1. e4 { [%clk 0:01:00] } 1-0

[Event "Rated Rapid game"]
[Site "https://lichess.org/abandon1"]
[White "erin"]
[Black "frank"]
[Result "1-0"]
[WhiteElo "1500"]
[BlackElo "1500"]
[TimeControl "600+0"]
[Termination "Abandoned"]

1. e4 1-0
"""

CHESSCOM_GAME = """[Event "Live Chess"]
[Site "Chess.com"]
[White "Me_On_ChessCom"]
[Black "someone"]
[Result "1-0"]
[WhiteElo "1450"]
[BlackElo "1480"]
[TimeControl "600"]
[Termination "Me_On_ChessCom won by checkmate"]

1. e4 {[%clk 0:09:58.5]} 1... e5 {[%clk 0:09:57.1]} 2. Qh5 {[%clk 0:09:50]} 2... Nc6 {[%clk 0:09:40]}
3. Bc4 {[%clk 0:09:45]} 3... Nf6?? {[%clk 0:09:30]} 4. Qxf7# {[%clk 0:09:40]} 1-0
"""


# ---------- encoding ----------

def _random_positions(n=300, seed=0):
    rng = random.Random(seed)
    boards = []
    for _ in range(n):
        b = chess.Board()
        for _ in range(rng.randint(0, 80)):
            moves = list(b.legal_moves)
            if not moves:
                break
            b.push(rng.choice(moves))
        boards.append(b)
    # promotions (incl. under-promotions and captures) for both colors
    boards.append(chess.Board("1n2k3/P7/8/8/8/8/7p/4K1N1 w - - 0 1"))
    boards.append(chess.Board("1n2k3/P7/8/8/8/8/7p/4K1N1 b - - 0 1"))
    return boards


def test_move_index_round_trip_for_all_legal_moves():
    for board in _random_positions():
        seen = set()
        for move in board.legal_moves:
            idx = move_to_index(move, board.turn)
            assert 0 <= idx < NUM_MOVES
            assert idx not in seen, "two moves share an index"
            seen.add(idx)
            assert index_to_move(idx, board) == move


def test_legal_mask_matches_legal_moves():
    board = chess.Board()
    assert legal_move_mask(board).sum() == 20


def test_position_is_seen_from_side_to_move():
    white_view = encode_position(chess.Board())
    black_view = encode_position(chess.Board("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR b KQkq - 0 1"))
    # Start position is symmetric, so both sides see exactly the same thing.
    assert np.array_equal(white_view[0], black_view[0])
    assert white_view[1] == black_view[1] == 0b1111


def test_planes_layout():
    board = chess.Board("4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 2")
    pieces, castling, ep = encode_position(board)
    planes = planes_from_arrays(pieces[None], np.array([castling], np.uint8), np.array([ep], np.int8))[0]
    assert planes.shape == (17, 8, 8)
    assert planes[0, 4, 4] == 1          # our pawn on e5 -> rank index 4, file index 4
    assert planes[6, 4, 3] == 1          # their pawn on d5
    assert planes[5, 0, 4] == 1          # our king on e1
    assert planes[16, 5, 3] == 1         # en passant on d6
    assert planes[12:16].sum() == 0      # no castling rights


# ---------- PGN ----------

def test_read_and_filter_games():
    games = list(read_games(io.StringIO(LICHESS_GAME)))
    assert len(games) == 3
    filt = GameFilter(speeds=("blitz", "rapid"))
    assert [filt.accepts(g.headers) for g in games] == [True, False, False]
    assert parse_time_control("180+2") == (180, 2)
    assert parse_time_control("-") is None


def test_encode_game_with_clocks_and_results():
    game = next(read_games(io.StringIO(LICHESS_GAME)))
    enc = encode_game(game).columns
    assert len(enc["move"]) == 4
    assert enc["clock"] == [180, 178, 179, 175]
    assert enc["elo"] == [1500, 1620, 1500, 1620]
    assert enc["result"] == [-1, 1, -1, 1]  # black won
    assert enc["base_time"][0] == 180 and enc["increment"][0] == 2


def test_encode_only_my_moves_from_chesscom_pgn():
    game = next(read_games(io.StringIO(CHESSCOM_GAME)))
    enc = encode_game(game, player={"me_on_chesscom"}).columns
    assert len(enc["move"]) == 4  # only White's moves
    assert enc["clock"][0] == 598
    assert enc["elo"][0] == 1450
    assert encode_game(game, player={"nobody"}) is None


def test_broken_movetext_is_skipped():
    game = next(read_games(io.StringIO(LICHESS_GAME.replace("Qh4#", "Qh9#"))))
    assert encode_game(game) is None


# ---------- build + load ----------

@pytest.fixture
def pgn_file(tmp_path):
    path = tmp_path / "games.pgn"
    blitz = LICHESS_GAME.split("\n\n[Event")[0] + "\n\n"
    games = [blitz.replace("abcd1234", f"g{i:04d}") for i in range(60)]
    path.write_text("".join(games))
    return path


def test_build_shards_and_load_batches(pgn_file, tmp_path):
    out = tmp_path / "positions"
    m = build([pgn_file], out, GameFilter(speeds=("blitz",)), val_percent=20, workers=2,
              shard_positions=50, progress=False)
    assert m["games_kept"] == 60
    assert m["train"]["positions"] + m["val"]["positions"] == 240
    assert m["val"]["games"] > 0 and m["train"]["shards"] > 1

    train = PositionSet.from_dir(out, "train")
    assert len(train) == m["train"]["positions"]
    batches = list(train.batches(32, seed=1))
    assert sum(len(b["move"]) for b in batches) == len(train)
    b = batches[0]
    assert b["planes"].shape[1:] == (17, 8, 8)
    assert b["planes"].dtype == np.float32 and b["move"].dtype == np.int64

    first = train.select(train.arrays["ply"] == 0)
    start = train.batch(first[:1])
    assert index_to_move(int(start["move"][0]), chess.Board()) == chess.Move.from_uci("f2f3")


def test_chesscom_games_split_by_link(tmp_path):
    games = "".join(
        CHESSCOM_GAME.replace('[Site "Chess.com"]', f'[Site "Chess.com"]\n[Link "https://www.chess.com/game/live/{i}"]')
        + "\n"
        for i in range(100)
    )
    path = tmp_path / "cc.pgn"
    path.write_text(games)
    m = build([path], tmp_path / "out", None, player={"me_on_chesscom"}, val_percent=10,
              workers=1, progress=False)
    assert 0 < m["val"]["games"] < 30


def test_batches_to_torch(pgn_file, tmp_path):
    import torch

    out = tmp_path / "positions"
    build([pgn_file], out, GameFilter(speeds=("blitz",)), val_percent=0, workers=1, progress=False)
    b = next(PositionSet.from_dir(out).batches(16, device="cpu"))
    assert isinstance(b["planes"], torch.Tensor) and b["planes"].shape == (16, 17, 8, 8)
