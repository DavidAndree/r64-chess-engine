import json
import sqlite3

import chess
import pytest
import torch

from r64.game import GameService
from r64.model.network import MoveNet, NetConfig
from r64.sparring import (
    SparringBot,
    adaptive_rating,
    focus_shares,
    my_score,
    next_focus,
    pick_drill,
    position_tags,
    targets_from_report,
)
from r64.sparring_service import SparringService
from r64.storage import GameStore
from tests.test_game import ScriptedOpponent

DRILL_FEN = "rnbq1rk1/p1p1ppbp/1p1p1np1/8/P1P1P3/1PNP1P2/6PP/R1BQKBNR b KQ - 2 7"


def test_adaptive_rating_moves_toward_fifty_percent():
    assert adaptive_rating([], 900) == (900, [900])
    up, hist = adaptive_rating([(900, 1.0)] * 5, 900)
    assert up > 900 and hist[0] == 900 and len(hist) == 6 and hist == sorted(hist)
    down, _ = adaptive_rating([(900, 0.0)] * 5, 900)
    assert down < 900
    assert adaptive_rating([(900, 0.5)] * 10, 900)[0] == 900
    assert adaptive_rating([(600, 0.0)] * 50, 600)[0] == 600  # clamped to the bot's range


def test_my_score():
    assert my_score("1-0", "white") == 1.0 and my_score("1-0", "black") == 0.0
    assert my_score("1/2-1/2", "black") == 0.5 and my_score(None, "white") is None


def test_targets_and_focus_rotation():
    report = {"weak_spots": [
        {"tag": "position: winning", "extra_per_game": 0.10},
        {"tag": "piece moved: rook", "extra_per_game": 0.05},   # not steerable
        {"tag": "mistake type: hung material", "extra_per_game": 0.05},
    ]}
    targets = targets_from_report(report)
    assert targets == [("position: winning", 1.0), ("mistake type: hung material", 0.5)]
    past: list = []
    for _ in range(6):
        past.append(next_focus(targets, past))
    assert past.count("position: winning") == 4 and past.count("mistake type: hung material") == 2
    assert next_focus([], []) is None


def test_position_tags():
    # Black to move, a knight attacked by a pawn: piece under attack.
    b = chess.Board("4k3/8/2n5/3P4/8/8/8/4K3 b - - 0 30")
    tags = position_tags(b, None)
    assert "situation: a piece under attack" in tags and "phase: endgame" in tags
    # Mid-exchange: White just took a knight with a defended pawn... I can recapture: not "behind".
    b = chess.Board("4k3/8/2P5/1p6/8/8/8/4K3 b - - 0 30")
    assert "material: even" in position_tags(b, None)
    # A free queen to take counts as material to win.
    b = chess.Board("4k3/8/8/3q4/8/8/8/3RK3 w - - 0 1")
    assert "mistake type: missed material" in position_tags(b, None)
    assert "position: winning" in position_tags(chess.Board(), 0.7)
    assert "position: losing" in position_tags(chess.Board(), 0.3)


def test_steering_picks_the_most_on_target_candidate():
    torch.manual_seed(0)
    model = MoveNet(NetConfig(blocks=1, channels=8)).eval()
    bot = SparringBot(model, targets_fn=lambda: [("situation: in check", 1.0)], strength=50.0,
                      think_time=(0, 0), seed=0, temperature=1.0, top_p=1.0)
    # White can give check (Qe2+, Qh5+...) — "in check" for me (Black) is the target.
    board = chess.Board("4k3/8/8/8/8/8/8/3QK3 w - - 0 1")
    for _ in range(5):
        move = bot.choose_move(board, 1000)
        after = board.copy()
        after.push(move)
        assert after.is_check()
    bot.strength = 0.0
    bot.focus = None
    assert any(not _gives_check(board, bot.choose_move(board, 1000)) for _ in range(30))


def _gives_check(board, move):
    b = board.copy()
    b.push(move)
    return b.is_check()


def test_focus_shares():
    shares = focus_shares(None, ["e2e4", "e7e5", "d1h5", "b8c6"], "black", ["situation: a piece under attack"], None)
    # Black's turns: after 1.e4 (nothing attacked) and after 2.Qh5 (e5 pawn attacked, but pawns don't count)
    assert shares == {"situation: a piece under attack": 0.0}


def test_store_adds_new_columns_to_old_databases(tmp_path):
    db = tmp_path / "old.sqlite"
    with sqlite3.connect(db) as conn:
        conn.execute("""CREATE TABLE games (id INTEGER PRIMARY KEY AUTOINCREMENT, created_at TEXT NOT NULL,
            finished_at TEXT, player_color TEXT NOT NULL, opponent TEXT NOT NULL, opponent_elo INTEGER,
            moves TEXT NOT NULL DEFAULT '', status TEXT NOT NULL DEFAULT 'ongoing', result TEXT,
            termination TEXT, pgn TEXT)""")
        conn.execute("INSERT INTO games (created_at, player_color, opponent) VALUES ('x', 'white', 'bot')")
    store = GameStore(db)
    g = store.get(1)
    assert g.mode == "normal" and g.start_fen is None


def test_drill_game_from_my_position(tmp_path):
    report = {
        "weak_spots": [{"tag": "position: winning", "extra_per_game": 0.1}],
        "drills": {"position: winning": [{"fen": DRILL_FEN, "wp_before": 0.82}]},
    }
    path = tmp_path / "weakness.json"
    path.write_text(json.dumps(report))
    svc = GameService(GameStore(":memory:"), {"bot": ScriptedOpponent(), "sparring": ScriptedOpponent()})
    sp = SparringService(svc, path)
    record, _ = sp.new_drill("position: winning")
    assert record.start_fen == DRILL_FEN and record.player_color == "black" and record.moves == []
    assert record.mode == "drill" and record.start_wp == 0.82
    svc.play(record.id, "c7c5")          # I'm to move first, the bot answers
    assert len(svc.get(record.id).moves) == 2
    g = svc.resign(record.id)
    assert '[FEN "' + DRILL_FEN + '"]' in g.pgn and "Mode" in g.pgn
    status = sp.status()
    assert status["drills"][0] == {"tag": "position: winning", "positions": 1, "played": 1,
                                   "score": 0.0, "expected": 0.82}
    with pytest.raises(ValueError):
        sp.new_drill("phase: endgame")


def test_sparring_game_uses_adaptive_rating_and_rotates_focus(tmp_path, monkeypatch):
    monkeypatch.setattr("r64.sparring_service.current_rating", lambda: 900)
    path = tmp_path / "weakness.json"
    path.write_text(json.dumps({"weak_spots": [{"tag": "position: winning", "extra_per_game": 0.1},
                                               {"tag": "material: behind", "extra_per_game": 0.1}]}))
    svc = GameService(GameStore(":memory:"), {"sparring": ScriptedOpponent(), "bot": ScriptedOpponent()})
    sp = SparringService(svc, path)
    g1 = sp.new_sparring_game("white")
    assert (g1.mode, g1.opponent_elo, g1.drill_tag) == ("sparring", 900, "position: winning")
    svc.resign(g1.id)                     # a loss: the partner gets easier
    g2 = sp.new_sparring_game("white")
    assert g2.opponent_elo < 900 and g2.drill_tag == "material: behind"
    g3 = sp.new_sparring_game("black", focus="position: winning")
    assert g3.drill_tag == "position: winning"
    with pytest.raises(ValueError):
        sp.new_sparring_game("white", focus="piece moved: rook")
    assert pick_drill([], set()) is None
