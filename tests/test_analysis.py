import shutil

import chess
import chess.engine
import pytest

from r64.analysis import store
from r64.analysis.analyze import MATE, analyze_moves, read_my_games
from r64.analysis.tags import describe, is_fork, severity, win_prob
from r64.analysis.weakness import build_report

FOOLS_MATE = ["f2f3", "e7e5", "g2g4", "d8h4"]


def test_win_prob_and_severity():
    assert win_prob(0) == pytest.approx(0.5)
    assert win_prob(300) == pytest.approx(1 - win_prob(-300))
    assert win_prob(MATE - 3) == 1.0 and win_prob(-(MATE - 3)) == 0.0
    assert [severity(x) for x in (0.02, 0.06, 0.12, 0.4)] == [None, "inaccuracy", "mistake", "blunder"]


def test_hung_queen():
    # Qd7+?? walks into the king and rook.
    f = describe("4k3/4r3/8/8/8/8/8/3QK3 w - - 0 1", "d1d7", "e1e2", 400, -900, "e7d7", None, -1)
    assert f.severity == "blunder" and f.cause == "hung material"


def test_missed_free_queen():
    f = describe("4k3/8/8/3q4/8/8/8/3RK3 w - - 0 1", "e1e2", "d1d5", 800, -400, None, None, -1)
    assert f.cause == "missed material"


def test_missed_fork():
    board = chess.Board("r3k3/8/8/1N6/8/8/8/4K3 w - - 0 1")
    assert is_fork(board, chess.Move.from_uci("b5c7"))
    f = describe(board.fen(), "e1e2", "b5c7", 400, -50, None, None, -1)
    assert f.cause == "missed fork"


def test_allowed_and_missed_mate():
    start = chess.Board()
    for m in FOOLS_MATE[:3]:
        start.push_uci(m)
    # Black has mate in 1 but plays something else.
    f = describe(start.fen(), "a7a6", "d8h4", -(MATE - 1), 0, None, None, -1)
    assert f.cause == "missed mate"
    # White's 2.g4 allows mate.
    b = chess.Board()
    b.push_uci("f2f3")
    b.push_uci("e7e5")
    f = describe(b.fen(), "g2g4", "b1c3", -50, -(MATE - 1), "d8h4", None, -1)
    assert f.cause == "allowed mate"


def test_position_tags():
    tags = describe(chess.STARTING_FEN, "e2e4", "e2e4", 30, 30, None, 20.0, 600).tags
    assert "phase: opening" in tags and "clock: time trouble (<10% or <30s)" in tags
    assert "piece moved: pawn" in tags and "material: even" in tags
    tags = describe("8/5k2/8/3p4/3P4/8/5K2/8 w - - 0 50", "f2e3", "f2e3", 0, 0, None, 400.0, 600).tags
    assert "phase: endgame" in tags and "endgame type: pawn endgame" in tags
    assert "clock: enough time" in tags


class FakeEngine:
    """Material count as the evaluation; first legal move as best move."""

    def analyse(self, board, limit):
        for move in board.legal_moves:  # sees mate in 1
            board.push(move)
            mate = board.is_checkmate()
            board.pop()
            if mate:
                return {"score": chess.engine.PovScore(chess.engine.Mate(1), board.turn), "pv": [move]}
        score = sum(len(board.pieces(pt, chess.WHITE)) - len(board.pieces(pt, chess.BLACK))
                    for pt in range(1, 6)) * 100
        return {"score": chess.engine.PovScore(chess.engine.Cp(score), chess.WHITE),
                "pv": [next(iter(board.legal_moves))]}


PGN = """[Event "Live Chess"]
[Site "Chess.com"]
[White "Me_Online"]
[Black "rival"]
[Result "0-1"]
[WhiteElo "900"]
[BlackElo "910"]
[TimeControl "600"]
[UTCDate "2026.10.01"]
[UTCTime "10:00:00"]
[Link "https://www.chess.com/game/live/1"]

1. f3 {[%clk 0:09:58]} 1... e5 {[%clk 0:09:57]} 2. g4 {[%clk 0:09:50]} 2... Qh4# {[%clk 0:09:55]} 0-1

[Event "Live Chess"]
[Site "Chess.com"]
[White "someone"]
[Black "Me_Online"]
[Result "1-0"]
[UTCDate "2026.10.02"]
[TimeControl "180"]
[Link "https://www.chess.com/game/live/2"]

1. e4 e5 2. Qh5 Nc6 3. Bc4 Nf6 4. Qxf7# 1-0
"""


def test_read_analyze_store_and_report(tmp_path):
    path = tmp_path / "chesscom_me_online.pgn"
    path.write_text(PGN)
    games = list(read_my_games([path], {"me_online"}))
    assert [(g.my_color, g.result, g.base_time) for g in games] == [("white", "loss", 600), ("black", "loss", 180)]
    assert games[0].clocks[0] == 598.0

    db = tmp_path / "a.sqlite"
    conn = store.connect(db)
    for g in games:
        positions = analyze_moves(g.moves, 100, engine=FakeEngine())
        assert positions[-1]["cp"] == (-(MATE) if g.my_color == "white" else MATE)  # checkmate
        for p in positions:
            p["move"] = g.moves[p["ply"]] if p["ply"] < len(g.moves) else None
            p["clock"] = None
        store.save_game(conn, {
            "key": g.key, "source": g.source, "played_at": g.played_at, "my_color": g.my_color,
            "my_elo": g.my_elo, "opp_elo": g.opp_elo, "result": g.result, "base_time": g.base_time,
            "increment": g.increment, "eco": g.eco, "plies": len(g.moves), "nodes": 100,
            "analyzed_at": "now"}, positions)
    assert store.analyzed_keys(conn) == {"https://www.chess.com/game/live/1", "https://www.chess.com/game/live/2"}
    conn.close()

    r = build_report(recent=10, db=db)
    assert r["games"] == 2 and r["me"]["moves"] == 2 + 3 and r["peers"]["moves"] == 2 + 4
    assert r["me"]["blunders_per_100"] > 0  # 2.g4 and 3...Nf6 walk into mate
    assert any(c["cause"] == "allowed mate" and c["me_per_100"] > 0 for c in r["causes"])


@pytest.mark.skipif(shutil.which("stockfish") is None, reason="Stockfish not installed")
def test_stockfish_finds_mate_in_fools_mate():
    eng = chess.engine.SimpleEngine.popen_uci(shutil.which("stockfish"))
    try:
        positions = analyze_moves(FOOLS_MATE, 2000, engine=eng)
    finally:
        eng.quit()
    assert positions[3]["best"] == "d8h4" and positions[3]["cp"] == -(MATE - 1)
    assert positions[4]["cp"] == -MATE and positions[4]["best"] is None


def test_weakness_api_and_page(tmp_path):
    from fastapi.testclient import TestClient

    from r64.analysis.service import WeaknessService
    from r64.server.app import create_app
    from r64.storage import GameStore

    app = create_app(make_store=lambda: GameStore(":memory:"), make_opponents=dict,
                     make_trainer=lambda s: None, make_weakness=lambda: WeaknessService(tmp_path))
    with TestClient(app) as client:
        assert client.get("/weakness").status_code == 200
        r = client.get("/api/weakness?recent=50").json()
        assert r["games"] == 0 and r["me"] == {"moves": 0} and r["analyzed_total"] == 0
        s = client.get("/api/weakness/status").json()
        assert s == {"running": False, "error": None, "analyzed": 0, "last_added": None}
