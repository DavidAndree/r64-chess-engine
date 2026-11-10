import shutil

import chess
import pytest
from fastapi.testclient import TestClient

from r64.game import GameOver, GameService, IllegalMove
from r64.opponents import StockfishOpponent
from r64.server.app import create_app
from r64.storage import GameStore


class ScriptedOpponent:
    """Plays a fixed list of moves, then the first legal move."""

    name = "scripted"
    label = "Scripted"
    min_elo, max_elo = 100, 3000

    def __init__(self, script: list[str] | None = None):
        self.script = list(script or [])

    def choose_move(self, board: chess.Board, elo: int | None) -> chess.Move:
        if self.script:
            return chess.Move.from_uci(self.script.pop(0))
        return next(iter(board.legal_moves))

    def close(self) -> None:
        pass


def make_service(script=None) -> GameService:
    return GameService(GameStore(":memory:"), {"scripted": ScriptedOpponent(script)})


def test_player_move_gets_reply_and_is_saved():
    svc = make_service(["e7e5"])
    g = svc.new_game("white", "scripted", None)
    assert g.moves == []
    g = svc.play(g.id, "e2e4")
    assert g.moves == ["e2e4", "e7e5"]
    assert svc.store.get(g.id).moves == ["e2e4", "e7e5"]


def test_opponent_moves_first_when_player_is_black():
    g = make_service(["d2d4"]).new_game("black", "scripted", None)
    assert g.moves == ["d2d4"]


def test_illegal_and_out_of_turn_moves_rejected():
    svc = make_service()
    g = svc.new_game("white", "scripted", None)
    with pytest.raises(IllegalMove):
        svc.play(g.id, "e2e5")
    with pytest.raises(IllegalMove):
        svc.play(g.id, "zzzz")
    black = svc.new_game("black", "scripted", None)
    svc.store.set_moves(black.id, [])  # pretend it's white to move
    with pytest.raises(IllegalMove):
        svc.play(black.id, "e2e4")


def test_checkmate_finishes_game_with_pgn():
    # Fool's mate: opponent (white) plays f3 and g4, I (black) mate with Qh4#.
    svc = make_service(["f2f3", "g2g4"])
    g = svc.new_game("black", "scripted", None)
    g = svc.play(g.id, "e7e5")
    g = svc.play(g.id, "d8h4")
    assert g.status == "finished"
    assert g.result == "0-1"
    assert g.termination == "checkmate"
    assert '[Black "me"]' in g.pgn and "Qh4#" in g.pgn
    with pytest.raises(GameOver):
        svc.play(g.id, "a7a6")


def test_resign():
    svc = make_service()
    g = svc.resign(svc.new_game("white", "scripted", None).id)
    assert (g.status, g.result, g.termination) == ("finished", "0-1", "resignation")


def test_api_flow():
    app = create_app(
        make_store=lambda: GameStore(":memory:"),
        make_opponents=lambda: {"scripted": ScriptedOpponent(["e7e5"])},
        make_trainer=lambda service: None,
    )
    with TestClient(app) as client:
        g = client.post("/api/games", json={"color": "white", "opponent": "scripted", "elo": 1500}).json()
        assert g["turn"] == "white" and "e2" in g["dests"]
        g = client.post(f"/api/games/{g['id']}/move", json={"uci": "e2e4"}).json()
        assert g["san"] == ["e4", "e5"] and g["last_move"] == ["e7", "e5"]
        assert client.post(f"/api/games/{g['id']}/move", json={"uci": "e1e3"}).status_code == 400
        assert client.get(f"/api/games/{g['id']}/pgn").status_code == 409
        client.post(f"/api/games/{g['id']}/resign")
        assert "1-0" not in client.get(f"/api/games/{g['id']}/pgn").text
        assert client.get("/api/games/999").status_code == 404
        assert len(client.get("/api/games").json()) == 1
        assert client.get("/").status_code == 200
        assert client.get("/api/personal").json() == {"available": False}
        assert client.post("/api/personal/train").status_code == 409
        assert client.get("/api/opponents").json() == [
            {"name": "scripted", "label": "Scripted", "min_elo": 100, "max_elo": 3000}
        ]
        bad = client.post("/api/games", json={"color": "white", "opponent": "scripted", "elo": 50})
        assert bad.status_code == 400 and "100-3000" in bad.text


@pytest.mark.skipif(shutil.which("stockfish") is None, reason="Stockfish not installed")
def test_stockfish_plays_legal_move_at_limited_strength():
    sf = StockfishOpponent(think_time=0.05)
    try:
        board = chess.Board()
        move = sf.choose_move(board, 1400)
        assert move in board.legal_moves
    finally:
        sf.close()
