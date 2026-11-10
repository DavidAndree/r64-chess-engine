import json
import sqlite3

import chess
import numpy as np
import torch

from r64.data import my_games
from r64.data.dataset import PositionSet
from r64.game import GameService
from r64.model import personal
from r64.model.bot import personal_bot
from r64.model.network import MoveNet, NetConfig, save
from r64.storage import GameStore
from tests.test_game import ScriptedOpponent


def _fake_version(root, number, r64_games=0, rating=900):
    d = root / f"v{number:03d}"
    d.mkdir(parents=True)
    torch.manual_seed(0)
    save(MoveNet(NetConfig(blocks=1, channels=8)), d / "model.pt")
    meta = {"version": number, "created_at": "2026-11-10T12:00:00+00:00", "games": 3000 + number,
            "r64_games": r64_games, "rating": rating, "val_acc": 0.46, "base_val_acc": 0.42}
    (d / "meta.json").write_text(json.dumps(meta))
    return personal.Version(number, d, meta)


def test_versions_skip_unfinished_training(tmp_path):
    _fake_version(tmp_path, 1)
    _fake_version(tmp_path, 2)
    (tmp_path / "v003.training").mkdir()
    vs = personal.versions(tmp_path)
    assert [v.number for v in vs] == [1, 2]
    assert personal.latest(tmp_path).key == "me-v2"
    assert vs[0].label == "Me v1 · Nov 10 · 3,001 games"


def test_personal_bot_plays_at_its_rating(tmp_path):
    bot = personal_bot(_fake_version(tmp_path, 1, rating=850))
    bot.think_time = (0, 0)
    assert (bot.name, bot.min_elo, bot.max_elo, bot.temperature) == ("me-v1", 850, 850, 1.0)
    board = chess.Board()
    assert bot.choose_move(board, 850) in board.legal_moves


def test_auto_trainer_retrains_after_every_n_games(tmp_path):
    finished = {"n": 0}
    added = []

    def fake_train():
        n = len(personal.versions(tmp_path)) + 1
        return _fake_version(tmp_path, n, r64_games=finished["n"])

    trainer = personal.AutoTrainer(lambda: finished["n"], added.append, every=3,
                                   train_fn=fake_train, root=tmp_path)
    for _ in range(2):
        finished["n"] += 1
        assert trainer.game_finished() is False
    finished["n"] += 1
    assert trainer.game_finished() is True
    trainer.wait()
    assert [v.key for v in added] == ["me-v1"]
    s = trainer.status()
    assert s["games_since_last"] == 0 and not s["training"] and s["versions"][0]["key"] == "me-v1"
    finished["n"] += 2
    assert trainer.game_finished() is False and trainer.status()["games_since_last"] == 2


def test_auto_trainer_reports_errors(tmp_path):
    def boom():
        raise FileNotFoundError("No games found")

    trainer = personal.AutoTrainer(lambda: 0, lambda v: None, train_fn=boom, root=tmp_path)
    assert trainer.start()
    trainer.wait()
    assert "No games found" in trainer.status()["error"] and not trainer.training


def test_finish_hook_runs_on_mate_and_resign():
    svc = GameService(GameStore(":memory:"), {"scripted": ScriptedOpponent(["f2f3", "g2g4"])})
    seen = []
    svc.on_finish.append(lambda r: seen.append((r.id, r.termination)))
    g = svc.new_game("black", "scripted", None)
    svc.play(g.id, "e7e5")
    svc.play(g.id, "d8h4")
    g2 = svc.new_game("white", "scripted", None)
    svc.resign(g2.id)
    assert seen == [(g.id, "checkmate"), (g2.id, "resignation")]
    assert svc.store.count_finished() == 2


def test_r64_games_exported_with_ratings(tmp_path):
    db = tmp_path / "r64.sqlite"
    store = GameStore(db)
    svc = GameService(store, {"scripted": ScriptedOpponent(["f2f3", "g2g4"])})
    g = svc.new_game("black", "scripted", 1400)
    svc.play(g.id, "e7e5")
    svc.play(g.id, "d8h4")
    store.close()
    out = tmp_path / "r64.pgn"
    assert my_games.export_r64_games(db, out, my_rating=870) == 1
    text = out.read_text()
    assert '[BlackElo "870"]' in text and '[WhiteElo "1400"]' in text and '[TimeControl "600"]' in text
    with sqlite3.connect(db) as conn:  # the stored PGN is untouched
        assert "Elo" not in conn.execute("SELECT pgn FROM games").fetchone()[0]


def test_current_rating_uses_my_latest_games(tmp_path, monkeypatch):
    monkeypatch.setattr(my_games, "MY_GAMES_DIR", tmp_path)
    games = []
    for day, elo in [(1, 500), (2, 600), (3, 700), (4, 800), (5, 900)]:
        games.append(f'[Event "Live Chess"]\n[White "Me_Online"]\n[Black "x"]\n'
                     f'[WhiteElo "{elo}"]\n[BlackElo "1000"]\n[UTCDate "2026.10.0{day}"]\n\n1. e4 *\n')
    (tmp_path / "chesscom_me.pgn").write_text("\n".join(reversed(games)))
    profile = {"chesscom": "me_online", "lichess": None}
    assert my_games.current_rating(profile, recent=3) == 800
    assert my_games.current_rating({**profile, "rating": 1234}) == 1234
    assert my_games.current_rating({"chesscom": None, "lichess": None}) == 1500


def test_game_ids_follow_ply_restarts():
    ps = PositionSet.__new__(PositionSet)
    ps.arrays = {"ply": np.array([0, 2, 4, 1, 3, 0, 0, 2], np.int16)}
    assert ps.game_ids().tolist() == [0, 0, 0, 1, 1, 2, 3, 3]
