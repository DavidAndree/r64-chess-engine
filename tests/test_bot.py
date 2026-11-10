import chess
import pytest
import torch

from r64.game import GameService
from r64.model.bot import BOT_MAX_ELO, BOT_MIN_ELO, HumanBot, temperature_for
from r64.model.match import play_game
from r64.model.network import MoveNet, NetConfig
from r64.storage import GameStore


@pytest.fixture
def bot():
    torch.manual_seed(0)
    return HumanBot(MoveNet(NetConfig(blocks=1, channels=16)).eval(), think_time=(0, 0), seed=1)


@pytest.mark.parametrize("fen", [
    chess.STARTING_FEN,
    "r3k2r/pppq1ppp/2n2n2/3pp3/1b1PP3/2N2N2/PPPQ1PPP/R3K2R b KQkq - 0 8",
    "1n2k3/P7/8/8/8/8/7p/4K1N1 b - - 0 1",  # black to move, can promote
])
def test_probabilities_cover_exactly_the_legal_moves(bot, fen):
    board = chess.Board(fen)
    probs = bot.move_probabilities(board, 1200)
    assert set(probs) == set(board.legal_moves)
    assert sum(probs.values()) == pytest.approx(1.0, abs=1e-4)


def test_sampling_respects_temperature_and_top_p(bot):
    a, b, c = (chess.Move.from_uci(u) for u in ("e2e4", "d2d4", "a2a3"))
    probs = {a: 0.6, b: 0.38, c: 0.02}
    assert {bot.sample(probs, 0) for _ in range(20)} == {a}
    seen = {bot.sample(probs, 1.0) for _ in range(300)}
    assert seen == {a, b}  # a2a3 is outside the top 95%


def test_temperature_follows_rating():
    assert temperature_for(BOT_MIN_ELO) == pytest.approx(1.0)
    assert temperature_for(BOT_MAX_ELO) == pytest.approx(0.1)
    assert temperature_for(100) == temperature_for(BOT_MIN_ELO)
    assert temperature_for(1000) > temperature_for(1800)


def test_bot_plays_full_legal_games(bot):
    result, plies = play_game((bot, 800), (bot, 2000))
    assert result in ("1-0", "0-1", "1/2-1/2") and plies > 0


def test_bot_games_are_saved_and_rating_range_enforced(bot):
    svc = GameService(GameStore(":memory:"), {"bot": bot})
    g = svc.new_game("black", "bot", 900)
    assert len(g.moves) == 1 and svc.store.get(g.id).opponent == "bot"
    with pytest.raises(ValueError, match="600-2200"):
        svc.new_game("white", "bot", 2500)
