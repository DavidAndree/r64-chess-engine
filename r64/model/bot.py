"""The human-like bot: the trained move-prediction network as an opponent.

For a position and a rating, the network gives a probability for every legal move — "how likely
is a player of this rating to play it?". The bot **samples** from that distribution instead of
always taking the top move, so it makes the same kinds of mistakes real players make, and
doesn't play the same game twice.

    temperature  < 1 sharpens the distribution (fewer odd moves), > 1 flattens it
    top_p        drop the unlikely tail: keep the most likely moves covering this much probability

The rating input alone changes the network's strength only a little (bot 800 vs bot 2000 scored
38%), so by default the temperature also follows the rating: low ratings sample loosely (more
of the mistakes people at that level make), high ratings mostly play the most likely move.
"""

from __future__ import annotations

import random
import threading
import time
from pathlib import Path

import chess
import numpy as np
import torch

from r64.data.encoding import encode_position, index_to_move, legal_move_mask, planes_from_arrays
from r64.model.network import MoveNet, features, load

BOT_MIN_ELO = 600   # the training data has few games outside this range
BOT_MAX_ELO = 2200
CLOCK_SECONDS = 600.0  # the web UI has no clock: play as in a 10-minute game with time to spare
TEMP_AT_MIN, TEMP_AT_MAX = 1.0, 0.1


def temperature_for(elo: float) -> float:
    """Default sampling temperature for a rating: linear from 1.0 at 600 to 0.1 at 2200."""
    x = (min(max(elo, BOT_MIN_ELO), BOT_MAX_ELO) - BOT_MIN_ELO) / (BOT_MAX_ELO - BOT_MIN_ELO)
    return TEMP_AT_MIN + x * (TEMP_AT_MAX - TEMP_AT_MIN)


class HumanBot:
    name = "bot"
    label = "Human-like bot"
    min_elo = BOT_MIN_ELO
    max_elo = BOT_MAX_ELO

    def __init__(
        self,
        checkpoint: Path | str | MoveNet,
        temperature: float | None = None,  # None: follow the rating (temperature_for)
        top_p: float = 0.95,
        think_time: tuple[float, float] = (0.4, 1.2),
        seed: int | None = None,
    ):
        self.temperature = temperature
        self.top_p = top_p
        self.think_time = think_time
        self._rng = random.Random(seed)
        self._lock = threading.Lock()
        self._model: MoveNet | None = checkpoint if isinstance(checkpoint, MoveNet) else None
        self._checkpoint = None if self._model is not None else Path(checkpoint)
        if self._model is None:
            # Load in the background so the server starts right away.
            threading.Thread(target=self._get_model, daemon=True).start()

    def _get_model(self) -> MoveNet:
        with self._lock:
            if self._model is None:
                self._model, _ = load(self._checkpoint, "cpu")  # one position at a time: CPU is fastest
            return self._model

    def move_probabilities(self, board: chess.Board, elo: int | None) -> dict[chess.Move, float]:
        """Probability of every legal move for a player of this rating."""
        model = self._get_model()
        elo = float(max(BOT_MIN_ELO, min(BOT_MAX_ELO, elo or 1500)))
        pieces, castling, ep = encode_position(board)
        planes = planes_from_arrays(
            pieces[None], np.array([castling], np.uint8), np.array([ep], np.int8)
        )
        t = lambda v: torch.tensor([v], dtype=torch.float32)  # noqa: E731
        feats = features(t(elo), t(elo), t(CLOCK_SECONDS), t(CLOCK_SECONDS))
        mask = torch.from_numpy(legal_move_mask(board))
        with self._lock, torch.inference_mode():
            logits, _ = model(torch.from_numpy(planes), feats)
        logits = logits[0].masked_fill(~mask, float("-inf"))
        probs = torch.softmax(logits, 0)
        idx = torch.nonzero(mask).flatten().tolist()
        return {index_to_move(i, board): float(probs[i]) for i in idx}

    def sample(self, probs: dict[chess.Move, float], temperature: float) -> chess.Move:
        moves = sorted(probs, key=probs.get, reverse=True)
        p = np.array([probs[m] for m in moves], dtype=np.float64)
        keep = int(np.searchsorted(np.cumsum(p), self.top_p) + 1)  # always at least the top move
        moves, p = moves[:keep], p[:keep]
        if temperature <= 0:
            return moves[0]
        p = p ** (1.0 / temperature)
        return self._rng.choices(moves, weights=p.tolist())[0]

    def choose_move(self, board: chess.Board, elo: int | None) -> chess.Move:
        start = time.monotonic()
        temp = self.temperature if self.temperature is not None else temperature_for(elo or 1500)
        move = self.sample(self.move_probabilities(board, elo), temp)
        lo, hi = self.think_time
        pause = self._rng.uniform(lo, hi) - (time.monotonic() - start)
        if pause > 0:
            time.sleep(pause)  # instant replies feel robotic
        return move

    def close(self) -> None:
        pass


def personal_bot(version) -> HumanBot:
    """A personal model version as an opponent. It plays at my rating when it was trained and
    samples at temperature 1 — the move distribution it learned from me, mistakes included."""
    bot = HumanBot(version.model_path, temperature=1.0)
    bot.name = version.key
    bot.label = version.label
    bot.min_elo = bot.max_elo = int(version.meta["rating"])
    return bot
