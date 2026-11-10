"""The personal model: the base model fine-tuned on my own games, in numbered versions.

    checkpoints/me/v001/model.pt    the network
    checkpoints/me/v001/meta.json   when, how many games, my rating then, how well it predicts me

Every version is kept, so I can play against "me, 6 months ago". A new version is trained:
- by hand:        uv run python -m r64.model personal
- automatically:  by the server, after every RETRAIN_EVERY new games played in r64.

Each version is always fine-tuned from the *base* model on *all* my games so far (a few minutes),
rather than from the previous version, so versions don't drift and are directly comparable.
"""

from __future__ import annotations

import json
import shutil
import threading
import time
import traceback
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from r64 import config
from r64.data import my_games

PERSONAL_DIR = config.CHECKPOINT_DIR / "me"
MY_POSITIONS = config.DATA_DIR / "positions" / "me"
RETRAIN_EVERY = 20  # new r64 games

# Fine-tuning recipe (chosen with a learning-rate sweep on my held-out games).
EPOCHS = 4.0
LR = 1e-4
BATCH_SIZE = 256


@dataclass
class Version:
    number: int
    dir: Path
    meta: dict

    @property
    def key(self) -> str:
        return f"me-v{self.number}"

    @property
    def model_path(self) -> Path:
        return self.dir / "model.pt"

    @property
    def label(self) -> str:
        day = datetime.fromisoformat(self.meta["created_at"]).strftime("%b %-d")
        return f"Me v{self.number} · {day} · {self.meta['games']:,} games"


def versions(root: Path = PERSONAL_DIR) -> list[Version]:
    out = []
    for d in sorted(root.glob("v[0-9][0-9][0-9]")):
        meta_path = d / "meta.json"
        if meta_path.exists() and (d / "model.pt").exists():
            out.append(Version(int(d.name[1:]), d, json.loads(meta_path.read_text())))
    return out


def latest(root: Path = PERSONAL_DIR) -> Version | None:
    vs = versions(root)
    return vs[-1] if vs else None


def compare_on_my_games(model_paths: dict[str, Path], positions: Path = MY_POSITIONS) -> dict[str, float]:
    """Move-match accuracy of each model on my held-out games (the same positions for all)."""
    from r64.data.dataset import PositionSet
    from r64.model.network import load
    from r64.model.train import EvalSet, predict_eval

    device = config.device()
    ev = EvalSet(PositionSet.from_dir(positions, "val"), 50_000, device, seed=0)
    out = {}
    for name, path in model_paths.items():
        model, _ = load(path, device)
        out[name] = float(predict_eval(model, ev)[0].mean())
    return out


def train_version(
    base: Path = config.BOT_CHECKPOINT,
    root: Path = PERSONAL_DIR,
    positions: Path = MY_POSITIONS,
    fetch: bool = False,
    log: Callable[[str], None] = print,
) -> Version:
    """Collect my games, fine-tune the base model on them, and save the next version."""
    from r64.model.train import TrainConfig, train

    profile = my_games.load_profile()
    if fetch:
        if profile.get("chesscom"):
            my_games.fetch_chesscom(profile["chesscom"])
        if profile.get("lichess"):
            my_games.fetch_lichess(profile["lichess"])
    rating = my_games.current_rating(profile)
    manifest = my_games.encode_mine(positions, progress=False)
    games = manifest["train"]["games"] + manifest["val"]["games"]
    log(f"Fine-tuning on {manifest['train']['positions']:,} of my moves ({games:,} games)")

    prev = latest(root)
    number = prev.number + 1 if prev else 1
    work = root / f"v{number:03d}.training"
    shutil.rmtree(work, ignore_errors=True)
    start = time.time()
    train(TrainConfig(
        data=[str(positions)], name=f"me-v{number}", epochs=EPOCHS, batch_size=BATCH_SIZE, lr=LR,
        warmup_steps=50, eval_every=10**9, eval_positions=50_000, init_from=str(base),
        out_dir=str(work),
    ))
    # The final weights after a fixed recipe — not the best checkpoint on my held-out games —
    # so the reported accuracy is an honest estimate.
    (work / "latest.pt").rename(work / "model.pt")
    (work / "best.pt").unlink(missing_ok=True)

    acc = compare_on_my_games({"base": Path(base), "me": work / "model.pt"}, positions)
    meta = {
        "version": number,
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "games": games,
        "r64_games": manifest["r64_games"],
        "train_positions": manifest["train"]["positions"],
        "val_positions": manifest["val"]["positions"],
        "rating": rating,
        "val_acc": acc["me"],
        "base_val_acc": acc["base"],
        "minutes": round((time.time() - start) / 60, 1),
        "base": str(base),
    }
    (work / "meta.json").write_text(json.dumps(meta, indent=2))
    final = root / f"v{number:03d}"
    work.rename(final)
    log(f"Saved {final}: predicts {acc['me']:.2%} of my moves (base model {acc['base']:.2%})")
    return Version(number, final, meta)


def learning_curve(
    sizes: list[int] | None = None,
    base: Path = config.BOT_CHECKPOINT,
    positions: Path = MY_POSITIONS,
    seed: int = 0,
) -> list[dict]:
    """Fine-tune on growing random subsets of my games and measure each on the same held-out games.
    If the personal model "learns more the more I play", accuracy rises with the number of games."""
    from r64.data.dataset import PositionSet
    from r64.model.train import TrainConfig, train

    n_games = int(PositionSet.from_dir(positions, "train").game_ids()[-1]) + 1
    sizes = sorted({min(s, n_games) for s in (sizes or [100, 250, 500, 1000, 2000, n_games])})
    work = config.CHECKPOINT_DIR / "me-curve"
    rows = []
    for n in sizes:
        out = work / f"{n}"
        train(TrainConfig(
            data=[str(positions)], name=f"curve-{n}", epochs=EPOCHS, batch_size=BATCH_SIZE, lr=LR,
            warmup_steps=min(50, max(1, n // 10)), eval_every=10**9, eval_positions=1000,
            init_from=str(base), out_dir=str(out), train_games=n, seed=seed,
        ))
        acc = compare_on_my_games({"me": out / "latest.pt"}, positions)["me"]
        rows.append({"games": n, "acc": acc})
        print(f"\n== {n:,} games → {acc:.2%}\n", flush=True)
    base_acc = compare_on_my_games({"base": Path(base)}, positions)["base"]
    print(f"Base model (0 of my games): {base_acc:.2%}")
    for r in rows:
        print(f"  {r['games']:>5,} games: {r['acc']:.2%}  ({(r['acc'] - base_acc) * 100:+.2f} points)")
    result = [{"games": 0, "acc": base_acc}, *rows]
    (work / "curve.json").write_text(json.dumps(result, indent=2))
    return result


class AutoTrainer:
    """Retrains the personal model in the background after every `every` new r64 games."""

    def __init__(
        self,
        count_finished: Callable[[], int],
        on_version: Callable[[Version], None],
        every: int = RETRAIN_EVERY,
        train_fn: Callable[[], Version] = train_version,
        root: Path = PERSONAL_DIR,
    ):
        self._count_finished = count_finished
        self._on_version = on_version
        self._train_fn = train_fn
        self._root = root
        self.every = every
        self.training = False
        self.error: str | None = None
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None

    def games_since_last(self) -> int:
        v = latest(self._root)
        return self._count_finished() - (v.meta.get("r64_games", 0) if v else 0)

    def status(self) -> dict:
        vs = versions(self._root)
        return {
            "training": self.training,
            "error": self.error,
            "every": self.every,
            "games_since_last": self.games_since_last(),
            "versions": [{"key": v.key, "label": v.label, **v.meta} for v in reversed(vs)],
        }

    def game_finished(self, _record=None) -> bool:
        """Call after each finished game. Starts training when enough new games piled up."""
        if self.games_since_last() >= self.every:
            return self.start()
        return False

    def start(self) -> bool:
        with self._lock:
            if self.training:
                return False
            self.training, self.error = True, None
            self._thread = threading.Thread(target=self._run, daemon=True)
            self._thread.start()
            return True

    def wait(self, timeout: float | None = None) -> None:
        if self._thread:
            self._thread.join(timeout)

    def _run(self) -> None:
        try:
            self._on_version(self._train_fn())
        except Exception as e:  # noqa: BLE001 - reported in the UI
            traceback.print_exc()
            self.error = f"{type(e).__name__}: {e}"
        finally:
            self.training = False
