"""Project paths and runtime settings."""

import os
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"          # downloaded databases, encoded shards (git-ignored)
CHECKPOINT_DIR = ROOT / "checkpoints"  # trained models (git-ignored)
# Played games and model versions. Override with R64_DB to keep test games out of your history.
DB_PATH = Path(os.environ.get("R64_DB", DATA_DIR / "r64.sqlite"))
# The network behind the human-like bot.
BOT_CHECKPOINT = Path(os.environ.get("R64_BOT", CHECKPOINT_DIR / "base" / "best.pt"))


def device() -> "torch.device":
    """Best available device: Apple GPU (MPS), then CUDA, then CPU."""
    import torch  # imported here so the web server starts fast

    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")
