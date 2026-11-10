"""Smoke tests: dependencies install and work together."""

import chess
import torch

import r64
from r64 import config


def test_package_imports():
    assert r64.__version__


def test_chess_rules_available():
    board = chess.Board()
    assert board.legal_moves.count() == 20


def test_torch_runs_on_best_device():
    dev = config.device()
    x = torch.ones(8, 8, device=dev)
    assert (x * 2).sum().item() == 128.0


def test_paths_point_into_repo():
    assert config.DATA_DIR.parent == config.ROOT
    assert (config.ROOT / "pyproject.toml").exists()
