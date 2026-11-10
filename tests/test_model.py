import chess
import numpy as np
import torch
import torch.nn.functional as F

from r64.data.encoding import (
    NUM_MOVES,
    board_from_arrays,
    encode_position,
    legal_move_mask,
    move_to_index,
    planes_from_arrays,
)
from r64.model.network import MoveNet, NetConfig, features


def _planes(board):
    pieces, castling, ep = encode_position(board)
    return torch.from_numpy(
        planes_from_arrays(pieces[None], np.array([castling], np.uint8), np.array([ep], np.int8))
    )


def _feats(n=1, elo=1500.0):
    t = torch.full((n,), elo)
    return features(t, t, torch.full((n,), 100.0), torch.full((n,), 180.0))


def test_output_shapes():
    model = MoveNet(NetConfig(blocks=2, channels=16))
    planes = torch.zeros(4, 17, 8, 8)
    policy, value = model(planes, _feats(4))
    assert policy.shape == (4, NUM_MOVES) and value.shape == (4, 3)


def test_features_handle_unknowns():
    f = features(torch.tensor([-1.0]), torch.tensor([-1.0]), torch.tensor([-1.0]), torch.tensor([-1.0]))
    assert torch.isfinite(f).all() and f[0, 0] == 0.0


def test_board_from_arrays_gives_same_legal_moves():
    for fen in [
        chess.STARTING_FEN,
        "r3k2r/pppq1ppp/2n2n2/3pp3/1b1PP3/2N2N2/PPPQ1PPP/R3K2R b KQkq - 0 8",
        "4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 2",
        "1n2k3/P7/8/8/8/8/7p/4K1N1 b - - 0 1",
    ]:
        board = chess.Board(fen)
        rebuilt = board_from_arrays(*encode_position(board))
        assert np.array_equal(legal_move_mask(rebuilt), legal_move_mask(board))


def test_policy_head_layout_can_learn_specific_moves():
    """Overfit two positions whose targets live in different parts of the policy vector
    (a normal move and an under-promotion). Proves the head's index mapping is consistent."""
    torch.manual_seed(0)
    b1 = chess.Board()
    b2 = chess.Board("1n2k3/P7/8/8/8/8/7p/4K1N1 w - - 0 1")
    targets = torch.tensor([
        move_to_index(chess.Move.from_uci("g1f3"), chess.WHITE),
        move_to_index(chess.Move.from_uci("a7b8n"), chess.WHITE),  # capture + under-promote
    ])
    planes = torch.cat([_planes(b1), _planes(b2)])
    model = MoveNet(NetConfig(blocks=1, channels=16))
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    for _ in range(150):
        policy, _ = model(planes, _feats(2))
        loss = F.cross_entropy(policy, targets)
        opt.zero_grad()
        loss.backward()
        opt.step()
    model.eval()
    assert torch.equal(model(planes, _feats(2))[0].argmax(1), targets)
