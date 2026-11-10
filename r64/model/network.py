"""The move-prediction network.

A residual CNN over the 8x8 board (AlphaZero / Maia style), built from standard PyTorch layers.
It is **conditioned on rating and clock**: a small MLP turns those numbers into a per-channel
scale and shift (FiLM) applied in every residual block, so one network can play like a
900-rated or a 1900-rated player.

Outputs:
    policy  logits over NUM_MOVES (4168) — "which move would this player make?"
    value   logits over (win, draw, loss) for the player to move — used later for steering
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import nn

from r64.data.encoding import NUM_MOVES, NUM_PLANES

NUM_FEATURES = 5


@dataclass
class NetConfig:
    blocks: int = 6
    channels: int = 64
    cond_hidden: int = 64


def features(elo, opp_elo, clock, base_time) -> torch.Tensor:
    """Normalize the conditioning numbers. Inputs: float tensors [B]; -1 means unknown."""
    elo = torch.where(elo < 0, torch.full_like(elo, 1500.0), elo)
    opp_elo = torch.where(opp_elo < 0, elo, opp_elo)
    known_clock = (clock >= 0) & (base_time > 0)
    frac = torch.where(known_clock, clock / base_time.clamp(min=1), torch.ones_like(clock))
    log_clock = torch.where(clock >= 0, torch.log1p(clock) / 8.0, torch.full_like(clock, 0.7))
    log_base = torch.where(base_time > 0, torch.log1p(base_time) / 8.0, torch.full_like(clock, 0.7))
    return torch.stack(
        [(elo - 1500) / 500, (opp_elo - elo) / 500, frac.clamp(0, 2), log_clock, log_base], dim=1
    )


class FiLMBlock(nn.Module):
    def __init__(self, channels: int, cond_hidden: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.film = nn.Linear(cond_hidden, 2 * channels)
        nn.init.zeros_(self.film.weight)  # start as a plain residual block
        nn.init.zeros_(self.film.bias)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        y = torch.relu(self.bn1(self.conv1(x)))
        y = self.bn2(self.conv2(y))
        gamma, beta = self.film(cond).chunk(2, dim=1)
        y = y * (1 + gamma[:, :, None, None]) + beta[:, :, None, None]
        return torch.relu(x + y)


class MoveNet(nn.Module):
    def __init__(self, cfg: NetConfig | None = None):
        super().__init__()
        cfg = cfg or NetConfig()
        self.cfg = cfg
        c = cfg.channels
        self.cond = nn.Sequential(
            nn.Linear(NUM_FEATURES, cfg.cond_hidden), nn.ReLU(),
            nn.Linear(cfg.cond_hidden, cfg.cond_hidden), nn.ReLU(),
        )
        self.stem = nn.Sequential(
            nn.Conv2d(NUM_PLANES, c, 3, padding=1, bias=False), nn.BatchNorm2d(c), nn.ReLU()
        )
        self.blocks = nn.ModuleList(FiLMBlock(c, cfg.cond_hidden) for _ in range(cfg.blocks))
        # Policy: for every from-square, 64 to-square logits (+9 under-promotion channels).
        self.policy = nn.Sequential(
            nn.Conv2d(c, c, 3, padding=1, bias=False), nn.BatchNorm2d(c), nn.ReLU(),
            nn.Conv2d(c, 64 + 9, 1),
        )
        self.value = nn.Sequential(
            nn.Conv2d(c, 4, 1, bias=False), nn.BatchNorm2d(4), nn.ReLU(), nn.Flatten(),
            nn.Linear(4 * 64 + cfg.cond_hidden, 128), nn.ReLU(), nn.Linear(128, 3),
        )

    def forward(self, planes: torch.Tensor, feats: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        cond = self.cond(feats)
        x = self.stem(planes)
        for block in self.blocks:
            x = block(x, cond)

        p = self.policy(x)                                   # [B, 73, 8(rank), 8(file)]
        b = p.shape[0]
        main = p[:, :64].reshape(b, 64, 64).transpose(1, 2)  # [B, from, to]
        under = p[:, 64:, 6, :].reshape(b, 3, 3, 8).permute(0, 1, 3, 2)  # [B, piece, file, dir]
        policy = torch.cat([main.reshape(b, 4096), under.reshape(b, 72)], dim=1)

        v_in = torch.cat([self.value[:4](x), cond], dim=1)
        value = self.value[4:](v_in)
        return policy, value

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


assert NUM_MOVES == 4096 + 72


def save(model: MoveNet, path, **extra) -> None:
    torch.save({"config": asdict(model.cfg), "state": model.state_dict(), **extra}, path)


def load(path, device="cpu") -> tuple[MoveNet, dict]:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = MoveNet(NetConfig(**ckpt["config"])).to(device)
    model.load_state_dict(ckpt["state"])
    model.eval()
    return model, ckpt
