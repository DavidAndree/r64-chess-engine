"""Train the move-prediction network on encoded positions."""

from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from r64 import config
from r64.data.dataset import PositionSet
from r64.data.encoding import board_from_arrays, legal_move_mask
from r64.model.network import MoveNet, NetConfig, features, load, save

VALUE_WEIGHT = 0.25  # the result prediction is a side task; moves matter most


@dataclass
class TrainConfig:
    data: list[str]                       # positions folders
    name: str = "base"                    # checkpoints/<name>/
    net: NetConfig = field(default_factory=NetConfig)
    epochs: float = 3.0
    batch_size: int = 1024
    lr: float = 2e-3
    weight_decay: float = 1e-4
    warmup_steps: int = 500
    eval_every: int = 2000
    eval_positions: int = 20_000
    init_from: str | None = None          # start from an existing checkpoint (fine-tuning)
    seed: int = 0
    out_dir: str | None = None            # default: checkpoints/<name>/
    train_games: int | None = None        # train on a random subset of this many games


# ---------- evaluation ----------

class EvalSet:
    """A fixed sample of held-out positions, with legal-move masks, kept on the device."""

    def __init__(self, ps: PositionSet, n: int, device, seed: int = 0):
        idx = np.sort(np.random.default_rng(seed).choice(len(ps), min(n, len(ps)), replace=False))
        a = ps.arrays
        masks = np.stack([
            legal_move_mask(board_from_arrays(a["pieces"][i], int(a["castling"][i]), int(a["ep"][i])))
            for i in idx
        ])
        self.elo = a["elo"][idx]
        self.batch = {k: torch.from_numpy(v).to(device) for k, v in ps.batch(idx).items()}
        self.mask = torch.from_numpy(masks).to(device)

    def __len__(self) -> int:
        return len(self.elo)


@torch.no_grad()
def predict_eval(model: MoveNet, ev: EvalSet, elo_override: float | None = None, bs: int = 4096):
    """Per-position: correct (legal-masked top-1), loss. Optionally condition on a fixed rating."""
    model.eval()
    correct, losses = [], []
    b = ev.batch
    for s in range(0, len(ev), bs):
        sl = slice(s, s + bs)
        elo = b["elo"][sl] if elo_override is None else torch.full_like(b["elo"][sl], elo_override)
        feats = features(elo, b["opp_elo"][sl], b["clock"][sl], b["base_time"][sl])
        logits, _ = model(b["planes"][sl], feats)
        logits = logits.masked_fill(~ev.mask[sl], float("-inf"))
        target = b["move"][sl]
        correct.append((logits.argmax(1) == target).float())
        losses.append(F.cross_entropy(logits, target, reduction="none"))
    model.train()
    return torch.cat(correct).cpu().numpy(), torch.cat(losses).cpu().numpy()


# ---------- training ----------

def lr_at(step: int, total: int, cfg: TrainConfig) -> float:
    if step < cfg.warmup_steps:
        return cfg.lr * (step + 1) / cfg.warmup_steps
    progress = (step - cfg.warmup_steps) / max(1, total - cfg.warmup_steps)
    return cfg.lr * 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))


def train(cfg: TrainConfig) -> Path:
    torch.manual_seed(cfg.seed)
    device = config.device()
    out = Path(cfg.out_dir) if cfg.out_dir else config.CHECKPOINT_DIR / cfg.name
    out.mkdir(parents=True, exist_ok=True)
    log = open(out / "log.jsonl", "a")  # noqa: SIM115

    train_paths = sorted(p for d in cfg.data for p in Path(d).glob("train-*.npz"))
    val_paths = sorted(p for d in cfg.data for p in Path(d).glob("val-*.npz"))
    print(f"Loading {len(train_paths)} train shards from {', '.join(cfg.data)}")
    train_set = PositionSet(train_paths)
    train_idx = None
    if cfg.train_games is not None:
        games = train_set.game_ids()
        n_games = int(games[-1]) + 1
        keep = np.random.default_rng(cfg.seed).choice(n_games, min(cfg.train_games, n_games), replace=False)
        train_idx = np.nonzero(np.isin(games, keep))[0]
        print(f"Using {len(keep):,} of {n_games:,} games ({len(train_idx):,} positions)")
    n_train = len(train_set) if train_idx is None else len(train_idx)
    print(f"Building eval set ({cfg.eval_positions:,} held-out positions)")
    ev = EvalSet(PositionSet(val_paths), cfg.eval_positions, device, seed=cfg.seed)

    if cfg.init_from:
        model, _ = load(cfg.init_from, device)
        cfg.net = model.cfg
    else:
        model = MoveNet(cfg.net).to(device)
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    steps_per_epoch = math.ceil(n_train / cfg.batch_size)
    total = int(steps_per_epoch * cfg.epochs)
    print(
        f"Device {device} · {model.num_params():,} params · {n_train:,} positions · "
        f"{total:,} steps ({cfg.epochs} epochs × {steps_per_epoch:,})"
    )
    (out / "train_config.json").write_text(json.dumps(asdict(cfg), indent=2))

    step, best, start = 0, -1.0, time.time()
    run_loss, run_acc, run_n = 0.0, 0.0, 0
    epoch = 0
    while step < total:
        for b in train_set.batches(cfg.batch_size, train_idx, seed=cfg.seed + epoch, device=device):
            if step >= total:
                break
            for g in opt.param_groups:
                g["lr"] = lr_at(step, total, cfg)
            feats = features(b["elo"], b["opp_elo"], b["clock"], b["base_time"])
            policy, value = model(b["planes"], feats)
            value_target = (1 - b["result"]).long()  # win 0, draw 1, loss 2
            loss_p = F.cross_entropy(policy, b["move"])
            loss = loss_p + VALUE_WEIGHT * F.cross_entropy(value, value_target)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            step += 1

            run_loss += loss_p.item()
            run_acc += (policy.argmax(1) == b["move"]).float().mean().item()
            run_n += 1
            if step % 200 == 0:
                elapsed = time.time() - start
                eta = elapsed / step * (total - step)
                print(
                    f"\rstep {step:,}/{total:,} · loss {run_loss / run_n:.3f} · "
                    f"train acc {run_acc / run_n:.1%} · {step * cfg.batch_size / elapsed:,.0f} pos/s · "
                    f"eta {eta / 60:.0f} min",
                    end="", flush=True,
                )
            if step % cfg.eval_every == 0 or step == total:
                correct, losses = predict_eval(model, ev)
                acc = float(correct.mean())
                rec = {
                    "step": step, "epoch": step / steps_per_epoch,
                    "train_loss": run_loss / max(run_n, 1), "train_acc": run_acc / max(run_n, 1),
                    "val_loss": float(losses.mean()), "val_acc": acc,
                    "lr": lr_at(step, total, cfg), "minutes": (time.time() - start) / 60,
                }
                log.write(json.dumps(rec) + "\n")
                log.flush()
                print(f"\n  eval @ {step:,}: val acc {acc:.2%} · val loss {rec['val_loss']:.3f}")
                run_loss, run_acc, run_n = 0.0, 0.0, 0
                save(model, out / "latest.pt", step=step, val_acc=acc)
                if acc > best:
                    best = acc
                    save(model, out / "best.pt", step=step, val_acc=acc)
        epoch += 1
    log.close()
    print(f"Done. Best val acc {best:.2%} → {out / 'best.pt'}")
    return out / "best.pt"
