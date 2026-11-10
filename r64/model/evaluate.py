"""How human-like is the model, and at which ratings?

1. Move-match accuracy by rating band on held-out lichess games.
2. Rating sensitivity: for players of a given band, condition the model on different ratings.
   If it learned rating-specific play, accuracy peaks when the conditioning rating matches.
3. (Optional) My games: how well it predicts me, and which rating I "play like".
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from r64 import config
from r64.data.dataset import PositionSet
from r64.model.network import load
from r64.model.train import EvalSet, predict_eval

BANDS = list(range(600, 2400, 200))  # band starts; each band is [start, start + 200)


def _band_rows(elo: np.ndarray, start: int) -> np.ndarray:
    return (elo >= start) & (elo < start + 200)


def evaluate(checkpoint: Path, data_dir: Path, me_dir: Path | None, positions: int = 30_000) -> dict:
    device = config.device()
    model, ckpt = load(checkpoint, device)
    print(f"Model {checkpoint} (step {ckpt.get('step')}, {model.num_params():,} params) on {device}")
    report: dict = {"checkpoint": str(checkpoint)}

    ev = EvalSet(PositionSet.from_dir(data_dir, "val"), positions, device, seed=1)
    correct, _ = predict_eval(model, ev)
    report["overall"] = float(correct.mean())
    print(f"\nOverall move-match accuracy: {correct.mean():.2%} on {len(ev):,} positions\n")

    print("By rating band (player to move):")
    report["by_band"] = {}
    for start in BANDS:
        rows = _band_rows(ev.elo, start)
        if rows.sum() >= 200:
            acc = float(correct[rows].mean())
            report["by_band"][start] = {"acc": acc, "n": int(rows.sum())}
            print(f"  {start}-{start + 199}: {acc:6.2%}  ({rows.sum():,} positions)")

    print("\nRating sensitivity — rows: real rating band, columns: rating the model is told")
    cond = [800, 1100, 1400, 1700, 2000, 2300]
    matrix = {c: predict_eval(model, ev, elo_override=float(c))[0] for c in cond}
    print("  band       " + "".join(f"{c:>8}" for c in cond))
    report["sensitivity"] = {}
    for start in BANDS:
        rows = _band_rows(ev.elo, start)
        if rows.sum() < 200:
            continue
        accs = [float(matrix[c][rows].mean()) for c in cond]
        best = int(np.argmax(accs))
        cells = "".join(f"{a:8.2%}" if i != best else f"{'*' + format(a, '.2%'):>8}" for i, a in enumerate(accs))
        print(f"  {start}-{start + 199}  {cells}")
        report["sensitivity"][start] = dict(zip(cond, accs, strict=True))

    if me_dir is not None and any(Path(me_dir).glob("val-*.npz")):
        me = EvalSet(PositionSet.from_dir(me_dir, "val"), positions, device, seed=1)
        own, _ = predict_eval(model, me)
        print(f"\nMy games: {own.mean():.2%} move-match using my recorded rating "
              f"(median {int(np.median(me.elo))}) on {len(me):,} positions")
        grid = list(range(600, 2201, 100))
        accs = [float(predict_eval(model, me, elo_override=float(r))[0].mean()) for r in grid]
        best = grid[int(np.argmax(accs))]
        print("  told rating → accuracy: " + ", ".join(f"{r}: {a:.1%}" for r, a in zip(grid, accs, strict=True)))
        print(f"  → I play most like a lichess {best} (accuracy {max(accs):.2%})")
        report["me"] = {"acc_recorded_elo": float(own.mean()), "best_rating": best,
                        "by_rating": dict(zip(grid, accs, strict=True))}

    out = Path(checkpoint).parent / "eval.json"
    out.write_text(json.dumps(report, indent=2))
    print(f"\nSaved {out}")
    return report
