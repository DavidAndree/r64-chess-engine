# r64

A chess sparring partner that learns how you play. Instead of trying to beat you like Stockfish, it
tries to be the most useful opponent for your level:

- **Human-like bot.** A neural network trained on lichess games plays like a human at any rating
  from 600 to 2200, mistakes included.
- **Personal model.** Fine-tuned on your own games, it plays like you, and it retrains as you play.
- **Weakness map.** Stockfish analyzes your games and shows where you lose more than players at your
  rating do, with real examples from your games.
- **Targeted sparring.** The sparring partner steers games into your weak spots and adapts its level
  to keep you around 50%. Drills start from positions where you went wrong.

Built with [python-chess](https://github.com/niklasf/python-chess), PyTorch, FastAPI,
[chessground](https://github.com/lichess-org/chessground) and Stockfish.

## Setup

Requires [uv](https://docs.astral.sh/uv/) and [Stockfish](https://stockfishchess.org/). Training
uses the Apple GPU on Apple Silicon Macs, or CUDA where available.

```bash
brew install stockfish   # or your platform's package
uv sync                  # install dependencies
uv run pytest            # run the tests
```

## Quick start

Trained models aren't included, so train the base model first:

```bash
# 1. Human-like base model: download ~200k lichess games, then train (a few hours on a laptop)
uv run python -m r64.data lichess-db --month 2026-10 --games 200000 --name lichess-base
uv run python -m r64.model train --data data/positions/lichess-base --name base --epochs 2

# 2. Your games: put in your own usernames, and leave out a site you don't use
uv run python -m r64.data import-mine --chesscom YOUR_CHESSCOM_NAME --lichess YOUR_LICHESS_NAME
uv run python -m r64.data encode-mine
uv run python -m r64.model personal       # your personal model (a few minutes)
uv run python -m r64.analysis analyze     # Stockfish over your games (about 1 s per game per core)

# 3. Play
uv run python -m r64.server               # then open http://127.0.0.1:8064
```

The server works at any stage: without a trained model you can still play Stockfish.

## What's on the page

- **Play** (`/`): choose the sparring partner, the human-like bot, your personal model, or
  Stockfish. The page also has drills and the status of your personal model.
- **Weakness map** (`/weakness`): your weak spots and strengths compared with your opponents, plus
  example positions, a monthly trend, opening results and mistake types.

## Other commands

| Command | What it does |
|---|---|
| `uv run python -m r64.model evaluate` | Base model accuracy by rating band |
| `uv run python -m r64.model match bot:1200 bot:1800` | Play two opponents against each other |
| `uv run python -m r64.model learning-curve` | Personal model accuracy vs number of your games |
| `uv run python -m r64.analysis report` | The weakness map as text |
| `uv run python -m r64.sparring_sim` | Measure how much steering focuses games |

Games are stored in `data/r64.sqlite`, and models in `checkpoints/`. Both are git-ignored. Set
`R64_DB` to use another games database, or `R64_BOT` to use another base model.

## Layout

```
r64/data/       game downloads, PGN parsing, position encoding
r64/model/      network, training, bots, personal model
r64/analysis/   Stockfish analysis and the weakness map
r64/sparring*   steering, adaptive level, drills
r64/server/     web server
web/            browser UI
tests/          tests
```

See [PLAN.md](PLAN.md) for the design.

## License

[GPL-3.0-or-later](LICENSE).
