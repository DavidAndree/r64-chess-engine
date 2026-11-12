# r64

A personal chess sparring partner. It doesn't try to beat Stockfish — it tries to be the most
useful opponent for **you**:

- **Plays like a human** at a chosen rating, with realistic mistakes.
- **Learns how you play** — the more games you play, the better it models you.
- **Finds your weak spots** and steers games toward them.

See [PLAN.md](PLAN.md) for the full design and roadmap.

## Status

Phase 6 done — a weakness map: Stockfish analyzes all my games and shows, at
http://127.0.0.1:8064/weakness, the kinds of positions where I lose more winning chances than
players at my rating do, with real examples from my games. Phase 5 gave a personal model that
predicts 46.2% of my moves (base model: 42.0%) and retrains itself as I play.

## Data

```bash
# Stream games from the lichess open database (only the needed part is downloaded)
uv run python -m r64.data lichess-db --month 2026-10 --games 200000

# Download my games and encode my moves (for the personal model)
uv run python -m r64.data import-mine --lichess <name> --chesscom <name>
uv run python -m r64.data encode-mine
```

## My personal model

```bash
uv run python -m r64.model personal          # train the next version (~3 min on an M2)
uv run python -m r64.model personal --fetch  # re-download my chess.com/lichess games first
uv run python -m r64.model learning-curve    # accuracy vs number of my games
```

Each version is the base model fine-tuned (4 epochs, lr 1e-4) on **all** my games so far: my
imported games plus every finished r64 game. Versions are kept in `checkpoints/me/vNNN/`, so I
can play against older versions of myself. The web server also retrains automatically after every
20 new r64 games, or when you click "Retrain now". It plays at my current rating (median of my last
50 rated games, or `"rating"` in `data/profile.json`) and samples my move distribution as is.

How well it predicts my moves (held-out games, same positions for all):

| My games used | 0 (base) | 100 | 250 | 500 | 1,000 | 2,000 | 3,067 |
|---|---|---|---|---|---|---|---|
| Move-match | 42.0% | 43.6% | 45.3% | 45.2% | 45.4% | 45.7% | 46.2% |

Most of the gain comes from the first few hundred games (my level and openings, e.g. it learned
that I answer 1.e4 with 1...c6: 5% → 41%); after that it keeps improving slowly.

## Weakness map

```bash
uv run python -m r64.analysis analyze     # Stockfish over every game not analyzed yet (resumable)
uv run python -m r64.analysis report      # the weakness map as text (also at /weakness)
```

Stockfish evaluates every position of my games (10,000 nodes ≈ depth 13; ~1 s per game per core,
all cores in parallel; 3,448 games ≈ 1 hour). Results go to `data/analysis.sqlite`; the "Analyze
new games" button on the page adds new chess.com/r64 games.

How it measures:
- **Cost of a move** = the winning chances it gives away (Stockfish eval before vs after, through
  lichess's eval → win-probability curve). Blunder ≥ 15 points, mistake ≥ 10, inaccuracy ≥ 5.
- **Players at my rating** = my opponents in the same games: same rating (chess.com matchmaking),
  same time controls. Games against r64's bots don't count as opponents.
- Every move is tagged (phase, winning/balanced/losing, material, clock, check, attacked piece,
  king in the center, piece moved, endgame type), and every mistake gets a cause (hung material,
  missed material, walked into / missed a fork, walked into / missed mate, other).
- A **weak spot** is a tag where I lose clearly more than my opponents (≥ 2 standard errors),
  ranked by the extra winning chances it costs per game.

## Train

```bash
# Human-like base model (6 residual blocks x 64 channels, conditioned on rating + clock)
uv run python -m r64.model train --data data/positions/lichess-base --name base --epochs 2

# Move-match accuracy by rating band, rating sensitivity, and how well it predicts me
uv run python -m r64.model evaluate --checkpoint checkpoints/base/best.pt

# Does the rating setting change the bot's strength? (alternating colors)
uv run python -m r64.model match bot:800 bot:2000 --games 60
uv run python -m r64.model match bot:1600 stockfish:1320 --games 30
```

Training on an M2 MacBook (8 GB) runs at ~3,000 positions/s, so 2 epochs over 14M positions
takes about 3 hours. Checkpoints are saved at every evaluation; stopping early is safe.

Data output goes to `data/positions/<name>/` as `train-*.npz` / `val-*.npz` shards
(~115 bytes per position; 1M positions ≈ 115 MB). Speed on an 8-core M-series Mac:
~1,700 games/s (≈100k positions/s) to encode, ~400k positions/s to load onto the GPU.

## Play

```bash
uv run python -m r64.server
```

Then open http://127.0.0.1:8064 and pick an opponent:

- **Human-like bot** (600-2200): the trained network. It samples a move from its prediction of
  what a player of that rating would play, so it makes human mistakes and varies its games.
  Lower ratings sample more loosely; higher ratings mostly play the most likely move.
  Uses `checkpoints/base/best.pt` (set `R64_BOT=path/to/model.pt` to use another).
- **Stockfish** (1320-3190): limited-strength Stockfish, if installed.

Measured strength (bot vs bot, 60 games): 800 scores 20% against 2000 and 1200 scores 33%
against 1800. Against Stockfish 1320: 27% at 1200, 38% at 1600, 47% at 2000. Its ceiling is around
Stockfish 1320, so the top of the range plays a bit below its label.

Games are stored in `data/r64.sqlite`
(set `R64_DB=/some/other.sqlite` to use a different database, e.g. for testing).

## Setup

Requires [uv](https://docs.astral.sh/uv/) and macOS on Apple Silicon (other platforms should work,
training just won't use the Apple GPU).

Keep the repo **outside** iCloud-synced folders (Desktop, Documents): with "Optimize Mac Storage",
macOS offloads files from `.venv` and `data/` to iCloud, and Python, uv and ruff then stall
reading them. `~/Developer` works well.

```bash
uv sync            # create .venv and install dependencies
uv run pytest      # run the tests
```

Later phases also need Stockfish:

```bash
brew install stockfish
```

## Layout

```
r64/            Python package
  data/         lichess database download, parsing, position encoding
  model/        move-prediction network and training
  analysis/     Stockfish analysis, weakness map
  server/       FastAPI web server
web/            browser UI
tests/          pytest tests
legacy/         original OpenGL/GLUT board prototype (not used)
data/           downloaded and generated data (git-ignored)
checkpoints/    trained models (git-ignored)
```

## License

MIT
