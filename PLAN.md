# r64 — a personal chess sparring partner

## The idea

Stockfish already beats every human, so r64 does **not** compete on strength.
It competes on being **useful to play against**:

1. **Plays like a human** at a chosen rating — realistic mistakes, not random blunders.
2. **Learns how *I* play** — a model of my style that gets more accurate with every game.
3. **Knows my weaknesses** — tracks the kinds of positions where I go wrong.
4. **Trains me on them** — steers games toward my weak spots, at a level that keeps games competitive.

"The more I play, the more it learns" is literal: my games are the most valuable data it has.

## Decisions

| Topic      | Choice |
|------------|--------|
| Learning   | Train a **human-move prediction** model on real human games (lichess open database), then **fine-tune on my games**. No self-play. |
| Stack      | **Python** throughout: PyTorch (Apple Silicon GPU / MPS), python-chess, FastAPI. *Changed from C++ + Python:* without self-play there's no heavy search, so C++ isn't needed. It can come back later if a search component needs speed. |
| Interface  | **Web UI** in the browser: board, move list, stats and weakness dashboard. |
| Priority   | Use proven libraries for heavy lifting; write the parts that make it personal myself. |

### What I write vs. what I use

| Use a library | Write myself |
|---|---|
| Chess rules: `python-chess` | Data pipeline: filter games, encode positions, build training shards |
| Neural network layers, training: PyTorch | Rating-conditioned move-prediction model setup + training loop |
| Web board: `chessground` (lichess's board) | Personal fine-tuning + automatic retraining as I play |
| Position analysis: Stockfish (to find my mistakes) | Weakness map: classify positions, measure where I lose |
| Game data: lichess open database, lichess / chess.com APIs | Targeted sparring + adaptive difficulty |
| Server: FastAPI; storage: SQLite | Web app and dashboards |

## How it works

```
lichess database (millions of human games)
        │  filter by rating / time control, encode positions
        ▼
  base model: "given this position and a rating, what would a human play?"
        │  fine-tune on my games (played in r64 + imported from lichess / chess.com)
        ▼
  personal model: "what would *I* play here?"   ← retrained automatically every N new games

my games ──► Stockfish analysis ──► mistakes, tagged by position type ──► weakness map
                                                                              │
web UI: play vs human-like bot at rating X, vs "me", or a targeted sparring ◄─┘
        session; difficulty adjusts to keep my score near 50%
```

## Prior work

- **Maia** (2020, University of Toronto / Microsoft): human-like engines trained on lichess games, one per rating band; predicts human moves ~50% of the time.
- **Individual models** (follow-up research, 2022): fine-tuning on one player's games predicts *that* player noticeably better.

So the core ideas are proven. What's new here is putting them together as a personal training tool
that keeps adapting, finds weaknesses, and targets them.

## Expectations

- **Base model:** a few million positions and several hours of training on the Mac should give solid
  human-like play. More data improves it further.
- **Personal model:** needs a few hundred of my games to differ meaningfully from the base model;
  thousands is better. **Importing my existing lichess / chess.com games** gives it a head start.
- **Strength:** a move-prediction model without search plays roughly at the rating it imitates,
  which is exactly what a sparring partner needs.

## Phases and milestones

### Phase 0 — Repo setup
- Folders: `r64/` (Python package: data, model, analysis, server), `web/`, `tests/`, `data/` (git-ignored).
- Python project (`uv` or `pip`): `torch`, `python-chess`, `fastapi`, `zstandard`.
- `.gitignore` (data, checkpoints, binaries, `.dSYM`), README.
- Move the GLUT prototype to `legacy/`.
- **Done when:** a fresh clone installs and runs the tests.

### Phase 1 — Web UI: play vs Stockfish
- FastAPI server using `python-chess`: new game, make move, engine move, history.
- Browser page with `chessground`: drag-and-drop, legal-move highlights, promotion, move list.
- Stockfish (via Homebrew) as the temporary opponent at a chosen strength.
- Every finished game saved to SQLite (exportable as PGN).
- **Done when:** I can play full games in the browser and they're saved.

### Phase 2 — Data pipeline
- Download one month of the lichess database (`.pgn.zst`), stream-parse without unpacking it all.
- Filter: rated games, blitz/rapid, both players in a rating range, no very short games.
- Encode each position as 8×8 input planes + the mover's rating; label = the move played.
- Write compact training shards; held-out test set split by game.
- Import my own games from lichess / chess.com.
- **Done when:** millions of encoded positions on disk, plus my games, and a loader that feeds PyTorch fast.

### Phase 3 — Human-like base model
- Residual CNN (PyTorch layers) → policy over moves; **conditioned on rating**, so one model covers all levels.
- Optional second output: predicted game result (useful later for steering).
- Metric: **move-match accuracy** on held-out games, broken down by rating band.
- **Done when:** top-1 move-match ≥ 45% and accuracy is noticeably different across rating bands (proof it learned rating-specific play).

### Phase 4 — Play vs the human-like bot
- Web UI opponent = base model at a chosen rating, sampling moves from its predictions.
- Games vs the bot are saved like any other game.
- **Done when:** playing it at 1200 vs 1800 feels clearly different, and both feel human.

### Phase 5 — Personal model
- Fine-tune the base model on my games → "me" model. Compare against the base model on my held-out games.
- Automatic retraining after every N new games; keep old versions ("me, 6 months ago").
- Mode: play against myself / an old version of me.
- **Done when:** the personal model predicts my moves better than the base model does, and improves as I add games.

### Phase 6 — Weakness map
- Stockfish analyzes my games → find mistakes (eval drops).
- Tag each position: opening, game phase, material balance, pawn structure, piece types, simple tactic patterns.
- Dashboard: where I lose the most per move, compared with players at my rating.
- **Done when:** the dashboard shows clear, believable weak spots.

### Phase 7 — Targeted sparring + adaptive difficulty
- Steering: among human-plausible moves, the bot prefers ones leading toward my weak position types.
- Drills: start games from real positions (from the database) that match my weak spots.
- Adaptive level: adjust the bot's rating to keep my score around 50%.
- **Done when:** a session visibly focuses on my weak areas, and my error rate there trends down over weeks.

### Later (optional)
- Explain mistakes in plain words (Stockfish line + position tags → short explanation).
- Lichess bot account so others can play the human-like bot.
- Light search on top of the model for stronger levels.

## Open questions
- My lichess / chess.com usernames for importing games.
- Main time control to model (blitz vs rapid — players behave differently).
- How many games before the first personal fine-tune, and how often to retrain.
