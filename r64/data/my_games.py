"""Collect my own games: import from lichess / chess.com and export games played in r64.

All of them end up as PGN files in data/my_games/, then get encoded (only my moves) into
data/positions/me/ for fine-tuning the personal model.
"""

from __future__ import annotations

import io
import json
import re
import sqlite3
import statistics
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import chess.pgn

from r64 import config
from r64.data.pgn import USER_AGENT

MY_GAMES_DIR = config.DATA_DIR / "my_games"
PROFILE_PATH = config.DATA_DIR / "profile.json"
R64_PLAYER_NAME = "me"  # how I appear in PGNs of games played inside r64


# ---------- profile: my usernames ----------

def load_profile() -> dict:
    if PROFILE_PATH.exists():
        return json.loads(PROFILE_PATH.read_text())
    return {"lichess": None, "chesscom": None}


def save_profile(profile: dict) -> None:
    PROFILE_PATH.parent.mkdir(parents=True, exist_ok=True)
    PROFILE_PATH.write_text(json.dumps(profile, indent=2))


def my_names(profile: dict) -> set[str]:
    names = {R64_PLAYER_NAME}
    for key in ("lichess", "chesscom"):
        if profile.get(key):
            names.add(profile[key].lower())
    return names


def current_rating(profile: dict | None = None, recent: int = 50) -> int:
    """My playing strength now: `rating` in the profile if set, else the median of my rating in
    my most recent imported games, else 1500."""
    profile = profile if profile is not None else load_profile()
    if profile.get("rating"):
        return int(profile["rating"])
    names = my_names(profile) - {R64_PLAYER_NAME}
    rated: list[tuple[str, int]] = []
    for path in MY_GAMES_DIR.glob("*.pgn"):
        if path.name == "r64.pgn":
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        for block in text.split("\n\n[Event ")[0:]:
            h = dict(re.findall(r'^\[(\w+) "([^"]*)"\]', block, re.M))
            for side in ("White", "Black"):
                if h.get(side, "").lower() in names and h.get(f"{side}Elo", "").isdigit():
                    when = h.get("UTCDate", h.get("Date", "")) + h.get("UTCTime", h.get("StartTime", ""))
                    rated.append((when, int(h[f"{side}Elo"])))
    if not rated:
        return 1500
    rated.sort()
    return int(statistics.median(e for _, e in rated[-recent:]))


# ---------- HTTP ----------

def _get(url: str, accept: str) -> urllib.request.addinfourl:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, "Accept": accept})
    for attempt in range(5):
        try:
            return urllib.request.urlopen(req, timeout=60)  # noqa: S310 - fixed API hosts
        except urllib.error.HTTPError as e:
            if e.code == 429 and attempt < 4:  # rate limited: wait and retry
                time.sleep(60 if "lichess" in url else 10)
                continue
            raise
    raise RuntimeError("unreachable")


# ---------- lichess ----------

def fetch_lichess(username: str, out_path: Path | None = None) -> int:
    """Download all my rated lichess games (with clock times) as PGN. Returns game count."""
    out_path = out_path or MY_GAMES_DIR / f"lichess_{username.lower()}.pgn"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    query = urllib.parse.urlencode({"clocks": "true", "rated": "true", "perfType": "bullet,blitz,rapid,classical"})
    url = f"https://lichess.org/api/games/user/{urllib.parse.quote(username)}?{query}"
    count = 0
    with _get(url, "application/x-chess-pgn") as resp, open(out_path, "w", encoding="utf-8") as out:
        for raw in resp:
            line = raw.decode("utf-8", errors="replace")
            if line.startswith("[Event "):
                count += 1
                if count % 200 == 0:
                    print(f"\rlichess: {count:,} games", end="", flush=True)
            out.write(line)
    print(f"\rlichess: {count:,} games → {out_path}")
    return count


# ---------- chess.com ----------

def fetch_chesscom(username: str, out_path: Path | None = None) -> int:
    """Download all my standard (non-daily) chess.com games as PGN. Returns game count."""
    out_path = out_path or MY_GAMES_DIR / f"chesscom_{username.lower()}.pgn"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    base = f"https://api.chess.com/pub/player/{urllib.parse.quote(username.lower())}"
    with _get(f"{base}/games/archives", "application/json") as resp:
        archives = json.load(resp)["archives"]
    count = 0
    with open(out_path, "w", encoding="utf-8") as out:
        for i, month_url in enumerate(archives, 1):
            with _get(month_url, "application/json") as resp:
                games = json.load(resp).get("games", [])
            for g in games:
                if g.get("rules") != "chess" or g.get("time_class") == "daily" or not g.get("pgn"):
                    continue
                out.write(g["pgn"].strip() + "\n\n")
                count += 1
            print(f"\rchess.com: month {i}/{len(archives)} · {count:,} games", end="", flush=True)
    print(f"\rchess.com: {count:,} games → {out_path}")
    return count


# ---------- r64 ----------

def export_r64_games(
    db_path: Path | None = None, out_path: Path | None = None, my_rating: int | None = None
) -> int:
    """Write all finished games played in r64 to one PGN file. Returns game count.

    r64 games have no ratings or clock; they get `my_rating` for me and the opponent's set
    strength, and a 10-minute time control (how the bots treat the clock), so the network sees
    them like my other games instead of as an unknown 1500 player."""
    db_path = db_path or config.DB_PATH
    out_path = out_path or MY_GAMES_DIR / "r64.pgn"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not Path(db_path).exists():
        out_path.write_text("")
        return 0
    with sqlite3.connect(db_path) as conn:
        rows = conn.execute(
            "SELECT pgn, player_color, opponent_elo FROM games"
            " WHERE status = 'finished' AND pgn IS NOT NULL ORDER BY id"
        ).fetchall()
    my_rating = my_rating if my_rating is not None else current_rating()
    out = []
    for pgn, color, opp_elo in rows:
        game = chess.pgn.read_game(io.StringIO(pgn))
        me, opp = ("White", "Black") if color == "white" else ("Black", "White")
        game.headers[f"{me}Elo"] = str(my_rating)
        game.headers[f"{opp}Elo"] = str(opp_elo or my_rating)
        game.headers["TimeControl"] = "600"
        out.append(str(game).strip() + "\n\n")
    out_path.write_text("".join(out), encoding="utf-8")
    return len(rows)


def encode_mine(out_dir: Path | None = None, progress: bool = True) -> dict:
    """Export r64 games, then encode my moves from all my PGNs into data/positions/me/."""
    from r64.data.build import build  # avoids an import cycle

    n = export_r64_games()
    if progress:
        print(f"r64: {n} finished games exported")
    sources = sorted(MY_GAMES_DIR.glob("*.pgn"))
    if not sources or all(p.stat().st_size == 0 for p in sources):
        raise FileNotFoundError("No games found. Run import-mine or play some games first.")
    names = my_names(load_profile())
    if progress:
        print(f"Encoding my moves (as {', '.join(sorted(names))}) from {len(sources)} files")
    out_dir = out_dir or config.DATA_DIR / "positions" / "me"
    manifest = build(sources, out_dir, game_filter=None, player=names, val_percent=10, progress=progress)
    manifest["r64_games"] = n
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest
