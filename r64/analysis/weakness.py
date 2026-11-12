"""The weakness map: where I lose the most, compared with players at my rating.

"Players at my rating" = my opponents in the same games (chess.com pairs me with players of about
my rating, at the same time control), so the comparison needs no extra data. Games against r64's
bots count for me but not as opponents.

For every position tag (phase, clock, material, piece moved, ...) it compares the winning chances
I lose per move with what my opponents lose in the same kind of position. A **weak spot** is a tag
where I lose clearly more (at least 2 standard errors), ranked by the extra winning chances it costs
me per game. Each weak spot comes with real examples from my recent games.
"""

from __future__ import annotations

import json
import math
import statistics
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import chess

from r64 import config
from r64.analysis import store
from r64.analysis.tags import CAUSES, MISTAKE, describe

TAGS_VERSION = 2   # bump when tags.py changes, to invalidate the cached report
CACHE_PATH = config.DATA_DIR / "weakness.json"
MIN_MOVES = 150    # ignore tags with fewer of my moves than this
Z_CLEAR = 2.0


@dataclass
class Move:
    game_id: int
    mine: bool
    peer: bool            # counts in the "players at my rating" baseline
    loss: float
    severity: str | None
    cause: str | None
    tags: list[str]
    ply: int
    fen: str
    move: str
    best: str | None
    played_at: str
    link: str | None


def load_moves(conn, recent: int) -> tuple[list[dict], list[Move]]:
    games = [dict(r) for r in conn.execute(
        "SELECT * FROM games ORDER BY played_at DESC LIMIT ?", (recent,)
    )]
    moves: list[Move] = []
    for g in games:
        rows = conn.execute(
            "SELECT ply, fen, move, best, cp, clock FROM positions WHERE game_id = ? ORDER BY ply",
            (g["id"],),
        ).fetchall()
        my_white = g["my_color"] == "white"
        clocks = {chess.WHITE: g["base_time"], chess.BLACK: g["base_time"]}
        g["opening_moves"] = [r["move"] for r in rows[:2] if r["move"]]
        for i in range(len(rows) - 1):
            r, nxt = rows[i], rows[i + 1]
            white_to_move = r["ply"] % 2 == 0
            color = chess.WHITE if white_to_move else chess.BLACK
            clock_before = clocks[color] if clocks[color] is not None and clocks[color] >= 0 else None
            if r["clock"] is not None:
                clocks[color] = r["clock"]
            facts = describe(r["fen"], r["move"], r["best"], r["cp"], nxt["cp"], nxt["best"],
                             clock_before, g["base_time"])
            mine = white_to_move == my_white
            moves.append(Move(
                game_id=g["id"], mine=mine, peer=not mine and g["source"] != "r64",
                loss=facts.loss, severity=facts.severity, cause=facts.cause, tags=facts.tags,
                ply=r["ply"], fen=r["fen"], move=r["move"], best=r["best"],
                played_at=g["played_at"], link=g["key"] if g["key"].startswith("http") else None,
            ))
    return games, moves


def _summary(ms: list[Move]) -> dict:
    n = len(ms)
    if not n:
        return {"moves": 0}
    count = lambda sev: sum(m.severity == sev for m in ms)
    return {
        "moves": n,
        "loss_per_move": sum(m.loss for m in ms) / n,
        "blunders_per_100": 100 * count("blunder") / n,
        "mistakes_per_100": 100 * count("mistake") / n,
        "inaccuracies_per_100": 100 * count("inaccuracy") / n,
    }


def _mean_var(xs: list[float]) -> tuple[float, float]:
    if len(xs) < 2:
        return (xs[0] if xs else 0.0), 0.0
    return statistics.fmean(xs), statistics.variance(xs)


def _san(fen: str, uci: str | None) -> str | None:
    if not uci:
        return None
    board = chess.Board(fen)
    try:
        return board.san(chess.Move.from_uci(uci))
    except (ValueError, AssertionError):
        return uci


def _example(m: Move) -> dict:
    board = chess.Board(m.fen)
    num = f"{board.fullmove_number}{'.' if board.turn == chess.WHITE else '...'}"
    return {
        "fen": m.fen, "move": m.move, "best": m.best,
        "played": f"{num} {_san(m.fen, m.move)}", "better": _san(m.fen, m.best),
        "loss": m.loss, "severity": m.severity, "cause": m.cause,
        "date": m.played_at[:10], "link": m.link,
        "orientation": "white" if board.turn == chess.WHITE else "black",
    }


def _opening_name(game: dict) -> str | None:
    ms = game["opening_moves"]
    if len(ms) < 2:
        return None
    board = chess.Board()
    first = board.san(chess.Move.from_uci(ms[0]))
    board.push_uci(ms[0])
    second = board.san(chess.Move.from_uci(ms[1]))
    return f"As {game['my_color']}: 1.{first} {second}"


def build_report(recent: int = 1000, db: Path | None = None) -> dict:
    conn = store.connect(db)
    games, moves = load_moves(conn, recent)
    conn.close()
    mine = [m for m in moves if m.mine]
    peers = [m for m in moves if m.peer]
    report: dict = {
        "tags_version": TAGS_VERSION,
        "games": len(games),
        "analyzed_total": None,
        "period": [games[-1]["played_at"][:10], games[0]["played_at"][:10]] if games else None,
        "my_rating": int(statistics.median([g["my_elo"] for g in games if g["my_elo"]] or [0])),
        "opp_rating": int(statistics.median([g["opp_elo"] for g in games if g["opp_elo"]] or [0])),
        "me": _summary(mine),
        "peers": _summary(peers),
    }
    if not mine:
        return report
    n_games = len(games)
    my_total_loss = sum(m.loss for m in mine)
    report["me"]["loss_per_game"] = my_total_loss / n_games

    # ----- every tag: me vs peers -----
    by_tag_me: dict[str, list[Move]] = defaultdict(list)
    by_tag_peer: dict[str, list[float]] = defaultdict(list)
    for m in mine:
        for t in m.tags:
            by_tag_me[t].append(m)
    for m in peers:
        for t in m.tags:
            by_tag_peer[t].append(m.loss)
    def compare(tag: str, me_vals: list[float], peer_vals: list[float], moves_in_tag: int) -> dict:
        mean_me, var_me = _mean_var(me_vals)
        mean_peer, var_peer = _mean_var(peer_vals)
        se = math.sqrt(var_me / len(me_vals) + var_peer / len(peer_vals)) or 1e-9
        excess = mean_me - mean_peer
        dim, _, value = tag.partition(": ")
        return {
            "tag": tag, "dimension": dim, "value": value,
            "moves": moves_in_tag, "moves_per_game": moves_in_tag / n_games,
            "loss_me": mean_me, "loss_peer": mean_peer, "excess": excess, "z": excess / se,
            # Extra winning chances lost per game, vs an opponent at my level in my shoes.
            "extra_per_game": excess * len(me_vals) / n_games,
            "share_of_my_loss": sum(me_vals) / my_total_loss,
        }

    rows = []
    for tag, ms in by_tag_me.items():
        peer_losses = by_tag_peer.get(tag, [])
        if len(ms) < MIN_MOVES or len(peer_losses) < MIN_MOVES:
            continue
        rows.append(compare(tag, [m.loss for m in ms], peer_losses, len(ms)))
    # Mistake types, compared over all moves: winning chances lost to that kind of mistake.
    for cause in CAUSES[:-1]:
        tag = f"mistake type: {cause}"
        mm = [m for m in mine if m.cause == cause]
        if not mm:
            continue
        by_tag_me[tag] = mm
        rows.append(compare(tag, [m.loss if m.cause == cause else 0.0 for m in mine],
                            [m.loss if m.cause == cause else 0.0 for m in peers], len(mm)))
    rows.sort(key=lambda r: (r["dimension"], -r["moves"]))
    report["tags"] = rows

    def with_examples(row: dict) -> dict:
        ms = [m for m in by_tag_me[row["tag"]] if m.loss >= MISTAKE]
        ms.sort(key=lambda m: m.played_at, reverse=True)
        return {**row, "examples": [_example(m) for m in ms[:4]]}

    weak = [r for r in rows if r["z"] >= Z_CLEAR and r["excess"] > 0]
    weak.sort(key=lambda r: r["extra_per_game"], reverse=True)
    report["weak_spots"] = [with_examples(r) for r in weak[:6]]
    strong = [r for r in rows if r["z"] <= -Z_CLEAR]
    strong.sort(key=lambda r: r["extra_per_game"])
    report["strengths"] = strong[:4]

    # ----- what went wrong in mistakes -----
    causes = []
    my_mistakes = [m for m in mine if m.cause]
    mistake_loss = sum(m.loss for m in my_mistakes) or 1e-9
    for cause in CAUSES:
        mm = [m for m in my_mistakes if m.cause == cause]
        pm = sum(m.cause == cause for m in peers)
        causes.append({
            "cause": cause,
            "me_per_100": 100 * len(mm) / len(mine),
            "peer_per_100": 100 * pm / max(len(peers), 1),
            "share_of_mistake_loss": sum(m.loss for m in mm) / mistake_loss,
            "examples": [_example(m) for m in sorted(mm, key=lambda m: m.played_at, reverse=True)[:3]],
        })
    report["causes"] = causes

    # ----- openings: my score and early losses -----
    loss_by_game_me: dict[int, list[float]] = defaultdict(list)
    loss_by_game_peer: dict[int, list[float]] = defaultdict(list)
    for m in moves:
        if m.ply < 20:
            (loss_by_game_me if m.mine else loss_by_game_peer)[m.game_id].append(m.loss)
    openings: dict[str, list[dict]] = defaultdict(list)
    for g in games:
        name = _opening_name(g)
        if name:
            openings[name].append(g)
    op_rows = []
    for name, gs in openings.items():
        if len(gs) < 15:
            continue
        pts = [1.0 if g["result"] == "win" else 0.5 if g["result"] == "draw" else 0.0 for g in gs]
        me_l = [x for g in gs for x in loss_by_game_me[g["id"]]]
        peer_l = [x for g in gs for x in loss_by_game_peer[g["id"]]]
        op_rows.append({
            "opening": name, "games": len(gs), "score": sum(pts) / len(pts),
            "loss_me": statistics.fmean(me_l) if me_l else None,
            "loss_peer": statistics.fmean(peer_l) if peer_l else None,
        })
    op_rows.sort(key=lambda r: r["games"], reverse=True)
    report["openings"] = op_rows

    # ----- trend by month -----
    month_of = {g["id"]: g["played_at"][:7].replace(".", "-") for g in games}
    elo_by_month: dict[str, list[int]] = defaultdict(list)
    for g in games:
        if g["my_elo"]:
            elo_by_month[month_of[g["id"]]].append(g["my_elo"])
    trend: dict[str, dict] = defaultdict(lambda: {"me": [], "peer": []})
    for m in moves:
        if m.mine or m.peer:
            trend[month_of[m.game_id]]["me" if m.mine else "peer"].append(m)
    report["trend"] = [
        {"month": mo, "games": len({m.game_id for m in d["me"]}),
         "loss_me": _summary(d["me"]).get("loss_per_move"),
         "loss_peer": _summary(d["peer"]).get("loss_per_move"),
         "blunders_me_per_100": _summary(d["me"]).get("blunders_per_100"),
         "rating": int(statistics.median(elo_by_month[mo])) if elo_by_month[mo] else None}
        for mo, d in sorted(trend.items()) if len(d["me"]) >= 100
    ]
    return report


def cached_report(recent: int = 1000, db: Path | None = None, cache: Path = CACHE_PATH) -> dict:
    """The report, recomputed only when new games were analyzed (it takes ~10-30 s)."""
    conn = store.connect(db)
    analyzed = conn.execute("SELECT COUNT(*) FROM games").fetchone()[0]
    conn.close()
    stamp = {"analyzed_total": analyzed, "recent": recent, "tags_version": TAGS_VERSION}
    if cache.exists():
        cached = json.loads(cache.read_text())
        if all(cached.get(k) == v for k, v in stamp.items()):
            return cached
    report = {**build_report(recent, db), **stamp}
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(report))
    return report


def pct(x: float | None) -> str:
    return "  –  " if x is None else f"{100 * x:5.2f}"


def print_report(recent: int = 1000) -> None:
    r = cached_report(recent)
    if not r.get("me", {}).get("moves"):
        print("No analyzed games yet. Run: uv run python -m r64.analysis analyze")
        return
    me, pe = r["me"], r["peers"]
    print(f"Weakness map — my last {r['games']:,} analyzed games ({r['period'][0]} to {r['period'][1]})")
    print(f"My rating ~{r['my_rating']}, opponents ~{r['opp_rating']}. "
          "Numbers are winning chances lost per move, in %.\n")
    print("                    me     opponents")
    print(f"  lost per move   {pct(me['loss_per_move'])}   {pct(pe['loss_per_move'])}")
    print(f"  blunders /100   {me['blunders_per_100']:5.2f}   {pe['blunders_per_100']:5.2f}")
    print(f"  mistakes /100   {me['mistakes_per_100']:5.2f}   {pe['mistakes_per_100']:5.2f}")

    print("\nWeak spots (I lose clearly more than my opponents in these positions):")
    for w in r["weak_spots"] or []:
        print(f"  {w['tag']:<45} me {pct(w['loss_me'])}  opp {pct(w['loss_peer'])}  "
              f"+{100 * w['extra_per_game']:.1f}%/game  (z {w['z']:.1f}, {w['moves']:,} moves)")
        for ex in w["examples"][:2]:
            print(f"      e.g. {ex['date']}: {ex['played']} (better {ex['better']}), "
                  f"-{100 * ex['loss']:.0f}% · {ex['cause']}")
    if not r["weak_spots"]:
        print("  none clearly worse than my opponents")
    print("\nStrengths (I lose clearly less):")
    for s in r["strengths"] or []:
        print(f"  {s['tag']:<45} me {pct(s['loss_me'])}  opp {pct(s['loss_peer'])}")
    print("\nWhat goes wrong in my mistakes (per 100 moves, me vs opponents):")
    for c in r["causes"]:
        print(f"  {c['cause']:<16} {c['me_per_100']:5.2f}  vs {c['peer_per_100']:5.2f}   "
              f"{100 * c['share_of_mistake_loss']:4.0f}% of what my mistakes cost")
    print("\nOpenings (15+ games):")
    for o in r["openings"][:10]:
        print(f"  {o['opening']:<28} {o['games']:4} games  score {100 * o['score']:3.0f}%  "
              f"first 10 moves: me {pct(o['loss_me'])} opp {pct(o['loss_peer'])}")
    print("\nBy month:")
    for t in r["trend"]:
        print(f"  {t['month']}  {t['games']:4} games  rating {t['rating'] or '–':>5}  "
              f"me {pct(t['loss_me'])}  opp {pct(t['loss_peer'])}  blunders/100 {t['blunders_me_per_100']:.1f}")
