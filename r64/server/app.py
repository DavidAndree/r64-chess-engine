"""FastAPI web server: game API + the browser UI."""

from __future__ import annotations

import logging
from collections.abc import Callable
from contextlib import asynccontextmanager
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from r64 import config
from r64.analysis.service import WeaknessService
from r64.game import GameNotFound, GameOver, GameService, IllegalMove, state
from r64.model.personal import AutoTrainer, Version, versions
from r64.opponents import Opponent, StockfishNotFound, StockfishOpponent
from r64.sparring_service import SparringService
from r64.storage import GameStore

WEB_DIR = config.ROOT / "web"
log = logging.getLogger("r64")


class NewGame(BaseModel):
    color: Literal["white", "black", "random"] = "white"
    opponent: str = "stockfish"
    elo: int | None = 1500  # checked against the opponent's range; ignored by the sparring partner
    focus: str | None = None  # sparring: which weak spot to target ("auto" / None = rotate)


class NewDrill(BaseModel):
    tag: str


class Move(BaseModel):
    uci: str = Field(min_length=4, max_length=5)


def default_opponents() -> dict[str, Opponent]:
    """Every opponent that's available: the bot if a trained model exists, Stockfish if installed."""
    opponents: dict[str, Opponent] = {}
    if config.BOT_CHECKPOINT.exists():
        from r64.model.bot import (  # imports torch: only when there is a model
            HumanBot,
            personal_bot,
        )
        from r64.sparring import SparringBot

        # Steering targets are set once the sparring service exists (see create_app).
        opponents["sparring"] = SparringBot(config.BOT_CHECKPOINT, targets_fn=list)
        for v in reversed(versions()):  # newest first
            opponents[v.key] = personal_bot(v)
        opponents["bot"] = HumanBot(config.BOT_CHECKPOINT)
    else:
        log.warning("No model at %s: the human-like bot is unavailable", config.BOT_CHECKPOINT)
    try:
        opponents["stockfish"] = StockfishOpponent()
    except StockfishNotFound as e:
        log.warning("%s", e)
    if not opponents:
        raise RuntimeError("No opponents available: train a model or install Stockfish")
    return opponents


def default_trainer(service: GameService) -> AutoTrainer | None:
    """Retrain my personal model in the background as I play (needs the base model)."""
    if not config.BOT_CHECKPOINT.exists():
        return None

    def add_version(v: Version) -> None:
        from r64.model.bot import personal_bot

        # Newest version first in the opponent list.
        rest = dict(service.opponents)
        service.opponents.clear()
        service.opponents[v.key] = personal_bot(v)
        service.opponents.update(rest)

    return AutoTrainer(service.store.count_finished, add_version)


def create_app(
    make_store: Callable[[], GameStore] = lambda: GameStore(config.DB_PATH),
    make_opponents: Callable[[], dict[str, Opponent]] = default_opponents,
    make_trainer: Callable[[GameService], AutoTrainer | None] = default_trainer,
    make_weakness: Callable[[], WeaknessService] = WeaknessService,
) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        store = make_store()
        opponents = make_opponents()
        service = GameService(store, opponents)
        trainer = make_trainer(service)
        if trainer is not None:
            service.on_finish.append(trainer.game_finished)
        app.state.service, app.state.trainer = service, trainer
        app.state.weakness = make_weakness()
        sparring = None
        if "sparring" in opponents and "bot" in opponents:
            sparring = SparringService(service, app.state.weakness.cache)
            opponents["sparring"].targets_fn = sparring.targets
        app.state.sparring = sparring
        yield
        if trainer is not None:
            trainer.wait()  # don't leave a half-written version behind
        for opp in opponents.values():
            opp.close()
        store.close()

    app = FastAPI(title="r64", lifespan=lifespan)

    @app.middleware("http")
    async def revalidate_ui(request, call_next):
        # Browsers must re-check the page and its scripts, or they run stale code after an update.
        response = await call_next(request)
        if not request.url.path.startswith("/api/"):
            response.headers["Cache-Control"] = "no-cache"
        return response

    def service() -> GameService:
        return app.state.service

    @app.get("/api/opponents")
    def opponents():
        return [
            {"name": key, "label": o.label, "min_elo": o.min_elo, "max_elo": o.max_elo}
            for key, o in service().opponents.items()
        ]

    @app.get("/api/personal")
    def personal_status():
        trainer = app.state.trainer
        if trainer is None:
            return {"available": False}
        return {"available": True, **trainer.status()}

    @app.post("/api/personal/train")
    def personal_train():
        trainer = app.state.trainer
        if trainer is None:
            raise HTTPException(409, "train the base model first")
        return {"started": trainer.start()}

    @app.post("/api/games")
    def new_game(req: NewGame):
        try:
            if req.opponent == "sparring" and app.state.sparring is not None:
                return state(app.state.sparring.new_sparring_game(req.color, req.focus))
            return state(service().new_game(req.color, req.opponent, req.elo))
        except ValueError as e:
            raise HTTPException(400, str(e)) from e

    def sparring_service() -> SparringService:
        if app.state.sparring is None:
            raise HTTPException(409, "sparring needs the trained model")
        return app.state.sparring

    @app.get("/api/sparring")
    def sparring_status():
        if app.state.sparring is None:
            return {"available": False}
        return {"available": True, **app.state.sparring.status()}

    @app.post("/api/drills")
    def new_drill(req: NewDrill):
        try:
            record, drill = sparring_service().new_drill(req.tag)
        except ValueError as e:
            raise HTTPException(400, str(e)) from e
        return {**state(record), "drill": drill}

    @app.get("/api/games/{game_id}/focus")
    def game_focus(game_id: int):
        sp = sparring_service()
        record = _found(lambda: service().get(game_id))
        bot = service().opponents["bot"]
        return sp.focus(record.id, lambda boards: bot.expected_scores(boards, record.opponent_elo))

    @app.get("/api/games")
    def list_games(limit: int = 50):
        return [
            {
                "id": r.id,
                "created_at": r.created_at,
                "player_color": r.player_color,
                "opponent": r.opponent,
                "opponent_elo": r.opponent_elo,
                "status": r.status,
                "result": r.result,
                "termination": r.termination,
                "num_moves": len(r.moves),
            }
            for r in service().store.list(limit)
        ]

    @app.get("/api/games/{game_id}")
    def get_game(game_id: int):
        return state(_found(lambda: service().get(game_id)))

    @app.post("/api/games/{game_id}/move")
    def move(game_id: int, req: Move):
        return state(_found(lambda: service().play(game_id, req.uci)))

    @app.post("/api/games/{game_id}/resign")
    def resign(game_id: int):
        return state(_found(lambda: service().resign(game_id)))

    @app.get("/api/games/{game_id}/pgn", response_class=PlainTextResponse)
    def pgn(game_id: int):
        record = _found(lambda: service().get(game_id))
        if record.pgn is None:
            raise HTTPException(409, "game is not finished")
        return record.pgn

    @app.get("/api/weakness")
    def weakness(recent: int = 1000):
        return app.state.weakness.report(max(1, min(recent, 100_000)))

    @app.get("/api/weakness/status")
    def weakness_status():
        return app.state.weakness.status()

    @app.post("/api/weakness/update")
    def weakness_update():
        return {"started": app.state.weakness.update()}

    @app.get("/")
    def index():
        return FileResponse(WEB_DIR / "index.html")

    @app.get("/weakness")
    def weakness_page():
        return FileResponse(WEB_DIR / "weakness.html")

    app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")
    return app


def _found(fn):
    try:
        return fn()
    except GameNotFound as e:
        raise HTTPException(404, "game not found") from e
    except (IllegalMove, GameOver) as e:
        raise HTTPException(400, str(e)) from e
