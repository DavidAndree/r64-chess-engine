"""Game flow: start a game, apply the player's move, let the opponent reply, detect the end."""

from __future__ import annotations

import random
from collections.abc import Callable

import chess
import chess.pgn

from r64.opponents import Opponent
from r64.storage import GameRecord, GameStore


class IllegalMove(ValueError):
    pass


class GameOver(ValueError):
    pass


class GameNotFound(LookupError):
    pass


def board_from_moves(moves: list[str]) -> chess.Board:
    board = chess.Board()
    for uci in moves:
        board.push_uci(uci)
    return board


def outcome(board: chess.Board) -> tuple[str, str] | None:
    """(result, termination) if the game is over, else None.

    Threefold repetition and the 50-move rule end the game automatically.
    """
    o = board.outcome(claim_draw=True)
    if o is None:
        return None
    return o.result(), o.termination.name.lower()


def to_pgn(record: GameRecord, board: chess.Board, result: str, termination: str) -> str:
    game = chess.pgn.Game.from_board(board)
    opp = record.opponent + (f" ({record.opponent_elo})" if record.opponent_elo else "")
    game.headers["Event"] = "r64 game"
    game.headers["Site"] = "r64"
    game.headers["Date"] = record.created_at[:10].replace("-", ".")
    game.headers["White"] = "me" if record.player_color == "white" else opp
    game.headers["Black"] = "me" if record.player_color == "black" else opp
    game.headers["Result"] = result
    game.headers["Termination"] = termination
    return str(game)


class GameService:
    def __init__(self, store: GameStore, opponents: dict[str, Opponent]):
        self.store = store
        self.opponents = opponents
        self.on_finish: list[Callable[[GameRecord], None]] = []  # called after a game ends

    def new_game(self, color: str, opponent: str, elo: int | None) -> GameRecord:
        if color == "random":
            color = random.choice(["white", "black"])
        if color not in ("white", "black"):
            raise ValueError(f"bad color: {color}")
        if opponent not in self.opponents:
            raise ValueError(f"unknown opponent: {opponent}")
        opp = self.opponents[opponent]
        if elo is not None and not opp.min_elo <= elo <= opp.max_elo:
            span = opp.min_elo if opp.min_elo == opp.max_elo else f"{opp.min_elo}-{opp.max_elo}"
            raise ValueError(f"{opp.label} plays at {span}, not {elo}")
        game_id = self.store.create(color, opponent, elo)
        if color == "black":
            self._opponent_reply(self._get(game_id))
        return self._get(game_id)

    def play(self, game_id: int, uci: str) -> GameRecord:
        record = self._get(game_id)
        if record.status != "ongoing":
            raise GameOver("game is over")
        board = board_from_moves(record.moves)
        if _color_name(board.turn) != record.player_color:
            raise IllegalMove("not your turn")
        try:
            move = chess.Move.from_uci(uci)
        except ValueError as e:
            raise IllegalMove(f"bad move format: {uci}") from e
        if move not in board.legal_moves:
            raise IllegalMove(f"illegal move: {uci}")

        board.push(move)
        moves = record.moves + [uci]
        if self._finish_if_over(record, board, moves):
            return self._get(game_id)
        self.store.set_moves(game_id, moves)
        self._opponent_reply(self._get(game_id))
        return self._get(game_id)

    def resign(self, game_id: int) -> GameRecord:
        record = self._get(game_id)
        if record.status != "ongoing":
            raise GameOver("game is over")
        board = board_from_moves(record.moves)
        result = "0-1" if record.player_color == "white" else "1-0"
        pgn = to_pgn(record, board, result, "resignation")
        self.store.finish(game_id, record.moves, result, "resignation", pgn)
        self._finished(game_id)
        return self._get(game_id)

    def get(self, game_id: int) -> GameRecord:
        return self._get(game_id)

    def _get(self, game_id: int) -> GameRecord:
        record = self.store.get(game_id)
        if record is None:
            raise GameNotFound(game_id)
        return record

    def _opponent_reply(self, record: GameRecord) -> None:
        board = board_from_moves(record.moves)
        move = self.opponents[record.opponent].choose_move(board, record.opponent_elo)
        board.push(move)
        moves = record.moves + [move.uci()]
        if not self._finish_if_over(record, board, moves):
            self.store.set_moves(record.id, moves)

    def _finish_if_over(self, record: GameRecord, board: chess.Board, moves: list[str]) -> bool:
        over = outcome(board)
        if over is None:
            return False
        result, termination = over
        self.store.finish(record.id, moves, result, termination, to_pgn(record, board, result, termination))
        self._finished(record.id)
        return True

    def _finished(self, game_id: int) -> None:
        record = self._get(game_id)
        for fn in self.on_finish:
            fn(record)


def _color_name(turn: chess.Color) -> str:
    return "white" if turn == chess.WHITE else "black"


def state(record: GameRecord) -> dict:
    """JSON-friendly game state for the web UI."""
    board = board_from_moves(record.moves)
    san, replay = [], chess.Board()
    for uci in record.moves:
        move = chess.Move.from_uci(uci)
        san.append(replay.san(move))
        replay.push(move)

    dests: dict[str, list[str]] = {}
    if record.status == "ongoing":
        for m in board.legal_moves:
            dests.setdefault(chess.square_name(m.from_square), []).append(
                chess.square_name(m.to_square)
            )

    last = record.moves[-1] if record.moves else None
    return {
        "id": record.id,
        "fen": board.fen(),
        "turn": _color_name(board.turn),
        "player_color": record.player_color,
        "opponent": record.opponent,
        "opponent_elo": record.opponent_elo,
        "moves": record.moves,
        "san": san,
        "last_move": [last[:2], last[2:4]] if last else None,
        "check": board.is_check(),
        "dests": dests,
        "status": record.status,
        "result": record.result,
        "termination": record.termination,
    }
