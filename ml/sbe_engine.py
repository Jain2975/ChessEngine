"""Thin wrapper around the C++ engine for the ML pipeline.

Responsibilities:
  * start/stop the engine process (via python-chess' UCI protocol)
  * select a learned evaluation weight file at runtime (`EvalFile` option)
  * analyse a position (single PV or MultiPV -> per-move scores) = training targets
  * play a game between two weight files (used by the arena)

Uses python-chess only as a UCI transport + board model; a separate interpreter
process is used for each weight file so weights never have to be reloaded
between moves of a single game.
"""
import math
import os
import random

import chess
import chess.engine

HERE = os.path.dirname(os.path.abspath(__file__))

ENGINE_PATH = os.environ.get(
    "SBE_ENGINE",
    os.path.join(os.path.dirname(HERE), "chessengine.exe"),
)

# Book of UCI opening lines used to start self-play / arena / GUI games.
BOOK_PATH = os.path.join(HERE, "openings.txt")

# A few short, well-known opening lines so self-play games are not all identical.
OPENINGS = [
    [],
    ["e2e4", "e7e5"],
    ["e2e4", "c7c5"],
    ["e2e4", "e7e6"],
    ["d2d4", "d7d5"],
    ["d2d4", "g8f6"],
    ["c2c4", "e7e5"],
    ["g1f3", "d7d5"],
    ["e2e3", "g8f6"],
    ["d2d4", "f7f5"],
]


def random_opening(rng=None, max_plies=2):
    rng = rng or random
    line = list(rng.choice(OPENINGS))
    return line[:max_plies]


# ---------------------------------------------------------------------------
# Opening sources: a real book file, or random legal "playout" openings
# ---------------------------------------------------------------------------
_MATERIAL = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3,
             chess.ROOK: 5, chess.QUEEN: 9}


def load_book(path=None):
    """Read UCI opening lines (one per line, '#' comments) -> list of move lists.

    Falls back to the built-in OPENINGS list when the file is missing or empty,
    so the pipeline still works if the book file is deleted.
    """
    path = path or BOOK_PATH
    lines = []
    try:
        with open(path, encoding="utf-8") as fh:
            for raw in fh:
                line = raw.split("#", 1)[0].strip()
                if line:
                    lines.append(line.split())
    except OSError:
        return [list(o) for o in OPENINGS]
    return lines or [list(o) for o in OPENINGS]


def _material_diff(board):
    diff = 0
    for _sq, piece in board.piece_map().items():
        val = _MATERIAL.get(piece.piece_type, 0)
        diff += val if piece.color == chess.WHITE else -val
    return diff


def random_playout(rng=None, plies=6, balanced_only=True, attempts=12):
    """`plies` random legal moves from the start position.

    This is an *unlimited* opening source: every call gives a fresh position,
    unlike a fixed book.  With `balanced_only` (the default) a playout that
    already captured material is retried, so training data is not polluted by
    positions where a queen was blundered away in the opening.
    """
    rng = rng or random
    plies = max(1, int(plies))
    best = None
    for _ in range(max(1, attempts)):
        board = chess.Board()
        moves = []
        for _ply in range(plies):
            if board.is_game_over(claim_draw=False):
                break
            mv = rng.choice(list(board.legal_moves))
            moves.append(mv.uci())
            board.push(mv)
        if best is None or len(moves) > len(best):
            best = moves
        if not balanced_only or _material_diff(board) == 0:
            return moves
    return best or []


def pick_opening(rng=None, mode="book", plies=2, book=None):
    """Return a list of UCI moves to start a game with.

    mode: 'book'   -> a random line from the book (book=path or list of lines),
                      truncated to `plies` moves (plies <= 0 = whole line)
          'random' -> `plies` random legal moves (see random_playout)
          'none'   -> empty opening (plain start position)
    """
    rng = rng or random
    if mode == "none":
        return []
    if mode == "random":
        return random_playout(rng, plies if plies and plies > 0 else 6)
    lines = book if isinstance(book, list) else load_book(book)
    line = list(rng.choice(lines))
    return line if not plies or plies < 0 else line[:plies]


class EngineHandle:
    """One engine process, configured with a specific evaluation."""

    def __init__(self, weights=None, hash_mb=16, use_learned=None):
        self.weights = weights
        opts = {"Hash": int(hash_mb)}
        if weights:
            opts["EvalFile"] = os.path.abspath(weights)
            opts["UseLearnedEval"] = True
        elif use_learned is False:
            opts["UseLearnedEval"] = False
        self.engine = chess.engine.SimpleEngine.popen_uci(ENGINE_PATH)
        self.engine.configure(opts)

    def analyse(self, board, depth=None, movetime=None, multipv=None):
        """Return list of (move, score_cp_or_None, mate_or_None, pv) sorted by score."""
        limit = chess.engine.Limit(depth=depth, time=movetime)
        kwargs = {}
        if multipv:
            kwargs["multipv"] = multipv
        infos = self.engine.analyse(board, limit, **kwargs)
        if isinstance(infos, dict):
            infos = [infos]
        out = []
        for info in infos:
            pv = info.get("pv") or []
            if not pv:
                continue
            cp = info["score"].white().score()
            mate = info["score"].white().score(mate_score=100000)
            out.append((pv[0], cp, mate, pv))
        return out

    def play(self, board, depth=None, movetime=None):
        res = self.engine.play(board, chess.engine.Limit(depth=depth, time=movetime))
        return res.move

    def play_noisy(self, board, depth=None, movetime=None, multipv=4,
                   temperature=0.7, rng=None):
        """Sample a move from the top-`multipv` root moves instead of the best.

        p(move) ~ exp((score_move - score_best) / (100 * temperature)), scores in
        centipawns from the side-to-move's point of view.  This is the self-play
        exploration knob: temperature 0 (or multipv <= 1) always plays the best
        move, larger values spread the games out (temperature 1 = a one-pawn
        difference makes the weaker move e times less likely).
        """
        rng = rng or random
        multipv = int(multipv or 0)
        if multipv <= 1 or not temperature or temperature <= 0:
            return self.play(board, depth=depth, movetime=movetime)
        infos = self.analyse(board, depth=depth, movetime=movetime, multipv=multipv)
        if not infos:
            return self.play(board, depth=depth, movetime=movetime)
        sign = 1.0 if board.turn == chess.WHITE else -1.0
        scored = []
        for mv, cp, mate, _pv in infos:
            val = cp if cp is not None else mate
            scored.append((mv, sign * float(val or 0)))
        best = max(v for _mv, v in scored)
        weights = [math.exp((v - best) / (100.0 * temperature))
                   for _mv, v in scored]
        total = sum(weights)
        if total <= 0.0:
            return scored[0][0]
        r = rng.random() * total
        acc = 0.0
        for (mv, _v), w in zip(scored, weights):
            acc += w
            if r <= acc:
                return mv
        return scored[0][0]

    def eval_cp(self, board):
        """Static evaluation of the current position (engine's own eval)."""
        return None      # not exposed over UCI; use analyse() instead

    def close(self):
        try:
            self.engine.quit()
        except Exception:
            try:
                self.engine.close()
            except Exception:
                pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def play_game(white: EngineHandle, black: EngineHandle, depth=3, max_plies=160,
              opening=None, start_fen=None, rng=None, noise=0, temperature=0.7):
    """Play one game. Returns (result_white_pov, list_of_boards_after_each_move, moves).

    `noise` > 1 samples each move from that many top root moves (see
    EngineHandle.play_noisy), which makes repeated games from the same opening
    diverge instead of producing identical copies.
    """
    rng = rng or random
    board = chess.Board(start_fen) if start_fen else chess.Board()
    for mv in (opening or []):
        board.push_uci(mv)
    moves = []
    boards = []
    while not board.is_game_over(claim_draw=False) and len(moves) < max_plies:
        eng = white if board.turn == chess.WHITE else black
        if noise and noise > 1:
            move = eng.play_noisy(board, depth=depth, multipv=noise,
                                  temperature=temperature, rng=rng)
        else:
            move = eng.play(board, depth=depth)
        if move is None or move not in board.legal_moves:
            return None, boards, moves
        board.push(move)
        moves.append(move.uci())
        boards.append(board.copy())
    result = board.result(claim_draw=False)
    if result == "1-0":
        res = 1.0
    elif result == "0-1":
        res = 0.0
    elif result == "1/2-1/2":
        res = 0.5
    else:
        # game not finished (ply cap): score by material as a weak label
        res = 0.5
    return res, boards, moves
