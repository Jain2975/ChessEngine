"""Diversity / exploration checks for the self-play data generator.

Before these features existed, both sides of a self-play game used the same
weights, the engine is deterministic, and only a 10-line opening list was
available -- so 20 "games" per generation were really ~10 distinct games
repeated.  These tests verify the remedies:

  * ml/openings.txt  -- a real 75-line opening book (external, editable file)
  * random playouts  -- unlimited random-but-sane opening positions
  * move noise       -- sample moves from the top-N root moves (softmax over
                        eval scores) so even the same opening diverges
  * asymmetric self-play -- different weights for White and Black

Run:  python tests/test_ml_diversity.py
"""
import os
import random
import shutil
import sys
import tempfile
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ML = os.path.join(ROOT, "ml")
sys.path.insert(0, ML)

import chess                                                        # noqa: E402

from sbe_engine import (EngineHandle, OPENINGS, load_book,           # noqa: E402
                        pick_opening, play_game, random_playout)
from selfplay import generate_games                                  # noqa: E402

WEIGHTS_INIT = os.path.join(ML, "weights_init.txt")
WEIGHTS_GEN2 = os.path.join(ML, "weights_gen2.txt")
VALUES = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3,
          chess.ROOK: 5, chess.QUEEN: 9}

FAILS = []


def check(name, ok, detail=""):
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""),
          flush=True)
    if not ok:
        FAILS.append(name)
    return ok


def replay(moves):
    """Return the board after playing `moves` (raises on illegal moves)."""
    board = chess.Board()
    for mv in moves:
        board.push_uci(mv)
    return board


def material_diff(moves):
    diff = 0
    for _sq, piece in replay(moves).piece_map().items():
        val = VALUES.get(piece.piece_type, 0)
        diff += val if piece.color == chess.WHITE else -val
    return diff


# ---------------------------------------------------------------- 1 opening book
def test_book():
    book = load_book(os.path.join(ML, "openings.txt"))
    check("opening book loads", len(book) >= 50, f"{len(book)} lines")
    check("opening book lines are unique", len({tuple(l) for l in book}) == len(book))
    bad = []
    for i, line in enumerate(book):
        try:
            replay(line)
        except Exception as exc:                      # illegal UCI in the book
            bad.append(f"line {i + 1}: {' '.join(line)} ({str(exc)[:40]})")
    check("every book line is legal", not bad, "; ".join(bad[:3]))
    check("missing book falls back to built-in lines",
          len(load_book(os.path.join(ML, "no_such_book.txt"))) == len(OPENINGS))
    check("opening-mode none = empty opening",
          pick_opening(random.Random(0), "none", 6) == [])
    whole = pick_opening(random.Random(0), "book", 0)
    check("opening-plies 0 plays the whole line", len(whole) >= 3, " ".join(whole))


# --------------------------------------------------- 2 random (playout) openings
def test_random_playouts():
    rng = random.Random(4)
    outs = [random_playout(rng, 8) for _ in range(12)]
    legal, err = True, ""
    for moves in outs:
        try:
            replay(moves)
        except Exception as exc:
            legal, err = False, str(exc)[:60]
    check("random playouts are legal", legal, err)
    distinct = len({tuple(o) for o in outs})
    check("random playouts are all distinct", distinct == len(outs),
          f"{distinct}/{len(outs)} distinct")
    diffs = [abs(material_diff(o)) for o in outs]
    check("random playouts keep the material balanced", max(diffs) == 0,
          f"max material difference = {max(diffs)}")
    check("playouts reproduce with the same seed",
          random_playout(random.Random(9), 6) == random_playout(random.Random(9), 6))


# ------------------------------------------------------------- 3 move-level noise
def test_move_noise():
    # quiet, non-trivial middlegame so MultiPV has something to spread over
    fen = "r1bqkbnr/pppp1ppp/2n5/4p3/4B3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4"
    board = chess.Board(fen)
    eng = EngineHandle(hash_mb=8)
    try:
        best = eng.play(board, depth=3)
        off = {eng.play_noisy(board, depth=3, multipv=5, temperature=0.0)
               for _ in range(6)}
        check("noise off is deterministic and plays the best move",
              off == {best}, f"played {off}")
        rng = random.Random(3)
        noisy = [eng.play_noisy(board, depth=3, multipv=5, temperature=1.0, rng=rng)
                 for _ in range(25)]
        check("noise on explores moves other than the best",
              len(set(noisy)) > 1 and set(noisy) <= set(board.legal_moves),
              f"{len(set(noisy))} distinct moves in 25 samples")
        check("engine without noise is a constant (why the noise exists)",
              len({eng.play(board, depth=3) for _ in range(6)}) == 1)
    finally:
        eng.close()


# ------------------------------------------- 4 repeated games from one opening
def test_repeat_games():
    # One engine process per side, exactly like selfplay.py / arena.py do it.
    eng_w = EngineHandle(hash_mb=8)
    eng_b = EngineHandle(hash_mb=8)
    opening = ["e2e4", "e7e5"]
    try:
        plain = [play_game(eng_w, eng_b, depth=2, max_plies=30, opening=opening,
                           rng=random.Random(11))[2] for _ in range(3)]
        check("same opening without noise -> literally the same game",
              len({tuple(m) for m in plain}) == 1,
              f"{len({tuple(m) for m in plain})} distinct of {len(plain)}")
        rng = random.Random(11)
        noisy = [m for m in
                 (play_game(eng_w, eng_b, depth=2, max_plies=30, opening=opening,
                            rng=rng, noise=5, temperature=1.0)[2]
                  for _ in range(4)) if m]
        check("same opening with noise -> the games diverge",
              len({tuple(m) for m in noisy}) == len(noisy),
              f"{len({tuple(m) for m in noisy})} distinct of {len(noisy)}")
    finally:
        eng_w.close()
        eng_b.close()


# ------------------------------------------------------ 5 asymmetric self-play
def test_gauntlet():
    if not os.path.exists(WEIGHTS_GEN2):
        check("gauntlet self-play (skipped: weights_gen2.txt missing)", True)
        return
    a = EngineHandle(weights=WEIGHTS_INIT, hash_mb=8)
    b = EngineHandle(weights=WEIGHTS_GEN2, hash_mb=8)
    opening = ["e2e4", "e7e5"]
    try:
        mirror = play_game(a, a, depth=3, max_plies=30, opening=opening,
                           rng=random.Random(5))[2]
        gauntlet = play_game(a, b, depth=3, max_plies=30, opening=opening,
                             rng=random.Random(5))[2]
        check("gauntlet game is legal and non-empty", bool(gauntlet),
              f"{len(gauntlet)} plies")
        check("different weights for Black change the game", mirror != gauntlet,
              f"mirror {len(mirror)} plies vs gauntlet {len(gauntlet)} plies")
    finally:
        a.close()
        b.close()


# ------------------------------- 6 measured A/B: how much more data we get now
def test_diversity_ab():
    workdir = tempfile.mkdtemp(prefix="sbe_div_")
    legacy_path = os.path.join(workdir, "legacy.jsonl")
    diverse_path = os.path.join(workdir, "diverse.jsonl")
    games = 16
    try:
        # BEFORE: the previous behaviour -- built-in 10 line book, no noise
        _n1, legacy = generate_games(legacy_path, games=games, depth=1, max_plies=24,
                                     seed=1, verbose=False, opening_mode="book",
                                     book=OPENINGS, opening_plies=0, noise=0)
        # AFTER: random 8-ply openings + softmax move noise
        _n2, diverse = generate_games(diverse_path, games=games, depth=1, max_plies=24,
                                      seed=1, verbose=False, opening_mode="random",
                                      opening_plies=8, noise=4, temperature=1.0)
        print(f"       before: {legacy['distinct_games']}/{legacy['games_played']} "
              f"distinct games ({legacy['positions']} positions)")
        print(f"       after : {diverse['distinct_games']}/{diverse['games_played']} "
              f"distinct games ({diverse['positions']} positions)")
        check("legacy settings repeat games (built-in book pigeonhole)",
              legacy["distinct_games"] <= len(OPENINGS),
              f"{legacy['distinct_games']} <= {len(OPENINGS)}")
        check("new settings give far more distinct games",
              diverse["distinct_games"] > legacy["distinct_games"],
              f"{legacy['distinct_games']} -> {diverse['distinct_games']}")
        check("new settings still produce labelled positions",
              diverse["positions"] > 0 and diverse["games_played"] == games)
        # the dataset must stay readable by train.py (fen + result per row)
        import json
        rows, keys_ok = 0, True
        with open(diverse_path, encoding="utf-8") as fh:
            for line in fh:
                rec = json.loads(line)
                rows += 1
                if "fen" not in rec or "result" not in rec or "game" not in rec:
                    keys_ok = False
                    break
        check("dataset rows keep the training keys (fen/result/game)", keys_ok,
              f"{rows} rows")
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


def main():
    t0 = time.time()
    for test in (test_book, test_random_playouts, test_move_noise,
                 test_repeat_games, test_gauntlet, test_diversity_ab):
        print(f"\n=== {test.__name__} ===")
        test()
    print(f"\n{len(FAILS)} failure(s) in {time.time() - t0:.1f}s")
    if FAILS:
        print("FAILED: " + ", ".join(FAILS))
        return 1
    print("ALL DIVERSITY CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())

