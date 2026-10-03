"""Match two evaluation weight files head to head.

    python ml/arena.py --a ml/weights_init.txt --b ml/gen1.txt --games 20 --depth 4

Each game is played by two engine processes (one per weight file), so weights
never reload mid-game.  Colours alternate, openings are drawn from a small book.
Prints the score (from A's point of view), the Elo difference and a verdict.
"""
import argparse
import json
import math
import os
import random
import sys

import chess

from sbe_engine import EngineHandle, pick_opening


def elo_from_score(score):
    """Elo difference of a player who scored `score` against its opponent.

    score = 1.0 -> very large, 0.5 -> 0, 0.0 -> very negative.
    """
    s = max(1e-6, min(1 - 1e-6, score))
    return -400.0 * math.log10(1.0 / s - 1.0)


def play_match(weights_a, weights_b, games=20, depth=4, max_plies=200,
               hash_mb=16, opening_plies=4, seed=7, verbose=True,
               book=None, opening_mode="book"):
    """A vs B.  Games are played in pairs: the same opening twice, with the
    colours swapped, which removes the opening's own bias from the score."""
    rng = random.Random(seed)
    eng_a = EngineHandle(weights=weights_a, hash_mb=hash_mb)
    eng_b = EngineHandle(weights=weights_b, hash_mb=hash_mb)
    results = []
    tally = {"A": 0, "B": 0, "draw": 0}
    opening = []
    try:
        for g in range(games):
            a_is_white = (g % 2 == 0)
            white, black = (eng_a, eng_b) if a_is_white else (eng_b, eng_a)
            board = chess.Board()
            if g % 2 == 0 or not opening:          # new pair -> new opening
                opening = pick_opening(rng, opening_mode, opening_plies, book)
            for mv in opening:
                board.push_uci(mv)
            plies = 0
            while not board.is_game_over(claim_draw=False) and plies < max_plies:
                eng = white if board.turn == chess.WHITE else black
                mv = eng.play(board, depth=depth)
                if mv is None or mv not in board.legal_moves:
                    break
                board.push(mv)
                plies += 1
            res = board.result(claim_draw=False)
            if res == "1-0":
                a_score = 1.0 if a_is_white else 0.0
            elif res == "0-1":
                a_score = 0.0 if a_is_white else 1.0
            elif res == "1/2-1/2":
                a_score = 0.5
            else:
                a_score = 0.5      # unfinished: count as a draw
            results.append({"game": g + 1, "a_white": a_is_white, "result": res,
                            "plies": plies, "a_score": a_score,
                            "opening": " ".join(opening), "fen": board.fen()})
            if a_score == 1.0:
                tally["A"] += 1
            elif a_score == 0.0:
                tally["B"] += 1
            else:
                tally["draw"] += 1
            if verbose:
                print(f"  game {g + 1}/{games}: A({'W' if a_is_white else 'B'}) "
                      f"score={a_score} ({res}) plies={plies}", flush=True)
    finally:
        eng_a.close()
        eng_b.close()
    score = sum(r["a_score"] for r in results) / max(1, len(results))
    candidate_score = 1.0 - score
    summary = {
        "games": len(results),
        "a_score": score,
        "candidate_score": candidate_score,
        "a_wins": tally["A"],
        "b_wins": tally["B"],
        "draws": tally["draw"],
        # positive => the NEW/candidate weights are stronger than A (the old ones)
        "elo_candidate_minus_current": elo_from_score(candidate_score),
        "weights_a": weights_a or "hand-tuned",
        "weights_b": weights_b or "hand-tuned",
        "results": results,
    }
    return summary


def main():
    ap = argparse.ArgumentParser(description="Arena match between weights")
    ap.add_argument("--a", default="weights_init.txt")
    ap.add_argument("--b", required=True)
    ap.add_argument("--games", type=int, default=20)
    ap.add_argument("--depth", type=int, default=4)
    ap.add_argument("--max-plies", type=int, default=200)
    ap.add_argument("--opening-plies", type=int, default=4,
                    help="moves of the book line to play (0 = whole line)")
    ap.add_argument("--opening-mode", choices=("book", "random", "none"),
                    default="book")
    ap.add_argument("--book", default=None,
                    help="opening book file (default ml/openings.txt)")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args()

    s = play_match(args.a, args.b, games=args.games, depth=args.depth,
                   max_plies=args.max_plies, opening_plies=args.opening_plies,
                   seed=args.seed, book=args.book, opening_mode=args.opening_mode)
    print(f"\nA={args.a}  B={args.b}")
    print(f"games={s['games']}  A wins={s['a_wins']}  B wins={s['b_wins']}  "
          f"draws={s['draws']}")
    print(f"A score={s['a_score']:.3f}  B score={s['candidate_score']:.3f}  ->  "
          f"B is {s['elo_candidate_minus_current']:+.1f} Elo vs A "
          f"({'B stronger' if s['elo_candidate_minus_current'] > 0 else 'A stronger'})")
    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as fh:
            json.dump(s, fh, indent=2)
    return 0


if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    sys.exit(main())
