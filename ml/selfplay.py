"""Self-play data generation (the engine is both player and teacher).

For every game we record, for each position:
  * the FEN (from which features are recomputed later)
  * the game result from White's point of view (training label)
  * optional MultiPV scores of all root moves (policy target for later use)

Position labels are the *game outcome*, which is exactly what makes this
reinforcement-style self-play learning rather than supervised hand-tuning.
"""
import argparse
import json
import os
import random
import time

import chess

from sbe_engine import EngineHandle, pick_opening, play_game


def generate_games(out_path, games=20, depth=3, weights=None, max_plies=140,
                   multipv=0, opening_plies=2, seed=1, verbose=True,
                   weights_white=None, weights_black=None, noise=0,
                   temperature=0.7, opening_mode="book", book=None):
    """Self-play games -> labelled positions.

    Diversity knobs (all optional; the defaults keep the previous behaviour):
      * opening_mode / opening_plies / book -- where each game starts
      * noise / temperature -- sample moves from the top-N root moves so games
        from the same opening diverge instead of being carbon copies
      * weights_white / weights_black -- asymmetric (gauntlet) self-play
    """
    rng = random.Random(seed)
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    white = EngineHandle(weights=weights_white or weights)
    black = EngineHandle(weights=weights_black or weights)
    stats = {"1-0": 0, "0-1": 0, "1/2-1/2": 0, "unfinished": 0,
             "games_played": 0, "distinct_games": 0, "positions": 0,
             "avg_plies": 0.0}
    seen = set()
    written = 0
    plies_total = 0
    t0 = time.time()
    try:
        with open(out_path, "a", encoding="utf-8") as out:
            for g in range(games):
                opening = pick_opening(rng, opening_mode, opening_plies, book)
                res, boards, moves = play_game(white, black, depth=depth,
                                               max_plies=max_plies, opening=opening,
                                               rng=rng, noise=noise,
                                               temperature=temperature)
                if res is None or not boards:
                    continue
                stats["games_played"] += 1
                plies_total += len(moves)
                if res in (0.0, 1.0):
                    stats["1-0" if res == 1.0 else "0-1"] += 1
                else:
                    stats["unfinished"] += 1
                seen.add(tuple(moves))
                stats["distinct_games"] = len(seen)
                for i, board in enumerate(boards):
                    # Skip the very last position of a finished game: it is
                    # already terminal and adds no information.
                    rec = {
                        "fen": board.fen(),
                        "result": res,
                        "ply": i,
                        "game": g,
                        "opening": opening,
                        "depth": depth,
                    }
                    if multipv > 0 and i % max(1, multipv) == 0:
                        infos = (white if board.turn == chess.WHITE else black).analyse(
                            board, depth=max(2, depth - 1), multipv=multipv)
                        rec["targets"] = [[m.uci(), cp] for m, cp, _mate, _pv in infos]
                    out.write(json.dumps(rec) + "\n")
                    written += 1
                out.flush()
                if verbose:
                    done = g + 1
                    rate = written / max(1e-6, time.time() - t0)
                    print(f"  game {done}/{games}  result={res}  plies={len(moves)}  "
                          f"positions={written}  ({rate:.0f} pos/s)  "
                          f"distinct={stats['distinct_games']}", flush=True)
    finally:
        white.close()
        black.close()
    stats["positions"] = written
    stats["avg_plies"] = round(plies_total / max(1, stats["games_played"]), 1)
    if verbose:
        tally = {k: v for k, v in stats.items()
                 if k in ("1-0", "0-1", "1/2-1/2", "unfinished")}
        print(f"self-play done: {written} positions -> {out_path}")
        print(f"  results={tally}")
        print(f"  games={stats['games_played']}  distinct game lines="
              f"{stats['distinct_games']}  avg plies={stats['avg_plies']}")
    return written, stats


def main():
    ap = argparse.ArgumentParser(description="Generate self-play training data")
    ap.add_argument("--out", default="data/games.jsonl")
    ap.add_argument("--games", type=int, default=20)
    ap.add_argument("--depth", type=int, default=3)
    ap.add_argument("--weights", default=None, help="eval weight file for BOTH sides "
                                                   "(default: hand-tuned)")
    ap.add_argument("--weights-white", default=None,
                    help="eval file for White only (asymmetric / gauntlet self-play)")
    ap.add_argument("--weights-black", default=None,
                    help="eval file for Black only (asymmetric / gauntlet self-play)")
    ap.add_argument("--max-plies", type=int, default=140)
    ap.add_argument("--multipv", type=int, default=0,
                    help="record MultiPV policy targets every N positions (0=off)")
    ap.add_argument("--noise", type=int, default=0, metavar="N",
                    help="sample each move from the N best root moves (0=off; "
                         "4-6 gives diverse games)")
    ap.add_argument("--temperature", type=float, default=0.7,
                    help="exploration temperature for --noise (small = strong and "
                         "repetitive, large = random; default 0.7)")
    ap.add_argument("--opening-mode", choices=("book", "random", "none"),
                    default="book",
                    help="book = a line from --book; random = N random legal moves; "
                         "none = always start from the initial position")
    ap.add_argument("--book", default=None,
                    help="opening book file (default ml/openings.txt)")
    ap.add_argument("--opening-plies", type=int, default=2,
                    help="with 'book': how many moves of the line to play "
                         "(0 = whole line); with 'random': how many random moves")
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()
    generate_games(args.out, games=args.games, depth=args.depth,
                   weights=args.weights, max_plies=args.max_plies,
                   multipv=args.multipv, opening_plies=args.opening_plies,
                   seed=args.seed, weights_white=args.weights_white,
                   weights_black=args.weights_black, noise=args.noise,
                   temperature=args.temperature, opening_mode=args.opening_mode,
                   book=args.book)


if __name__ == "__main__":
    main()
