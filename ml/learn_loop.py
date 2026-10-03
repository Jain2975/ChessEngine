"""The learning loop: self-play -> train -> verify (arena) -> keep or reject.

This is the "learns by playing itself" driver.  Each generation:

  1. the current weights play themselves (both sides) to produce fresh data
  2. a trainer fits new weights on all data collected so far
  3. an arena match (candidate vs current, colour-swapped) decides whether the
     candidate is accepted, so the loop can never silently regress
  4. progress is appended to ml/data/history.jsonl (the GUI reads this file)

    python ml/learn_loop.py --generations 5 --games-per-gen 10 --depth 3 \
                            --arena-games 12 --arena-depth 4

Starting point is whatever you pass with --start (default: hand-tuned weights
exported by the engine's `dumpweights` command).
"""
import argparse
import json
import os
import shutil
import sys
import time

import features as F
from arena import play_match
from selfplay import generate_games
from train import train

HERE = os.path.dirname(os.path.abspath(__file__))

LOCK_STALE_HOURS = 12


def append_history(rec, path):
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(rec) + "\n")


def acquire_lock(path, force=False):
    """Refuse to run two loops in the same workdir.

    Two concurrent loops would fight over the dataset and the current weights
    (and would overwrite each other's generations), so one run owns a workdir.
    A lock older than LOCK_STALE_HOURS is treated as left over from a crash.
    """
    if os.path.exists(path) and not force:
        age_h = (time.time() - os.path.getmtime(path)) / 3600.0
        if age_h < LOCK_STALE_HOURS:
            with open(path, encoding="utf-8") as fh:
                info = fh.read().strip()
            raise SystemExit(
                f"another learning loop looks alive in this workdir:\n"
                f"  lock  : {path} ({age_h:.1f} h old)\n"
                f"  holder: {info}\n"
                f"Stop that run, delete the lock file, or pass --force to take over.")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(f"pid={os.getpid()} started={time.strftime('%Y-%m-%d %H:%M:%S')}")


def release_lock(path):
    try:
        os.remove(path)
    except OSError:
        pass


def main():
    ap = argparse.ArgumentParser(description="Self-play learning loop")
    ap.add_argument("--start", default=os.path.join(HERE, "weights_init.txt"))
    ap.add_argument("--workdir", default=os.path.join(HERE, "data"))
    ap.add_argument("--generations", type=int, default=5,
                    help="number of generations to run; 0 = run forever (stop with Ctrl+C)")
    ap.add_argument("--games-per-gen", type=int, default=10)
    ap.add_argument("--depth", type=int, default=3, help="self-play search depth")
    ap.add_argument("--opponent", default=None,
                    help="weights for the BLACK side during self-play (default: same "
                         "as the current weights).  Point it at an older generation to "
                         "play a gauntlet instead of a mirror match")
    ap.add_argument("--noise", type=int, default=0, metavar="N",
                    help="self-play explores: sample each move from the N best root "
                         "moves instead of always playing the best (0=off, 4-6 "
                         "recommended).  Without it every game from the same opening "
                         "is a carbon copy")
    ap.add_argument("--temperature", type=float, default=0.7,
                    help="exploration temperature for --noise (default 0.7)")
    ap.add_argument("--opening-mode", choices=("book", "random", "none"),
                    default="book", help="where self-play games start (default book)")
    ap.add_argument("--book", default=None,
                    help="opening book file (default ml/openings.txt)")
    ap.add_argument("--opening-plies", type=int, default=6,
                    help="moves to play from the opening source: book line length "
                         "(0 = whole line) or number of random moves with "
                         "--opening-mode random")
    ap.add_argument("--arena-games", type=int, default=12)
    ap.add_argument("--arena-depth", type=int, default=4)
    ap.add_argument("--min-gain", type=float, default=0.5,
                    help="minimum score the candidate must achieve against the "
                         "current weights to be accepted (0.5 = must not lose)")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--max-positions", type=int, default=60000)
    ap.add_argument("--fresh", action="store_true", help="delete previous data first")
    ap.add_argument("--force", action="store_true",
                    help="take over the workdir even if another loop holds its lock")
    args = ap.parse_args()

    os.makedirs(args.workdir, exist_ok=True)
    data_path = os.path.join(args.workdir, "games.jsonl")
    history_path = os.path.join(args.workdir, "history.jsonl")
    lock_path = os.path.join(args.workdir, "learn_loop.lock")
    if args.fresh:
        for p in (data_path, history_path):
            if os.path.exists(p):
                os.remove(p)
    acquire_lock(lock_path, force=args.force)

    current = args.start
    if not os.path.exists(current):
        release_lock(lock_path)
        raise SystemExit(f"start weights not found: {current}\n"
                         f"create them with:  chessengine dumpweights {current}")

    append_history({"event": "start", "weights": current}, history_path)
    infinite = args.generations <= 0
    print(f"learning loop: {'infinite (stop with Ctrl+C)' if infinite else args.generations} "
          f"generations, start weights = {current}")
    print(f"dataset : {data_path}")
    print(f"weights : {os.path.join(args.workdir, 'weights_genN.txt')} "
          f"and {os.path.join(args.workdir, 'weights_current.txt')}")

    gen = 0
    try:
        while infinite or gen < args.generations:
            gen += 1
            t0 = time.time()
            print(f"\n=== generation {gen} ===")
            n_new, stats = generate_games(data_path, games=args.games_per_gen,
                                          depth=args.depth, weights=current,
                                          seed=1000 + gen,
                                          weights_black=args.opponent,
                                          noise=args.noise,
                                          temperature=args.temperature,
                                          opening_mode=args.opening_mode,
                                          opening_plies=args.opening_plies,
                                          book=args.book)
            candidate = os.path.join(args.workdir, f"weights_gen{gen}.txt")
            train(data_path, current, candidate, epochs=args.epochs, lr=args.lr,
                  max_positions=args.max_positions)

            print(f"arena: gen{gen} (B) vs current (A), {args.arena_games} games at depth "
                  f"{args.arena_depth}")
            summary = play_match(current, candidate, games=args.arena_games,
                                 depth=args.arena_depth, seed=500 + gen,
                                 opening_mode=args.opening_mode,
                                 opening_plies=args.opening_plies, book=args.book)
            # A = current weights, B = candidate.  Accept the candidate when it is
            # at least as good as the current generation (score from B's side).
            accepted = summary["candidate_score"] >= args.min_gain
            if accepted:
                kept = os.path.join(args.workdir, "weights_current.txt")
                shutil.copy(candidate, kept)
                current_next = kept
            else:
                current_next = current

            rec = {
                "event": "generation",
                "generation": gen,
                "positions_total": n_new,
                "selfplay_stats": stats,
                "distinct_games": stats.get("distinct_games"),
                "games_played": stats.get("games_played"),
                "arena": {k: v for k, v in summary.items() if k != "results"},
                "accepted": accepted,
                "elo_candidate_minus_current": summary["elo_candidate_minus_current"],
                "candidate": candidate,
                "current_after": current_next,
                "seconds": round(time.time() - t0, 1),
            }
            append_history(rec, history_path)
            print(f"generation {gen}: {stats.get('distinct_games', 0)}/"
                  f"{stats.get('games_played', 0)} distinct self-play games, "
                  f"candidate {'ACCEPTED' if accepted else 'REJECTED'} "
                  f"({summary['elo_candidate_minus_current']:+.1f} Elo), {rec['seconds']}s")
            current = current_next
    except KeyboardInterrupt:
        print("\nstopped by user (Ctrl+C)")
    finally:
        release_lock(lock_path)

    print(f"\nlearning loop finished after {gen} generation(s). current weights: {current}")
    print(f"history: {history_path}")
    return 0


if __name__ == "__main__":
    sys.path.insert(0, HERE)
    sys.exit(main())
