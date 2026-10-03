# Simple Bitboard Chess Engine (C++17)

A self-contained UCI chess engine written in one translation unit (`main.cpp`,
~1.8k lines) using only the C++ standard library. It prioritises **correctness
first** (validated with perft and with cross-checks against independent
implementations) and then **speed** (magic-bitboard move generation and an
alpha-beta search with a transposition table).

```
main.cpp          the whole engine (single translation unit)
Makefile          make / mingw32-make build (g++ -O2 -std=c++17)
CMakeLists.txt    equivalent CMake build
README.md         this file
COMMANDS.md       every command: build, test, UCI, self-play, training, GUI
tests/
  validate_uci.py           python-chess based end-to-end UCI validation harness
  test_weight_roundtrip.py  learned eval must reproduce the hand-tuned eval
  test_gui_smoke.py         headless GUI smoke test (needs a display)
  test_ml_diversity.py      opening book, random openings, move noise, gauntlet
ml/                         machine-learning layer (self-play / training / GUI)
  README.md                 how the learning loop works + measured results
  openings.txt              75 real opening lines used to start games
  features.py  sbe_engine.py  selfplay.py  train.py  arena.py  learn_loop.py  gui.py
```

**In a hurry? [COMMANDS.md](COMMANDS.md) has a 6-command cheat sheet** (build →
test → export weights → learn → watch the GUI).

---

## 1. Build

### Make (Linux/macOS/MinGW)

```sh
make                 # g++ -O2 -std=c++17 -Wall -Wextra -o chessengine(.exe)
make native          # additionally uses -O3 -march=native (fastest)
make test            # perft suites + internal consistency check
make bench           # aggregate perft benchmark
```

On Windows with MinGW use `mingw32-make` instead of `make`.

### CMake

```sh
cmake -S . -B build && cmake --build build --config Release
ctest --test-dir build
```

### Manual

```sh
g++ -O2 -std=c++17 -Wall -Wextra -o chessengine main.cpp
```

Adding `-march=native` (or `-mpopcnt -mbmi`) speeds up the bit intrinsics
noticeably; the code falls back to portable operations otherwise.

---

## 2. Architecture

### 2.1 Board representation — bitboards

* `U64 pieces[2][6]` — one bitboard per colour/piece type (12 total).
* `U64 occ[3]` — white occupancy, black occupancy, and their union.
* `int board[64] / colorOf[64]` — a small mailbox kept in sync for O(1)
  "what is on this square" queries used by move ordering, eval and make/unmake.
* Square indexing: `a1 = 0 … h8 = 63`, `file = sq & 7`, `rank = sq >> 3`.
* `Position` also carries side to move, castling-rights bitmask, en-passant
  square, halfmove/fullmove counters, the incremental Zobrist key and a ring of
  position keys (for repetition detection).
* Standard queries: `pieceAt`, `isAttacked(sq, by)`, `inCheck(colour)`,
  `kingSq(colour)`, `computeHash()`.

**Sliding attacks: magic bitboards (chosen over classical ray-based lookup).**
A magic lookup is a single `(occ & mask) * magic >> shift` followed by one table
read, with a tiny memory footprint on cache-friendly hot squares; ray-based
(hyperbola/kogge-stone) lookups need several shifts/masks and are measurably
slower in this make/unmake-heavy perft/search loop. The classic 64-bit magics
are **generated at startup** by the standard collision-search procedure using a
fixed-seed xorshift PRNG (deterministic, so no magic constant table is trusted
and no external data is needed). Tables: 64×4096 rook entries (2 MB) and
64×512 bishop entries (256 KB), with knight/king/pawn attack tables precomputed.

### 2.2 Move generation

Moves are packed in 32 bits: `from`, `to`, `promotion piece`, capture flag and a
2-bit flag (`1 = double push`, `2 = en passant`, `3 = castling`).

Generation is **pseudo-legal**, then filtered by legality:

```
genPseudo()  ->  for each move: make(); if (!inCheck(mover)) keep; unmake();
```

`genLegal()` (used by the UCI layer and search) plus the same filter inline in
`perft()`/`alphabeta()`/`qsearch()`. This "make and test" scheme costs one
make/unmake per candidate move, but it guarantees full legality — including the
three hard cases that break naive generators:

* **Castling** — rights are tracked in a 4-bit mask and maintained with a
  `CR_CLEAR[from] & CR_CLEAR[to]` table (`&=` on every make). Generation also
  requires the target squares to be empty and the king's origin, transit and
  destination squares to be unattacked, so castling out of, through or into
  check is impossible (the make/test filter re-checks the destination).
* **En passant** — the capture is encoded with its own flag; make/unmake remove
  and restore the captured pawn on a square that is *not* the destination.
  Because legality is verified after the move is applied, exotic cases such as
  en-passant discoveries along a rank or a diagonal double pin are handled
  automatically (perft positions 3 and 4 exercise these).
* **Promotion** — all four pieces (queen, rook, bishop, knight) are generated
  for both pushes and captures, and are treated as capture-type moves in
  quiescence search.

Perft counts leaves with bulk counting at depth 1 (the number of legal moves is
returned instead of recursing), which is standard and does not change the counts.

### 2.3 Evaluation (pluggable)

`evaluate(const Position&)` returns a score in centipawns **from the side to
move's point of view**. It uses PeSTO-style tapered evaluation: midgame/endgame
material values plus midgame/endgame piece-square tables, interpolated by the
remaining game phase (0…24). It is a plain free function with no hidden state, so
it can be replaced (or extended) without touching search — search reaches it only
through three `evaluate(pos)` calls.

### 2.4 Search

* Negamax **alpha-beta** (fail-soft). No PVS/LMR/null-move pruning: simplicity
  was preferred over knock-out power since correctness came first.
* **Iterative deepening** with a wall-clock limit; the best move of the last
---

## 3. Running the perft tests

```sh
./chessengine test                          # all perft suites + consistency check
./chessengine perft 6                       # perft(6) from the start position
./chessengine perft 5 "<FEN>"               # perft(5) from any FEN
```

Inside the UCI loop, `position …` followed by `perft <d>` prints a
`divide`-style per-move breakdown and a `Total:` line.

`test` prints PASS/FAIL for every depth:

| position | depths | expected |
|---|---|---|
| start position | 1–6 | 20 / 400 / 8,902 / 197,281 / 4,865,609 / 119,060,324 |
| Kiwipete `r3k2r/p1ppqpb1/…` | 1–4 | 48 / 2,039 / 97,862 / 4,085,603 |
| CPW position 3 (EP endgame) | 1–5 | 14 / 191 / 2,812 / 43,238 / 674,624 |
| CPW position 4 (promotions) | 1–4 | 6 / 264 / 9,467 / 422,333 |
| CPW position 5 (castling/pins) | 1–4 | 44 / 1,486 / 62,379 / 2,103,487 |
| CPW position 6 (middlegame) | 1–4 | 46 / 2,079 / 89,890 / 3,894,594 |

`test` also runs `consistencyCheck()`, which plays random games and verifies at
every ply that

1. the incremental Zobrist key equals a from-scratch recomputation,
2. `board[]`/`colorOf[]` agree with the piece bitboards and occupancy, and
3. `unmakeMove` restores the previous position exactly (keys, counters, castling
   and en-passant state).

An independent end-to-end check using **python-chess** (no code shared with the
engine) is provided:

```sh
python -m pip install chess
python tests/validate_uci.py
```

It validates the UCI handshake/options, bestmove legality in tricky positions,
mate-in-1 reporting, `0000` for stalemate/checkmate, `movetime` accuracy,
`go infinite` + `stop` responsiveness, a full 80-ply self-play game, and it
compares the engine's `perft` output against python-chess' own legal move
generator on six standard positions.

---

## 4. Measured performance

Measured on this machine: **Intel Core Ultra 7 255H**, Windows 11 (NT 10.0.26200),
built with MinGW **g++ 6.3.0 -O2 -std=c++17** (no `-march=native`).

### Perft (move generation + make/unmake only)

| workload | nodes | time | rate |
|---|---|---|---|
| perft(6), start position | 119,060,324 | 2.5–2.8 s | **43–47 Mnodes/s** |
| perft(5), Kiwipete | 193,690,690 | 4.1–4.4 s | 44–47 Mnodes/s |
| `bench` aggregate (6 positions) | 319,846,052 | 7.0–7.4 s | **43–46 Mnodes/s** |

So `perft(6)` runs at roughly **45 million nodes/second** here, i.e. ≈ 2.7 s wall
clock (bulk counting at depth 1 is used, which does not change the counts).
Expect different numbers on your machine; the aggregate `make bench` figure
(~320M nodes) is the most stable number to compare.

### Search

Alpha-beta with TT + quiescence + MVV-LVA/killer/history ordering, single
thread, start position:

| depth | nodes | time | search rate |
|---|---|---|---|
| 8 | 2.3 M | 0.76 s | 3.1 Mnodes/s |
| 9 | 11.8 M | 1.79 s | 6.6 Mnodes/s |
| 10 | 61.2 M | 12.0 s | 5.1 Mnodes/s |
| 11 | 459.9 M | 71.5 s | 6.4 Mnodes/s |

Search nodes are several times more expensive than perft leaves (evaluation at
every node, move ordering, TT probing), and the effective branching factor stays
around 7–8 because there is no LMR/null-move pruning — that is why deeper
iterations grow quickly in time. This is expected for the chosen feature set;
because evaluation is a pure function reached from exactly three call sites,
adding pruning techniques or a better eval does not disturb the rest.

---

## 5. Validation status (all green)

* `chessengine test` — **all perft suites PASS** (6 positions, 27 depth checks)
  and **CONSISTENCY CHECK PASSED**.
* `python tests/validate_uci.py` — **ALL UCI VALIDATION CHECKS PASSED**, including
  perft(3) equality with python-chess on six positions, mate detection, `0000`
  handling, `movetime` accuracy (1500 ms requested → 1.44 s), `stop` during
  `go infinite`, and 80 plies of self-play with no illegal or null moves.
* `python tests/test_weight_roundtrip.py` — **ROUND-TRIP OK**: with the exported
  hand-tuned weights the learnable evaluation reproduces the hand-tuned
  evaluation exactly on every test position (the ML layer cannot silently change
  the engine's behaviour).
* `python tests/test_gui_smoke.py` — **GUI SMOKE TEST PASSED**: the tkinter GUI
  builds, both engines start, a random opening is played, a move is played, the
  learning curve redraws.
* `python tests/test_ml_diversity.py` — **ALL DIVERSITY CHECKS PASSED**: all 75
  opening lines are legal, random openings are legal and material-balanced, move
  noise explores beyond the best move, identical openings repeat exactly without
  noise but diverge with it, and the measured distinct-games comparison
  (8/16 with the old settings → **16/16** with the new ones).
* ML pipeline: self-play, training with a held-out validation split, arena
  matches and the full `learn_loop.py` generation were all executed on this
  machine — measured numbers are in [`ml/README.md`](ml/README.md).

---

## 6. Machine-learning layer (both sides learn by playing)

The evaluation is the *pluggable* part of this engine, so it can be **learned**
instead of hand-tuned:

* `main.cpp` exposes the hand-tuned evaluation as a 1562-parameter tapered linear
  model (midgame/endgame material, 12 piece-square planes, tempo) that can be
  exported with `chessengine dumpweights <path>`, replaced with
  `setoption name EvalFile value <path>` and toggled with
  `setoption name UseLearnedEval value true|false`.
* `MultiPV` support reports every root move with its score, which is what makes
  policy/knowledge distillation from the engine's own search possible.
* Exporting the hand-tuned weights and loading them back reproduces the original
  evaluation exactly (`tests/test_weight_roundtrip.py`), so learning always
  starts from a sane engine and can only be judged in the arena.

`ml/` contains the learning loop: `selfplay.py` has the engine play itself and
labels every position with the game result, `train.py` fits the weights with
Adam logistic regression (held-out validation split), `arena.py` plays
colour-swapped matches to estimate the Elo difference and `learn_loop.py` chains
self-play → train → arena-gate → keep/reject. `ml/gui.py` is a tkinter GUI where
**two weight files play each other** (the board, the live evaluation and the
learning curve are all visible) and a "Learn 1 generation" button runs the loop.

```sh
cd ml
../chessengine dumpweights weights_init.txt
python selfplay.py --out data/main.jsonl --games 60 --depth 3 \
                   --opening-mode random --opening-plies 6 --noise 4 --temperature 1.0
python train.py --data data/main.jsonl --init weights_init.txt --out weights_gen1.txt
python arena.py  --a weights_init.txt --b weights_gen1.txt --games 20 --depth 4
python learn_loop.py --generations 5 --games-per-gen 40 --arena-games 20 --noise 4   # or 0 = forever
python gui.py
```

Self-play needs **diverse games** to learn anything: the search is deterministic,
so without `--noise`/random openings every game from the same opening is a
carbon copy (measured 8/16 distinct games with the old settings, 16/16 with the
new ones).  `learn_loop.py` owns its workdir through `learn_loop.lock`, and
writes `games.jsonl`, `history.jsonl`, `weights_genN.txt` and
`weights_current.txt` there; see [`COMMANDS.md`](COMMANDS.md) §4 for every knob.

See [`ml/README.md`](ml/README.md) for the weight layout, the feature mirror, and
honest expectations: a linear model of this size is worth tens of Elo, not
hundreds, and it needs 10^5–10^6 self-play positions to be trained properly.

---

## 7. Limitations / notes

* Single-threaded; no SMP, no pondering (`ponderhit` is ignored).
* No LMR, null-move pruning, aspiration windows, SEE or PVS; evaluation is
  material + PST only, so playing strength is modest (a few hundred Elo below a
  tuned engine).
* `go infinite` is stopped by `stop` or `quit`; a bare EOF does not interrupt a
  search (deliberate, so piped command files run to completion).
* Repetition detection uses the position-key history and is correct within the
  current irreversible-move segment; the engine scores repetitions as draws but
  does not *claim* draws over UCI.
* Magic bitboards are generated at startup (a few milliseconds) with a fixed
  seed, deterministically; nothing is cached to disk.
* Perft uses make/unmake + legality testing instead of a specialised
  legality-checked generator. That is a bit slower than the state of the art
  (bulk-counted perft can reach 100–300 Mnps) but far easier to verify.
  *completed* iteration is returned and the engine always falls back to a legal
  move if stopped during depth 1.
* **Transposition table** (~32 MB default, 16-byte entries, resizable with
  `setoption name Hash`) keyed by the incremental **Zobrist** hash, with
  EXACT/LOWER/UPPER flags, depth-preferred replacement and generation aging;
  mate scores are normalised by ply.
* **Quiescence search** at leaf nodes: stand-pat, then captures and promotions
  only, ordered by MVV-LVA — avoiding the horizon effect on exchanges.
* Move ordering: TT move → captures/promotions (MVV-LVA) → two killers per ply →
  history heuristic.
* Draw handling: fifty-move rule and repetition (against the position-history
  ring) score 0.
* The node counter is sampled every 2048 nodes to check the time budget and poll
  for `stop`.

### 2.5 UCI protocol

Supported: `uci`, `isready`, `ucinewgame`,
`setoption name Hash value <MB>`, `setoption name Clear Hash`,
`setoption name MultiPV value <1..64>`, `setoption name EvalFile value <path>`,
`setoption name UseLearnedEval value true|false`,
`position [startpos | fen <FEN>] [moves …]`,
`go depth | movetime | wtime/btime/winc/binc | movestogo | infinite`, `stop`,
`quit`. `bestmove` is always emitted (with promotion suffix), `0000` when the
side to move has no legal move, and `info depth … score cp|mate … nodes … nps …
time … pv …` is printed for every completed iteration (`multipv k` is added when
MultiPV > 1).

`stop` works while thinking: a background reader thread (Win32 `CreateThread`,
`std::thread` on POSIX) feeds stdin lines into a queue that the search polls.
Commands received during a search are queued and executed afterwards, so nothing
is dropped. This behaves identically for GUI pipes, redirected files and
interactive consoles.

Extra (non-standard, handy) commands: `perft <d>`, `d` (print board), `eval`,
`hash` (incremental vs. recomputed Zobrist), `check` (internal consistency test).

The engine can also be driven from the command line without a GUI:

```sh
./chessengine test                      # perft suites + consistency check
./chessengine check [games]             # random-game validation only
./chessengine bench                     # aggregate perft benchmark
./chessengine perft <d> ["<FEN>"]       # perft a single position
./chessengine dumpweights <path>        # export evaluation weights (for ml/)
```
