# Command Reference

Every command you need for the SimpleBitboard chess engine and its machine-learning
pipeline, in one place.  Copy-paste friendly.  Commands are shown for **Windows
PowerShell**; on Linux/macOS drop the `.exe` and replace `mingw32-make` with `make`.

---

## 0. Prerequisites

| Tool | Needed for | Install |
|---|---|---|
| g++ (MinGW-w64) or any C++17 compiler | building the engine | <https://www.mingw-w64.org/> or MSYS2 |
| Python 3.10+ | tests, ML pipeline, GUI | <https://www.python.org/> |
| python-chess | tests, ML pipeline, GUI | `pip install python-chess` |
| numpy | training | `pip install numpy` |
| tkinter | GUI | included with CPython on Windows (`sudo apt install python3-tk` on Debian/Ubuntu) |

```powershell
pip install python-chess numpy          # one-time Python setup
```

---

## 1. Build the engine

From the repository root (`D:\ChessEngine`):

```powershell
# MinGW (Windows)
mingw32-make            # -> chessengine.exe  (uses -O2 -std=c++17 -Wall -Wextra)
mingw32-make clean      # delete the binary

# or, directly
g++ -O2 -std=c++17 -Wall -Wextra -o chessengine.exe main.cpp

# or, CMake (any platform / any generator)
cmake -S . -B build
cmake --build build --config Release      # -> build/chessengine(.exe)
```

Expected: a single binary, ~200 KB, no external dependencies.

---

## 2. Verify the engine (run after every build)

```powershell
# 1) perft suites (CPW reference positions) + 20-game internal consistency check
.\chessengine.exe test
#    expect: ALL PERFT TESTS PASSED / ALL EXTRA PERFT TESTS PASSED / CONSISTENCY CHECK PASSED

# 2) longer internal consistency check (self-play vs movegen agreement)
.\chessengine.exe check 50

# 3) full UCI validation against python-chess (handshake, perft parity, mates,
#    movetime, stop during infinite search, 80 plies of self-play)
python tests\validate_uci.py
#    expect: ALL UCI VALIDATION CHECKS PASSED

# 4) ML weight round-trip (learned eval must reproduce the hand-tuned eval exactly)
python tests\test_weight_roundtrip.py
#    expect: ROUND-TRIP OK

# 5) GUI smoke test (builds the window, plays one engine move, shuts down)
python tests\test_gui_smoke.py
#    expect: GUI SMOKE TEST PASSED

# 6) self-play diversity: opening book, random openings, move noise, gauntlet
python tests\test_ml_diversity.py
#    expect: ALL DIVERSITY CHECKS PASSED
```

---

## 3. Engine CLI

```powershell
.\chessengine.exe                    # enter UCI mode (speaks to any UCI GUI)
.\chessengine.exe test               # perft suites + consistency (exit 0 = pass)
.\chessengine.exe check [games]      # internal consistency self-play (default 50)
.\chessengine.exe perft <depth> ["<FEN>"]   # node count for one position
.\chessengine.exe bench              # aggregate perft benchmark / speed test
.\chessengine.exe dumpweights <path> # export the 1562 hand-tuned eval weights
```

Examples:

```powershell
.\chessengine.exe perft 6
.\chessengine.exe perft 5 "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1"
.\chessengine.exe dumpweights ml\weights_init.txt
```

### Useful UCI commands (type these into the engine or any UCI GUI)

```
uci                                        # handshake, lists all options
isready                                    # -> readyok
ucinewgame                                 # reset position, clear TT
position startpos                          # set the initial position
position startpos moves e2e4 e7e5          # ... with moves applied
position fen <fen-string> moves e2e4       # arbitrary FEN
go depth 8                                 # fixed-depth search
go movetime 1500                           # search ~1.5 s
go infinite  +  stop                       # search until told to stop
d                                          # print the board
perft 5                                    # perft from the current position
setoption name Hash value 64               # transposition table in MB
setoption name Threads value 2             # NPS benchmark threads (perft/bench)
setoption name MultiPV value 3             # report best 3 root moves
setoption name EvalFile value ml\weights_gen1.txt
setoption name UseLearnedEval value true   # switch to the learned evaluation
quit
```

---

## 4. Machine-learning pipeline (`ml/`)

All ML commands run **from the `ml\` directory**:

```powershell
cd ml
```

### 4.1 One-time setup — export the hand-tuned weights

```powershell
..\chessengine.exe dumpweights weights_init.txt
```

### 4.2 Self-play — generate labelled training data

```powershell
# recommended: the engine explores, and every game starts from a different
# random-but-sane opening, so no two games are carbon copies
python selfplay.py --out data\main.jsonl --games 60 --depth 3 `
                   --opening-mode random --opening-plies 6 --noise 4 --temperature 1.0

# classic/reproducible: a line from the opening book, always the best move
python selfplay.py --out data\main.jsonl --games 60 --depth 3 --opening-mode book

# self-play with already-trained weights (generation N+1 learns from N)
python selfplay.py --out data\gen2.jsonl --games 60 --depth 3 --weights weights_gen1.txt

# gauntlet: White uses the new weights, Black an older generation
python selfplay.py --out data\gen2.jsonl --games 60 --depth 3 `
                   --weights-white weights_gen2.txt --weights-black weights_init.txt

# also record the engine's own root-move scores as policy targets (slower)
python selfplay.py --out data\policy.jsonl --games 20 --depth 3 --multipv 3
```

Options: `--out` (default `data/games.jsonl`), `--games` (20), `--depth` (3),
`--weights` (both sides; hand-tuned if omitted), `--weights-white` /
`--weights-black` (asymmetric self-play), `--max-plies` (140), `--multipv`
(0=off), `--noise N` (0=off — sample moves from the N best root moves),
`--temperature` (0.7), `--opening-mode book|random|none`,
`--opening-plies` (book line length, or number of random moves),
`--book` (default `ml/openings.txt`), `--seed` (1).

Every run prints `games=… distinct game lines=…` — that is the number that
tells you whether the data is varied or repetitive.

### 4.2b How many distinct games you get (measured)

| settings | distinct games out of 16 (16 positions/game) |
|---|---|
| book, 2 plies of the 10-line built-in list, no noise | **8/16** |
| random openings (6 plies) + `--noise 4 --temperature 1.0` | **16/16** |

The engine is deterministic, so with the same weights and the same opening two
games are *identical* (verified: 3 games -> 1 distinct game). Diversity has to
come from the opening and from exploration:

* `--opening-mode random` — unlimited fresh start positions (material kept
  balanced, so no queen was already blundered away in the "opening")
* `--opening-mode book` — 75 real opening lines in `ml/openings.txt`
  (`--opening-plies 0` plays the whole line: far more variety than `2`)
* `--noise 4` — sample each move from the 4 best root moves, softmax-weighted by
  their eval scores (`p(move) ∝ exp((score−best)/(100·temperature))`); it only
  picks moves the search already considers reasonable
* `--weights-white/--weights-black` — the two sides use different evaluations


### 4.3 Train — fit the 1562 evaluation weights

```powershell
python train.py --data data\main.jsonl --init weights_init.txt --out weights_gen1.txt
```

Options: `--epochs` (40), `--batch-size` (256), `--lr` (0.01), `--l2` (1e-6),
`--pull` (2e-7, L2 pull towards `--init`), `--val-frac` (0.1 held-out
validation), `--max-positions` (all).

Watch the printed **held-out val loss** — it should go *down*.  That is the
honest signal that learning is working.

### 4.4 Arena — measure whether the new weights actually play better

```powershell
# 60 games, alternating colours, depth 4 (~10 min)
python arena.py --a weights_init.txt --b weights_gen1.txt --games 60 --json-out data\arena_gen1.json
```

Options: `--games` (20), `--depth` (4), `--max-plies` (200),
`--opening-plies` (4), `--seed` (7), `--json-out` (save full results).
`--a` defaults to `weights_init.txt`; `--b` is required.

The printed Elo difference has a huge error bar at 60 games (±80 Elo) — use
`--games 200+` for decisions you care about.

### 4.5 Learn loop — the whole cycle, automated, with a safety gate

```powershell
# 5 generations; each = self-play -> train -> arena vs current -> accept/reject
python learn_loop.py --start weights_init.txt --generations 5 --games-per-gen 20 --arena-games 30

# run FOREVER: keep learning generation after generation until you press Ctrl+C
python learn_loop.py --generations 0 --games-per-gen 20 --arena-games 30 `
                     --opening-mode random --opening-plies 6 --noise 4 --temperature 1.0

# gauntlet: self-play against a fixed older generation instead of a mirror match
python learn_loop.py --generations 0 --opponent weights_gen10.txt --noise 4
```

`--generations 0` (or any negative number) means **infinite**: the loop keeps
playing, training and gating forever.  It is safe to leave running — the arena
gate (`--min-gain 0.5`) discards any candidate that loses, so the current
weights can only stay the same or improve.  Stop it any time with **Ctrl+C**;
it finishes the current sentence, prints where it stopped and exits cleanly.
With the settings above, expect roughly **15–30 minutes per generation** on a
desktop CPU.

Where the files go:

```
<workdir>\games.jsonl           accumulating self-play dataset
<workdir>\history.jsonl         one JSON record per generation (the GUI curve)
<workdir>\weights_genN.txt      candidate of generation N
<workdir>\weights_current.txt   the newest ACCEPTED weights (= chain of learning)
<workdir>\learn_loop.lock       owned by the running loop, deleted on exit
```

Each generation prints `N/M distinct self-play games` — if that number stays low
(e.g. 8/20) your games are repeats and the generation adds almost no new
information; raise `--noise`, use `--opening-mode random`, or both.

**One loop per workdir.** A second run in the same workdir refuses to start
("another learning loop looks alive in this workdir") because the two would
overwrite each other's dataset and current weights.  If a run crashed and left a
stale lock behind (>12 h old) it is ignored automatically; use `--force` to take
over immediately.  Use different `--workdir` values to run experiments in
parallel.

Options: `--workdir` (default `ml\data`), `--depth` (3, self-play),
`--arena-depth` (4), `--min-gain` (0.5 = candidate may not lose or it is
rejected and current weights are kept), `--epochs` (20), `--lr` (0.01),
`--max-positions` (60000), `--fresh` (delete previous data first),
`--opponent` (weights for the Black side -> gauntlet instead of mirror),
`--noise N` / `--temperature` (self-play exploration, see 4.2b),
`--opening-mode book|random|none`, `--opening-plies`, `--book`,
`--force` (take over a workdir whose lock is still there).

Accepted generations are appended to `ml\data\history.jsonl` — that is what the
GUI's learning curve plots — and the winner is copied to `ml\weights_current.txt`.

### 4.6 Quick smoke run (minutes, not hours)

```powershell
..\chessengine.exe dumpweights weights_init.txt
python selfplay.py --out data\demo.jsonl --games 8 --depth 2
python train.py --data data\demo.jsonl --init weights_init.txt --out weights_demo.txt --epochs 10
python arena.py --b weights_demo.txt --games 8 --depth 2
python learn_loop.py --generations 1 --games-per-gen 4 --depth 2 --arena-games 4 --arena-depth 2
```

---

## 5. The GUI — watch two learned brains play

```powershell
cd D:\ChessEngine
python ml\gui.py
```

* Pick the evaluation for **White** and **Black** from the drop-downs
  (`hand-tuned`, `weights_init.txt`, `data/weights_gen41.txt`, ...; the newest
  24 generations are listed).
* **Random opening** (checkbox + *random plies*) starts every new game from a
  random-but-sane position, so "New game" no longer replays the same game.
* **Step** plays one engine move; **Auto** plays the whole game.
* **Flip board**, **New game**.
* **Learn 1 generation** runs a full self-play -> train -> arena cycle in the
  background; if the candidate wins its gate it becomes the current weights and
  appears on the learning-curve chart at the bottom.
* The panel on the right shows each engine's depth, score and principal
  variation after every move.

Requires a display; on a headless machine use the scripts in section 4 instead.

---

## 6. Cheat sheet — from zero to learning in 6 commands

```powershell
cd D:\ChessEngine
mingw32-make                                        # 1 build
.\chessengine.exe test                              # 2 verify movegen
python tests\validate_uci.py                        # 3 verify UCI
cd ml
..\chessengine.exe dumpweights weights_init.txt     # 4 export start weights
python learn_loop.py --generations 0 --games-per-gen 20 --arena-games 30 `
                     --opening-mode random --opening-plies 6 --noise 4   # 5 learn forever (Ctrl+C)
cd ..
python ml\gui.py                                    # 6 watch it play
```

---

## 7. Troubleshooting

| Symptom | Fix |
|---|---|
| `python-chess not found` | `pip install python-chess numpy` |
| `start weights not found` | run `chessengine dumpweights ml\weights_init.txt` first |
| GUI does not open | no display — use the scripts; verify with `python -c "import tkinter; tkinter.Tk()"` |
| arena/learn loop feels slow | lower `--depth`/`--games`; every game is a real engine-vs-engine search |
| val loss goes up | too much `--lr` or too little data — collect more self-play games first |
| `weights_current.txt` missing | no candidate has passed the arena gate yet — normal on small runs |
| `another learning loop looks alive in this workdir` | a loop owns that workdir (see 4.5) — stop it, use another `--workdir`, or `--force` |
| generations print `8/20 distinct ... games` | data is repetitive — add `--noise 4` and/or `--opening-mode random` |
| want a clean slate | delete the workdir (`ml\data\`) — or pass `--fresh` to learn_loop |

