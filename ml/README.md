# Machine-learning layer (`ml/`)

This directory turns the engine into a **learning system**: two ML agents play
each other in a GUI, the games they play become training data, and the trained
evaluation is loaded back into the *same* C++ engine. Generation over generation
the playing strength improves (or the candidate is rejected — see the arena gate).

Nothing about the C++ engine is replaced: it stays the fast, perft-verified
referee (move generation, alpha-beta, transposition table, quiescence). Only the
**evaluation weights** are learned, which is the classic, tractable way to make a
chess engine "learn" without a GPU.

```
                 ┌──────────────────────────────────────────────┐
   self-play     │  chessengine.exe  (C++)                      │
   games         │   - legal move gen, alpha-beta, TT, quiesce   │
   ┌───────────► │   - evaluate() = learnable linear model       │
   │             └──────────────┬───────────────────────────────┘
   │                            │  weights file  (1562 doubles)
   │                            ▼
   │             ┌──────────────────────────────────────────────┐
   │             │  Python trainer (numpy)                      │
   │             │   - features.py  mirrors the engine exactly  │
   │             │   - train.py     logistic regression on       │
   │             │                  game results (Texel style)  │
   │             └──────────────┬───────────────────────────────┘
   │                            │  candidate weights
   └──────────── arena gate ────┘  (must not lose vs the current generation)
```

## 1. The learnable evaluation (C++ side)

`main.cpp` exposes the hand-tuned eval as a parameter vector:

| option | meaning |
|---|---|
| `EvalFile <path>` | load a weight file (also enables it) |
| `UseLearnedEval true/false` | switch between hand-tuned and learned eval |
| `MultiPV <1..64>` | report the best N root moves with scores (policy targets) |
| `dumpweights <path>` (CLI) | export the current weights as a training start point |

Weight layout (version 1, `EVAL_NF = 1562` doubles):

```
[   0 ..  11]  mg material      [colour * 6 + pieceType]
[  12 ..  23]  eg material      [colour * 6 + pieceType]
[  24 .. 791]  mg piece-square  [colour][pieceType][square in own frame]
[ 792 ..1559]  eg piece-square  [colour][pieceType][square in own frame]
[1560]         mg tempo bonus
[1561]         eg tempo bonus
```

`evaluate()` interpolates midgame/endgame by the game phase, exactly like the
hand-tuned version, so **exporting the hand-tuned weights and loading them back
reproduces the original evaluation bit for bit** (verified by
`tests/test_weight_roundtrip.py`). Training therefore starts from a sane,
already reasonable engine and can only be judged by the arena.

## 2. Files

| file | purpose |
|---|---|
| `features.py` | feature extraction mirroring the C++ model (train/val matrices, `load_weights`, `save_weights`) |
| `sbe_engine.py` | UCI wrapper (python-chess transport), self-play driver, opening book / random openings, `play_noisy` (top-N sampling) |
| `openings.txt` | 75 real opening lines (UCI, editable) used to start games |
| `selfplay.py` | generate games with both sides driven by the engine; record FEN + game result (+ optional MultiPV targets); reports distinct-game count |
| `train.py` | Adam logistic regression on game outcomes with a held-out validation split |
| `arena.py` | match two weight files (colour-swapped *pairs on the same opening*) and compute Elo |
| `learn_loop.py` | generations: self-play → train → arena gate → keep/reject; workdir ownership lock; writes `data/history.jsonl` |
| `gui.py` | tkinter GUI: board, two weight-file players, random openings, live eval, "Learn 1 generation", learning curve |

## 3. Quick start

```sh
cd ml

# 0) export the hand-tuned weights as the starting point (once)
../chessengine dumpweights weights_init.txt          # or chessengine.exe on Windows

# 1) self-play: the engine plays itself, every position is labelled with the result
#    (--noise + --opening-mode random give genuinely different games; see 4.1)
python selfplay.py --out data/main.jsonl --games 60 --depth 3 --max-plies 120 \
                   --opening-mode random --opening-plies 6 --noise 4 --temperature 1.0

# 2) train: logistic regression of "White wins" on the engine's own features
python train.py --data data/main.jsonl --init weights_init.txt --out weights_gen1.txt \
                --epochs 40 --lr 0.01

# 3) verify: does the new generation really play better?
python arena.py --a weights_init.txt --b weights_gen1.txt --games 20 --depth 4

# 4) or run the whole loop with automatic accept/reject
python learn_loop.py --generations 5 --games-per-gen 40 --arena-games 20 \
                     --depth 3 --arena-depth 4 --noise 4 --opening-mode random

#    ... or forever, until you press Ctrl+C (safe: losing candidates are rejected)
python learn_loop.py --generations 0 --games-per-gen 40 --arena-games 20 --noise 4

# 5) watch it in the GUI (board + learning curve + "Learn 1 generation" button)
python gui.py
```

The loop writes `<workdir>/games.jsonl`, `<workdir>/history.jsonl`,
`<workdir>/weights_genN.txt` and `<workdir>/weights_current.txt`, and owns the
workdir via `<workdir>/learn_loop.lock` — a second loop in the same workdir
refuses to start (use `--force` or another `--workdir`).

## 4. What "learning" means here, and how far it can go

Two learning signals are available to the trainer:

1. **Game outcome (value learning, used by default).** `selfplay.py` labels every
   position with the final result of the game it came from, so the model learns
   "positions that tend to lead to wins are good". This is the standard Texel /
   logistic-regression tuning signal and it is genuine self-play reinforcement:
   no human knowledge is injected beyond the feature set itself.
2. **Search preference (policy distillation, optional).** With
   `--multipv N` the generator also records the engine's *own* root-move scores
   for every position, i.e. what its deeper search preferred. Those targets can
   be used to re-weight samples or to train a policy head later. They are stored
   in the data file (`"targets": [[move, score], ...]`) and each engine query
   only costs a shallow (`depth-1`) search.

### 4.1 Self-play diversity — the real bottleneck (fixed)

The C++ search is **deterministic**: same position, same weights, same depth →
same move.  With one engine process per side (how the pipeline runs), the same
weights and the same opening therefore give a *byte-identical* game — measured:
3 games → 1 distinct game.  The original pipeline only had a 10-line built-in
opening list, so 20 "games" per generation were really ~10 unique games, and the
dataset grew mostly by duplication (a 48k-position run repeated the same lines
often; the generator now prints `distinct=N` so you can see it).

Four knobs now create genuinely new games:

| knob | what it does |
|---|---|
| `--book ml/openings.txt` | 75 real opening lines (editable file; `--opening-plies 0` plays whole lines instead of 2 moves) |
| `--opening-mode random --opening-plies 6` | 6 random legal moves from the start, retried until no material was captured — unlimited distinct start positions that are still sane |
| `--noise 4 --temperature 1.0` | sample each move from the 4 best root moves, softmax-weighted by eval (`p ∝ exp((score−best)/(100·T))`); temperature 0 disables it, so it degenerates to "always best" |
| `--weights-white/--weights-black` (selfplay), `--opponent` (learn loop) | the two sides use different evaluations — a gauntlet, which diverges immediately |

Measured A/B (16 games per configuration, `tests/test_ml_diversity.py`):

| configuration | distinct games |
|---|---|
| built-in 10-line book, no noise (previous behaviour) | **8 / 16** |
| random 6-ply openings + `--noise 4 --temperature 1.0` | **16 / 16** |

Curiosity worth knowing: if a *single* engine process is used for both sides,
killer/history/TT state carries over between moves and can change the choice
among equally-scored moves (measured 3/3 distinct games).  The pipeline always
uses one process per side, which is why it is exactly reproducible.

Cost: `--noise N` makes self-play roughly `N`x more expensive (every move runs a
MultiPV search instead of a normal one), so use 4–6; random openings are free.

Realistic expectations (please read before judging results):

* The model is a **1562-parameter linear evaluation** (material + PST + tempo).
  That is exactly what a hand-tuned engine has, so the ceiling is "the same kind
  of eval, but with better numbers" — typically tens of Elo, not hundreds.
  Because the features are linear, it cannot learn interactions such as
  "rook on an open file is better with a queen on the board"; those need extra
  features (mobility, pawn structure, king safety, …) or a small neural net.
* A meaningful fit needs on the order of **10^5–10^6 positions**. On this machine
  self-play produces roughly **60–70 positions per second** at depth 3, so
  100k positions is about 25 minutes of self-play and a few seconds of training.
  Small datasets (a few thousand positions) mostly fit noise: the arena gate
  will then correctly *reject* the candidate.
* Everything is CPU-only and single-threaded per engine, so this is a
  "watch it learn" project rather than a stockfish-scale training run. The gate
  makes it safe to leave running: a rejected generation is simply discarded.

## 5. How the pieces correspond to "a human learning from games"

| human analogue | implementation |
|---|---|
| playing games | `selfplay.py` (both sides are the engine, one per colour) |
| remembering positions that worked | JSONL dataset of (position, result) |
| forming an intuition about good pieces/squares | logistic regression fitting the material/PST weights |
| noticing a habit was wrong | gradient step away from the hand-tuned prior |
| proving you improved before believing it | `arena.py` gate (colour-swapped matches, Elo estimate) |
| watching progress accumulate | `data/history.jsonl` + the GUI's learning curve |

To imitate **human** games instead (behaviour cloning), only the data loader has
to change: python-chess can read PGN, so a corpus of human games can be turned
into the same (position, result) records, and the per-move targets can be used as
policy labels. That is a natural next step; it needs an external PGN corpus.

## 6. Verification

```sh
# from the repository root
python tests/test_weight_roundtrip.py   # learned model reproduces the hand-tuned eval exactly
python tests/test_ml_diversity.py       # book legality, random openings, move noise, gauntlet,
                                        # and the measured distinct-games A/B (8/16 -> 16/16)
python tests/test_gui_smoke.py          # the GUI builds, random opening works, engines play
```

and the engine's own test suites are unaffected by any of this:

```sh
./chessengine test        # perft + consistency (C++ only)
python tests/validate_uci.py
```

## 7. Measured results on this machine

Hardware: Intel Core Ultra 7 255H, Windows 11; engine built with g++ -O2.
Self-play: depth 3, ~64 positions/s. Arena: depth 4, colour-swapped, 4-ply
opening book.

| run | data | train/val loss (held-out) | val accuracy | arena (60 games, depth 4) |
|---|---|---|---|---|
| demo (8 games) | 407 pos | 0.2117 / — | — | n/a (candidate correctly rejected) |
| gen1 | 12,017 pos | 0.1971 → **0.2094** | 0.888 | **+11.6 Elo** vs hand-tuned (0.517) |
| gen2 (from gen1) | 12,017 pos | 0.1838 → **0.1911** | 0.925 | **+11.6 Elo** vs gen1 (0.517); **0.500 (dead even)** vs hand-tuned |
| unattended loop, 46 generations | 48,186 pos | (per-generation gates only) | — | 30-game gate per generation, all accepted at `--min-gain 0.5` |

The 46-generation run (started before the diversity features landed, so with the
old 10-line book and no noise) shows what the machinery does when left running:
the dataset grows to ~48k positions and every generation either passes the gate
or is discarded. Its *own* gates are 30-game comparisons (±100 Elo), so "all
accepted" means "never clearly worse", not "monotonically stronger" — for a real
strength claim, run `arena.py` with a few hundred games between the first and
last generation.

Interpretation (please read):

* The validation column is the honest one: a **held-out** set that is never
  trained on, and it improves in every generation (0.2094 → 0.1911). The model is
  therefore learning something real about which positions lead to wins rather
  than memorising the training rows.
* The arena Elo is **statistically thin**. With 60 games the 95% confidence
  interval is roughly ±80 Elo, so "+11.6" means "not worse, probably slightly
  better", and the 30-30 result of gen2 against the hand-tuned eval means the
  cumulative gain is still inside the noise. Measuring a genuine 10–20 Elo
  improvement needs several hundred games per comparison (`--games`), or
  hundreds of thousands of self-play positions.
* This is expected: a 1562-parameter **linear** model over material/PST features
  is the same *kind* of function as the hand-tuned eval, so it cannot be much
  stronger. What is demonstrated end to end is the learning machinery:
  self-play → labelled data → fit → arena gate → keep/reject, plus a GUI where
  two learned evaluations play each other and improve (or are rejected).
* The loop is safe to leave running: a candidate must not lose the arena
  (`--min-gain 0.5`) or it is discarded and the current weights stay.

Ways to scale this into a genuinely stronger "learning" engine:

1. **More data** — self-play is the bottleneck (~64 positions/s at depth 3):
   10^5–10^6 positions is the range where the fit stops being noise-limited.
2. **Better features** — add mobility counts, pawn structure (passed/doubled/
   isolated), rook on open file, king safety; all are linear in the same
   framework and would lift the ceiling well above the hand-tuned eval.
3. **Small neural net** — the plot point `evaluate()` / `EvalFile` can host an
   MLP over the same features (or a per-square 12×64 input tensor), giving the
   model the non-linear interactions the linear version cannot express.
4. **Search-guided labels** — `--multipv N` already records the engine's own
   root-move scores; distilling those (a policy target) usually converges faster
   than outcome-only labels.
5. **Deeper self-play** — depth 4–6 games give cleaner labels (weaker opponents
   produce noisier outcomes), at a proportional cost in time.



