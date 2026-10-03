"""Texel-style tuning of the engine's evaluation weights from self-play data.

Model:      P(White wins) = sigmoid(alpha * eval_white(pos))
Loss:       logistic log-loss + L2 + a soft pull towards the hand-tuned weights
Optimiser:  Adam (numpy), mini-batches, deterministic shuffling

Features are extracted once and cached (dense float32 matrix), so epochs are
pure numpy work: a few thousand positions train in well under a second.

    python ml/train.py --data ml/data/games.jsonl --init ml/weights_init.txt \
                       --out ml/weights_gen1.txt --epochs 40 --lr 0.01

The output is directly loadable by the engine:
    setoption name EvalFile value ml/weights_gen1.txt
"""
import argparse
import json
import math
import os
import random
import sys
import time

import chess
import numpy as np

import features as F

ALPHA = math.log(10.0) / 400.0        # 400 cp == 10:1 odds (standard scaling)


def load_data(path, max_positions=None):
    fens, labels = [], []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            fens.append(rec["fen"])
            labels.append(float(rec["result"]))
    if max_positions and len(fens) > max_positions:
        idx = list(range(len(fens)))
        random.Random(0).shuffle(idx)
        idx = idx[:max_positions]
        fens = [fens[i] for i in idx]
        labels = [labels[i] for i in idx]
    return fens, np.array(labels, dtype=np.float64)


def build_cache(fens):
    """Dense feature matrix (float32) + phase per position."""
    boards = [chess.Board(f) for f in fens]
    X = F.features_batch(boards).astype(np.float32)
    phases = np.array([F.phase_of(b) for b in boards], dtype=np.float32)
    return X, phases


def effective_batch(X, phases, idx):
    """Phase-scaled design matrix for the rows in `idx` (mg/eg blocks scaled)."""
    Xb = X[idx]
    ph = phases[idx]
    Xe = Xb.astype(np.float32)
    Xe[:, F._MG] = Xb[:, F._MG] * (ph / 24.0)[:, None]
    Xe[:, F._EG] = Xb[:, F._EG] * ((24.0 - ph) / 24.0)[:, None]
    return Xe


def eval_split(X, phases, y, w, chunk=2048):
    """Mean log-loss and accuracy of weights `w` on the given rows."""
    if len(y) == 0:
        return 0.0, 0.0
    total, correct = 0.0, 0
    for i in range(0, len(y), chunk):
        idx = np.arange(i, min(i + chunk, len(y)))
        Xe = effective_batch(X, phases, idx)
        p = 1.0 / (1.0 + np.exp(-ALPHA * Xe.dot(w)))
        ysub = y[idx]
        total += float(-(ysub * np.log(p + 1e-12) +
                         (1 - ysub) * np.log(1 - p + 1e-12)).sum())
        correct += int(((p > 0.5) == (ysub > 0.5)).sum())
    return total / len(y), correct / len(y)


def train(data_path, init_path, out_path, epochs=40, batch_size=256, lr=0.01,
          l2=1e-6, pull=2e-7, max_positions=None, val_frac=0.1, verbose=True):
    fens, y = load_data(data_path, max_positions)
    if len(fens) == 0:
        raise SystemExit(f"no training positions found in {data_path}")

    # Deterministic split; the validation rows are never trained on, so the
    # reported val loss/accuracy measure generalisation rather than memorisation.
    order = list(range(len(fens)))
    random.Random(99).shuffle(order)
    n_val = int(len(fens) * val_frac)
    val_rows = np.array(sorted(order[:n_val]), dtype=np.int64)
    tr_rows = np.array(sorted(order[n_val:]), dtype=np.int64)
    if len(tr_rows) == 0:
        tr_rows, val_rows = np.array(order, dtype=np.int64), np.array([], dtype=np.int64)

    t_build = time.time()
    X, phases = build_cache(fens)
    Xtr, ptr = X[tr_rows], phases[tr_rows]
    ytr = y[tr_rows]
    Xval, pval = X[val_rows], phases[val_rows]
    yval = y[val_rows]
    del X, phases

    w0 = F.load_weights(init_path)
    w = w0.copy()

    if verbose:
        print(f"training on {len(tr_rows)} positions ({len(val_rows)} held out) "
              f"for {epochs} epochs (lr={lr}, batch={batch_size}); "
              f"features built in {time.time() - t_build:.1f}s")

    def report(prefix):
        tl, ta = eval_split(Xtr, ptr, ytr, w)
        vl, va = eval_split(Xval, pval, yval, w)
        print(f"{prefix}: train loss={tl:.4f} acc={ta:.3f} | "
              f"val loss={vl:.4f} acc={va:.3f}")

    if verbose:
        report("initial ")

    m = np.zeros_like(w)
    v = np.zeros_like(w)
    b1, b2, eps = 0.9, 0.999, 1e-8
    step = 0
    rng = random.Random(12345)
    n = len(tr_rows)
    order_local = list(range(n))

    for epoch in range(1, epochs + 1):
        rng.shuffle(order_local)
        total_loss, total_acc, seen = 0.0, 0, 0
        t0 = time.time()
        for i in range(0, n, batch_size):
            idx = np.array(order_local[i:i + batch_size], dtype=np.int64)
            Xe = effective_batch(Xtr, ptr, idx)
            ysub = ytr[idx]
            p = 1.0 / (1.0 + np.exp(-ALPHA * Xe.dot(w)))
            grad = Xe.T.dot(p - ysub) * ALPHA / len(idx)
            grad += l2 * w + pull * (w - w0)
            step += 1
            m = b1 * m + (1 - b1) * grad
            v = b2 * v + (1 - b2) * (grad * grad)
            mhat = m / (1 - b1 ** step)
            vhat = v / (1 - b2 ** step)
            w -= lr * mhat / (np.sqrt(vhat) + eps)

            total_loss += float(-(ysub * np.log(p + 1e-12) +
                                  (1 - ysub) * np.log(1 - p + 1e-12)).sum())
            total_acc += int(((p > 0.5) == (ysub > 0.5)).sum())
            seen += len(idx)
        if verbose and (epoch % max(1, epochs // 8) == 0 or epoch == epochs):
            print(f"epoch {epoch:3d}  train loss={total_loss / max(1, seen):.4f}  "
                  f"acc={total_acc / max(1, seen):.3f}  ({time.time() - t0:.2f}s)")
    if verbose:
        report("final   ")

    if out_path:
        os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
        F.save_weights(out_path, w)
        if verbose:
            delta = np.abs(w - w0)
            print(f"wrote {out_path}  |dw| mean={delta.mean():.3f} max={delta.max():.3f}")
    return w


def main():
    ap = argparse.ArgumentParser(description="Train evaluation weights")
    ap.add_argument("--data", default="data/games.jsonl")
    ap.add_argument("--init", default="weights_init.txt")
    ap.add_argument("--out", default="weights_gen1.txt")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--l2", type=float, default=1e-6)
    ap.add_argument("--pull", type=float, default=2e-7,
                    help="L2 pull towards the initial weights")
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--max-positions", type=int, default=None)
    args = ap.parse_args()
    train(args.data, args.init, args.out, epochs=args.epochs,
          batch_size=args.batch_size, lr=args.lr, l2=args.l2, pull=args.pull,
          max_positions=args.max_positions, val_frac=args.val_frac)


if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    main()

