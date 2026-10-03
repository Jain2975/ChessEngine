"""Feature extraction that mirrors the C++ `evaluateLearned` exactly.

The engine's learnable model is a tapered linear model

    eval_white = (mg . phase + eg . (24 - phase)) / 24   (integer division)

where the feature vector is, per position:

    [   0 ..  11] mg material   : [colour * 6 + pieceType]
    [  12 ..  23] eg material   : [colour * 6 + pieceType]
    [  24 .. 791] mg PST        : [colour][pieceType][square-in-own-frame]
    [ 792 .. 1559] eg PST       : [colour][pieceType][square-in-own-frame]
    [1560]        mg tempo bonus
    [1561]        eg tempo bonus

Features are collected from **White's point of view**: a White piece contributes
+1, a Black piece -1 (with the same relative square index, because the tables are
mirrored).  Tempo contributes +1 when White is to move, -1 when Black is to move.
"""
import numpy as np
import chess

EVAL_NF_MAT = 2 * 12
EVAL_NF_PST = 2 * 2 * 6 * 64
EVAL_NF_TEMPO = 2
EVAL_NF = EVAL_NF_MAT + EVAL_NF_PST + EVAL_NF_TEMPO          # 1562

GAME_PHASE = np.array([0, 1, 1, 2, 4, 0], dtype=np.int32)     # pawn..king

# python-chess piece_type is 1..6 (PAWN..KING); the engine uses 0..5.
PT = {
    chess.PAWN: 0, chess.KNIGHT: 1, chess.BISHOP: 2,
    chess.ROOK: 3, chess.QUEEN: 4, chess.KING: 5,
}


def w_mat(ph, c, p):
    return ph * 12 + c * 6 + p


def w_pst(ph, c, p, s):
    return EVAL_NF_MAT + ph * 768 + c * 384 + p * 64 + s


def w_tempo(ph):
    return EVAL_NF_MAT + EVAL_NF_PST + ph


def own_frame_square(square, colour):
    """White-frame index: rank-8-first for White, unchanged for Black."""
    if colour == chess.WHITE:
        return square ^ 56
    return square


def feature_indices(board):
    """Return a list of (feature_index, coefficient) pairs (white-positive)."""
    feats = []
    for square, piece in board.piece_map().items():
        sign = 1.0 if piece.color == chess.WHITE else -1.0
        p = PT[piece.piece_type]
        c = 0 if piece.color == chess.WHITE else 1
        ts = own_frame_square(square, piece.color)
        feats.append((w_mat(0, c, p), sign))
        feats.append((w_mat(1, c, p), sign))
        feats.append((w_pst(0, c, p, ts), sign))
        feats.append((w_pst(1, c, p, ts), sign))
    t = 1.0 if board.turn == chess.WHITE else -1.0
    feats.append((w_tempo(0), t))
    feats.append((w_tempo(1), t))
    return feats


def feature_vector(board):
    """Dense float64 feature vector (1562,)."""
    f = np.zeros(EVAL_NF, dtype=np.float64)
    for idx, val in feature_indices(board):
        f[idx] += val
    return f


def phase_of(board):
    phase = 0
    for piece in board.piece_map().values():
        phase += GAME_PHASE[PT[piece.piece_type]]
    return min(phase, 24)


def features_batch(boards):
    """Dense matrix (n, 1562) for a list of boards."""
    X = np.zeros((len(boards), EVAL_NF), dtype=np.float64)
    for i, b in enumerate(boards):
        for idx, val in feature_indices(b):
            X[i, idx] += val
    return X


def static_eval_white(board, w):
    """White-perspective evaluation from weights, mirroring the engine model."""
    ph, Xeff = effective_matrix([board])
    return float(Xeff.dot(w)[0])


def mg_indices():
    """Indices belonging to the midgame block (material mg, PST mg, tempo mg)."""
    idx = list(range(0, 12)) + list(range(EVAL_NF_MAT, EVAL_NF_MAT + 768)) \
        + [w_tempo(0)]
    return np.array(idx, dtype=np.int64)


def eg_indices():
    """Indices belonging to the endgame block (material eg, PST eg, tempo eg)."""
    idx = list(range(12, 24)) + list(range(EVAL_NF_MAT + 768, EVAL_NF_MAT + EVAL_NF_PST)) \
        + [w_tempo(1)]
    return np.array(idx, dtype=np.int64)


_MG = mg_indices()
_EG = eg_indices()


def effective_matrix(boards):
    """Phase-scaled design matrix such that  X_eff . w  equals the tapered
    white-POV evaluation of each board for weights `w` in the engine layout.

    Returns (phases, X_eff) with X_eff of shape (n, EVAL_NF).
    """
    X = features_batch(boards)
    phases = np.array([phase_of(b) for b in boards], dtype=np.float64)
    Xeff = np.zeros_like(X)
    mgScale = (phases / 24.0)[:, None]
    egScale = ((24.0 - phases) / 24.0)[:, None]
    Xeff[:, _MG] = X[:, _MG] * mgScale
    Xeff[:, _EG] = X[:, _EG] * egScale
    return phases, Xeff


def load_weights(path):
    w = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.split("#", 1)[0]
            for tok in line.split():
                w.append(float(tok))
    if len(w) == EVAL_NF + 1:
        w = w[1:]
    if len(w) != EVAL_NF:
        raise ValueError(f"{path}: expected {EVAL_NF} weights, found {len(w)}")
    return np.array(w, dtype=np.float64)


def save_weights(path, w):
    if len(w) != EVAL_NF:
        raise ValueError(f"expected {EVAL_NF} weights, got {len(w)}")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(f"# SBEVAL v1 n={EVAL_NF}\n")
        for i in range(0, EVAL_NF, 6):
            fh.write(" ".join(f"{v:.6f}" for v in w[i:i + 6]))
            fh.write("\n")
