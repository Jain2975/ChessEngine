"""Independent UCI validation harness for the engine, using python-chess.

Checks:
  1. UCI handshake, options, readyok
  2. bestmove legality in a variety of positions (raw 'moves' path)
  3. mate detection + 'score mate 1' reporting
  4. stalemate  -> 'bestmove 0000'
  5. 'go movetime'  respects the time budget
  6. 'go infinite' + 'stop'  returns a legal move quickly
  7. full self-play game through python-chess (FEN path), all moves legal
"""
import subprocess
import sys
import time

import chess
import chess.engine

ENGINE = r"D:\ChessEngine\chessengine.exe"

failures = []


def check(name, cond, detail=""):
    status = "PASS" if cond else "FAIL"
    print(f"[{status}] {name}{(' -- ' + detail) if detail else ''}")
    if not cond:
        failures.append(name)


class RawEngine:
    def __init__(self, path):
        self.p = subprocess.Popen([path], stdin=subprocess.PIPE,
                                  stdout=subprocess.PIPE, text=True, bufsize=1)

    def send(self, cmd):
        self.p.stdin.write(cmd + "\n")
        self.p.stdin.flush()

    def read_until(self, token, timeout=30.0):
        lines = []
        t0 = time.time()
        while True:
            line = self.p.stdout.readline()
            if not line:
                raise RuntimeError("engine closed stdout unexpectedly")
            line = line.strip()
            if line:
                lines.append(line)
            if line.startswith(token):
                return lines
            if time.time() - t0 > timeout:
                raise RuntimeError(f"timeout waiting for {token!r}; got {lines[-3:]}")

    def quit(self):
        try:
            self.send("quit")
            self.p.wait(timeout=5)
        except Exception:
            self.p.kill()


def perft_py(board, depth):
    """Independent perft via python-chess' own legal move generator."""
    if depth == 0:
        return 1
    total = 0
    for m in board.legal_moves:
        board.push(m)
        total += perft_py(board, depth - 1)
        board.pop()
    return total


def engine_perft(e, position_cmd, depth):
    e.send(position_cmd)
    e.send(f"perft {depth}")
    out = e.read_until("Total:")
    return int(out[-1].split()[-1])


def main():
    # ---------------------------------------------------------- 1 handshake
    e = RawEngine(ENGINE)
    e.send("uci")
    lines = e.read_until("uciok")
    check("uci handshake", any(l.startswith("id name") for l in lines))
    check("advertises Hash option",
          any("option name Hash type spin" in l for l in lines))
    e.send("isready")
    check("isready/readyok", e.read_until("readyok")[-1] == "readyok")

    # --------------------------------------------------- 2 bestmove legality
    cases = [
        ("startpos", "startpos"),
        ("kiwipete", "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1"),
        ("ep-available", "8/8/8/2k5/2pP4/8/B7/4K3 b - d3 0 3"),
        ("promotion", "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8"),
        ("in-check", "8/8/8/8/8/2k5/1q6/K7 w - - 0 1"),
    ]
    for name, fen in cases:
        if fen == "startpos":
            e.send("position startpos")
            board = chess.Board()
        else:
            e.send(f"position fen {fen}")
            board = chess.Board(fen)
        e.send("go depth 6")
        out = e.read_until("bestmove")
        mv = out[-1].split()[1]
        if board.legal_moves.count() == 0:
            check(f"no-legal-moves -> 0000 ({name})", mv == "0000", f"move={mv}")
        else:
            legal = mv != "0000" and chess.Move.from_uci(mv) in board.legal_moves
            check(f"legal bestmove ({name})", legal, f"move={mv}")

    seq = ["e2e4", "e7e5", "g1f3", "b8c6", "f1b5", "a7a6"]
    e.send("position startpos moves " + " ".join(seq))
    e.send("go depth 8")
    out = e.read_until("bestmove")
    mv = out[-1].split()[1]
    board = chess.Board()
    for m in seq:
        board.push_uci(m)
    check("legal bestmove (moves path)",
          chess.Move.from_uci(mv) in board.legal_moves, f"move={mv}")

    # ------------------------------------------------------- 3 mate detection
    e.send("position fen 6k1/5ppp/8/8/8/8/5PPP/R5K1 w - - 0 1")
    e.send("go depth 6")
    out = e.read_until("bestmove")
    mate = any("score mate 1" in l for l in out)
    mv = out[-1].split()[1]
    b = chess.Board("6k1/5ppp/8/8/8/8/5PPP/R5K1 w - - 0 1")
    b.push_uci(mv)
    check("mate-in-1 found and reported", mate and b.is_checkmate(), f"move={mv}")

    # ----------------------------------------------------------- 4 stalemate
    e.send("position fen 7k/5Q2/6K1/8/8/8/8/8 b - - 0 1")
    e.send("go depth 4")
    out = e.read_until("bestmove")
    check("stalemate -> bestmove 0000", out[-1] == "bestmove 0000", out[-1])

    # checkmated position must also report 0000 and not hang
    mate_fen = "rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3"
    e.send(f"position fen {mate_fen}")
    check("fool's mate FEN is checkmate", chess.Board(mate_fen).is_checkmate())
    e.send("go depth 4")
    out = e.read_until("bestmove")
    check("checkmated -> bestmove 0000", out[-1] == "bestmove 0000", out[-1])

    # ------------------------------------------------------------- 5 movetime
    e.send("position startpos")
    t0 = time.time()
    e.send("go movetime 1500")
    e.read_until("bestmove")
    dt = time.time() - t0
    check("movetime 1500 respected", 1.2 <= dt <= 3.0, f"{dt:.2f}s")

    # ------------------------------------------------------- 6 infinite + stop
    e.send("position startpos moves d2d4 g8f6")
    e.send("go infinite")
    time.sleep(1.0)
    t0 = time.time()
    e.send("stop")
    out = e.read_until("bestmove")
    dt = time.time() - t0
    mv = out[-1].split()[1]
    b = chess.Board()
    b.push_uci("d2d4")
    b.push_uci("g8f6")
    check("infinite+stop returns quickly", dt < 2.0, f"{dt:.2f}s")
    check("infinite+stop move legal",
          chess.Move.from_uci(mv) in b.legal_moves, f"move={mv}")

    e.send("ucinewgame")
    e.send("setoption name Hash value 16")
    e.send("isready")
    check("ucinewgame/setoption/readyok", e.read_until("readyok")[-1] == "readyok")

    e.quit()
    check("clean quit", e.p.returncode is not None)

    # --------------------------------------------------------- 7 self-play
    board = chess.Board()
    eng = chess.engine.SimpleEngine.popen_uci(ENGINE)
    illegal = 0
    plies = 0
    while not board.is_game_over(claim_draw=False) and plies < 80:
        result = eng.play(board, chess.engine.Limit(depth=6))
        if result.move is None or result.move not in board.legal_moves:
            illegal += 1
            break
        board.push(result.move)
        plies += 1
    check("self-play: no illegal/null moves", illegal == 0, f"{plies} plies")
    print(f"       (self-play reached: {board.fen()} result={board.result(claim_draw=False)})")
    eng.quit()

    # the engine must actually deliver a mate inside a game context
    mboard = chess.Board("6k1/5ppp/8/8/8/8/5PPP/R5K1 w - - 0 1")
    eng2 = chess.engine.SimpleEngine.popen_uci(ENGINE)
    res = eng2.play(mboard, chess.engine.Limit(depth=5))
    mboard.push(res.move)
    check("delivers mate in a game context", mboard.is_checkmate(), str(res.move))
    eng2.quit()

    # ------------------------------------------- 8 perft vs. python-chess
    print("\n-- perft cross-checked against python-chess' move generator --")
    perft_cases = [
        ("startpos", "position startpos", "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"),
        ("kiwipete", "position fen r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
         "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1"),
        ("pos3-ep", "position fen 8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
         "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1"),
        ("pos4-promo", "position fen r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1",
         "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1"),
        ("pos5", "position fen rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8",
         "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8"),
        ("pos6", "position fen r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10",
         "r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10"),
    ]
    depth = 3
    e2 = RawEngine(ENGINE)
    for name, cmd, fen in perft_cases:
        got = engine_perft(e2, cmd, depth)
        want = perft_py(chess.Board(fen), depth)
        check(f"perft({depth}) matches python-chess ({name})", got == want,
              f"engine={got} python-chess={want}")
    e2.quit()

    print()
    if failures:
        print("FAILED CHECKS:", ", ".join(failures))
        return 1
    print("ALL UCI VALIDATION CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())