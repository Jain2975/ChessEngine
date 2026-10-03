import subprocess
import sys

ENGINE = r"D:\ChessEngine\chessengine.exe"
FENS = [
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
    "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
    "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1",
    "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8",
    "4k3/8/8/8/8/8/8/4K2R w K - 0 1",
    "8/8/8/2k5/2pP4/8/B7/4K3 b - d3 0 3",
]


def run(cmds):
    p = subprocess.run([ENGINE], input="\n".join(cmds) + "\n",
                       capture_output=True, text=True)
    return p.stdout.splitlines()


def main():
    # 1) hand-tuned eval values
    cmds = []
    for fen in FENS:
        cmds += [f"position fen {fen}", "eval"]
    base = [l.split()[-2] for l in run(cmds) if l.startswith("eval:")]

    # 2) learned-eval values with the exported (hand-tuned) weights
    cmds = ["setoption name EvalFile value D:\\ChessEngine\\ml\\weights_init.txt"]
    for fen in FENS:
        cmds += [f"position fen {fen}", "eval"]
    learned = [l.split()[-2] for l in run(cmds) if l.startswith("eval:")]

    ok = True
    for fen, a, b in zip(FENS, base, learned):
        same = (a == b)
        ok = ok and same
        print(f"[{'PASS' if same else 'FAIL'}] hand={a:>6} learned={b:>6}  {fen}")
    print("\nROUND-TRIP " + ("OK: learned model reproduces the hand-tuned eval exactly"
                             if ok else "MISMATCH"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
