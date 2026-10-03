"""Tkinter GUI: watch two (learning) engines play, and drive the learning loop.

Features
  * chessboard with Unicode pieces, move list, last-move highlight
  * Side A / Side B each pick an evaluation weight file (or "hand-tuned")
  * autoplay with adjustable search depth and speed, plus step / pause / new game
  * live engine evaluation of the position
  * "Learn 1 generation" button runs ml/learn_loop.py in the background
    (self-play -> train -> arena gate) and streams its log into the window
  * learning curve drawn from ml/data/history.jsonl (Elo of each new generation
    against the previous one)

Run:  python ml/gui.py
"""
import glob
import json
import os
import queue
import random
import subprocess
import sys
import threading
import tkinter as tk
from tkinter import ttk

import chess

from sbe_engine import EngineHandle, pick_opening

HERE = os.path.dirname(os.path.abspath(__file__))
HISTORY = os.path.join(HERE, "data", "history.jsonl")
UNICODE = chess.UNICODE_PIECE_SYMBOLS
SQ = 58
LIGHT, DARK = "#f0d9b5", "#b58863"
HILITE = "#f6f669"


MAX_WEIGHT_CHOICES = 24


def list_weights():
    """Hand-tuned plus the available weight files, newest first.

    Files come from ml/ (exported by hand, experimental) and from ml/data/
    (generations written by learn_loop.py).  Only the newest generations are
    offered so a long training run cannot flood the dropdown.
    """
    paths = glob.glob(os.path.join(HERE, "weights*.txt"))
    paths += glob.glob(os.path.join(HERE, "data", "weights*.txt"))
    paths.sort(key=os.path.getmtime, reverse=True)          # newest first
    gens, others = [], []
    for p in paths:
        name = os.path.relpath(p, HERE).replace("\\", "/")
        stem = os.path.basename(name).replace("weights_", "")
        (gens if stem.startswith("gen") else others).append(name)
    opts = ["hand-tuned"]
    for name in others + gens[:MAX_WEIGHT_CHOICES]:
        if name not in opts:
            opts.append(name)
    return opts


class BoardView(tk.Canvas):
    def __init__(self, master):
        super().__init__(master, width=SQ * 8, height=SQ * 8, highlightthickness=0)
        self.board = chess.Board()
        self.last_move = None
        self.flipped = False
        self.redraw()

    def set_board(self, board, last_move=None):
        self.board = board.copy()
        self.last_move = last_move
        self.redraw()

    def xy(self, square):
        f = chess.square_file(square)
        r = chess.square_rank(square)
        if self.flipped:
            f, r = 7 - f, 7 - r
        return f * SQ, (7 - r) * SQ

    def redraw(self):
        self.delete("all")
        for sq in chess.SQUARES:
            x, y = self.xy(sq)
            dark = (chess.square_file(sq) + chess.square_rank(sq)) % 2 == 0
            colour = DARK if dark else LIGHT
            if self.last_move and sq in (self.last_move.from_square,
                                        self.last_move.to_square):
                colour = HILITE
            self.create_rectangle(x, y, x + SQ, y + SQ, fill=colour, width=0)
        for sq, piece in self.board.piece_map().items():
            x, y = self.xy(sq)
            self.create_text(x + SQ / 2, y + SQ / 2 + 2,
                             text=UNICODE[piece.symbol()],
                             font=("Segoe UI Symbol", int(SQ * 0.62)))


class App:
    def __init__(self, root):
        self.root = root
        root.title("Simple Bitboard - learning arena")
        self.msgq = queue.Queue()
        self.white_handle = None
        self.black_handle = None
        self.playing = False
        self.learning = False
        self.depth = tk.IntVar(value=3)
        self.delay = tk.IntVar(value=300)
        self.mirror = tk.BooleanVar(value=True)
        self.weights_a = tk.StringVar(value="hand-tuned")
        self.weights_b = tk.StringVar(value="hand-tuned")
        self.games_var = tk.IntVar(value=6)
        self.arena_var = tk.IntVar(value=8)
        self.status = tk.StringVar(value="ready")
        self.rand_open = tk.BooleanVar(value=False)
        self.open_plies = tk.IntVar(value=6)

        self.board = chess.Board()
        self.moves = []
        self.rng = random.Random()
        self._build_widgets()
        self.reload_players()
        self.root.after(100, self._poll)
        self._refresh_curve()

    # ------------------------------------------------------------------ widgets
    def _build_widgets(self):
        left = ttk.Frame(self.root, padding=8)
        left.grid(row=0, column=0, sticky="n")
        self.view = BoardView(left)
        self.view.pack()

        right = ttk.Frame(self.root, padding=8)
        right.grid(row=0, column=1, sticky="n")

        box = ttk.LabelFrame(right, text="Players (evaluation weights)", padding=6)
        box.pack(fill="x")
        ttk.Label(box, text="Side A (White)").grid(row=0, column=0, sticky="w")
        ttk.Combobox(box, textvariable=self.weights_a, values=list_weights(),
                     width=26, state="readonly").grid(row=0, column=1)
        ttk.Label(box, text="Side B (Black)").grid(row=1, column=0, sticky="w")
        ttk.Combobox(box, textvariable=self.weights_b, values=list_weights(),
                     width=26, state="readonly").grid(row=1, column=1)
        ttk.Checkbutton(box, text="Mirror (identical = self-play)",
                        variable=self.mirror).grid(row=2, column=0, columnspan=2,
                                                   sticky="w")
        ttk.Label(box, text="Search depth").grid(row=3, column=0, sticky="w")
        ttk.Scale(box, from_=1, to=6, orient="horizontal", variable=self.depth,
                  length=150).grid(row=3, column=1, sticky="w")
        ttk.Label(box, text="Move delay (ms)").grid(row=4, column=0, sticky="w")
        ttk.Scale(box, from_=0, to=2000, orient="horizontal", variable=self.delay,
                  length=150).grid(row=4, column=1, sticky="w")

        ctl = ttk.Frame(right, padding=6)
        ctl.pack(fill="x")
        ttk.Button(ctl, text="New game", command=self.new_game).grid(row=0, column=0)
        self.play_btn = ttk.Button(ctl, text="Play", command=self.toggle_play)
        self.play_btn.grid(row=0, column=1)
        ttk.Button(ctl, text="Step", command=self.step).grid(row=0, column=2)
        ttk.Button(ctl, text="Flip", command=self.flip).grid(row=0, column=3)
        ttk.Button(ctl, text="Reload players", command=self.reload_players).grid(
            row=1, column=0, columnspan=4, sticky="we", pady=(4, 0))
        ttk.Checkbutton(ctl, text="Random opening",
                        variable=self.rand_open).grid(row=2, column=0, columnspan=2,
                                                      sticky="w", pady=(4, 0))
        ttk.Label(ctl, text="random plies").grid(row=2, column=2, sticky="e")
        ttk.Spinbox(ctl, from_=2, to=12, width=4,
                    textvariable=self.open_plies).grid(row=2, column=3)

        lrn = ttk.LabelFrame(right, text="Learning loop", padding=6)
        lrn.pack(fill="x", pady=(6, 0))
        self.learn_btn = ttk.Button(lrn, text="Learn 1 generation "
                                              "(self-play -> train -> arena)",
                                    command=self.run_generation)
        self.learn_btn.pack(fill="x")
        row = ttk.Frame(lrn)
        row.pack(fill="x", pady=2)
        ttk.Label(row, text="self-play games").pack(side="left")
        ttk.Entry(row, textvariable=self.games_var, width=5).pack(side="left")
        ttk.Label(row, text="arena games").pack(side="left")
        ttk.Entry(row, textvariable=self.arena_var, width=5).pack(side="left")
        self.progress = ttk.Progressbar(lrn, mode="indeterminate")
        self.progress.pack(fill="x", pady=2)

        info = ttk.LabelFrame(right, text="Status", padding=6)
        info.pack(fill="x", pady=(6, 0))
        ttk.Label(info, textvariable=self.status, wraplength=320,
                  justify="left").pack(anchor="w")

        nb = ttk.Notebook(right)
        nb.pack(fill="both", expand=True, pady=(6, 0))
        self.move_list = tk.Listbox(nb, width=36, height=9)
        nb.add(self.move_list, text="Moves")
        self.log = tk.Text(nb, width=36, height=9, wrap="none")
        nb.add(self.log, text="Learning log")

        curve_box = ttk.LabelFrame(right, text="Learning curve: Elo of each new "
                                               "generation vs previous", padding=6)
        curve_box.pack(fill="x", pady=(6, 0))
        self.curve = tk.Canvas(curve_box, width=330, height=120, bg="white",
                               highlightthickness=1)
        self.curve.pack()

    # ------------------------------------------------------------------ helpers
    def log_line(self, text):
        self.log.insert("end", text.rstrip() + "\n")
        self.log.see("end")

    def resolve_weights(self, name):
        if not name or name == "hand-tuned":
            return None
        return os.path.join(HERE, name)

    def reload_players(self):
        for h in (self.white_handle, self.black_handle):
            if h:
                h.close()
        wa = self.resolve_weights(self.weights_a.get())
        wb = wa if self.mirror.get() else self.resolve_weights(self.weights_b.get())
        try:
            self.white_handle = EngineHandle(weights=wa)
            self.black_handle = EngineHandle(weights=wb)
        except Exception as exc:                    # pragma: no cover
            self.status.set(f"engine start failed: {exc}")
            return
        self.status.set(f"A={self.weights_a.get()}   B="
                        f"{self.weights_a.get() if self.mirror.get() else self.weights_b.get()}")

    def new_game(self):
        self.playing = False
        self.play_btn.config(text="Play")
        self.board = chess.Board()
        self.moves = []
        self.move_list.delete(0, "end")
        opening = []
        if self.rand_open.get():
            try:
                opening = pick_opening(self.rng, "random", self.open_plies.get())
            except Exception as exc:                    # pragma: no cover
                self.status.set(f"random opening failed: {exc}")
                opening = []
            for mv in opening:
                self.board.push_uci(mv)
                self.moves.append(mv)
        self._render_moves()
        self.view.set_board(self.board)
        if opening:
            self.status.set("random opening: " + " ".join(opening))
        else:
            self.status.set(f"new game - depth {self.depth.get()}")

    def _render_moves(self):
        """Rebuild the move-list widget from self.moves (UCI strings)."""
        self.move_list.delete(0, "end")
        for i, uci in enumerate(self.moves):
            if i % 2 == 0:
                self.move_list.insert("end", f"{i // 2 + 1}. {uci}")
            else:
                last = self.move_list.get("end")
                self.move_list.delete("end")
                self.move_list.insert("end", f"{last}  {uci}")

    def flip(self):
        self.view.flipped = not self.view.flipped
        self.view.redraw()

    def toggle_play(self):
        self.playing = not self.playing
        self.play_btn.config(text="Pause" if self.playing else "Play")
        if self.playing:
            self.root.after(50, self._autoplay_tick)

    def step(self):
        if self.board.is_game_over(claim_draw=False):
            self.status.set(f"game over: {self.board.result(claim_draw=False)}")
            self.playing = False
            self.play_btn.config(text="Play")
            return
        handle = self.white_handle if self.board.turn == chess.WHITE else self.black_handle
        board_snapshot = self.board.copy()
        depth = self.depth.get()
        self.status.set(f"thinking (depth {depth})...")

        def work():
            try:
                move = handle.play(board_snapshot, depth=depth)
                line = ""
                try:
                    infos = handle.analyse(board_snapshot, depth=min(4, depth))
                    if infos:
                        mv, cp, mate, pv = infos[0]
                        score = (f"mate {mate // 1000}" if cp is None and mate else
                                 f"{cp / 100.0:+.2f}")
                        line = "pv " + " ".join(m.uci() for m in pv[:6])
                        self.msgq.put(("eval", f"eval {score} (white POV)   {line}"))
                except Exception:
                    pass
                self.msgq.put(("move", move))
            except Exception as exc:                # pragma: no cover
                self.msgq.put(("error", str(exc)))

        threading.Thread(target=work, daemon=True).start()

    def _autoplay_tick(self):
        if not self.playing:
            return
        if not self.board.is_game_over(claim_draw=False):
            self.step()
        self.root.after(max(30, self.delay.get()), self._autoplay_tick)

    # ------------------------------------------------------------------ polling
    def _apply_move(self, move):
        if move is None or move not in self.board.legal_moves:
            self.status.set("engine produced no/illegal move")
            self.playing = False
            self.play_btn.config(text="Play")
            return
        self.board.push(move)
        self.moves.append(move.uci())
        n = len(self.moves)
        if n % 2 == 1:
            self.move_list.insert("end", f"{n // 2 + 1}. {move.uci()}")
        else:
            last = self.move_list.get("end")
            self.move_list.delete("end")
            self.move_list.insert("end", f"{last}  {move.uci()}")
        self.view.set_board(self.board, move)
        if self.board.is_game_over(claim_draw=False):
            res = self.board.result(claim_draw=False)
            self.status.set(f"game over: {res}")
            self.playing = False
            self.play_btn.config(text="Play")
            self.log_line(f"game finished {res} in {n} plies")
        else:
            side = "A(white)" if self.board.turn == chess.BLACK else "B(black)"
            self.status.set(f"{side} to move - {n} plies played")

    def _poll(self):
        try:
            while True:
                msg = self.msgq.get_nowait()
                kind = msg[0]
                if kind == "move":
                    self._apply_move(msg[1])
                elif kind == "eval":
                    self.eval_text = msg[1]
                    self.status.set(msg[1])
                elif kind == "error":
                    self.status.set("engine error: " + msg[1])
                elif kind == "log":
                    self.log_line(msg[1])
                elif kind == "gen_done":
                    self.learning = False
                    self.progress.stop()
                    self.learn_btn.config(state="normal")
                    self.reload_players()
                    self._refresh_curve()
                    self.status.set("learning generation finished")
        except queue.Empty:
            pass
        self.root.after(100, self._poll)

    # ------------------------------------------------------------------ learning
    def run_generation(self):
        if self.learning:
            return
        self.learning = True
        self.learn_btn.config(state="disabled")
        self.progress.start(12)
        self.status.set("running one learning generation...")
        cmd = [sys.executable, os.path.join(HERE, "learn_loop.py"),
               "--generations", "1",
               "--games-per-gen", str(self.games_var.get()),
               "--arena-games", str(self.arena_var.get()),
               "--depth", "3", "--arena-depth", "4",
               # diverse self-play: random openings + exploration noise
               "--opening-mode", "random", "--opening-plies", "6",
               "--noise", "4", "--temperature", "1.0"]
        # continue from the last kept weights instead of restarting from scratch,
        # so repeated clicks really do learn generation after generation
        current = os.path.join(HERE, "data", "weights_current.txt")
        if os.path.exists(current):
            cmd += ["--start", current]
        self.log_line("$ " + " ".join(cmd))

        def work():
            try:
                proc = subprocess.Popen(cmd, cwd=HERE, stdout=subprocess.PIPE,
                                        stderr=subprocess.STDOUT, text=True, bufsize=1)
                for line in proc.stdout:
                    self.msgq.put(("log", line))
                proc.wait()
            except Exception as exc:                # pragma: no cover
                self.msgq.put(("log", f"error: {exc}"))
            self.msgq.put(("gen_done", None))

        threading.Thread(target=work, daemon=True).start()

    def _refresh_curve(self):
        self.curve.delete("all")
        rows = []
        if os.path.exists(HISTORY):
            with open(HISTORY, "r", encoding="utf-8") as fh:
                for line in fh:
                    try:
                        rec = json.loads(line)
                    except Exception:
                        continue
                    if rec.get("event") == "generation":
                        rows.append(rec)
        w, h = 330, 120
        if not rows:
            self.curve.create_text(w / 2, h / 2, text="no generations yet",
                                   fill="#888")
            return
        elos = [float(r.get("elo_candidate_minus_current", 0.0)) for r in rows]
        lo, hi = min(elos + [0.0]), max(elos + [0.0])
        if hi - lo < 40:
            hi, lo = lo + 40, lo - 5
        def px(i):
            return 30 + (w - 40) * (i / max(1, len(elos) - 1))
        def py(v):
            return h - 20 - (h - 35) * (v - lo) / (hi - lo)
        self.curve.create_line(30, py(0), w - 5, py(0), fill="#bbb")
        self.curve.create_text(28, py(0), text="0", anchor="e", fill="#888")
        if len(elos) >= 2:                     # Tk needs >= 2 points for a line
            pts = []
            for i, v in enumerate(elos):
                pts += [px(i), py(v)]
            self.curve.create_line(*pts, fill="#2b6cb0", width=2)
        for i, v in enumerate(elos):
            col = "#2f855a" if rows[i].get("accepted") else "#c53030"
            self.curve.create_oval(px(i) - 3, py(v) - 3, px(i) + 3, py(v) + 3,
                                   fill=col, outline=col)
            self.curve.create_text(px(i), h - 8, text=str(i + 1), fill="#666")


def main():
    root = tk.Tk()
    App(root)
    root.mainloop()
    return 0


if __name__ == "__main__":
    sys.path.insert(0, HERE)
    sys.exit(main())


