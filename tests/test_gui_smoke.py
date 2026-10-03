"""Headless smoke test for the tkinter GUI (no human interaction needed).

Builds the whole window, plays one move with each engine, redraws the learning
curve and shuts down.  Requires a display; skip it on a headless CI box.

    python tests/test_gui_smoke.py
"""
import os
import sys
import time

ML = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "ml")
sys.path.insert(0, ML)

print("step 1: import gui", flush=True)
import gui                                  # noqa: E402
print("ok: weight choices =", gui.list_weights(), flush=True)

print("step 2: build window", flush=True)
import tkinter as tk                        # noqa: E402
root = tk.Tk()
app = gui.App(root)
root.update()
print("ok: players =", app.white_handle.weights, app.black_handle.weights, flush=True)

print("step 3: random opening + one engine move", flush=True)
app.rand_open.set(True)
app.open_plies.set(6)
app.new_game()
root.update()
opening_ok = len(app.moves) == 6 and app.board.fen() != gui.chess.Board().fen()
print(f"ok: random opening played {app.moves} (fen {app.board.fen()})", flush=True)

t0 = time.time()
app.step()
while time.time() - t0 < 15 and len(app.moves) == 6:
    root.update()
    time.sleep(0.05)
moved = len(app.moves) > 6
print(f"ok: moves={app.moves} after {time.time() - t0:.1f}s", flush=True)

print("step 4: flip + curve + shutdown", flush=True)
app.view.flipped = True
app.view.redraw()
app._refresh_curve()
root.update()
app.white_handle.close()
app.black_handle.close()
root.destroy()
ok = moved and opening_ok
print("GUI SMOKE TEST PASSED" if ok else "GUI SMOKE TEST FAILED",
      f"(random opening={opening_ok}, engine move={moved})", flush=True)
sys.exit(0 if ok else 1)
