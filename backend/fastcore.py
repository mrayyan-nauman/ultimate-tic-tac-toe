"""Numba-compiled game core for the self-play hot loop.

The scalar game ops (apply_move / winner / legal_moves / small-board winner) are
the Python bottleneck in MCTS self-play. Here they are @njit-compiled to machine
code over a single packed int8 state array, so there is no numpy per-call
dispatch overhead (which is what made an earlier pure-numpy attempt *slower*).

Packed state: int8[92]
    [0:81]  board cells, row-major bi*9+ci   (0 empty, 1 X, 2 O)
    [81:90] sub-board winners                 (0 none, 1 X, 2 O, 3 draw)
    [90]    active sub-board 0..8, or 9 = any
    [91]    player to move                    (1 X, 2 O)

Moves are flat indices 0..80 (bi*9+ci). `validate_against_engine()` asserts this
matches engine.py / net.encode_state exactly.
"""
import numpy as np
from numba import njit

_LINES = np.array([
    [0, 1, 2], [3, 4, 5], [6, 7, 8],
    [0, 3, 6], [1, 4, 7], [2, 5, 8],
    [0, 4, 8], [2, 4, 6],
], dtype=np.int64)


def initial_state():
    s = np.zeros(92, dtype=np.int8)
    s[90] = 9  # active = any
    s[91] = 1  # X to move
    return s


@njit(cache=True)
def _small_winner(state, base):
    for li in range(8):
        a = _LINES[li, 0]; b = _LINES[li, 1]; c = _LINES[li, 2]
        va = state[base + a]
        if va != 0 and va == state[base + b] and va == state[base + c]:
            return va
    for k in range(9):
        if state[base + k] == 0:
            return 0
    return 3


@njit(cache=True)
def nb_winner(state):
    for li in range(8):
        a = _LINES[li, 0]; b = _LINES[li, 1]; c = _LINES[li, 2]
        va = state[81 + a]
        if (va == 1 or va == 2) and va == state[81 + b] and va == state[81 + c]:
            return va
    for k in range(9):
        if state[81 + k] == 0:
            return 0
    return 3


@njit(cache=True)
def nb_legal(state):
    active = state[90]
    out = np.empty(81, dtype=np.int64)
    n = 0
    if active != 9 and state[81 + active] == 0:
        base = active * 9
        for ci in range(9):
            if state[base + ci] == 0:
                out[n] = base + ci; n += 1
    else:
        for bi in range(9):
            if state[81 + bi] == 0:
                base = bi * 9
                for ci in range(9):
                    if state[base + ci] == 0:
                        out[n] = base + ci; n += 1
    return out[:n]


@njit(cache=True)
def nb_apply(state, mv):
    ns = state.copy()
    player = state[91]
    ns[mv] = player
    bi = mv // 9
    if ns[81 + bi] == 0:
        ns[81 + bi] = _small_winner(ns, bi * 9)
    ci = mv - bi * 9
    ns[90] = ci if ns[81 + ci] == 0 else 9
    ns[91] = 2 if player == 1 else 1
    return ns


@njit(cache=True)
def nb_terminal_value(state):
    w = nb_winner(state)
    if w == 0:
        return 2.0   # sentinel: not terminal
    if w == 3:
        return 0.0
    return 1.0 if state[91] == w else -1.0


def encode_batch(states):
    """Vectorized batch encoder -> (B,486) float32, identical to net.encode_state."""
    B = len(states)
    arr = np.stack(states)                        # (B,92) int8
    boards = arr[:, :81].reshape(B, 9, 9)
    sw = arr[:, 81:90]                            # (B,9)
    actives = arr[:, 90].astype(np.int64)
    players = arr[:, 91]

    planes = np.zeros((B, 6, 9, 9), np.float32)
    planes[:, 0] = boards == 1
    planes[:, 1] = boards == 2
    planes[:, 2] = (sw == 1)[:, :, None]
    planes[:, 3] = (sw == 2)[:, :, None]

    has_active = actives != 9
    aw = np.zeros(B, np.int8)
    rows = np.arange(B)
    aw[has_active] = sw[rows[has_active], actives[has_active]]
    play_anywhere = (~has_active) | (aw != 0)
    bi_idx = np.arange(9)
    board_playable = (sw == 0) & (play_anywhere[:, None] | (bi_idx[None, :] == actives[:, None]))
    planes[:, 4] = (boards == 0) & board_playable[:, :, None]
    planes[:, 5] = (players == 1)[:, None, None]
    return planes.reshape(B, 486)


def from_canonical(cstate):
    cb, cw, ca, cp = cstate
    s = np.zeros(92, dtype=np.int8)
    for bi in range(9):
        for ci in range(9):
            v = cb[bi][ci]
            s[bi * 9 + ci] = 1 if v == "X" else 2 if v == "O" else 0
    for i in range(9):
        w = cw[i]
        s[81 + i] = 1 if w == "X" else 2 if w == "O" else 3 if w == "D" else 0
    s[90] = 9 if ca is None else ca
    s[91] = 1 if cp == "X" else 2
    return s


def validate_against_engine(n=300, seed=0):
    import random as _r
    from engine import initial_state as cinit, legal_moves as cleg, apply_move as cap, winner as cwin
    from net import encode_state as cenc

    _r.seed(seed)
    for _ in range(n):
        cs = cinit()
        for _ in range(_r.randint(0, 55)):
            cm = cleg(cs)
            if not cm or cwin(cs) is not None:
                break
            cs = cap(cs, _r.choice(cm))
        fs = from_canonical(cs)

        cleg_idx = sorted(bi * 9 + ci for (bi, ci) in cleg(cs))
        assert cleg_idx == sorted(int(m) for m in nb_legal(fs)), "legal mismatch"
        cw = cwin(cs)
        cw_i = 0 if cw is None else (1 if cw == "X" else 2 if cw == "O" else 3)
        assert cw_i == nb_winner(fs), "winner mismatch"
        assert np.array_equal(cenc(cs), encode_batch([fs])[0]), "encode mismatch"
    return True


if __name__ == "__main__":
    validate_against_engine()
    print("fastcore validated against canonical engine OK")
