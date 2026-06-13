"""Tiny AlphaZero-style policy+value network for Ultimate Tic-Tac-Toe.

Input encoding (6 planes of 9x9 = 486 floats):
  0: X stones
  1: O stones
  2: sub-board won by X (broadcast across its 9 cells)
  3: sub-board won by O (broadcast across its 9 cells)
  4: legal-cell mask (1 where current player may play)
  5: side-to-move (all 1s if current player is X, else 0)

Output:
  policy: 81 logits (one per (boardIndex*9 + cellIndex))
  value : scalar in [-1, 1], from current player's perspective
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from engine import legal_moves, WIN_LINES

NUM_PLANES = 6
INPUT_DIM = NUM_PLANES * 81
NUM_ACTIONS = 81

# v2 encoding adds 5 tactical feature planes (total 11):
#   6  free-move cells   (cell whose target board is already decided)
#   7  X sub-win threats  (empty cell that completes a small board for X)
#   8  O sub-win threats
#   9  macro-threat X     (capturing this board completes a macro line for X)
#   10 macro-threat O
NUM_PLANES_V2 = 11
INPUT_DIM_V2 = NUM_PLANES_V2 * 81


def encode_state(state):
    boards, winners, active, player = state
    planes = np.zeros((NUM_PLANES, 9, 9), dtype=np.float32)

    for bi in range(9):
        for ci in range(9):
            v = boards[bi][ci]
            if v == "X":
                planes[0, bi, ci] = 1.0
            elif v == "O":
                planes[1, bi, ci] = 1.0

    for bi in range(9):
        if winners[bi] == "X":
            planes[2, bi, :] = 1.0
        elif winners[bi] == "O":
            planes[3, bi, :] = 1.0

    # Legal-cell mask
    for (bi, ci) in legal_moves(state):
        planes[4, bi, ci] = 1.0

    if player == "X":
        planes[5, :, :] = 1.0

    return planes.reshape(-1)  # flat 486


def _winning_cells(cells, mark):
    """Empty-cell indices where placing `mark` completes a small-board line."""
    res = set()
    for a, b, c in WIN_LINES:
        trio = (cells[a], cells[b], cells[c])
        if trio.count(mark) == 2 and trio.count(None) == 1:
            for idx in (a, b, c):
                if cells[idx] is None:
                    res.add(idx)
    return res


def encode_state_v2(state):
    """11-plane encoding: the 6 base planes + 5 tactical feature planes."""
    boards, winners, active, player = state
    planes = np.zeros((NUM_PLANES_V2, 9, 9), dtype=np.float32)

    # Base planes 0-5 (identical to encode_state).
    for bi in range(9):
        for ci in range(9):
            v = boards[bi][ci]
            if v == "X":
                planes[0, bi, ci] = 1.0
            elif v == "O":
                planes[1, bi, ci] = 1.0
    for bi in range(9):
        if winners[bi] == "X":
            planes[2, bi, :] = 1.0
        elif winners[bi] == "O":
            planes[3, bi, :] = 1.0
    for (bi, ci) in legal_moves(state):
        planes[4, bi, ci] = 1.0
    if player == "X":
        planes[5, :, :] = 1.0

    # Plane 6: free-move cells — playing a cell in column ci sends the opponent
    # to board ci; if that board is already decided they get a free choice.
    for ci in range(9):
        if winners[ci] is not None:
            planes[6, :, ci] = 1.0

    # Planes 7/8: immediate sub-board-winning cells for X / O (undecided boards).
    for bi in range(9):
        if winners[bi] is None:
            for ci in _winning_cells(boards[bi], "X"):
                planes[7, bi, ci] = 1.0
            for ci in _winning_cells(boards[bi], "O"):
                planes[8, bi, ci] = 1.0

    # Planes 9/10: macro-threat boards — capturing board bi completes a macro
    # line for X / O (the other two boards in some macro line are already theirs).
    macro = [w if w in ("X", "O") else None for w in winners]
    for bi in range(9):
        if winners[bi] is None:
            for a, b, c in WIN_LINES:
                if bi in (a, b, c):
                    o1, o2 = [x for x in (a, b, c) if x != bi]
                    if macro[o1] == "X" and macro[o2] == "X":
                        planes[9, bi, :] = 1.0
                    if macro[o1] == "O" and macro[o2] == "O":
                        planes[10, bi, :] = 1.0

    return planes.reshape(-1)  # flat 891


def move_to_index(move):
    bi, ci = move
    return bi * 9 + ci


def index_to_move(idx):
    return (idx // 9, idx % 9)


class _ResBlock(nn.Module):
    """Pre-activation-free residual MLP block: x -> relu(x + W2 relu(W1 x))."""
    def __init__(self, hidden):
        super().__init__()
        self.fc1 = nn.Linear(hidden, hidden)
        self.fc2 = nn.Linear(hidden, hidden)

    def forward(self, x):
        h = F.relu(self.fc1(x))
        h = self.fc2(h)
        return F.relu(x + h)


class AZNet(nn.Module):
    """Residual MLP policy + value network.

    Bigger capacity than the original 3-layer trunk for a stronger ceiling, while
    keeping the exact I/O contract (486-float input -> 81 policy logits + 1 value),
    so the TypeScript encoder, parity tests, and ONNX export remain valid.
    """
    def __init__(self, input_dim=INPUT_DIM, hidden=512, blocks=4):
        super().__init__()
        self.stem = nn.Sequential(nn.Linear(input_dim, hidden), nn.ReLU())
        self.blocks = nn.ModuleList([_ResBlock(hidden) for _ in range(blocks)])
        self.policy_head = nn.Linear(hidden, NUM_ACTIONS)
        self.value_head = nn.Sequential(
            nn.Linear(hidden, 128), nn.ReLU(),
            nn.Linear(128, 1), nn.Tanh(),
        )

    def forward(self, x):
        h = self.stem(x)
        for blk in self.blocks:
            h = blk(h)
        return self.policy_head(h), self.value_head(h).squeeze(-1)


class OldAZNet(nn.Module):
    """The original pre-GPU 3-layer MLP. Kept so training can load the previous
    deployed model (az_net.pt.bak) as a fixed benchmark opponent for Elo gating.
    Same I/O contract (486 in -> 81 policy + 1 value)."""
    def __init__(self, hidden=256):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(INPUT_DIM, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.policy_head = nn.Linear(hidden, NUM_ACTIONS)
        self.value_head = nn.Sequential(
            nn.Linear(hidden, 64), nn.ReLU(),
            nn.Linear(64, 1), nn.Tanh(),
        )

    def forward(self, x):
        h = self.trunk(x)
        return self.policy_head(h), self.value_head(h).squeeze(-1)


@torch.no_grad()
def predict(net, state, device="cpu"):
    """Return (policy_probs over legal moves dict, value scalar)."""
    x = torch.from_numpy(encode_state(state)).unsqueeze(0).to(device)
    logits, value = net(x)
    logits = logits.squeeze(0).cpu().numpy()
    moves = legal_moves(state)
    if not moves:
        return {}, float(value.item())
    idxs = np.array([move_to_index(m) for m in moves])
    masked = logits[idxs]
    masked = masked - masked.max()  # numerical stability
    exp = np.exp(masked)
    probs = exp / exp.sum()
    return dict(zip(moves, probs.tolist())), float(value.item())