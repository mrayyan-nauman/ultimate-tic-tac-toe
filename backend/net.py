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

from engine import legal_moves

NUM_PLANES = 6
INPUT_DIM = NUM_PLANES * 81
NUM_ACTIONS = 81


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


def move_to_index(move):
    bi, ci = move
    return bi * 9 + ci


def index_to_move(idx):
    return (idx // 9, idx % 9)


class AZNet(nn.Module):
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