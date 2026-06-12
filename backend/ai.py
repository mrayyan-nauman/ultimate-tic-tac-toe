"""Flask-facing AI. Loads the trained AlphaZero net and picks a move via MCTS."""
import os
import random

import torch

from engine import legal_moves
from net import AZNet
from mcts import run_mcts

CKPT_PATH = os.path.join(os.path.dirname(__file__), "az_net.pt")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
INFERENCE_SIMULATIONS = 128  # bump this up for stronger (slower) play

_net = AZNet().to(DEVICE)
_loaded = False
if os.path.exists(CKPT_PATH):
    _net.load_state_dict(torch.load(CKPT_PATH, map_location=DEVICE))
    _loaded = True
_net.eval()


def _build_state(boards, board_winners, active_board, player):
    boards_t = tuple(tuple(b) for b in boards)
    winners_t = tuple(board_winners)
    return (boards_t, winners_t, active_board, player)


def get_ai_move(boards, board_winners, active_board, player="O"):
    state = _build_state(boards, board_winners, active_board, player)
    moves = legal_moves(state)
    if not moves:
        return None

    if not _loaded:
        # No checkpoint yet -> random fallback so the app still works pre-training.
        bi, ci = random.choice(moves)
        return {"boardIndex": bi, "cellIndex": ci}

    counts = run_mcts(state, _net, simulations=INFERENCE_SIMULATIONS,
                      device=DEVICE, add_dirichlet=False)
    # Pick the most-visited move (exploitation).
    best_move = max(counts.items(), key=lambda kv: kv[1])[0]
    bi, ci = best_move
    return {"boardIndex": bi, "cellIndex": ci}
