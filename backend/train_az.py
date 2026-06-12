"""AlphaZero-style self-play training loop.

Usage:
    python train_az.py 20          # run 20 self-play iterations

Each iteration:
  1) Play `GAMES_PER_ITER` self-play games using current net + MCTS.
  2) Collect (state_encoding, policy_target, value_target) samples.
  3) Train net for a few epochs on the buffer.
  4) Save checkpoint to az_net.pt.

Resumes automatically from az_net.pt if present.
"""
import os
import sys
import time
import random
from collections import deque

import numpy as np
import torch
import torch.nn.functional as F

from engine import initial_state, winner, apply_move
from net import AZNet, encode_state, move_to_index
from mcts import run_mcts, policy_from_counts

CKPT_PATH = os.path.join(os.path.dirname(__file__), "az_net.pt")

GAMES_PER_ITER = 20
MCTS_SIMULATIONS = 64
TEMP_MOVES = 12          # first N plies use temperature=1 (exploration)
REPLAY_CAPACITY = 50_000
BATCH_SIZE = 256
EPOCHS_PER_ITER = 4
LR = 1e-3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def play_self_play_game(net):
    """Return list of (encoded_state, policy_vector_81, player_to_move)."""
    state = initial_state()
    samples = []
    ply = 0
    while True:
        w = winner(state)
        if w is not None:
            break
        temperature = 1.0 if ply < TEMP_MOVES else 0.0
        counts = run_mcts(state, net, simulations=MCTS_SIMULATIONS,
                          device=DEVICE, add_dirichlet=True)
        moves, probs = policy_from_counts(counts, temperature=temperature)

        pi = np.zeros(81, dtype=np.float32)
        for m, p in zip(moves, probs):
            pi[move_to_index(m)] = p
        samples.append((encode_state(state), pi, state[3]))

        chosen = moves[np.random.choice(len(moves), p=probs)]
        state = apply_move(state, chosen)
        ply += 1

    # Assign value targets from each sample's perspective
    if w == "D":
        result_for = {"X": 0.0, "O": 0.0}
    else:
        result_for = {w: 1.0, ("O" if w == "X" else "X"): -1.0}

    finalized = [(enc, pi, result_for[player]) for (enc, pi, player) in samples]
    return finalized, w, len(samples)


def train_step(net, opt, batch):
    encs = torch.from_numpy(np.stack([b[0] for b in batch])).to(DEVICE)
    pis = torch.from_numpy(np.stack([b[1] for b in batch])).to(DEVICE)
    zs = torch.tensor([b[2] for b in batch], dtype=torch.float32, device=DEVICE)

    logits, values = net(encs)
    log_probs = F.log_softmax(logits, dim=1)
    policy_loss = -(pis * log_probs).sum(dim=1).mean()
    value_loss = F.mse_loss(values, zs)
    loss = policy_loss + value_loss

    opt.zero_grad()
    loss.backward()
    opt.step()
    return float(policy_loss.item()), float(value_loss.item())


def main():
    iterations = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    net = AZNet().to(DEVICE)
    if os.path.exists(CKPT_PATH):
        net.load_state_dict(torch.load(CKPT_PATH, map_location=DEVICE))
        print(f"[resume] loaded {CKPT_PATH}")
    opt = torch.optim.Adam(net.parameters(), lr=LR)
    buffer = deque(maxlen=REPLAY_CAPACITY)

    for it in range(1, iterations + 1):
        t0 = time.time()
        net.eval()
        results = {"X": 0, "O": 0, "D": 0}
        total_plies = 0
        for g in range(GAMES_PER_ITER):
            samples, w, plies = play_self_play_game(net)
            buffer.extend(samples)
            results[w] += 1
            total_plies += plies
        sp_time = time.time() - t0

        # Train
        net.train()
        pl_total, vl_total, steps = 0.0, 0.0, 0
        for _ in range(EPOCHS_PER_ITER):
            if len(buffer) < BATCH_SIZE:
                break
            random.shuffle_in_place = None  # noop
            batch = random.sample(list(buffer), BATCH_SIZE)
            pl, vl = train_step(net, opt, batch)
            pl_total += pl; vl_total += vl; steps += 1

        torch.save(net.state_dict(), CKPT_PATH)
        avg_pl = pl_total / max(steps, 1)
        avg_vl = vl_total / max(steps, 1)
        print(f"[iter {it}] games={GAMES_PER_ITER} plies={total_plies} "
              f"X/O/D={results['X']}/{results['O']}/{results['D']} "
              f"buffer={len(buffer)} policy_loss={avg_pl:.3f} "
              f"value_loss={avg_vl:.3f} time={sp_time:.1f}s")


if __name__ == "__main__":
    main()