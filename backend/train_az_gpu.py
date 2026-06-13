"""Batched, GPU-accelerated AlphaZero self-play training.

Plays many games in parallel and evaluates all of their MCTS leaves in a single
batched forward pass per simulation step, so the GPU does large batches instead
of thousands of tiny batch-1 calls. Tree management is plain Python on one core,
so the laptop stays responsive while the GPU is fed.

Tunables via env vars (used for calibration):
    UTTT_GAMES  parallel games per self-play cohort   (default 128)
    UTTT_SIMS   MCTS simulations per move             (default 96)

Usage:
    python train_az_gpu.py [iterations]
"""
import os
import sys
import math
import time
import random
from collections import deque

import numpy as np
import torch
import torch.nn.functional as F

from engine import initial_state, legal_moves, apply_move, winner
from net import AZNet, encode_state, move_to_index, NUM_ACTIONS

CKPT_PATH = os.path.join(os.path.dirname(__file__), "az_net.pt")

PARALLEL_GAMES = int(os.environ.get("UTTT_GAMES", 128))
MCTS_SIMULATIONS = int(os.environ.get("UTTT_SIMS", 96))
C_PUCT = 1.5
DIRICHLET_ALPHA = 0.3
DIRICHLET_FRAC = 0.25
TEMP_MOVES = 12            # first N plies sample ~ visit counts (exploration)
REPLAY_CAPACITY = 100_000
BATCH_SIZE = 1024
TRAIN_STEPS_PER_ITER = 16
LR = 1e-3
MAX_PLIES = 81            # a UTTT game can't exceed 81 moves

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class Node:
    __slots__ = ("prior", "N", "W", "Q", "children", "state", "expanded", "terminal_value")

    def __init__(self, prior, state):
        self.prior = prior
        self.N = 0
        self.W = 0.0
        self.Q = 0.0
        self.children = {}
        self.state = state
        self.expanded = False
        self.terminal_value = None


class Game:
    __slots__ = ("state", "root", "samples", "ply", "result_winner")

    def __init__(self):
        self.state = initial_state()
        self.root = None
        self.samples = []          # (encoded_state, pi_81, player_to_move)
        self.ply = 0
        self.result_winner = None


def terminal_value(state):
    """Value from the perspective of the player about to move at `state`."""
    w = winner(state)
    if w is None:
        return None
    if w == "D":
        return 0.0
    return 1.0 if state[3] == w else -1.0


def select_child(node):
    best_score, best = -1e30, None
    sqrt_sum = math.sqrt(node.N + 1e-8)
    for child in node.children.values():
        u = C_PUCT * child.prior * sqrt_sum / (1 + child.N)
        score = -child.Q + u          # child Q is from opponent's view; flip
        if score > best_score:
            best_score, best = score, child
    return best


def backup(path, value):
    v = value
    for n in reversed(path):
        n.N += 1
        n.W += v
        n.Q = n.W / n.N
        v = -v


def expand(node, logits_row):
    """Create children with masked-softmax priors from a policy-logits row."""
    moves = legal_moves(node.state)
    node.expanded = True
    if not moves:
        node.terminal_value = 0.0
        return
    idxs = np.fromiter((move_to_index(m) for m in moves), dtype=np.int64, count=len(moves))
    masked = logits_row[idxs]
    masked = masked - masked.max()
    exp = np.exp(masked)
    probs = exp / exp.sum()
    for m, p in zip(moves, probs):
        node.children[m] = Node(float(p), apply_move(node.state, m))


@torch.no_grad()
def batched_eval(net, states):
    arr = np.stack([encode_state(s) for s in states])
    x = torch.from_numpy(arr).to(DEVICE, non_blocking=True)
    logits, values = net(x)
    return logits.detach().cpu().numpy(), values.detach().cpu().numpy()


def play_cohort(net, n_games):
    games = [Game() for _ in range(n_games)]
    active = list(games)
    move_no = 0

    while active and move_no < MAX_PLIES:
        # Fresh root each move; expand all roots in one batch.
        for g in active:
            g.root = Node(1.0, g.state)
        logits, _ = batched_eval(net, [g.root.state for g in active])
        for i, g in enumerate(active):
            expand(g.root, logits[i])
            moves = list(g.root.children.keys())
            if moves:
                noise = np.random.dirichlet([DIRICHLET_ALPHA] * len(moves))
                for j, m in enumerate(moves):
                    c = g.root.children[m]
                    c.prior = (1 - DIRICHLET_FRAC) * c.prior + DIRICHLET_FRAC * noise[j]

        # Simulations: descend each active game to a leaf, batch-evaluate leaves.
        for _ in range(MCTS_SIMULATIONS):
            leaf_nodes, leaf_paths, leaf_states = [], [], []
            for g in active:
                node = g.root
                path = [node]
                while node.expanded and node.terminal_value is None and node.children:
                    node = select_child(node)
                    path.append(node)
                if node.terminal_value is not None:
                    backup(path, node.terminal_value)
                elif not node.expanded:
                    tv = terminal_value(node.state)
                    if tv is not None:
                        node.expanded = True
                        node.terminal_value = tv
                        backup(path, tv)
                    else:
                        leaf_nodes.append(node)
                        leaf_paths.append(path)
                        leaf_states.append(node.state)
                else:
                    backup(path, 0.0)

            if leaf_states:
                lg, lv = batched_eval(net, leaf_states)
                for k in range(len(leaf_nodes)):
                    expand(leaf_nodes[k], lg[k])
                    backup(leaf_paths[k], float(lv[k]))

        # Choose moves, record training samples, advance games.
        still_active = []
        for g in active:
            moves = list(g.root.children.keys())
            visits = np.array([g.root.children[m].N for m in moves], dtype=np.float64)
            total = visits.sum()
            pi = np.zeros(NUM_ACTIONS, dtype=np.float32)
            dist = visits / total if total > 0 else np.ones(len(moves)) / len(moves)
            for m, p in zip(moves, dist):
                pi[move_to_index(m)] = p
            g.samples.append((encode_state(g.state), pi, g.state[3]))

            if g.ply < TEMP_MOVES and total > 0:
                probs = dist / dist.sum()   # guard against float rounding in choice
                chosen = moves[np.random.choice(len(moves), p=probs)]
            else:
                chosen = moves[int(visits.argmax())]
            g.state = apply_move(g.state, chosen)
            g.ply += 1
            w = winner(g.state)
            if w is not None:
                g.result_winner = w
            else:
                still_active.append(g)
        active = still_active
        move_no += 1

    # Assign value targets from each sample's player perspective.
    samples, results = [], {"X": 0, "O": 0, "D": 0}
    for g in games:
        w = g.result_winner if g.result_winner is not None else (winner(g.state) or "D")
        results[w if w in results else "D"] += 1
        if w == "D":
            rf = {"X": 0.0, "O": 0.0}
        else:
            rf = {w: 1.0, ("O" if w == "X" else "X"): -1.0}
        for enc, pi, player in g.samples:
            samples.append((enc, pi, rf[player]))
    return samples, results


def train(net, opt, buffer, steps):
    if len(buffer) < BATCH_SIZE:
        return None
    data = list(buffer)
    net.train()
    pl_t = vl_t = 0.0
    for _ in range(steps):
        batch = random.sample(data, BATCH_SIZE)
        enc = torch.from_numpy(np.stack([b[0] for b in batch])).to(DEVICE)
        pis = torch.from_numpy(np.stack([b[1] for b in batch])).to(DEVICE)
        zs = torch.tensor([b[2] for b in batch], dtype=torch.float32, device=DEVICE)
        logits, values = net(enc)
        logp = F.log_softmax(logits, dim=1)
        policy_loss = -(pis * logp).sum(dim=1).mean()
        value_loss = F.mse_loss(values, zs)
        (policy_loss + value_loss).backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
        pl_t += policy_loss.item()
        vl_t += value_loss.item()
    net.eval()
    return pl_t / steps, vl_t / steps


def main():
    iterations = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)
    if DEVICE == "cuda":
        torch.backends.cudnn.benchmark = True

    net = AZNet().to(DEVICE)
    net.eval()
    opt = torch.optim.Adam(net.parameters(), lr=LR)
    buffer = deque(maxlen=REPLAY_CAPACITY)

    nparams = sum(p.numel() for p in net.parameters())
    print(f"[init] device={DEVICE} params={nparams:,} games/iter={PARALLEL_GAMES} "
          f"sims={MCTS_SIMULATIONS} iterations={iterations}", flush=True)

    t_start = time.time()
    for it in range(1, iterations + 1):
        t0 = time.time()
        samples, results = play_cohort(net, PARALLEL_GAMES)
        buffer.extend(samples)
        sp = time.time() - t0
        tr = train(net, opt, buffer, TRAIN_STEPS_PER_ITER)
        torch.save(net.state_dict(), CKPT_PATH)
        msg = (f"[iter {it}/{iterations}] X/O/D={results['X']}/{results['O']}/{results['D']} "
               f"samples={len(samples)} buffer={len(buffer)} selfplay={sp:.1f}s")
        if tr:
            msg += f" ploss={tr[0]:.3f} vloss={tr[1]:.3f}"
        if DEVICE == "cuda":
            msg += f" vram={torch.cuda.max_memory_allocated()/1e6:.0f}MB"
        msg += f" total={time.time() - t_start:.1f}s"
        print(msg, flush=True)

    print(f"[done] saved {CKPT_PATH} in {time.time() - t_start:.1f}s", flush=True)


if __name__ == "__main__":
    main()
