"""Batched, GPU-accelerated AlphaZero self-play training with resume + Elo gating.

Plays many games in parallel and evaluates all of their MCTS leaves in a single
batched forward pass per simulation step, so the GPU does large batches instead
of thousands of tiny batch-1 calls. Tree management is plain Python on one core,
so the laptop stays responsive while the GPU is fed.

Campaign features:
  * Resume: loads az_net.pt at startup (if architecture matches) and keeps the
    best net there, so deploys are always the strongest gated model.
  * Elo gating: a freshly trained candidate only replaces `best` if it beats it
    in a head-to-head match (>= win threshold). A running Elo estimate of the
    best net (vs the campaign's starting point) is tracked in az_net.meta.json.

Tunables via env vars:
    UTTT_GAMES        parallel self-play games per iteration   (default 256)
    UTTT_SIMS         MCTS sims per move in self-play           (default 64)
    UTTT_EVAL_GAMES   games per gating match                    (default 40)
    UTTT_EVAL_SIMS    MCTS sims per move in gating match        (default 64)
    UTTT_EVAL_EVERY   gate every N iterations                   (default 2)
    UTTT_WIN_THRESH   candidate score needed to promote         (default 0.55)
    UTTT_MINUTES      wall-clock budget (overrides iteration arg if > 0)
    UTTT_CKPT         checkpoint path                           (default az_net.pt)

Usage:
    python train_az_gpu.py [iterations]
"""
import os
import sys
import json
import math
import time
import random
from collections import deque

import numpy as np
import torch
import torch.nn.functional as F

from engine import initial_state, legal_moves, apply_move, winner
from net import AZNet, encode_state, move_to_index, NUM_ACTIONS

CKPT_PATH = os.environ.get("UTTT_CKPT", os.path.join(os.path.dirname(__file__), "az_net.pt"))
META_PATH = CKPT_PATH[:-3] + ".meta.json" if CKPT_PATH.endswith(".pt") else CKPT_PATH + ".meta.json"

PARALLEL_GAMES = int(os.environ.get("UTTT_GAMES", 256))
MCTS_SIMULATIONS = int(os.environ.get("UTTT_SIMS", 64))
EVAL_GAMES = int(os.environ.get("UTTT_EVAL_GAMES", 40))
EVAL_SIMS = int(os.environ.get("UTTT_EVAL_SIMS", 64))
EVAL_EVERY = int(os.environ.get("UTTT_EVAL_EVERY", 2))
WIN_THRESHOLD = float(os.environ.get("UTTT_WIN_THRESH", 0.55))
MINUTES = float(os.environ.get("UTTT_MINUTES", 0))

C_PUCT = 1.5
DIRICHLET_ALPHA = 0.3
DIRICHLET_FRAC = 0.25
TEMP_MOVES = 12
REPLAY_CAPACITY = 100_000
BATCH_SIZE = 1024
TRAIN_STEPS_PER_ITER = 16
LR = 1e-3
MAX_PLIES = 81

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
    __slots__ = ("state", "samples", "ply", "result_winner")

    def __init__(self):
        self.state = initial_state()
        self.samples = []
        self.ply = 0
        self.result_winner = None


def terminal_value(state):
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
        score = -child.Q + u
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


def batched_mcts(net, states, sims, add_dirichlet):
    """Run MCTS for a batch of (non-terminal) root states; return [(moves, visits)]."""
    roots = [Node(1.0, s) for s in states]
    logits, _ = batched_eval(net, states)
    for i, r in enumerate(roots):
        expand(r, logits[i])
        if add_dirichlet and r.children:
            moves = list(r.children.keys())
            noise = np.random.dirichlet([DIRICHLET_ALPHA] * len(moves))
            for j, m in enumerate(moves):
                c = r.children[m]
                c.prior = (1 - DIRICHLET_FRAC) * c.prior + DIRICHLET_FRAC * noise[j]

    for _ in range(sims):
        leaf_nodes, leaf_paths, leaf_states = [], [], []
        for r in roots:
            if not r.children:
                continue
            node, path = r, [r]
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

    out = []
    for r in roots:
        moves = list(r.children.keys())
        visits = np.array([r.children[m].N for m in moves], dtype=np.float64)
        out.append((moves, visits))
    return out


def self_play(net, n_games):
    games = [Game() for _ in range(n_games)]
    active = list(games)
    move_no = 0
    while active and move_no < MAX_PLIES:
        policies = batched_mcts(net, [g.state for g in active], MCTS_SIMULATIONS, add_dirichlet=True)
        still = []
        for g, (moves, visits) in zip(active, policies):
            total = visits.sum()
            dist = visits / total if total > 0 else np.ones(len(moves)) / len(moves)
            pi = np.zeros(NUM_ACTIONS, dtype=np.float32)
            for m, p in zip(moves, dist):
                pi[move_to_index(m)] = p
            g.samples.append((encode_state(g.state), pi, g.state[3]))
            if g.ply < TEMP_MOVES and total > 0:
                probs = dist / dist.sum()
                chosen = moves[np.random.choice(len(moves), p=probs)]
            else:
                chosen = moves[int(visits.argmax())]
            g.state = apply_move(g.state, chosen)
            g.ply += 1
            w = winner(g.state)
            if w is not None:
                g.result_winner = w
            else:
                still.append(g)
        active = still
        move_no += 1

    samples, results = [], {"X": 0, "O": 0, "D": 0}
    for g in games:
        w = g.result_winner if g.result_winner is not None else (winner(g.state) or "D")
        results[w if w in results else "D"] += 1
        rf = {"X": 0.0, "O": 0.0} if w == "D" else {w: 1.0, ("O" if w == "X" else "X"): -1.0}
        for enc, pi, player in g.samples:
            samples.append((enc, pi, rf[player]))
    return samples, results


def duel(net_a, net_b, n_games, sims):
    """Play net_a vs net_b (colors alternated). Return net_a's score fraction."""
    games = [(Game(), i % 2 == 0) for i in range(n_games)]   # a_is_x flag
    active = list(games)
    move_no = 0
    while active and move_no < MAX_PLIES:
        a_items, a_states, b_items, b_states = [], [], [], []
        for g, a_is_x in active:
            a_to_move = (g.state[3] == "X") == a_is_x
            if a_to_move:
                a_items.append((g, a_is_x)); a_states.append(g.state)
            else:
                b_items.append((g, a_is_x)); b_states.append(g.state)
        pol_a = batched_mcts(net_a, a_states, sims, add_dirichlet=True) if a_states else []
        pol_b = batched_mcts(net_b, b_states, sims, add_dirichlet=True) if b_states else []

        still = []
        for items, pols in ((a_items, pol_a), (b_items, pol_b)):
            for (g, a_is_x), (moves, visits) in zip(items, pols):
                if g.ply < 2 and visits.sum() > 0:           # a little opening variety
                    dist = visits / visits.sum()
                    chosen = moves[np.random.choice(len(moves), p=dist / dist.sum())]
                else:
                    chosen = moves[int(visits.argmax())]
                g.state = apply_move(g.state, chosen)
                g.ply += 1
                if winner(g.state) is None:
                    still.append((g, a_is_x))
        active = still
        move_no += 1

    points = 0.0
    for g, a_is_x in games:
        w = winner(g.state) or "D"
        if w == "D":
            points += 0.5
        elif w == ("X" if a_is_x else "O"):
            points += 1.0
    return points / n_games


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


def elo_diff(score):
    s = min(max(score, 0.01), 0.99)
    return 400.0 * math.log10(s / (1 - s))


def load_meta():
    if os.path.exists(META_PATH):
        try:
            with open(META_PATH) as f:
                return json.load(f)
        except Exception:
            pass
    return {"elo": 0.0, "promotions": 0, "iterations": 0}


def save_meta(meta):
    with open(META_PATH, "w") as f:
        json.dump(meta, f, indent=2)


def try_load(net, path):
    if os.path.exists(path):
        try:
            net.load_state_dict(torch.load(path, map_location=DEVICE))
            return True
        except Exception as e:
            print(f"[resume] checkpoint incompatible, starting fresh: {e}", flush=True)
    return False


def main():
    iterations = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)
    if DEVICE == "cuda":
        torch.backends.cudnn.benchmark = True

    net = AZNet().to(DEVICE)
    resumed = try_load(net, CKPT_PATH)
    net.eval()
    best = AZNet().to(DEVICE)
    best.load_state_dict(net.state_dict())
    best.eval()
    opt = torch.optim.Adam(net.parameters(), lr=LR)
    buffer = deque(maxlen=REPLAY_CAPACITY)
    meta = load_meta()

    nparams = sum(p.numel() for p in net.parameters())
    budget = f"{MINUTES:.0f}min" if MINUTES > 0 else f"{iterations} iters"
    print(f"[init] device={DEVICE} params={nparams:,} resumed={resumed} "
          f"start_elo={meta['elo']:.0f} games/iter={PARALLEL_GAMES} sims={MCTS_SIMULATIONS} "
          f"eval={EVAL_GAMES}g/{EVAL_SIMS}s every {EVAL_EVERY} thresh={WIN_THRESHOLD} budget={budget}",
          flush=True)

    t_start = time.time()
    it = 0
    while True:
        if MINUTES > 0:
            if (time.time() - t_start) >= MINUTES * 60:
                break
        elif it >= iterations:
            break
        it += 1

        t0 = time.time()
        samples, results = self_play(net, PARALLEL_GAMES)
        buffer.extend(samples)
        tr = train(net, opt, buffer, TRAIN_STEPS_PER_ITER)

        gate = ""
        if it % EVAL_EVERY == 0:
            score = duel(net, best, EVAL_GAMES, EVAL_SIMS)
            if score > WIN_THRESHOLD:
                d = elo_diff(score)
                meta["elo"] += d
                meta["promotions"] += 1
                best.load_state_dict(net.state_dict())
                torch.save(best.state_dict(), CKPT_PATH)
                meta["iterations"] = meta.get("iterations", 0) + EVAL_EVERY
                save_meta(meta)
                gate = f"PROMOTED score={score:.2f} +{d:.0f} -> elo={meta['elo']:.0f}"
            else:
                gate = f"rejected score={score:.2f} (keep best elo={meta['elo']:.0f})"

        msg = (f"[iter {it}] X/O/D={results['X']}/{results['O']}/{results['D']} "
               f"buffer={len(buffer)} selfplay={time.time() - t0:.1f}s")
        if tr:
            msg += f" ploss={tr[0]:.3f} vloss={tr[1]:.3f}"
        if gate:
            msg += f" | {gate}"
        msg += f" | total={time.time() - t_start:.0f}s"
        print(msg, flush=True)

    print(f"[done] iters={it} best_elo={meta['elo']:.0f} promotions={meta['promotions']} "
          f"checkpoint={CKPT_PATH} time={time.time() - t_start:.0f}s", flush=True)


if __name__ == "__main__":
    main()
