"""Batched, GPU-accelerated AlphaZero self-play training.

Plays many games in parallel and evaluates all of their MCTS leaves in a single
batched forward pass per simulation step. The scalar game core is Numba-compiled
in fastcore.py (validated bit-identical to engine.py), removing the Python
game-op bottleneck.

Two-level gating:
  * Lineage gate (internal): a candidate replaces the running `best` only if it
    wins a head-to-head match vs `best` (>= WIN_THRESHOLD). `best` therefore
    never regresses against its own lineage. Saved to the *training* checkpoint
    (az_net.train.pt, gitignored).
  * Deploy gate (external): the published model (az_net.pt, committed) is only
    updated when `best` beats the OLD pre-GPU net (az_net.pt.bak) by
    VS_OLD_THRESH. Every eval logs the best-vs-old score, so progress toward the
    real opponent is measurable. With --deploy, a passing net is exported +
    pushed (Vercel redeploys).

Tunables via env:
    UTTT_GAMES UTTT_SIMS UTTT_EVAL_GAMES UTTT_EVAL_SIMS UTTT_EVAL_EVERY
    UTTT_WIN_THRESH UTTT_VS_OLD_THRESH UTTT_FINAL_EVAL_GAMES UTTT_MINUTES UTTT_CKPT

Usage:
    python train_az_gpu.py [iterations] [--deploy]
"""
import os
import sys
import json
import math
import time
import random
import subprocess
from collections import deque

import numpy as np
import torch
import torch.nn.functional as F

import fastcore as fc
from net import AZNet, AZNetConv, OldAZNet, INPUT_DIM, INPUT_DIM_V2

ARCH = os.environ.get("UTTT_ARCH", "mlp")          # mlp | conv
CONV_CH = int(os.environ.get("UTTT_CH", 64))
CONV_BLOCKS = int(os.environ.get("UTTT_BLOCKS", 6))

HERE = os.path.dirname(os.path.abspath(__file__))
DEPLOY_CKPT = os.environ.get("UTTT_CKPT", os.path.join(HERE, "az_net.pt"))
DEPLOY_META = DEPLOY_CKPT[:-3] + ".meta.json" if DEPLOY_CKPT.endswith(".pt") else DEPLOY_CKPT + ".meta.json"
TRAIN_CKPT = DEPLOY_CKPT[:-3] + ".train.pt" if DEPLOY_CKPT.endswith(".pt") else DEPLOY_CKPT + ".train.pt"
TRAIN_META = DEPLOY_CKPT[:-3] + ".train.json" if DEPLOY_CKPT.endswith(".pt") else DEPLOY_CKPT + ".train.json"
OLD_PATH = os.environ.get("UTTT_OLD", DEPLOY_CKPT + ".bak")
PLANES = int(os.environ.get("UTTT_PLANES", 6))  # 6 = v1 encoding, 11 = v2 (tactical planes)

PARALLEL_GAMES = int(os.environ.get("UTTT_GAMES", 256))
MCTS_SIMULATIONS = int(os.environ.get("UTTT_SIMS", 64))
EVAL_GAMES = int(os.environ.get("UTTT_EVAL_GAMES", 40))
EVAL_SIMS = int(os.environ.get("UTTT_EVAL_SIMS", 64))
EVAL_EVERY = int(os.environ.get("UTTT_EVAL_EVERY", 2))
WIN_THRESHOLD = float(os.environ.get("UTTT_WIN_THRESH", 0.55))
VS_OLD_THRESH = float(os.environ.get("UTTT_VS_OLD_THRESH", 0.55))
FINAL_EVAL_GAMES = int(os.environ.get("UTTT_FINAL_EVAL_GAMES", 80))
MINUTES = float(os.environ.get("UTTT_MINUTES", 0))

NUM_ACTIONS = 81
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
        self.state = fc.initial_state()
        self.samples = []
        self.ply = 0
        self.result_winner = None


def terminal_value(state):
    v = fc.nb_terminal_value(state)
    return None if v == 2.0 else v


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
    moves = fc.nb_legal(node.state)
    node.expanded = True
    if moves.shape[0] == 0:
        node.terminal_value = 0.0
        return
    masked = logits_row[moves]
    masked = masked - masked.max()
    exp = np.exp(masked)
    probs = exp / exp.sum()
    ch, st = node.children, node.state
    for k in range(moves.shape[0]):
        mv = int(moves[k])
        ch[mv] = Node(float(probs[k]), fc.nb_apply(st, mv))


@torch.no_grad()
def batched_eval(net, states, encode_fn):
    x = torch.from_numpy(encode_fn(states)).to(DEVICE, non_blocking=True)
    logits, values = net(x)
    return logits.detach().cpu().numpy(), values.detach().cpu().numpy()


def batched_mcts(net, states, sims, add_dirichlet, encode_fn):
    roots = [Node(1.0, s) for s in states]
    logits, _ = batched_eval(net, states, encode_fn)
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
            lg, lv = batched_eval(net, leaf_states, encode_fn)
            for k in range(len(leaf_nodes)):
                expand(leaf_nodes[k], lg[k])
                backup(leaf_paths[k], float(lv[k]))

    out = []
    for r in roots:
        moves = list(r.children.keys())
        visits = np.array([r.children[m].N for m in moves], dtype=np.float64)
        out.append((moves, visits))
    return out


def self_play(net, n_games, encode_fn):
    games = [Game() for _ in range(n_games)]
    active = list(games)
    move_no = 0
    while active and move_no < MAX_PLIES:
        states = [g.state for g in active]
        root_enc = encode_fn(states)
        policies = batched_mcts(net, states, MCTS_SIMULATIONS, True, encode_fn)
        still = []
        for i, (g, (moves, visits)) in enumerate(zip(active, policies)):
            total = visits.sum()
            dist = visits / total if total > 0 else np.ones(len(moves)) / len(moves)
            pi = np.zeros(NUM_ACTIONS, dtype=np.float32)
            for k, mv in enumerate(moves):
                pi[mv] = dist[k]
            g.samples.append((root_enc[i].copy(), pi, int(g.state[91])))
            if g.ply < TEMP_MOVES and total > 0:
                chosen = moves[np.random.choice(len(moves), p=dist / dist.sum())]
            else:
                chosen = moves[int(visits.argmax())]
            g.state = fc.nb_apply(g.state, chosen)
            g.ply += 1
            w = fc.nb_winner(g.state)
            if w != 0:
                g.result_winner = w
            else:
                still.append(g)
        active = still
        move_no += 1

    samples, results = [], {1: 0, 2: 0, 3: 0}
    for g in games:
        w = g.result_winner if g.result_winner is not None else fc.nb_winner(g.state)
        if w == 0:
            w = 3
        results[w] += 1
        rf = {1: 0.0, 2: 0.0} if w == 3 else {w: 1.0, (2 if w == 1 else 1): -1.0}
        for enc, pi, player in g.samples:
            samples.append((enc, pi, rf[player]))
    return samples, results


def duel(net_a, net_b, n_games, sims, enc_a, enc_b):
    """net_a vs net_b (each with its own encoder), colors alternated.
    Returns net_a's score fraction."""
    games = [(Game(), i % 2 == 0) for i in range(n_games)]
    active = list(games)
    move_no = 0
    while active and move_no < MAX_PLIES:
        a_items, a_states, b_items, b_states = [], [], [], []
        for g, a_is_x in active:
            a_to_move = (int(g.state[91]) == 1) == a_is_x
            if a_to_move:
                a_items.append((g, a_is_x)); a_states.append(g.state)
            else:
                b_items.append((g, a_is_x)); b_states.append(g.state)
        pol_a = batched_mcts(net_a, a_states, sims, True, enc_a) if a_states else []
        pol_b = batched_mcts(net_b, b_states, sims, True, enc_b) if b_states else []
        still = []
        for items, pols in ((a_items, pol_a), (b_items, pol_b)):
            for (g, a_is_x), (moves, visits) in zip(items, pols):
                if g.ply < 2 and visits.sum() > 0:
                    dist = visits / visits.sum()
                    chosen = moves[np.random.choice(len(moves), p=dist / dist.sum())]
                else:
                    chosen = moves[int(visits.argmax())]
                g.state = fc.nb_apply(g.state, chosen)
                g.ply += 1
                if fc.nb_winner(g.state) == 0:
                    still.append((g, a_is_x))
        active = still
        move_no += 1

    points = 0.0
    for g, a_is_x in games:
        w = fc.nb_winner(g.state)
        if w == 0 or w == 3:
            points += 0.5
        elif w == (1 if a_is_x else 2):
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
    for p in (TRAIN_META, DEPLOY_META):
        if os.path.exists(p):
            try:
                with open(p) as f:
                    return json.load(f)
            except Exception:
                pass
    return {"elo": 0.0, "promotions": 0, "vs_old": None}


def save_json(path, obj):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def load_into(net, path):
    try:
        net.load_state_dict(torch.load(path, map_location=DEVICE))
        return True
    except Exception as e:
        print(f"[resume] {os.path.basename(path)} incompatible: {e}", flush=True)
        return False


def resume(net):
    for p in (TRAIN_CKPT, DEPLOY_CKPT):
        if os.path.exists(p) and load_into(net, p):
            return os.path.basename(p)
    return None


def load_benchmark(path):
    """Load the benchmark opponent, auto-detecting its architecture + encoder
    (v1 6-plane AZNet/OldAZNet or v2 11-plane AZNet). Returns (net, encode_fn)."""
    if not os.path.exists(path):
        print(f"[old] benchmark {os.path.basename(path)} not found — deploy gating disabled", flush=True)
        return None, None
    sd = torch.load(path, map_location=DEVICE)
    indim = None
    for k in ("stem.0.weight", "trunk.0.weight"):
        if k in sd:
            indim = int(sd[k].shape[1])
            break
    try:
        net = AZNet(input_dim=indim).to(DEVICE) if "stem.0.weight" in sd else OldAZNet().to(DEVICE)
        net.load_state_dict(sd)
        net.eval()
    except Exception as e:
        print(f"[old] could not load benchmark {os.path.basename(path)}: {e}", flush=True)
        return None, None
    enc = fc.encode_batch_v2 if indim == INPUT_DIM_V2 else fc.encode_batch
    return net, enc


def deploy_best(best, meta, vs_old):
    try:
        print("[deploy] best beats old — exporting ONNX + pushing...", flush=True)
        torch.save(best.state_dict(), DEPLOY_CKPT)
        save_json(DEPLOY_META, {"planes": PLANES, "elo": meta["elo"],
                                "promotions": meta["promotions"], "vs_old": vs_old})
        # Export the right architecture from the just-saved checkpoint (quantized).
        env = {**os.environ, "UTTT_PLANES": str(PLANES), "UTTT_EXPORT_SRC": DEPLOY_CKPT,
               "UTTT_ARCH": ARCH, "UTTT_CH": str(CONV_CH), "UTTT_BLOCKS": str(CONV_BLOCKS)}
        subprocess.run([sys.executable, os.path.join(HERE, "export_onnx.py")], check=True, cwd=HERE, env=env)
        repo = os.path.dirname(HERE)
        # az_net.pt is gitignored now — only the committed meta + onnx are deployed.
        subprocess.run(["git", "-C", repo, "add", os.path.relpath(DEPLOY_META, repo).replace("\\", "/"),
                        "frontend/public/az_net.onnx"], check=True)
        subprocess.run(["git", "-C", repo, "commit", "-m",
                        f"Train: deploy net that beats old (vs_old={vs_old:.2f}, elo {meta['elo']:.0f})"], check=True)
        subprocess.run(["git", "-C", repo, "-c", "http.version=HTTP/1.1", "push", "origin", "main"], check=True)
        print("[deploy] pushed — Vercel will redeploy.", flush=True)
    except Exception as e:
        print(f"[deploy] FAILED: {e}", flush=True)


def build_net(input_dim):
    if ARCH == "conv":
        return AZNetConv(input_dim=input_dim, channels=CONV_CH, blocks=CONV_BLOCKS).to(DEVICE)
    return AZNet(input_dim=input_dim).to(DEVICE)


def main():
    args = sys.argv[1:]
    deploy = "--deploy" in args
    nums = [a for a in args if a.lstrip("-").isdigit()]
    iterations = int(nums[0]) if nums else 10

    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)
    if DEVICE == "cuda":
        torch.backends.cudnn.benchmark = True

    fc.validate_against_engine(60)  # safety + warm up the JIT

    # Encoding: v1 (6 planes) or v2 (11 tactical planes). The benchmark opponent
    # is always a 6-plane net, so it uses the v1 encoder regardless.
    enc = fc.encode_batch_v2 if PLANES == 11 else fc.encode_batch
    input_dim = INPUT_DIM_V2 if PLANES == 11 else INPUT_DIM

    net = build_net(input_dim)
    src = resume(net)
    net.eval()
    best = build_net(input_dim)
    best.load_state_dict(net.state_dict())
    best.eval()
    old, enc_old = load_benchmark(OLD_PATH)
    opt = torch.optim.Adam(net.parameters(), lr=LR)
    buffer = deque(maxlen=REPLAY_CAPACITY)
    meta = load_meta()

    nparams = sum(p.numel() for p in net.parameters())
    budget = f"{MINUTES:.0f}min" if MINUTES > 0 else f"{iterations} iters"
    print(f"[init] device={DEVICE} arch={ARCH} planes={PLANES} params={nparams:,} resume={src} "
          f"start_elo={meta['elo']:.0f} games/iter={PARALLEL_GAMES} sims={MCTS_SIMULATIONS} "
          f"eval={EVAL_GAMES}g/{EVAL_SIMS}s every {EVAL_EVERY} winthr={WIN_THRESHOLD} "
          f"vs_old_thr={VS_OLD_THRESH} old={os.path.basename(OLD_PATH) if old is not None else 'no'} "
          f"budget={budget} deploy={deploy}", flush=True)

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
        samples, results = self_play(net, PARALLEL_GAMES, enc)
        buffer.extend(samples)
        tr = train(net, opt, buffer, TRAIN_STEPS_PER_ITER)

        gate = ""
        if it % EVAL_EVERY == 0:
            score = duel(net, best, EVAL_GAMES, EVAL_SIMS, enc, enc)
            if score > WIN_THRESHOLD:
                meta["elo"] += elo_diff(score)
                meta["promotions"] += 1
                best.load_state_dict(net.state_dict())
                torch.save(best.state_dict(), TRAIN_CKPT)
                save_json(TRAIN_META, meta)
                gate = f"PROMOTED {score:.2f} -> elo={meta['elo']:.0f}"
            else:
                gate = f"rejected {score:.2f}"
            if old is not None:
                vs_old = duel(best, old, EVAL_GAMES, EVAL_SIMS, enc, enc_old)
                meta["vs_old"] = vs_old
                save_json(TRAIN_META, meta)
                gate += f" | vs_old={vs_old:.2f}"

        msg = (f"[iter {it}] X/O/D={results[1]}/{results[2]}/{results[3]} "
               f"buffer={len(buffer)} selfplay={time.time() - t0:.1f}s")
        if tr:
            msg += f" ploss={tr[0]:.3f} vloss={tr[1]:.3f}"
        if gate:
            msg += f" | {gate}"
        msg += f" | total={time.time() - t_start:.0f}s"
        print(msg, flush=True)

    # Final, larger best-vs-old measurement → deploy decision.
    final_vs_old = None
    if old is not None:
        final_vs_old = duel(best, old, FINAL_EVAL_GAMES, EVAL_SIMS, enc, enc_old)
        print(f"[vs-old] final: best scores {final_vs_old:.0%} vs old over {FINAL_EVAL_GAMES} games "
              f"(need >= {VS_OLD_THRESH:.0%} to deploy)", flush=True)
    print(f"[done] iters={it} elo={meta['elo']:.0f} promotions={meta['promotions']} "
          f"time={time.time() - t_start:.0f}s", flush=True)

    if deploy and final_vs_old is not None and final_vs_old >= VS_OLD_THRESH:
        deploy_best(best, meta, final_vs_old)
    elif deploy:
        why = "no old benchmark" if final_vs_old is None else f"vs_old={final_vs_old:.2f} < {VS_OLD_THRESH}"
        print(f"[deploy] not deploying ({why}); live site unchanged.", flush=True)


if __name__ == "__main__":
    main()
