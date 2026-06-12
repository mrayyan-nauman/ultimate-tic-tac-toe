"""PUCT Monte Carlo Tree Search guided by the AZNet."""
import math
import numpy as np

from engine import legal_moves, apply_move, winner, state_key
from net import predict

C_PUCT = 1.5


class Node:
    __slots__ = ("prior", "N", "W", "Q", "children", "state", "terminal_value")

    def __init__(self, prior, state):
        self.prior = prior          # P(s,a) from parent's perspective
        self.N = 0
        self.W = 0.0
        self.Q = 0.0
        self.children = {}          # move -> Node
        self.state = state          # state AFTER the move that led here
        self.terminal_value = None  # value from the perspective of player-to-move at `state`


def _terminal_value(state):
    """Return value from the perspective of the player about to move at `state`.
    If state has a winner, the previous player just won, so the player to move loses.
    """
    w = winner(state)
    if w is None:
        return None
    if w == "D":
        return 0.0
    # `state[3]` is the player to move; if they don't match the winner, they lost.
    return 1.0 if state[3] == w else -1.0


def _expand(node, net, device):
    tv = _terminal_value(node.state)
    if tv is not None:
        node.terminal_value = tv
        return tv
    probs, value = predict(net, node.state, device=device)
    for move, p in probs.items():
        child_state = apply_move(node.state, move)
        node.children[move] = Node(prior=p, state=child_state)
    return value


def _select(node):
    best_score, best_move, best_child = -float("inf"), None, None
    sqrt_sum = math.sqrt(node.N + 1e-8)
    for move, child in node.children.items():
        u = C_PUCT * child.prior * sqrt_sum / (1 + child.N)
        # Child Q is from opponent's perspective; flip for us.
        score = -child.Q + u
        if score > best_score:
            best_score, best_move, best_child = score, move, child
    return best_move, best_child


def run_mcts(root_state, net, simulations=64, device="cpu", add_dirichlet=True):
    root = Node(prior=1.0, state=root_state)
    _expand(root, net, device)

    # Dirichlet noise on root priors for exploration during self-play.
    if add_dirichlet and root.children:
        moves = list(root.children.keys())
        noise = np.random.dirichlet([0.3] * len(moves))
        for m, n in zip(moves, noise):
            root.children[m].prior = 0.75 * root.children[m].prior + 0.25 * n

    for _ in range(simulations):
        node = root
        path = [node]
        # Selection: walk to a leaf
        while node.children and node.terminal_value is None:
            _, node = _select(node)
            path.append(node)

        # Expansion / evaluation
        if node.terminal_value is not None:
            value = node.terminal_value
        else:
            value = _expand(node, net, device)

        # Backup. `value` is from the perspective of the player to move at `node.state`.
        # As we walk up, the perspective flips each step.
        v = value
        for n in reversed(path):
            n.N += 1
            n.W += v
            n.Q = n.W / n.N
            v = -v

    # Visit-count policy
    counts = {m: c.N for m, c in root.children.items()}
    return counts


def policy_from_counts(counts, temperature=1.0):
    moves = list(counts.keys())
    visits = np.array([counts[m] for m in moves], dtype=np.float64)
    if temperature <= 1e-3:
        probs = np.zeros_like(visits)
        probs[visits.argmax()] = 1.0
    else:
        v = visits ** (1.0 / temperature)
        probs = v / v.sum()
    return moves, probs