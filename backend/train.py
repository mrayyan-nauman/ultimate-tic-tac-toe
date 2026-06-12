"""Self-play training loop for the Q-learning agent.

Run:  python train.py [num_games]
"""
import os
import sys
import time

from engine import initial_state, apply_move, winner, state_key
from qagent import QAgent

QTABLE_PATH = os.path.join(os.path.dirname(__file__), "qtable.pkl")


def play_one_game(agent, epsilon):
    state = initial_state()
    # Separate histories per player so we can reward correctly.
    histories = {"X": [], "O": []}

    while True:
        w = winner(state)
        if w is not None:
            break
        player = state[3]
        move = agent.choose_move(state, epsilon=epsilon)
        if move is None:
            break
        s_key = state_key(state)
        next_state = apply_move(state, move)
        histories[player].append([s_key, move, next_state])
        state = next_state

    w = winner(state)
    return w, histories


def reward_for(player, result):
    if result == "D" or result is None:
        return 0.0
    return 1.0 if result == player else -1.0


def train(num_games=50_000, resume=True):
    agent = QAgent()
    if resume:
        agent.load(QTABLE_PATH)
        if agent.Q:
            print(f"Resumed with {len(agent.Q)} known states.")

    eps_start, eps_end = 1.0, 0.05
    stats = {"X": 0, "O": 0, "D": 0}
    t0 = time.time()

    for g in range(1, num_games + 1):
        # Linear epsilon decay.
        frac = g / num_games
        epsilon = eps_start + (eps_end - eps_start) * frac

        result, histories = play_one_game(agent, epsilon)
        stats[result if result in stats else "D"] += 1

        # Apply rewards: walk each player's history; terminal move uses final reward,
        # earlier moves bootstrap via Q-learning rule.
        for player in ("X", "O"):
            hist = histories[player]
            if not hist:
                continue
            # Mark terminal move's next_state as None so update() uses final_reward.
            hist[-1][2] = None
            agent.update(hist, reward_for(player, result))

        if g % 2_000 == 0:
            dt = time.time() - t0
            print(f"[{g:>6}/{num_games}]  X:{stats['X']}  O:{stats['O']}  D:{stats['D']}  "
                  f"states={len(agent.Q)}  eps={epsilon:.3f}  {dt:.1f}s")
            agent.save(QTABLE_PATH)
            stats = {"X": 0, "O": 0, "D": 0}

    agent.save(QTABLE_PATH)
    print(f"Done. Q-table saved to {QTABLE_PATH} with {len(agent.Q)} states.")


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 50_000
    train(num_games=n)