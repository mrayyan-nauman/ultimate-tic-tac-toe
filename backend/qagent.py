"""Tabular Q-learning agent for Ultimate Tic-Tac-Toe."""
import os
import pickle
import random
from engine import state_key, legal_moves

ALPHA = 0.1     # learning rate
GAMMA = 0.95    # discount factor
DEFAULT_Q = 0.0


class QAgent:
    def __init__(self):
        # Q[state_key] -> dict[move_tuple -> float]
        self.Q = {}

    # ---------- persistence ----------
    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump(self.Q, f)

    def load(self, path):
        if os.path.exists(path):
            with open(path, "rb") as f:
                self.Q = pickle.load(f)

    # ---------- core ----------
    def _row(self, key, moves):
        row = self.Q.get(key)
        if row is None:
            row = {m: DEFAULT_Q for m in moves}
            self.Q[key] = row
        else:
            for m in moves:
                if m not in row:
                    row[m] = DEFAULT_Q
        return row

    def choose_move(self, state, epsilon=0.0):
        moves = legal_moves(state)
        if not moves:
            return None
        if epsilon > 0 and random.random() < epsilon:
            return random.choice(moves)
        key = state_key(state)
        row = self._row(key, moves)
        best_val = max(row[m] for m in moves)
        best = [m for m in moves if row[m] == best_val]
        return random.choice(best)

    def update(self, history, final_reward):
        """
        history: list of (state_key, move, next_state_or_None) recorded in order.
        final_reward: scalar reward for the LAST move in history.
        Earlier moves bootstrap from the next state's max Q.
        """
        for i in range(len(history) - 1, -1, -1):
            s_key, move, next_state = history[i]
            row = self.Q.setdefault(s_key, {})
            old = row.get(move, DEFAULT_Q)

            if i == len(history) - 1 or next_state is None:
                target = final_reward
            else:
                next_moves = legal_moves(next_state)
                if not next_moves:
                    target = final_reward
                else:
                    next_key = state_key(next_state)
                    next_row = self.Q.get(next_key, {})
                    next_max = max((next_row.get(m, DEFAULT_Q) for m in next_moves),
                                   default=DEFAULT_Q)
                    target = GAMMA * next_max

            row[move] = old + ALPHA * (target - old)