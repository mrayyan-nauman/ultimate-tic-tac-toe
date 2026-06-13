"""Generate cross-language parity fixtures for the TypeScript AI port.

Plays random self-play games with the canonical Python engine, samples states,
and writes each state together with the reference encode_state() vector, legal
moves, and overall winner. The frontend vitest suite loads this JSON and asserts
its TypeScript port produces identical results.

Run from backend/:
    python gen_test_fixtures.py
"""
import json
import os
import random

from engine import initial_state, legal_moves, apply_move, winner
from net import encode_state, encode_state_v2

OUT_PATH = os.path.join(
    os.path.dirname(__file__), "..", "frontend", "src", "test", "ai-fixtures.json"
)


def state_to_json(state):
    boards, winners, active, player = state
    return {
        "boards": [list(b) for b in boards],
        "boardWinners": list(winners),
        "activeBoard": active,
        "player": player,
    }


def main():
    random.seed(1234)
    samples = []

    # Always include the empty starting position.
    states = [initial_state()]

    # Plus a spread of mid-game positions from random self-play.
    for _ in range(40):
        state = initial_state()
        for _ in range(random.randint(1, 50)):
            moves = legal_moves(state)
            if not moves or winner(state) is not None:
                break
            state = apply_move(state, random.choice(moves))
        states.append(state)

    for state in states:
        samples.append({
            "state": state_to_json(state),
            "encoded": [float(x) for x in encode_state(state)],
            "encoded_v2": [float(x) for x in encode_state_v2(state)],
            "legalMoves": [list(m) for m in legal_moves(state)],
            "winner": winner(state),
        })

    os.makedirs(os.path.dirname(os.path.abspath(OUT_PATH)), exist_ok=True)
    with open(os.path.abspath(OUT_PATH), "w") as f:
        json.dump(samples, f)
    print(f"Wrote {len(samples)} fixtures -> {os.path.abspath(OUT_PATH)}")


if __name__ == "__main__":
    main()
