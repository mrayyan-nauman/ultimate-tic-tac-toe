# Ultimate Tic-Tac-Toe

A polished, neon-styled [Ultimate Tic-Tac-Toe](https://en.wikipedia.org/wiki/Ultimate_tic-tac-toe)
where you play against an **AlphaZero-style AI that runs entirely in your browser** —
no backend, no server costs, no cold starts.

🔗 **Live demo:** **https://ultimate-tic-tac-toe-alpha.vercel.app**

---

## What is Ultimate Tic-Tac-Toe?

It's tic-tac-toe nested inside tic-tac-toe. The board is a 3×3 grid of nine small
boards. Win a small board to claim that cell of the big board; win three small
boards in a row to win the game. The twist: **the cell you play in decides which
small board your opponent must play in next.** That single rule turns a solved
child's game into one with a genuinely deep search space.

## How the AI works

The opponent is a miniature **AlphaZero**:

- A small policy + value neural network (`net.py`) was trained from scratch by
  **self-play reinforcement learning** (`train_az.py`) — no human games, no opening book.
- At play time, the net guides a **PUCT Monte Carlo Tree Search** (`mcts.py`):
  128 simulations per move, picking the most-visited action.

The clever part is the deployment. Instead of hosting Python + PyTorch on a server,
the trained net is exported to **ONNX** (`export_onnx.py`) and the entire engine +
MCTS is re-implemented in TypeScript (`frontend/src/lib/ai/`). The network runs in
the browser via [ONNX Runtime Web](https://onnxruntime.ai/docs/get-started/with-javascript/web.html)
(WebAssembly). The result is a **single static site** — free to host, instant to
load, and identical in strength to the trained agent.

A 125-case test suite (`frontend/src/test/ai-engine.test.ts`) asserts the
TypeScript engine/encoder reproduces the canonical Python implementation
*exactly*, so the web AI can't silently drift from the trained model.

```
┌─────────────────── Browser (static site) ───────────────────┐
│  React UI ──▶ Web Worker ──▶ TS MCTS ──▶ ONNX Runtime (WASM) │
│                                            └─ az_net.onnx     │
└──────────────────────────────────────────────────────────────┘
   No backend at runtime. Flask + PyTorch are for training only.
   The AI runs in a Web Worker so even a long "think" never freezes the UI.
```

## Difficulty rating (1–2000) and sides

Before each game you pick an **AI rating from 1 to 2000** (a "compressed chess
rating" — higher is harder) and **which side you play** (X always moves first, so
choosing O lets the AI open). The rating maps (`frontend/src/lib/ai/difficulty.ts`)
to three strength knobs that together span the whole range:

| Range        | Behaviour                                                         |
| ------------ | ----------------------------------------------------------------- |
| 1            | Uniformly random moves                                            |
| ~1–600       | Mostly random → raw policy, no tree search                        |
| ~600–1400    | Light, then growing MCTS; temperature softens move choice         |
| ~1400–1999   | Argmax MCTS, thinking budget up to **10 s**                       |
| 2000         | The strongest the deployed net can play — up to **60 s** / move   |

**2000 is anchored to the *currently deployed* network.** Train and deploy a
stronger net and "2000" automatically re-anchors to that net's maximum, with the
rest of the scale shifting accordingly — no UI change needed.

### Raising the ceiling (making 2000 stronger)

The strength of the top end is set by the network, and the AlphaZero self-play
loop can keep improving it with more training. To push the ceiling:

```bash
cd backend
python train_az.py 200      # more self-play iterations = stronger net
python export_onnx.py       # re-export to frontend/public/az_net.onnx
```

Redeploy and the whole 1–2000 scale re-anchors to the new maximum.

## Tech stack

| Layer        | Tech                                                                 |
| ------------ | -------------------------------------------------------------------- |
| Frontend     | React 18, TypeScript, Vite, Tailwind CSS, shadcn/ui, Framer Motion   |
| In-browser AI| ONNX Runtime Web (WASM), custom MCTS in TypeScript                   |
| Training     | Python, PyTorch (AlphaZero self-play), tabular Q-learning baseline   |
| Dev API      | Flask + Flask-CORS (optional local server)                          |

---

## Getting started

### Play locally (frontend only)

```bash
cd frontend
npm install
npm run dev          # open http://localhost:8080
```

That's all you need — the AI runs in the browser.

### Run the tests

```bash
cd frontend
npm test             # engine/MCTS parity + unit tests
npm run lint
npm run build
```

---

## Backend (training & optional API)

The Python side is only needed to (re)train the model or run the original Flask
API. The exported `frontend/public/az_net.onnx` is already committed, so you don't
need any of this just to play.

```bash
cd backend
python -m venv venv && source venv/Scripts/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

| Task                         | Command                              |
| ---------------------------- | ------------------------------------ |
| Train the AlphaZero net      | `python train_az.py`                 |
| Export the net to ONNX       | `python export_onnx.py`              |
| Regenerate parity fixtures   | `python gen_test_fixtures.py`        |
| Run the original Flask API   | `python app.py` (serves `:5000`)     |
| Train the Q-learning baseline| `python train.py`                    |

> **Note:** `backend/qtable.pkl` (the 1.2 GB tabular Q-table) is **not** committed.
> It is regenerated by `train.py`.

---

## Project structure

```
ultimate-tic-tac-toe/
├── frontend/                 # React + Vite game (the deployed app)
│   ├── public/az_net.onnx    # trained net, exported for the browser
│   └── src/
│       ├── lib/ai/           # engine, encode, net (ONNX), mcts, difficulty,
│       │                     #   aiWorker + aiClient (Web Worker), entrypoint
│       ├── pages/            # Landing, Setup (difficulty + sides), Game
│       └── components/       # board UI + shadcn/ui
├── backend/                  # PyTorch training + Flask API (dev/training only)
│   ├── engine.py net.py mcts.py    # canonical game + AlphaZero net + MCTS
│   ├── train_az.py qagent.py train.py
│   ├── export_onnx.py gen_test_fixtures.py
│   └── az_net.pt requirements.txt
└── README.md
```

## Deployment

The site deploys to **Vercel** as a static build:

- **Framework preset:** Vite
- **Root directory:** `frontend`
- SPA routing is handled by `frontend/vercel.json`.

Pushing to `main` triggers an automatic production deploy.

## License

[MIT](./LICENSE)
