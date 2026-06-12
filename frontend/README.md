# Ultimate Tic-Tac-Toe — Frontend

React + Vite + TypeScript app with an in-browser AlphaZero AI (ONNX Runtime Web).
See the [root README](../README.md) for the full project overview.

## Commands

```bash
npm install
npm run dev      # dev server on http://localhost:8080
npm run build    # production build -> dist/
npm run preview  # preview the production build
npm test         # vitest (engine/MCTS parity + unit tests)
npm run lint     # eslint
```

## AI code

The browser AI lives in [`src/lib/ai/`](./src/lib/ai/):

| File         | Role                                                           |
| ------------ | -------------------------------------------------------------- |
| `engine.ts`  | Game rules (port of `backend/engine.py`)                       |
| `encode.ts`  | State → 486-float input tensor (port of `backend/net.py`)      |
| `net.ts`     | ONNX Runtime Web inference (loads `public/az_net.onnx`)        |
| `mcts.ts`    | PUCT Monte Carlo Tree Search (port of `backend/mcts.py`)       |
| `index.ts`   | `getAIMove()` entrypoint + random fallback                     |

The model file `public/az_net.onnx` is produced by `backend/export_onnx.py`.
