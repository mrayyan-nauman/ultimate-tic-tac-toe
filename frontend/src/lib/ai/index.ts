// Public entrypoint for the in-browser AlphaZero AI.
// Mirrors backend/ai.py: load the net, run MCTS, play the most-visited move.
// Falls back to a random legal move if the model can't load or inference fails,
// so the game is always playable.

import { State, Move, legalMoves } from "./engine";
import { loadModel } from "./net";
import { runMcts } from "./mcts";

// Matches INFERENCE_SIMULATIONS in backend/ai.py. Higher = stronger but slower.
const SIMULATIONS = 128;

function randomMove(moves: Move[]): { boardIndex: number; cellIndex: number } | null {
  if (moves.length === 0) return null;
  const [bi, ci] = moves[Math.floor(Math.random() * moves.length)];
  return { boardIndex: bi, cellIndex: ci };
}

/** Kick off model loading early (e.g. on the game screen mount) so the first
 *  move isn't delayed by the ~1 MB download + WASM init. Fire-and-forget. */
export function warmUpAI(): void {
  loadModel().catch(() => {
    /* fallback handled in getAIMove */
  });
}

/**
 * Choose the AI's move for the current position.
 * Accepts the frontend's loose board types and narrows internally.
 */
export async function getAIMove(
  boards: (string | null)[][],
  boardWinners: (string | null)[],
  activeBoard: number | null,
  player: "X" | "O" = "O",
): Promise<{ boardIndex: number; cellIndex: number } | null> {
  const state: State = {
    boards: boards as State["boards"],
    winners: boardWinners as State["winners"],
    active: activeBoard,
    player,
  };

  const moves = legalMoves(state);
  if (moves.length === 0) return null;

  let session;
  try {
    session = await loadModel();
  } catch (err) {
    console.warn("[AI] model load failed, playing randomly:", err);
    return randomMove(moves);
  }

  try {
    const { counts, moves: moveLookup } = await runMcts(state, session, SIMULATIONS);
    let bestKey: string | null = null;
    let bestN = -1;
    for (const [k, n] of counts) {
      if (n > bestN) {
        bestN = n;
        bestKey = k;
      }
    }
    if (bestKey === null) return randomMove(moves);
    const [bi, ci] = moveLookup.get(bestKey) as Move;
    return { boardIndex: bi, cellIndex: ci };
  } catch (err) {
    console.warn("[AI] inference failed, playing randomly:", err);
    return randomMove(moves);
  }
}
