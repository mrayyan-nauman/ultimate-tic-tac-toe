// Public entrypoint for the in-browser AlphaZero AI.
// Applies a difficulty config (epsilon / temperature / search budget) on top of
// the AlphaZero net + MCTS. Falls back to a random legal move if the model can't
// load or inference fails, so the game is always playable.

import { State, Move, legalMoves } from "./engine";
import { loadModel } from "./net";
import { predict } from "./net";
import { runMcts } from "./mcts";
import { AIConfig, difficultyToConfig } from "./difficulty";

export type { AIConfig } from "./difficulty";
export interface MoveResult {
  boardIndex: number;
  cellIndex: number;
}

// If the model + WASM don't finish loading within this window, fall back to a
// random move instead of hanging (e.g. very slow / blocked CDN). The session
// keeps loading in the background, so later moves use the real net once ready.
const MODEL_LOAD_TIMEOUT_MS = 20000;

function withTimeout<T>(promise: Promise<T>, ms: number, label: string): Promise<T> {
  return new Promise<T>((resolve, reject) => {
    const timer = setTimeout(() => reject(new Error(`${label} timed out after ${ms}ms`)), ms);
    promise.then(
      (v) => { clearTimeout(timer); resolve(v); },
      (e) => { clearTimeout(timer); reject(e); },
    );
  });
}

function uniformRandom(moves: Move[]): MoveResult {
  const [bi, ci] = moves[Math.floor(Math.random() * moves.length)];
  return { boardIndex: bi, cellIndex: ci };
}

/** Pick from parallel (move, weight) arrays using a temperature.
 *  temperature ~0 → argmax; 1 → proportional; large → more uniform. */
function sampleByTemperature(moves: Move[], weights: number[], temperature: number): MoveResult {
  if (temperature <= 1e-3) {
    let best = 0;
    for (let i = 1; i < weights.length; i++) if (weights[i] > weights[best]) best = i;
    const [bi, ci] = moves[best];
    return { boardIndex: bi, cellIndex: ci };
  }
  const adj = weights.map((w) => Math.pow(Math.max(w, 0), 1 / temperature));
  const sum = adj.reduce((a, b) => a + b, 0);
  if (!(sum > 0)) return uniformRandom(moves);
  let r = Math.random() * sum;
  for (let i = 0; i < moves.length; i++) {
    r -= adj[i];
    if (r <= 0) {
      const [bi, ci] = moves[i];
      return { boardIndex: bi, cellIndex: ci };
    }
  }
  const [bi, ci] = moves[moves.length - 1];
  return { boardIndex: bi, cellIndex: ci };
}

/** Kick off model loading early so the first move isn't delayed. Fire-and-forget. */
export function warmUpAI(): void {
  loadModel().catch(() => {
    /* fallback handled in getAIMove */
  });
}

/**
 * Choose the AI's move for the current position at the given difficulty.
 * `config` may be an AIConfig or a raw rating (1..2000); defaults to max.
 */
export async function getAIMove(
  boards: (string | null)[][],
  boardWinners: (string | null)[],
  activeBoard: number | null,
  player: "X" | "O" = "O",
  config?: AIConfig | number,
): Promise<MoveResult | null> {
  const state: State = {
    boards: boards as State["boards"],
    winners: boardWinners as State["winners"],
    active: activeBoard,
    player,
  };

  const moves = legalMoves(state);
  if (moves.length === 0) return null;

  const cfg: AIConfig =
    typeof config === "number"
      ? difficultyToConfig(config)
      : config ?? difficultyToConfig(2000);

  // 1) Blunder roll: with probability epsilon, play a uniformly random move.
  if (cfg.epsilon > 0 && Math.random() < cfg.epsilon) return uniformRandom(moves);

  // Only one legal move? Take it without spinning up the net.
  if (moves.length === 1) return uniformRandom(moves);

  let session;
  try {
    session = await withTimeout(loadModel(), MODEL_LOAD_TIMEOUT_MS, "model load");
  } catch (err) {
    console.warn("[AI] model load failed, playing randomly:", err);
    return uniformRandom(moves);
  }

  try {
    // 2) Low end: no tree search — sample the raw policy at temperature.
    if (cfg.timeBudgetMs <= 0) {
      const { moves: pMoves, probs } = await predict(session, state);
      if (pMoves.length === 0) return uniformRandom(moves);
      return sampleByTemperature(pMoves, probs, cfg.temperature);
    }

    // 3) Search: time-budgeted MCTS, then pick by visit counts at temperature.
    const { counts, moves: lookup } = await runMcts(state, session, {
      timeBudgetMs: cfg.timeBudgetMs,
      maxSims: cfg.maxSims,
    });
    const keys = [...counts.keys()];
    if (keys.length === 0) return uniformRandom(moves);
    const visits = keys.map((k) => counts.get(k) as number);
    const mvs = keys.map((k) => lookup.get(k) as Move);
    return sampleByTemperature(mvs, visits, cfg.temperature);
  } catch (err) {
    console.warn("[AI] inference failed, playing randomly:", err);
    return uniformRandom(moves);
  }
}
