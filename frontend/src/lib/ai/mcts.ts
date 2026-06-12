// PUCT Monte Carlo Tree Search guided by the AZNet — a port of backend/mcts.py.
// Dirichlet root noise is omitted (inference/exploitation only), matching
// ai.get_ai_move(..., add_dirichlet=False) on the backend.

import type { InferenceSession } from "onnxruntime-web";
import { State, Move, applyMove, getWinner, moveKey } from "./engine";
import { predict } from "./net";

const C_PUCT = 1.5;

class Node {
  prior: number;
  N = 0;
  W = 0;
  Q = 0;
  children = new Map<string, Node>();
  moves = new Map<string, Move>();
  state: State;
  terminalValue: number | null = null;

  constructor(prior: number, state: State) {
    this.prior = prior;
    this.state = state;
  }
}

/** Value from the perspective of the player about to move at `state`. */
function terminalValue(state: State): number | null {
  const w = getWinner(state);
  if (w === null) return null;
  if (w === "D") return 0;
  return state.player === w ? 1 : -1;
}

async function expand(node: Node, session: InferenceSession): Promise<number> {
  const tv = terminalValue(node.state);
  if (tv !== null) {
    node.terminalValue = tv;
    return tv;
  }
  const { moves, probs, value } = await predict(session, node.state);
  for (let i = 0; i < moves.length; i++) {
    const k = moveKey(moves[i]);
    node.children.set(k, new Node(probs[i], applyMove(node.state, moves[i])));
    node.moves.set(k, moves[i]);
  }
  return value;
}

function select(node: Node): Node {
  let bestScore = -Infinity;
  let best: Node | null = null;
  const sqrtSum = Math.sqrt(node.N + 1e-8);
  for (const child of node.children.values()) {
    const u = (C_PUCT * child.prior * sqrtSum) / (1 + child.N);
    // Child Q is from the opponent's perspective; flip for us.
    const score = -child.Q + u;
    if (score > bestScore) {
      bestScore = score;
      best = child;
    }
  }
  return best as Node;
}

export interface MctsBudget {
  /** Fixed simulation count (used when timeBudgetMs is 0/absent). */
  simulations?: number;
  /** Wall-clock budget in ms; runs until this elapses (or maxSims is hit). */
  timeBudgetMs?: number;
  /** Hard cap on simulations regardless of time. */
  maxSims?: number;
}

/**
 * Run MCTS and return visit counts keyed by "bi,ci", plus the move lookup.
 * Stops on whichever of {time budget, sim cap, fixed sim count} comes first.
 */
export async function runMcts(
  rootState: State,
  session: InferenceSession,
  budget: MctsBudget = {},
): Promise<{ counts: Map<string, number>; moves: Map<string, Move> }> {
  const root = new Node(1, rootState);
  await expand(root, session);

  const collect = () => {
    const counts = new Map<string, number>();
    for (const [k, child] of root.children) counts.set(k, child.N);
    return { counts, moves: root.moves };
  };

  // Terminal or forced move: no search needed.
  if (root.children.size <= 1) return collect();

  const timeBudgetMs = budget.timeBudgetMs ?? 0;
  const fixedSims = budget.simulations ?? 128;
  const maxSims = budget.maxSims && budget.maxSims > 0 ? budget.maxSims : Infinity;
  const start = performance.now();
  // Re-check the clock only every few sims to avoid performance.now() overhead.
  const TIME_CHECK_EVERY = 16;

  let s = 0;
  while (true) {
    if (timeBudgetMs > 0) {
      if (s >= maxSims) break;
      if (s % TIME_CHECK_EVERY === 0 && performance.now() - start >= timeBudgetMs) break;
    } else if (s >= fixedSims) {
      break;
    }

    let node = root;
    const path: Node[] = [node];

    while (node.children.size > 0 && node.terminalValue === null) {
      node = select(node);
      path.push(node);
    }

    const value = node.terminalValue !== null ? node.terminalValue : await expand(node, session);

    // Backup: value is from the perspective of the player to move at `node.state`;
    // the perspective flips at each step up the tree.
    let v = value;
    for (let i = path.length - 1; i >= 0; i--) {
      const n = path[i];
      n.N += 1;
      n.W += v;
      n.Q = n.W / n.N;
      v = -v;
    }
    s++;
  }

  return collect();
}
