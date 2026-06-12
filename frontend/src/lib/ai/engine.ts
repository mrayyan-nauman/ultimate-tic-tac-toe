// Pure rules engine for Ultimate Tic-Tac-Toe — a faithful port of backend/engine.py.
// No I/O, no randomness. Used by the in-browser MCTS so the web AI matches the
// trained PyTorch/MCTS agent exactly.

export type Cell = "X" | "O" | null;
export type SmallWinner = "X" | "O" | "D" | null;
export type Player = "X" | "O";
export type Move = [number, number]; // [boardIndex, cellIndex]

export interface State {
  boards: Cell[][]; // 9 sub-boards of 9 cells
  winners: SmallWinner[]; // winner of each sub-board
  active: number | null; // forced sub-board, or null = play anywhere
  player: Player; // side to move
}

export const WIN_LINES: ReadonlyArray<readonly [number, number, number]> = [
  [0, 1, 2], [3, 4, 5], [6, 7, 8],
  [0, 3, 6], [1, 4, 7], [2, 5, 8],
  [0, 4, 8], [2, 4, 6],
];

/** Winner of a single 9-cell board: 'X', 'O', 'D' (draw), or null. */
export function checkSmall(board: Cell[]): SmallWinner {
  for (const [a, b, c] of WIN_LINES) {
    if (board[a] !== null && board[a] === board[b] && board[a] === board[c]) {
      return board[a] as SmallWinner;
    }
  }
  if (board.every((c) => c !== null)) return "D";
  return null;
}

export function legalMoves(state: State): Move[] {
  const { boards, winners, active } = state;
  const playable =
    active !== null && winners[active] === null
      ? [active]
      : boards.map((_, i) => i).filter((i) => winners[i] === null);

  const moves: Move[] = [];
  for (const bi of playable) {
    for (let ci = 0; ci < 9; ci++) {
      if (boards[bi][ci] === null) moves.push([bi, ci]);
    }
  }
  return moves;
}

/** Apply a move immutably, returning a fresh state. */
export function applyMove(state: State, move: Move): State {
  const [bi, ci] = move;
  const { player } = state;

  const newBoards = state.boards.map((b) => b.slice());
  newBoards[bi][ci] = player;

  const newWinners = state.winners.slice();
  if (newWinners[bi] === null) newWinners[bi] = checkSmall(newBoards[bi]);

  // Opponent must play in sub-board `ci`, unless it's already decided.
  const nextActive = newWinners[ci] === null ? ci : null;
  const nextPlayer: Player = player === "X" ? "O" : "X";

  return { boards: newBoards, winners: newWinners, active: nextActive, player: nextPlayer };
}

/** Overall game winner: 'X', 'O', 'D', or null. Sub-board draws are neutral. */
export function getWinner(state: State): SmallWinner {
  const macro = state.winners.map((w) => (w === "X" || w === "O" ? w : null));
  for (const [a, b, c] of WIN_LINES) {
    if (macro[a] !== null && macro[a] === macro[b] && macro[a] === macro[c]) {
      return macro[a] as SmallWinner;
    }
  }
  if (state.winners.every((w) => w !== null)) return "D";
  return null;
}

export function moveKey(move: Move): string {
  return `${move[0]},${move[1]}`;
}
