// Local game logic for standalone play. The neural-net AI lives in `@/lib/ai`;
// getRandomAIMove below is only the fallback when that model can't load.

const WIN_LINES = [
  [0, 1, 2], [3, 4, 5], [6, 7, 8],
  [0, 3, 6], [1, 4, 7], [2, 5, 8],
  [0, 4, 8], [2, 4, 6],
];

export function checkWinner(board: (string | null)[]): string | null {
  for (const [a, b, c] of WIN_LINES) {
    if (board[a] && board[a] === board[b] && board[a] === board[c]) {
      return board[a];
    }
  }
  if (board.every((c) => c !== null)) return "D";
  return null;
}

/** Random legal move — local fallback when the neural-net AI is unavailable. */
export function getRandomAIMove(
  boards: (string | null)[][],
  boardWinners: (string | null)[],
  activeBoard: number | null
): { boardIndex: number; cellIndex: number } | null {
  const playableBoards = activeBoard !== null
    ? [activeBoard]
    : boards.map((_, i) => i).filter((i) => !boardWinners[i]);

  const validMoves: { boardIndex: number; cellIndex: number }[] = [];
  for (const bi of playableBoards) {
    for (let ci = 0; ci < 9; ci++) {
      if (!boards[bi][ci]) {
        validMoves.push({ boardIndex: bi, cellIndex: ci });
      }
    }
  }

  if (validMoves.length === 0) return null;
  return validMoves[Math.floor(Math.random() * validMoves.length)];
}

export function createEmptyBoards(): (string | null)[][] {
  return Array.from({ length: 9 }, () => Array(9).fill(null));
}

export function createEmptyWinners(): (string | null)[] {
  return Array(9).fill(null);
}
