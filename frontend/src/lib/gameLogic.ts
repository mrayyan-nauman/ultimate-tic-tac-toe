// Local game logic shared by the board UI. The neural-net AI (and its own random
// fallback) lives in `@/lib/ai`.

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

export function createEmptyBoards(): (string | null)[][] {
  return Array.from({ length: 9 }, () => Array(9).fill(null));
}

export function createEmptyWinners(): (string | null)[] {
  return Array(9).fill(null);
}
