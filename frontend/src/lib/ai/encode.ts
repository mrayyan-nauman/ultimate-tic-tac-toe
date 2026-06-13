// State encoding — a faithful port of encode_state() in backend/net.py.
//
// 6 planes of 9x9 = 486 floats, flattened row-major as plane*81 + bi*9 + ci:
//   0: X stones
//   1: O stones
//   2: sub-board won by X (broadcast across its 9 cells)
//   3: sub-board won by O (broadcast across its 9 cells)
//   4: legal-cell mask (1 where the side to move may play)
//   5: side-to-move (all 1s if current player is X, else 0)

import { State, Move, Cell, legalMoves, WIN_LINES } from "./engine";

export const NUM_PLANES = 6;
export const INPUT_DIM = NUM_PLANES * 81; // 486
export const NUM_ACTIONS = 81;

// v2 encoding adds 5 tactical planes (total 11) — see backend/net.encode_state_v2.
export const NUM_PLANES_V2 = 11;
export const INPUT_DIM_V2 = NUM_PLANES_V2 * 81; // 891

export function encodeState(state: State): Float32Array {
  const planes = new Float32Array(INPUT_DIM);
  const at = (plane: number, bi: number, ci: number) => plane * 81 + bi * 9 + ci;

  for (let bi = 0; bi < 9; bi++) {
    for (let ci = 0; ci < 9; ci++) {
      const v = state.boards[bi][ci];
      if (v === "X") planes[at(0, bi, ci)] = 1;
      else if (v === "O") planes[at(1, bi, ci)] = 1;
    }
  }

  for (let bi = 0; bi < 9; bi++) {
    if (state.winners[bi] === "X") for (let ci = 0; ci < 9; ci++) planes[at(2, bi, ci)] = 1;
    else if (state.winners[bi] === "O") for (let ci = 0; ci < 9; ci++) planes[at(3, bi, ci)] = 1;
  }

  for (const [bi, ci] of legalMoves(state)) planes[at(4, bi, ci)] = 1;

  if (state.player === "X") {
    for (let i = 0; i < 81; i++) planes[at(5, 0, 0) + i] = 1;
  }

  return planes;
}

/** Empty-cell indices where placing `mark` completes a small-board line. */
function winningCells(cells: Cell[], mark: "X" | "O"): number[] {
  const res: number[] = [];
  for (const [a, b, c] of WIN_LINES) {
    const trio = [cells[a], cells[b], cells[c]];
    const marks = trio.filter((x) => x === mark).length;
    const empties = trio.filter((x) => x === null).length;
    if (marks === 2 && empties === 1) {
      for (const idx of [a, b, c]) if (cells[idx] === null && !res.includes(idx)) res.push(idx);
    }
  }
  return res;
}

/** 11-plane encoding — a faithful port of encode_state_v2() in backend/net.py. */
export function encodeStateV2(state: State): Float32Array {
  const planes = new Float32Array(INPUT_DIM_V2);
  const at = (plane: number, bi: number, ci: number) => plane * 81 + bi * 9 + ci;
  const { boards, winners, player } = state;

  // Base planes 0-5.
  for (let bi = 0; bi < 9; bi++) {
    for (let ci = 0; ci < 9; ci++) {
      const v = boards[bi][ci];
      if (v === "X") planes[at(0, bi, ci)] = 1;
      else if (v === "O") planes[at(1, bi, ci)] = 1;
    }
  }
  for (let bi = 0; bi < 9; bi++) {
    if (winners[bi] === "X") for (let ci = 0; ci < 9; ci++) planes[at(2, bi, ci)] = 1;
    else if (winners[bi] === "O") for (let ci = 0; ci < 9; ci++) planes[at(3, bi, ci)] = 1;
  }
  for (const [bi, ci] of legalMoves(state)) planes[at(4, bi, ci)] = 1;
  if (player === "X") for (let i = 0; i < 81; i++) planes[at(5, 0, 0) + i] = 1;

  // 6: free-move cells (column ci where board ci is decided).
  for (let ci = 0; ci < 9; ci++) {
    if (winners[ci] !== null) for (let bi = 0; bi < 9; bi++) planes[at(6, bi, ci)] = 1;
  }

  // 7/8: immediate sub-board-winning cells for X / O (undecided boards).
  for (let bi = 0; bi < 9; bi++) {
    if (winners[bi] === null) {
      for (const ci of winningCells(boards[bi], "X")) planes[at(7, bi, ci)] = 1;
      for (const ci of winningCells(boards[bi], "O")) planes[at(8, bi, ci)] = 1;
    }
  }

  // 9/10: macro-threat boards for X / O (draws are neutral).
  const macro = winners.map((w) => (w === "X" || w === "O" ? w : null));
  for (let bi = 0; bi < 9; bi++) {
    if (winners[bi] === null) {
      for (const [a, b, c] of WIN_LINES) {
        if (a === bi || b === bi || c === bi) {
          const others = [a, b, c].filter((x) => x !== bi);
          const [o1, o2] = others;
          if (macro[o1] === "X" && macro[o2] === "X") for (let ci = 0; ci < 9; ci++) planes[at(9, bi, ci)] = 1;
          if (macro[o1] === "O" && macro[o2] === "O") for (let ci = 0; ci < 9; ci++) planes[at(10, bi, ci)] = 1;
        }
      }
    }
  }

  return planes;
}

export function moveToIndex(move: Move): number {
  return move[0] * 9 + move[1];
}

export function indexToMove(idx: number): Move {
  return [Math.floor(idx / 9), idx % 9];
}
