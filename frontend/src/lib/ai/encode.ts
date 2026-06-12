// State encoding — a faithful port of encode_state() in backend/net.py.
//
// 6 planes of 9x9 = 486 floats, flattened row-major as plane*81 + bi*9 + ci:
//   0: X stones
//   1: O stones
//   2: sub-board won by X (broadcast across its 9 cells)
//   3: sub-board won by O (broadcast across its 9 cells)
//   4: legal-cell mask (1 where the side to move may play)
//   5: side-to-move (all 1s if current player is X, else 0)

import { State, Move, legalMoves } from "./engine";

export const NUM_PLANES = 6;
export const INPUT_DIM = NUM_PLANES * 81; // 486
export const NUM_ACTIONS = 81;

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

export function moveToIndex(move: Move): number {
  return move[0] * 9 + move[1];
}

export function indexToMove(idx: number): Move {
  return [Math.floor(idx / 9), idx % 9];
}
