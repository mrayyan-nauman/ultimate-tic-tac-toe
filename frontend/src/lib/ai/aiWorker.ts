// Runs the AlphaZero AI off the main thread. onnxruntime-web loads and runs here,
// so even a 60-second "think" at max difficulty never freezes the UI.
//
// We cast `self` to a minimal worker interface rather than pulling in the
// WebWorker TS lib, which would clash with the app's DOM lib.

import { getAIMove, warmUpAI } from "./index";
import type { AIConfig } from "./difficulty";

type WorkerRequest =
  | { type: "warmup" }
  | {
      type: "move";
      id: number;
      boards: (string | null)[][];
      boardWinners: (string | null)[];
      activeBoard: number | null;
      player: "X" | "O";
      config: AIConfig;
    };

const ctx = self as unknown as {
  onmessage: ((e: MessageEvent<WorkerRequest>) => void) | null;
  postMessage: (message: unknown) => void;
};

ctx.onmessage = async (e: MessageEvent<WorkerRequest>) => {
  const msg = e.data;
  if (msg.type === "warmup") {
    warmUpAI();
    return;
  }
  if (msg.type === "move") {
    try {
      const move = await getAIMove(
        msg.boards,
        msg.boardWinners,
        msg.activeBoard,
        msg.player,
        msg.config,
      );
      ctx.postMessage({ type: "move", id: msg.id, move });
    } catch (err) {
      ctx.postMessage({ type: "move", id: msg.id, move: null, error: String(err) });
    }
  }
};
