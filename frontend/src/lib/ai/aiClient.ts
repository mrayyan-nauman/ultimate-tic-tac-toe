// Main-thread client for the AI Web Worker.
//
// `requestMove` always resolves to a move (or null if the game is over) — it
// degrades gracefully to main-thread compute if the Worker can't be created,
// errors, or exceeds a generous timeout. So callers never need to handle a
// rejected promise.

import type { AIConfig } from "./difficulty";
import type { MoveResult } from "./index";

export interface MoveRequest {
  boards: (string | null)[][];
  boardWinners: (string | null)[];
  activeBoard: number | null;
  player: "X" | "O";
  config: AIConfig;
}

interface Pending {
  req: MoveRequest;
  resolve: (m: MoveResult | null) => void;
  timer: ReturnType<typeof setTimeout>;
}

let worker: Worker | null = null;
let workerBroken = false;
let nextId = 1;
const pending = new Map<number, Pending>();

/** Compute a move on the main thread (fallback path). Never rejects. */
function mainThreadMove(req: MoveRequest): Promise<MoveResult | null> {
  return import("./index")
    .then((m) => m.getAIMove(req.boards, req.boardWinners, req.activeBoard, req.player, req.config))
    .catch(() => null);
}

function getWorker(): Worker | null {
  if (workerBroken) return null;
  if (worker) return worker;
  try {
    worker = new Worker(new URL("./aiWorker.ts", import.meta.url), { type: "module" });
    worker.onmessage = (e: MessageEvent) => {
      const data = e.data as { type?: string; id?: number; move?: MoveResult | null };
      if (data?.type === "move" && data.id !== undefined) {
        const p = pending.get(data.id);
        if (p) {
          clearTimeout(p.timer);
          pending.delete(data.id);
          p.resolve(data.move ?? null);
        }
      }
    };
    worker.onerror = () => {
      // Worker died — re-run everything in flight on the main thread.
      workerBroken = true;
      worker = null;
      const inflight = [...pending.values()];
      pending.clear();
      for (const p of inflight) {
        clearTimeout(p.timer);
        mainThreadMove(p.req).then(p.resolve);
      }
    };
    return worker;
  } catch {
    workerBroken = true;
    return null;
  }
}

/** Preload the model (in the Worker if possible). Fire-and-forget. */
export function warmUpAI(): void {
  const w = getWorker();
  if (w) {
    w.postMessage({ type: "warmup" });
    return;
  }
  import("./index").then((m) => m.warmUpAI()).catch(() => {});
}

/** Request the AI's move. Always resolves (move or null). */
export function requestMove(req: MoveRequest): Promise<MoveResult | null> {
  const w = getWorker();
  if (!w) return mainThreadMove(req);

  const id = nextId++;
  const timeoutMs = Math.max(15000, (req.config.timeBudgetMs || 0) + 20000);

  return new Promise<MoveResult | null>((resolve) => {
    const timer = setTimeout(() => {
      pending.delete(id);
      mainThreadMove(req).then(resolve);
    }, timeoutMs);
    pending.set(id, { req, resolve, timer });
    w.postMessage({ type: "move", id, ...req });
  });
}
