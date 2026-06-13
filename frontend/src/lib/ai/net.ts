// In-browser neural-net inference via onnxruntime-web.
// Loads the exported AlphaZero net (public/az_net.onnx) once and runs the
// masked-softmax policy + value head, mirroring predict() in backend/net.py.

import * as ort from "onnxruntime-web";
import { State, Move, legalMoves } from "./engine";
import { encodeStateV2, moveToIndex, INPUT_DIM_V2 } from "./encode";

// Serve the WASM runtime from the CDN build that matches the installed package
// version. Single-threaded so we don't need SharedArrayBuffer / COOP-COEP headers.
ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.1/dist/";
ort.env.wasm.numThreads = 1;

const MODEL_URL = `${import.meta.env.BASE_URL}az_net.onnx`;

let sessionPromise: Promise<ort.InferenceSession> | null = null;

/** Load (and cache) the ONNX session. Safe to call repeatedly. */
export function loadModel(): Promise<ort.InferenceSession> {
  if (!sessionPromise) {
    sessionPromise = ort.InferenceSession.create(MODEL_URL, {
      executionProviders: ["wasm"],
      graphOptimizationLevel: "all",
    });
  }
  return sessionPromise;
}

export interface Prediction {
  moves: Move[];
  /** Softmax probabilities over `moves` (same order). */
  probs: number[];
  /** Value in [-1, 1] from the side-to-move's perspective. */
  value: number;
}

export async function predict(
  session: ort.InferenceSession,
  state: State,
): Promise<Prediction> {
  const input = new ort.Tensor("float32", encodeStateV2(state), [1, INPUT_DIM_V2]);
  const out = await session.run({ input });

  const logits = out.policy.data as Float32Array; // 81 raw logits
  const value = (out.value.data as Float32Array)[0];

  const moves = legalMoves(state);
  if (moves.length === 0) return { moves, probs: [], value };

  // Masked softmax over legal moves only (numerically stable).
  const masked = moves.map((m) => logits[moveToIndex(m)]);
  const max = Math.max(...masked);
  const exp = masked.map((v) => Math.exp(v - max));
  const sum = exp.reduce((a, b) => a + b, 0);
  const probs = exp.map((v) => v / sum);

  return { moves, probs, value };
}
