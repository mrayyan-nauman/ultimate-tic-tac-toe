// Maps a difficulty rating (1..2000) to concrete AI strength knobs.
//
// The rating is a "compressed chess rating": higher = harder. It is anchored to
// the *currently deployed* network — rating 2000 always means "this net at full
// search", so retraining + redeploying a stronger net automatically re-anchors
// the top of the scale (and everything below shifts with it).
//
// Three knobs, combined, span the full range from "absolutely random" to "max":
//   epsilon      probability of a uniformly random move (true randomness at the
//                bottom; the only knob that can make play *worse than the net*).
//   temperature  softness of move selection over the policy / visit counts
//                (high = varied & weak, 0 = argmax = sharpest).
//   timeBudgetMs MCTS thinking budget (0 = policy-only, no tree search), capped
//                at MAX_THINK_MS (15 s) for every rating.
//   maxSims      hard safety cap on simulations regardless of time.

export interface AIConfig {
  /** Probability [0,1] of playing a uniformly random legal move. */
  epsilon: number;
  /** Move-selection temperature; 0 = argmax (strongest). */
  temperature: number;
  /** MCTS time budget in ms; 0 = no search (use the raw policy). */
  timeBudgetMs: number;
  /** Hard cap on MCTS simulations (memory / runaway guard). */
  maxSims: number;
}

export const MIN_RATING = 1;
export const MAX_RATING = 2000;

// Anchor points, interpolated linearly between. Tuned by hand so that strength is
// monotonic in the rating and the progression *feels* like a chess ladder.
// (rating, epsilon, temperature, timeBudgetMs, maxSims)
type Anchor = [number, number, number, number, number];
const ANCHORS: Anchor[] = [
  [1,    1.00, 1.50, 0,      0],      // pure random (temperature moot at epsilon=1)
  [100,  0.85, 1.50, 0,      0],
  [300,  0.60, 1.25, 0,      0],
  [500,  0.40, 1.00, 0,      0],      // policy-only, fairly loose
  [700,  0.22, 0.80, 150,    64],     // a little search appears
  [900,  0.10, 0.60, 350,    160],
  [1100, 0.04, 0.45, 700,    400],    // epsilon essentially gone
  [1300, 0.00, 0.30, 1200,   900],
  [1500, 0.00, 0.18, 2200,   2500],
  [1700, 0.00, 0.08, 4000,   6000],   // approaching argmax
  [1850, 0.00, 0.00, 6500,   15000],  // argmax from here up
  [1999, 0.00, 0.00, 12000,  40000],
  [2000, 0.00, 0.00, 15000,  120000], // max think
];

// Hard cap on AI think time at every rating (ms).
export const MAX_THINK_MS = 15000;

function lerp(a: number, b: number, t: number): number {
  return a + (b - a) * t;
}

/** Clamp+round a rating to a valid integer in [1, 2000]. */
export function clampRating(rating: number): number {
  if (!Number.isFinite(rating)) return MAX_RATING;
  return Math.min(MAX_RATING, Math.max(MIN_RATING, Math.round(rating)));
}

export function difficultyToConfig(rating: number): AIConfig {
  const r = clampRating(rating);

  // Find the bracketing anchors.
  let lo = ANCHORS[0];
  let hi = ANCHORS[ANCHORS.length - 1];
  for (let i = 0; i < ANCHORS.length - 1; i++) {
    if (r >= ANCHORS[i][0] && r <= ANCHORS[i + 1][0]) {
      lo = ANCHORS[i];
      hi = ANCHORS[i + 1];
      break;
    }
  }

  const span = hi[0] - lo[0];
  const t = span === 0 ? 0 : (r - lo[0]) / span;

  return {
    epsilon: lerp(lo[1], hi[1], t),
    temperature: lerp(lo[2], hi[2], t),
    timeBudgetMs: Math.min(MAX_THINK_MS, Math.round(lerp(lo[3], hi[3], t))),
    maxSims: Math.round(lerp(lo[4], hi[4], t)),
  };
}

export interface Tier {
  name: string;
  /** Inclusive lower bound of the tier. */
  min: number;
}

const TIERS: Tier[] = [
  { name: "Beginner", min: 1 },
  { name: "Casual", min: 400 },
  { name: "Intermediate", min: 900 },
  { name: "Skilled", min: 1400 },
  { name: "Expert", min: 1800 },
  { name: "Maximum", min: 2000 },
];

/** Human-readable tier label for a rating (e.g. "Skilled"). */
export function ratingTier(rating: number): string {
  const r = clampRating(rating);
  let label = TIERS[0].name;
  for (const tier of TIERS) {
    if (r >= tier.min) label = tier.name;
  }
  return label;
}
