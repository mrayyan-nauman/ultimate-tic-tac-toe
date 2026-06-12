import { describe, it, expect } from "vitest";
import {
  difficultyToConfig,
  clampRating,
  ratingTier,
  MIN_RATING,
  MAX_RATING,
} from "@/lib/ai/difficulty";

describe("difficultyToConfig", () => {
  it("rating 1 is fully random with no search", () => {
    const c = difficultyToConfig(1);
    expect(c.epsilon).toBe(1);
    expect(c.timeBudgetMs).toBe(0);
    expect(c.maxSims).toBe(0);
  });

  it("rating 2000 is argmax with the 60s budget", () => {
    const c = difficultyToConfig(2000);
    expect(c.epsilon).toBe(0);
    expect(c.temperature).toBe(0);
    expect(c.timeBudgetMs).toBe(60000);
    expect(c.maxSims).toBeGreaterThan(0);
  });

  it("only the very top (2000) is allowed past the 10s cap", () => {
    expect(difficultyToConfig(1999).timeBudgetMs).toBeLessThanOrEqual(10000);
    expect(difficultyToConfig(2000).timeBudgetMs).toBe(60000);
  });

  it("is monotonic in every knob across the full range", () => {
    let prevEps = Infinity;
    let prevTemp = Infinity;
    let prevBudget = -1;
    let prevSims = -1;
    for (let r = MIN_RATING; r <= MAX_RATING; r += 1) {
      const c = difficultyToConfig(r);
      // strength rises: epsilon and temperature fall, budget and sims rise
      expect(c.epsilon).toBeLessThanOrEqual(prevEps + 1e-9);
      expect(c.temperature).toBeLessThanOrEqual(prevTemp + 1e-9);
      expect(c.timeBudgetMs).toBeGreaterThanOrEqual(prevBudget);
      expect(c.maxSims).toBeGreaterThanOrEqual(prevSims);
      prevEps = c.epsilon;
      prevTemp = c.temperature;
      prevBudget = c.timeBudgetMs;
      prevSims = c.maxSims;
    }
  });

  it("produces valid configs everywhere", () => {
    for (let r = MIN_RATING; r <= MAX_RATING; r += 7) {
      const c = difficultyToConfig(r);
      expect(c.epsilon).toBeGreaterThanOrEqual(0);
      expect(c.epsilon).toBeLessThanOrEqual(1);
      expect(c.temperature).toBeGreaterThanOrEqual(0);
      expect(c.timeBudgetMs).toBeGreaterThanOrEqual(0);
      expect(c.maxSims).toBeGreaterThanOrEqual(0);
    }
  });
});

describe("clampRating", () => {
  it("clamps out-of-range and rounds", () => {
    expect(clampRating(-5)).toBe(MIN_RATING);
    expect(clampRating(99999)).toBe(MAX_RATING);
    expect(clampRating(1499.6)).toBe(1500);
  });
  it("falls back to max on non-finite input", () => {
    expect(clampRating(NaN)).toBe(MAX_RATING);
    expect(clampRating(Infinity)).toBe(MAX_RATING);
  });
});

describe("ratingTier", () => {
  it("labels the extremes", () => {
    expect(ratingTier(1)).toBe("Beginner");
    expect(ratingTier(2000)).toBe("Maximum");
  });
  it("is non-empty across the range", () => {
    for (let r = MIN_RATING; r <= MAX_RATING; r += 50) {
      expect(ratingTier(r).length).toBeGreaterThan(0);
    }
  });
});
