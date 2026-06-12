import { useState } from "react";
import { motion } from "framer-motion";
import { useNavigate } from "react-router-dom";
import { Slider } from "@/components/ui/slider";
import { Button } from "@/components/ui/button";
import { MIN_RATING, MAX_RATING, ratingTier, clampRating } from "@/lib/ai/difficulty";

const STORAGE_KEY = "uttt:setup";

type Side = "X" | "O";

function loadSaved(): { rating: number; side: Side } {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (raw) {
      const v = JSON.parse(raw);
      return {
        rating: clampRating(Number(v.rating) || 1500),
        side: v.side === "O" ? "O" : "X",
      };
    }
  } catch {
    /* ignore */
  }
  return { rating: 1500, side: "X" };
}

const Setup = () => {
  const navigate = useNavigate();
  const saved = loadSaved();
  const [rating, setRating] = useState<number>(saved.rating);
  const [side, setSide] = useState<Side>(saved.side);

  const tier = ratingTier(rating);

  const start = () => {
    const r = clampRating(rating);
    try {
      localStorage.setItem(STORAGE_KEY, JSON.stringify({ rating: r, side }));
    } catch {
      /* ignore */
    }
    navigate(`/game?level=${r}&side=${side}`);
  };

  return (
    <div className="min-h-screen flex flex-col items-center justify-center p-4">
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5 }}
        className="w-full max-w-md flex flex-col gap-8"
      >
        <div className="text-center">
          <h1 className="text-2xl md:text-3xl font-black tracking-tight font-[var(--font-display)]">
            CHOOSE YOUR OPPONENT
          </h1>
          <p className="text-muted-foreground text-sm mt-2">
            Set the AI's rating — higher is harder.
          </p>
        </div>

        {/* Rating display */}
        <div className="text-center">
          <div className="text-6xl md:text-7xl font-black text-player-blue text-glow-blue tabular-nums">
            {rating}
          </div>
          <div className="mt-1 text-sm tracking-widest uppercase text-muted-foreground font-[var(--font-display)]">
            {tier}
          </div>
        </div>

        {/* Slider + exact input */}
        <div className="flex flex-col gap-4">
          <Slider
            value={[rating]}
            min={MIN_RATING}
            max={MAX_RATING}
            step={1}
            onValueChange={(v) => setRating(v[0])}
            aria-label="AI difficulty rating"
          />
          <div className="flex items-center justify-between text-xs text-muted-foreground">
            <span>{MIN_RATING} · random</span>
            <div className="flex items-center gap-2">
              <input
                type="number"
                min={MIN_RATING}
                max={MAX_RATING}
                value={rating}
                onChange={(e) => setRating(clampRating(Number(e.target.value)))}
                className="w-20 rounded-md border border-input bg-background px-2 py-1 text-center text-sm tabular-nums focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
                aria-label="Exact rating"
              />
            </div>
            <span>{MAX_RATING} · max</span>
          </div>
        </div>

        {/* Side selection */}
        <div className="flex flex-col gap-2">
          <span className="text-xs uppercase tracking-widest text-muted-foreground text-center font-[var(--font-display)]">
            You play as
          </span>
          <div className="grid grid-cols-2 gap-3">
            <button
              onClick={() => setSide("X")}
              className={`rounded-xl border-2 py-4 font-bold text-lg transition-all font-[var(--font-display)] ${
                side === "X"
                  ? "border-player-blue text-player-blue glow-blue"
                  : "border-border text-muted-foreground hover:border-player-blue/50"
              }`}
            >
              X
              <span className="block text-[10px] font-normal tracking-wider mt-1">FIRST MOVE</span>
            </button>
            <button
              onClick={() => setSide("O")}
              className={`rounded-xl border-2 py-4 font-bold text-lg transition-all font-[var(--font-display)] ${
                side === "O"
                  ? "border-player-red text-player-red glow-red"
                  : "border-border text-muted-foreground hover:border-player-red/50"
              }`}
            >
              O
              <span className="block text-[10px] font-normal tracking-wider mt-1">AI MOVES FIRST</span>
            </button>
          </div>
        </div>

        <div className="flex gap-3">
          <Button
            variant="ghost"
            onClick={() => navigate("/")}
            className="font-[var(--font-display)] text-xs text-muted-foreground"
          >
            BACK
          </Button>
          <Button
            onClick={start}
            className="flex-1 font-[var(--font-display)] tracking-wider glow-blue"
          >
            START GAME
          </Button>
        </div>
      </motion.div>
    </div>
  );
};

export default Setup;
