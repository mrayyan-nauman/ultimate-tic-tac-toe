import { motion } from "framer-motion";
import { useNavigate } from "react-router-dom";
import { Gamepad2 } from "lucide-react";
import { Button } from "@/components/ui/button";

const Landing = () => {
  const navigate = useNavigate();

  return (
    <div className="min-h-screen flex flex-col items-center justify-center relative overflow-hidden">
      {/* Background grid effect */}
      <div className="absolute inset-0 opacity-5">
        <div className="grid grid-cols-3 h-full">
          {Array.from({ length: 9 }).map((_, i) => (
            <div key={i} className="border border-foreground/20 grid grid-cols-3">
              {Array.from({ length: 9 }).map((_, j) => (
                <div key={j} className="border border-foreground/10" />
              ))}
            </div>
          ))}
        </div>
      </div>

      <motion.div
        initial={{ opacity: 0, y: 30 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.8 }}
        className="relative z-10 flex flex-col items-center gap-8 px-4"
      >
        <motion.div
          initial={{ scale: 0 }}
          animate={{ scale: 1 }}
          transition={{ delay: 0.2, type: "spring", stiffness: 200 }}
          className="w-20 h-20 rounded-2xl bg-primary/10 border-2 border-primary flex items-center justify-center glow-blue"
        >
          <Gamepad2 size={40} className="text-primary" />
        </motion.div>

        <div className="text-center space-y-3">
          <h1 className="text-4xl md:text-6xl font-black tracking-tight">
            <span className="text-player-blue text-glow-blue">ULTIMATE</span>
            <br />
            <span className="text-foreground">TIC TAC TOE</span>
          </h1>
          <p className="text-muted-foreground text-lg max-w-md">
            Challenge the AI in the ultimate battle of strategy
          </p>
        </div>

        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.6 }}
        >
          <Button
            size="lg"
            onClick={() => navigate("/play")}
            className="text-lg px-10 py-6 bg-primary hover:bg-primary/90 text-primary-foreground glow-blue font-[var(--font-display)] tracking-wider"
          >
            START GAME
          </Button>
        </motion.div>

        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.9 }}
          className="flex gap-6 text-sm text-muted-foreground"
        >
          <span className="flex items-center gap-2">
            <div className="w-3 h-3 rounded-full bg-player-blue" />
            You
          </span>
          <span className="text-muted-foreground/40">vs</span>
          <span className="flex items-center gap-2">
            <div className="w-3 h-3 rounded-full bg-player-red" />
            AI
          </span>
        </motion.div>
      </motion.div>
    </div>
  );
};

export default Landing;
