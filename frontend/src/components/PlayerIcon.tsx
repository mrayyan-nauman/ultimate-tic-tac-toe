import { motion } from "framer-motion";
import { User, Bot } from "lucide-react";

interface PlayerIconProps {
  player: "X" | "O";
  isActive: boolean;
  isWinner?: boolean;
  size?: "sm" | "lg";
}

const PlayerIcon = ({ player, isActive, isWinner = false, size = "sm" }: PlayerIconProps) => {
  const isBlue = player === "X";
  const Icon = isBlue ? User : Bot;
  const sizeClass = size === "lg" ? "w-20 h-20" : "w-14 h-14";
  const iconSize = size === "lg" ? 36 : 24;

  return (
    <motion.div
      className={`${sizeClass} rounded-xl flex items-center justify-center border-2 transition-all duration-300 ${
        isBlue
          ? `border-player-blue bg-player-blue/10 ${isActive ? "glow-blue" : ""}`
          : `border-player-red bg-player-red/10 ${isActive ? "glow-red" : ""}`
      }`}
      animate={
        isWinner
          ? { scale: [1, 1.15, 1], opacity: [1, 0.3, 1] }
          : isActive
          ? { scale: [1, 1.05, 1] }
          : { scale: 1, opacity: 0.4 }
      }
      transition={
        isWinner
          ? { duration: 0.8, repeat: Infinity }
          : isActive
          ? { duration: 1.5, repeat: Infinity }
          : { duration: 0.3 }
      }
    >
      <Icon
        size={iconSize}
        className={isBlue ? "text-player-blue" : "text-player-red"}
      />
    </motion.div>
  );
};

export default PlayerIcon;
