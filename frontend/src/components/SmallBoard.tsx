import { motion } from "framer-motion";

interface SmallBoardProps {
  board: (string | null)[];
  winner: string | null;
  isPlayable: boolean;
  onCellClick: (cellIndex: number) => void;
  boardIndex: number;
}

const SmallBoard = ({ board, winner, isPlayable, onCellClick, boardIndex }: SmallBoardProps) => {
  if (winner) {
    return (
      <motion.div
        initial={{ scale: 0.8, opacity: 0 }}
        animate={{ scale: 1, opacity: 1 }}
        className={`w-full h-full rounded-lg flex items-center justify-center text-4xl font-bold font-[var(--font-display)] ${
          winner === "X"
            ? "bg-player-blue/20 text-player-blue text-glow-blue"
            : winner === "O"
            ? "bg-player-red/20 text-player-red text-glow-red"
            : "bg-muted/30 text-muted-foreground"
        }`}
      >
        {winner === "D" ? "—" : winner}
      </motion.div>
    );
  }

  return (
    <div
      className={`grid grid-cols-3 gap-0.5 w-full h-full rounded-lg p-0.5 transition-all duration-300 ${
        isPlayable ? "bg-cell-active ring-1 ring-primary/40" : "bg-board-bg"
      }`}
    >
      {board.map((cell, i) => (
        <motion.button
          key={`${boardIndex}-${i}`}
          whileHover={isPlayable && !cell ? { scale: 1.1 } : {}}
          whileTap={isPlayable && !cell ? { scale: 0.95 } : {}}
          onClick={() => onCellClick(i)}
          disabled={!isPlayable || !!cell}
          className={`aspect-square rounded-sm flex items-center justify-center text-sm font-bold font-[var(--font-display)] transition-colors duration-200 ${
            isPlayable && !cell
              ? "bg-cell-bg hover:bg-cell-hover cursor-pointer"
              : "bg-cell-bg cursor-default"
          } ${
            cell === "X"
              ? "text-player-blue"
              : cell === "O"
              ? "text-player-red"
              : "text-transparent"
          }`}
        >
          {cell || "·"}
        </motion.button>
      ))}
    </div>
  );
};

export default SmallBoard;
