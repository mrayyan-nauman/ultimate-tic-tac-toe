import { useState, useCallback, useEffect, useMemo } from "react";
import { motion, AnimatePresence } from "framer-motion";
import confetti from "canvas-confetti";
import PlayerIcon from "@/components/PlayerIcon";
import UltimateTicTacToe from "@/components/UltimateTicTacToe";
import { checkWinner, createEmptyBoards, createEmptyWinners } from "@/lib/gameLogic";
import { requestMove, warmUpAI } from "@/lib/ai/aiClient";
import { difficultyToConfig, ratingTier } from "@/lib/ai/difficulty";
import { Button } from "@/components/ui/button";
import { RotateCcw, SlidersHorizontal } from "lucide-react";
import { useNavigate, useSearchParams, Navigate } from "react-router-dom";

type Mark = "X" | "O";

const markColor = (m: Mark) => (m === "X" ? "text-player-blue" : "text-player-red");
const markGlow = (m: Mark) => (m === "X" ? "text-glow-blue" : "text-glow-red");

/** Parses + validates the difficulty/side params, redirecting to setup if absent. */
const Game = () => {
  const [params] = useSearchParams();
  const levelRaw = Number(params.get("level"));
  const sideRaw = params.get("side");

  const validLevel = Number.isFinite(levelRaw) && levelRaw >= 1 && levelRaw <= 2000;
  const validSide = sideRaw === "X" || sideRaw === "O";
  if (!validLevel || !validSide) return <Navigate to="/play" replace />;

  const level = Math.round(levelRaw);
  // Remount fresh state whenever the chosen game parameters change.
  return <GameBoard key={`${level}-${sideRaw}`} level={level} humanPlayer={sideRaw as Mark} />;
};

const GameBoard = ({ level, humanPlayer }: { level: number; humanPlayer: Mark }) => {
  const navigate = useNavigate();
  const aiPlayer: Mark = humanPlayer === "X" ? "O" : "X";
  const config = useMemo(() => difficultyToConfig(level), [level]);
  const tier = ratingTier(level);

  const [boards, setBoards] = useState(createEmptyBoards);
  const [boardWinners, setBoardWinners] = useState(createEmptyWinners);
  const [activeBoard, setActiveBoard] = useState<number | null>(null);
  const [currentPlayer, setCurrentPlayer] = useState<Mark>("X"); // X always moves first
  const [gameWinner, setGameWinner] = useState<string | null>(null);
  const [isAIThinking, setIsAIThinking] = useState(false);

  const fireConfetti = useCallback(() => {
    const duration = 3000;
    const end = Date.now() + duration;
    const colors = gameWinner === "X" ? ["#3b82f6", "#60a5fa"] : ["#ef4444", "#f87171"];
    (function frame() {
      confetti({ particleCount: 3, angle: 60, spread: 55, origin: { x: 0 }, colors });
      confetti({ particleCount: 3, angle: 120, spread: 55, origin: { x: 1 }, colors });
      if (Date.now() < end) requestAnimationFrame(frame);
    })();
  }, [gameWinner]);

  useEffect(() => {
    if (gameWinner && gameWinner !== "D") fireConfetti();
  }, [gameWinner, fireConfetti]);

  // Warm up the net (download + WASM init in the worker) on mount.
  useEffect(() => {
    warmUpAI();
  }, []);

  const makeMove = (boardIndex: number, cellIndex: number, player: Mark) => {
    const newBoards = boards.map((b) => [...b]);
    newBoards[boardIndex][cellIndex] = player;

    const newBoardWinners = [...boardWinners];
    const bw = checkWinner(newBoards[boardIndex]);
    if (bw) newBoardWinners[boardIndex] = bw;

    const overallWinner = checkWinner(newBoardWinners);

    let nextActive: number | null = cellIndex;
    if (newBoardWinners[cellIndex]) nextActive = null;

    setBoards(newBoards);
    setBoardWinners(newBoardWinners);
    setActiveBoard(nextActive);

    if (overallWinner) {
      setGameWinner(overallWinner);
      return;
    }
    setCurrentPlayer(player === "X" ? "O" : "X");
  };

  const handleCellClick = useCallback(
    (boardIndex: number, cellIndex: number) => {
      if (gameWinner || currentPlayer !== humanPlayer || isAIThinking) return;
      makeMove(boardIndex, cellIndex, humanPlayer);
    },
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [gameWinner, currentPlayer, isAIThinking, boards, boardWinners, activeBoard, humanPlayer]
  );

  // AI turn — runs the AlphaZero AI in a Web Worker at the chosen difficulty.
  useEffect(() => {
    if (currentPlayer !== aiPlayer || gameWinner) return;

    setIsAIThinking(true);
    let cancelled = false;

    (async () => {
      const move = await requestMove({
        boards,
        boardWinners,
        activeBoard,
        player: aiPlayer,
        config,
      });
      if (cancelled) return;
      if (move) makeMove(move.boardIndex, move.cellIndex, aiPlayer);
      setIsAIThinking(false);
    })();

    return () => {
      cancelled = true;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [currentPlayer, gameWinner]);

  const resetGame = () => {
    setBoards(createEmptyBoards());
    setBoardWinners(createEmptyWinners());
    setActiveBoard(null);
    setCurrentPlayer("X");
    setGameWinner(null);
    setIsAIThinking(false);
  };

  const winnerLabel =
    gameWinner === humanPlayer ? "You" : gameWinner === aiPlayer ? "AI" : null;

  return (
    <div className="min-h-screen flex flex-col items-center justify-center p-4 relative">
      {/* Top center: difficulty badge */}
      <div className="absolute top-4 left-1/2 -translate-x-1/2 text-center">
        <div className="text-xs tracking-widest uppercase text-muted-foreground font-[var(--font-display)]">
          AI Rating
        </div>
        <div className="text-lg font-black tabular-nums text-foreground">
          {level} <span className="text-muted-foreground font-normal text-xs">· {tier}</span>
        </div>
      </div>

      {/* Top left: human */}
      <div className="absolute top-4 left-4">
        <PlayerIcon
          player={humanPlayer}
          isActive={currentPlayer === humanPlayer && !gameWinner}
          isWinner={gameWinner === humanPlayer}
        />
        <span className={`text-xs ${markColor(humanPlayer)} font-[var(--font-display)] mt-1 block text-center`}>
          YOU
        </span>
      </div>

      {/* Bottom right: AI */}
      <div className="absolute bottom-4 right-4">
        <PlayerIcon
          player={aiPlayer}
          isActive={currentPlayer === aiPlayer && !gameWinner}
          isWinner={gameWinner === aiPlayer}
        />
        <span className={`text-xs ${markColor(aiPlayer)} font-[var(--font-display)] mt-1 block text-center`}>
          AI
        </span>
      </div>

      {/* Turn indicator */}
      <div className="mb-4 mt-10">
        <p
          className={`text-sm font-[var(--font-display)] tracking-widest ${
            currentPlayer === humanPlayer ? `${markColor(humanPlayer)} ${markGlow(humanPlayer)}` : `${markColor(aiPlayer)} ${markGlow(aiPlayer)}`
          }`}
        >
          {gameWinner ? "" : isAIThinking ? "AI IS THINKING..." : currentPlayer === humanPlayer ? "YOUR TURN" : ""}
        </p>
      </div>

      {/* Game board */}
      <UltimateTicTacToe
        boards={boards}
        boardWinners={boardWinners}
        activeBoard={activeBoard}
        onCellClick={handleCellClick}
        disabled={!!gameWinner || isAIThinking || currentPlayer !== humanPlayer}
      />

      {/* Controls */}
      <div className="mt-6 flex gap-3">
        <Button variant="outline" size="sm" onClick={resetGame} className="gap-2 font-[var(--font-display)] text-xs">
          <RotateCcw size={14} />
          PLAY AGAIN
        </Button>
        <Button variant="outline" size="sm" onClick={() => navigate("/play")} className="gap-2 font-[var(--font-display)] text-xs">
          <SlidersHorizontal size={14} />
          DIFFICULTY
        </Button>
        <Button variant="ghost" size="sm" onClick={() => navigate("/")} className="font-[var(--font-display)] text-xs text-muted-foreground">
          EXIT
        </Button>
      </div>

      {/* Winner overlay */}
      <AnimatePresence>
        {gameWinner && gameWinner !== "D" && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="absolute inset-0 flex items-center justify-center bg-background/80 backdrop-blur-sm z-50"
          >
            <motion.div
              initial={{ scale: 0.5, opacity: 0 }}
              animate={{ scale: 1, opacity: 1 }}
              transition={{ type: "spring", stiffness: 200 }}
              className="text-center"
            >
              <motion.h2
                animate={{ opacity: [1, 0.3, 1] }}
                transition={{ duration: 1, repeat: Infinity }}
                className={`text-5xl md:text-7xl font-black tracking-tight ${
                  gameWinner === "X" ? "text-player-blue text-glow-blue" : "text-player-red text-glow-red"
                }`}
              >
                {winnerLabel} WON!
              </motion.h2>
              <div className="mt-8 flex gap-4 justify-center">
                <Button onClick={resetGame} className="font-[var(--font-display)] glow-blue">
                  PLAY AGAIN
                </Button>
                <Button variant="outline" onClick={() => navigate("/play")} className="font-[var(--font-display)]">
                  CHANGE DIFFICULTY
                </Button>
              </div>
            </motion.div>
          </motion.div>
        )}
        {gameWinner === "D" && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="absolute inset-0 flex items-center justify-center bg-background/80 backdrop-blur-sm z-50"
          >
            <motion.div initial={{ scale: 0.5 }} animate={{ scale: 1 }} className="text-center">
              <h2 className="text-5xl md:text-7xl font-black text-muted-foreground">DRAW</h2>
              <div className="mt-8 flex gap-4 justify-center">
                <Button onClick={resetGame} className="font-[var(--font-display)]">
                  PLAY AGAIN
                </Button>
                <Button variant="outline" onClick={() => navigate("/play")} className="font-[var(--font-display)]">
                  CHANGE DIFFICULTY
                </Button>
              </div>
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
};

export default Game;
