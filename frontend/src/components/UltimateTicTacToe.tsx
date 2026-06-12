import SmallBoard from "./SmallBoard";

interface UltimateTicTacToeProps {
  boards: (string | null)[][];
  boardWinners: (string | null)[];
  activeBoard: number | null;
  onCellClick: (boardIndex: number, cellIndex: number) => void;
  disabled?: boolean;
}

const UltimateTicTacToe = ({
  boards,
  boardWinners,
  activeBoard,
  onCellClick,
  disabled = false,
}: UltimateTicTacToeProps) => {
  return (
    <div className="grid grid-cols-3 gap-2 p-3 bg-board-bg rounded-2xl w-full max-w-[min(85vw,500px)] aspect-square">
      {boards.map((board, boardIdx) => {
        const isPlayable =
          !disabled &&
          !boardWinners[boardIdx] &&
          (activeBoard === null || activeBoard === boardIdx);
        return (
          <div key={boardIdx} className="aspect-square">
            <SmallBoard
              board={board}
              winner={boardWinners[boardIdx]}
              isPlayable={isPlayable}
              onCellClick={(cellIdx) => onCellClick(boardIdx, cellIdx)}
              boardIndex={boardIdx}
            />
          </div>
        );
      })}
    </div>
  );
};

export default UltimateTicTacToe;
