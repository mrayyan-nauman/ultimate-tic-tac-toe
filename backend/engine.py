"""Pure rules engine for Ultimate Tic-Tac-Toe. No I/O, no randomness."""

WIN_LINES = [
    (0, 1, 2), (3, 4, 5), (6, 7, 8),
    (0, 3, 6), (1, 4, 7), (2, 5, 8),
    (0, 4, 8), (2, 4, 6),
]

# A state is a tuple:
#   (boards, board_winners, active_board, current_player)
# where:
#   boards         : tuple of 9 tuples of 9 cells. Each cell is "X", "O", or None.
#   board_winners  : tuple of 9 values. Each is "X", "O", "D", or None.
#   active_board   : int 0..8 or None (None = play anywhere).
#   current_player : "X" or "O".


def initial_state():
    boards = tuple(tuple([None] * 9) for _ in range(9))
    winners = tuple([None] * 9)
    return (boards, winners, None, "X")


def _check_small(board):
    """Return 'X', 'O', 'D', or None for a 9-cell board."""
    for a, b, c in WIN_LINES:
        if board[a] is not None and board[a] == board[b] == board[c]:
            return board[a]
    if all(c is not None for c in board):
        return "D"
    return None


def legal_moves(state):
    boards, winners, active, _ = state
    if active is not None and winners[active] is None:
        playable = [active]
    else:
        playable = [i for i in range(9) if winners[i] is None]
    moves = []
    for bi in playable:
        for ci in range(9):
            if boards[bi][ci] is None:
                moves.append((bi, ci))
    return moves


def apply_move(state, move):
    boards, winners, _active, player = state
    bi, ci = move

    # Update target sub-board (immutably).
    new_cells = list(boards[bi])
    new_cells[ci] = player
    new_board = tuple(new_cells)
    new_boards = boards[:bi] + (new_board,) + boards[bi + 1:]

    # Re-evaluate winner of the changed sub-board.
    new_winners = list(winners)
    if new_winners[bi] is None:
        new_winners[bi] = _check_small(new_board)
    new_winners = tuple(new_winners)

    # Active board for opponent = ci, unless that board is already decided.
    next_active = ci if new_winners[ci] is None else None

    next_player = "O" if player == "X" else "X"
    return (new_boards, new_winners, next_active, next_player)


def winner(state):
    """Return overall winner: 'X', 'O', 'D', or None."""
    _boards, winners, _active, _player = state
    # Treat draws on sub-boards as neutral when checking macro lines.
    macro = [w if w in ("X", "O") else None for w in winners]
    for a, b, c in WIN_LINES:
        if macro[a] is not None and macro[a] == macro[b] == macro[c]:
            return macro[a]
    if all(w is not None for w in winners):
        return "D"
    return None


_CELL_CHAR = {None: ".", "X": "X", "O": "O"}


def state_key(state):
    """Compact hashable string for use as a Q-table key."""
    boards, winners, active, player = state
    flat = "".join(_CELL_CHAR[c] for b in boards for c in b)
    wins = "".join((w if w in ("X", "O", "D") else "-") for w in winners)
    a = "-" if active is None else str(active)
    return f"{flat}|{wins}|{a}|{player}"