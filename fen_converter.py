from __future__ import annotations

from typing import Iterable, Literal

PieceName = str | None
Board = list[list[PieceName]]
SelfColor = Literal["red", "black", "unknown"]

PIECE_TO_FEN: dict[str, str] = {
    "red_king": "K",
    "red_advisor": "A",
    "red_bishop": "B",
    "red_rook": "R",
    "red_knight": "N",
    "red_cannon": "C",
    "red_pawn": "P",
    "black_king": "k",
    "black_advisor": "a",
    "black_bishop": "b",
    "black_rook": "r",
    "black_knight": "n",
    "black_cannon": "c",
    "black_pawn": "p",
}


def rotate_board_180(board: Board) -> Board:
    """Return a board rotated 180 degrees without mutating the input."""
    _validate_board_shape(board)
    return [list(reversed(row)) for row in reversed(board)]


def detect_self_color(screen_board: Board) -> SelfColor:
    """Infer the side at the bottom of the screen board.

    The MVP follows the product rule: count red and black pieces in the lower
    half of the screen board. More red means the user side is red; more black
    means the user side is black.
    """
    _validate_board_shape(screen_board)
    red_count = 0
    black_count = 0

    for row in screen_board[5:10]:
        for piece in row:
            if not piece:
                continue
            if piece.startswith("red_"):
                red_count += 1
            elif piece.startswith("black_"):
                black_count += 1

    if red_count > black_count:
        return "red"
    if black_count > red_count:
        return "black"
    return "unknown"


def board_to_fen(board: Board, side_to_move: Literal["w", "b"]) -> str:
    """Convert a canonical Xiangqi board to FEN.

    Canonical board orientation is black at the top and red at the bottom.
    Empty points are compressed by row. The trailing fields are accepted by
    UCI-style Xiangqi engines such as Pikafish.
    """
    _validate_board_shape(board)
    if side_to_move not in {"w", "b"}:
        raise ValueError("side_to_move must be 'w' or 'b'")

    fen_rows: list[str] = []
    for row in board:
        empty_count = 0
        fen_row_parts: list[str] = []
        for piece in row:
            if piece is None:
                empty_count += 1
                continue
            if empty_count:
                fen_row_parts.append(str(empty_count))
                empty_count = 0
            try:
                fen_row_parts.append(PIECE_TO_FEN[piece])
            except KeyError as exc:
                raise ValueError(f"unknown piece name: {piece}") from exc
        if empty_count:
            fen_row_parts.append(str(empty_count))
        fen_rows.append("".join(fen_row_parts) or "9")

    return f"{'/'.join(fen_rows)} {side_to_move} - - 0 1"


def screen_board_to_fen(
    screen_board: Board,
    self_color: Literal["red", "black"],
) -> tuple[str, Board]:
    """Convert a screen-oriented board to canonical FEN for the selected side."""
    if self_color == "red":
        canonical_board = [row[:] for row in screen_board]
        side_to_move: Literal["w", "b"] = "w"
    elif self_color == "black":
        canonical_board = rotate_board_180(screen_board)
        side_to_move = "b"
    else:
        raise ValueError("self_color must be 'red' or 'black'")

    return board_to_fen(canonical_board, side_to_move), canonical_board


def bestmove_to_screen_points(
    bestmove: str,
    self_color: Literal["red", "black"],
) -> tuple[tuple[int, int], tuple[int, int]] | None:
    """Convert a Pikafish move to screen board coordinates.

    Returns ((from_row, from_col), (to_row, to_col)). This is used only as a
    readable helper in the MVP; no automatic moving is implemented.
    """
    move = bestmove.strip().split()[0] if bestmove.strip() else ""
    if len(move) < 4:
        return None

    try:
        from_col, from_row = _uci_square_to_canonical_point(move[0:2])
        to_col, to_row = _uci_square_to_canonical_point(move[2:4])
    except ValueError:
        return None

    if self_color == "black":
        from_col, from_row = 8 - from_col, 9 - from_row
        to_col, to_row = 8 - to_col, 9 - to_row

    return (from_row, from_col), (to_row, to_col)


def _uci_square_to_canonical_point(square: str) -> tuple[int, int]:
    if len(square) != 2:
        raise ValueError("square must be 2 characters")
    file_char = square[0].lower()
    rank_char = square[1]
    if file_char < "a" or file_char > "i" or not rank_char.isdigit():
        raise ValueError(f"invalid Xiangqi square: {square}")
    rank = int(rank_char)
    if rank < 0 or rank > 9:
        raise ValueError(f"invalid Xiangqi rank: {square}")

    col = ord(file_char) - ord("a")
    row = 9 - rank
    return col, row


def _validate_board_shape(board: Iterable[Iterable[PieceName]]) -> None:
    rows = list(board)
    if len(rows) != 10:
        raise ValueError("board must contain 10 rows")
    for row in rows:
        if len(list(row)) != 9:
            raise ValueError("each board row must contain 9 columns")
