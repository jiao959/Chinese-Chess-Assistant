from __future__ import annotations

from typing import Literal

Board = list[list[str | None]]
SideColor = Literal["red", "black"]

CHINESE_NUMBERS = ["一", "二", "三", "四", "五", "六", "七", "八", "九"]

PIECE_TEXT: dict[str, str] = {
    "red_king": "帅",
    "red_advisor": "仕",
    "red_bishop": "相",
    "red_rook": "车",
    "red_knight": "马",
    "red_cannon": "炮",
    "red_pawn": "兵",
    "black_king": "将",
    "black_advisor": "士",
    "black_bishop": "象",
    "black_rook": "车",
    "black_knight": "马",
    "black_cannon": "炮",
    "black_pawn": "卒",
}

TARGET_FILE_PIECES = {
    "red_advisor",
    "red_bishop",
    "red_knight",
    "black_advisor",
    "black_bishop",
    "black_knight",
}


def bestmove_to_chinese(
    bestmove_line: str,
    canonical_board: Board,
    side_color: SideColor,
) -> str | None:
    move = extract_bestmove(bestmove_line)
    if not move:
        return None

    points = move_to_points(move)
    if not points:
        return None
    from_row, from_col, to_row, to_col = points

    if not _is_inside(from_row, from_col) or not _is_inside(to_row, to_col):
        return None

    piece = canonical_board[from_row][from_col]
    if not piece or not piece.startswith(f"{side_color}_"):
        return None

    piece_text = PIECE_TEXT.get(piece)
    if not piece_text:
        return None

    first = _piece_prefix(canonical_board, piece, from_row, from_col, side_color)
    action = _action_text(from_row, to_row, side_color)
    target = _target_text(piece, from_row, from_col, to_row, to_col, side_color, action)

    return f"{piece_text}{first}{action}{target}"


def extract_bestmove(bestmove_line: str) -> str:
    parts = bestmove_line.strip().split()
    if not parts:
        return ""
    if parts[0] == "bestmove" and len(parts) >= 2:
        return "" if parts[1] == "(none)" else parts[1]
    if "bestmove" in parts:
        index = parts.index("bestmove")
        if index + 1 < len(parts) and parts[index + 1] != "(none)":
            return parts[index + 1]
        return ""
    return parts[0]


def move_to_points(move: str) -> tuple[int, int, int, int] | None:
    if len(move) < 4:
        return None
    try:
        from_col, from_row = _uci_square_to_canonical_point(move[0:2])
        to_col, to_row = _uci_square_to_canonical_point(move[2:4])
    except ValueError:
        return None
    return from_row, from_col, to_row, to_col


def _piece_prefix(
    board: Board,
    piece: str,
    from_row: int,
    from_col: int,
    side_color: SideColor,
) -> str:
    same_file_rows = [
        row
        for row in range(10)
        if row != from_row and board[row][from_col] == piece
    ]
    if not same_file_rows:
        return _file_text(from_col, side_color)

    rows = same_file_rows + [from_row]
    rows = sorted(rows, reverse=(side_color == "black"))
    index = rows.index(from_row)
    if len(rows) == 2:
        return "前" if index == 0 else "后"
    if index == 0:
        return "前"
    if index == len(rows) - 1:
        return "后"
    return "中"


def _action_text(from_row: int, to_row: int, side_color: SideColor) -> str:
    if from_row == to_row:
        return "平"
    if side_color == "red":
        return "进" if to_row < from_row else "退"
    return "进" if to_row > from_row else "退"


def _target_text(
    piece: str,
    from_row: int,
    from_col: int,
    to_row: int,
    to_col: int,
    side_color: SideColor,
    action: str,
) -> str:
    if action == "平" or piece in TARGET_FILE_PIECES:
        return _file_text(to_col, side_color)
    return _number_text(abs(to_row - from_row))


def _file_text(col: int, side_color: SideColor) -> str:
    if side_color == "red":
        number = 9 - col
    else:
        number = col + 1
    return _number_text(number)


def _number_text(number: int) -> str:
    if not 1 <= number <= 9:
        raise ValueError(f"Xiangqi notation number out of range: {number}")
    return CHINESE_NUMBERS[number - 1]


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


def _is_inside(row: int, col: int) -> bool:
    return 0 <= row < 10 and 0 <= col < 9
