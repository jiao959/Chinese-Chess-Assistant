from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from board_recognizer import BoardGeometry

ScreenPoint = tuple[int, int]


def draw_bestmove_arrow(
    board_image: Image.Image,
    geometry: BoardGeometry,
    from_point: ScreenPoint,
    to_point: ScreenPoint,
    output_path: Path | None = None,
) -> Image.Image:
    """Draw a visual best-move arrow on a cropped board image.

    Points are screen board coordinates: (row, col), not pixel coordinates.
    """
    image_rgb = np.array(board_image.convert("RGB"))
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

    path = _screen_move_path(geometry, from_point, to_point)
    thickness = max(8, int(min(geometry.cell_w, geometry.cell_h) * 0.14))
    color = (0, 105, 255)

    if len(path) == 2:
        cv2.arrowedLine(
            image_bgr,
            path[0],
            path[1],
            color,
            thickness,
            cv2.LINE_AA,
            tipLength=0.18,
        )
    else:
        for start, end in zip(path[:-2], path[1:-1]):
            cv2.line(image_bgr, start, end, color, thickness, cv2.LINE_AA)
        cv2.arrowedLine(
            image_bgr,
            path[-2],
            path[-1],
            color,
            thickness,
            cv2.LINE_AA,
            tipLength=0.22,
        )

    result_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    result = Image.fromarray(result_rgb)
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        result.save(output_path)
    return result


def _screen_move_path(
    geometry: BoardGeometry,
    from_point: ScreenPoint,
    to_point: ScreenPoint,
) -> list[tuple[int, int]]:
    from_row, from_col = from_point
    to_row, to_col = to_point
    start = geometry.point(from_row, from_col)
    end = geometry.point(to_row, to_col)

    row_delta = abs(to_row - from_row)
    col_delta = abs(to_col - from_col)
    if row_delta and col_delta:
        # Knight moves read better as an angled route like many chess apps show.
        if row_delta > col_delta:
            corner = (start[0], end[1])
        else:
            corner = (end[0], start[1])
        return [start, corner, end]

    return [start, end]
