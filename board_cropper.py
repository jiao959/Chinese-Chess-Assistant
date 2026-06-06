from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image


@dataclass(frozen=True)
class CropResult:
    board_image: Image.Image
    preview_image: Image.Image
    box: tuple[int, int, int, int]
    score: float


@dataclass(frozen=True)
class AxisFit:
    start: float
    end: float
    cell: float
    hits: int
    mean_error: float
    score: float = 0.0


def auto_crop_board(full_screen_image: Image.Image) -> CropResult:
    rgb = np.array(full_screen_image.convert("RGB"))
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    height, width = bgr.shape[:2]

    candidates: list[tuple[tuple[int, int, int, int], float]] = []
    grid_candidate = _detect_board_by_grid(bgr)
    if grid_candidate:
        candidates.append(grid_candidate)

    for warm_box in _warm_rect_candidates(bgr):
        refined = _refine_box_by_inner_grid(bgr, warm_box)
        if refined:
            candidates.append(refined)
        else:
            candidates.append((_expand_box(warm_box, width, height, 0.01), _board_score(_crop_bgr(bgr, warm_box)) * 0.75))

    if not candidates:
        candidates.append(((0, 0, width, height), 0.0))

    box, score = max(candidates, key=lambda item: item[1] - _dark_bottom_penalty(bgr, item[0]))

    x, y, w, h = box
    board = full_screen_image.crop((x, y, x + w, y + h))
    preview = _draw_preview(full_screen_image, box, score)
    return CropResult(board_image=board, preview_image=preview, box=box, score=score)


def _crop_bgr(bgr: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    x, y, w, h = box
    return bgr[y : y + h, x : x + w]


def _detect_board_by_grid(bgr: np.ndarray) -> tuple[tuple[int, int, int, int], float] | None:
    height, width = bgr.shape[:2]
    scale = min(1.0, 1400.0 / max(width, height))
    work = bgr
    if scale < 1.0:
        work = cv2.resize(bgr, (int(width * scale), int(height * scale)), interpolation=cv2.INTER_AREA)

    work_height, work_width = work.shape[:2]
    gray = cv2.cvtColor(work, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    edges = cv2.Canny(gray, 45, 135)

    lines = cv2.HoughLinesP(
        edges,
        1,
        np.pi / 180,
        threshold=max(35, int(min(work_width, work_height) * 0.045)),
        minLineLength=max(55, int(min(work_width, work_height) * 0.08)),
        maxLineGap=max(8, int(min(work_width, work_height) * 0.018)),
    )
    if lines is None:
        return None

    vertical_positions: list[float] = []
    horizontal_positions: list[float] = []
    max_dx = max(5.0, work_width * 0.008)
    max_dy = max(5.0, work_height * 0.008)

    for line in lines[:, 0, :]:
        x1, y1, x2, y2 = [int(value) for value in line]
        dx = abs(x2 - x1)
        dy = abs(y2 - y1)
        if dy >= max(55, work_height * 0.10) and dx <= max_dx:
            vertical_positions.append((x1 + x2) / 2.0)
        elif dx >= max(55, work_width * 0.06) and dy <= max_dy:
            horizontal_positions.append((y1 + y2) / 2.0)

    x_groups = _group_positions(vertical_positions, tolerance=max(5.0, work_width * 0.006))
    y_groups = _group_positions(horizontal_positions, tolerance=max(5.0, work_height * 0.006))
    x_fits = _axis_fits(x_groups, expected_count=9, axis_length=work_width)
    y_fits = _axis_fits(y_groups, expected_count=10, axis_length=work_height)

    best: tuple[float, AxisFit, AxisFit] | None = None
    for x_fit in x_fits:
        for y_fit in y_fits:
            if not 0.70 <= x_fit.cell / max(1.0, y_fit.cell) <= 1.30:
                continue
            grid_aspect = (x_fit.end - x_fit.start) / max(1.0, y_fit.end - y_fit.start)
            if not 0.70 <= grid_aspect <= 1.05:
                continue

            hit_score = (x_fit.hits / 9.0 + y_fit.hits / 10.0) / 2.0
            error_score = 1.0 - min(
                1.0,
                (x_fit.mean_error / max(1.0, x_fit.cell) + y_fit.mean_error / max(1.0, y_fit.cell)) / 2.0,
            )
            warm_score = _warm_score(work, x_fit, y_fit)
            score = hit_score * 0.50 + error_score * 0.25 + warm_score * 0.25
            if best is None or score > best[0]:
                best = (score, x_fit, y_fit)

    if best is None:
        return None

    score, x_fit, y_fit = best
    if score < 0.52:
        return None

    pad_x = x_fit.cell * 0.65
    pad_top = y_fit.cell * 0.75
    pad_bottom = y_fit.cell * 0.75
    left = max(0, int(round((x_fit.start - pad_x) / scale)))
    top = max(0, int(round((y_fit.start - pad_top) / scale)))
    right = min(width, int(round((x_fit.end + pad_x) / scale)))
    bottom = min(height, int(round((y_fit.end + pad_bottom) / scale)))

    if right - left < 220 or bottom - top < 260:
        return None
    return (left, top, right - left, bottom - top), score


def _fallback_warm_board_crop(bgr: np.ndarray) -> tuple[tuple[int, int, int, int], float]:
    height, width = bgr.shape[:2]
    candidates = _warm_rect_candidates(bgr)
    if not candidates:
        return (0, 0, width, height), 0.0

    best_box = candidates[0]
    best_score = -1.0
    for box in candidates:
        x, y, w, h = _expand_box(box, width, height, 0.01)
        score = _board_score(bgr[y : y + h, x : x + w])
        if score > best_score:
            best_box = (x, y, w, h)
            best_score = score
    return best_box, best_score * 0.75


def _refine_box_by_inner_grid(
    bgr: np.ndarray,
    box: tuple[int, int, int, int],
) -> tuple[tuple[int, int, int, int], float] | None:
    image_height, image_width = bgr.shape[:2]
    x, y, w, h = _expand_box(box, image_width, image_height, 0.01)
    crop = bgr[y : y + h, x : x + w]
    if crop.size == 0:
        return None

    try:
        from board_recognizer import estimate_board_geometry

        geometry = estimate_board_geometry(crop)
    except Exception:
        return None

    cell_ratio = geometry.cell_w / max(1.0, geometry.cell_h)
    if not 0.78 <= cell_ratio <= 1.22:
        return None

    pad_x = geometry.cell_w * 0.65
    pad_top = geometry.cell_h * 0.80
    pad_bottom = geometry.cell_h * 0.80
    left = max(0, int(round(x + geometry.left - pad_x)))
    top = max(0, int(round(y + geometry.top - pad_top)))
    right = min(image_width, int(round(x + geometry.right + pad_x)))
    bottom = min(image_height, int(round(y + geometry.bottom + pad_bottom)))
    if right - left < 220 or bottom - top < 260:
        return None

    refined_box = (left, top, right - left, bottom - top)
    crop_score = _board_score(bgr[top:bottom, left:right])
    geometry_score = 1.0 - min(1.0, abs(1.0 - cell_ratio))
    return refined_box, crop_score * 0.65 + geometry_score * 0.35


def _dark_bottom_penalty(bgr: np.ndarray, box: tuple[int, int, int, int]) -> float:
    crop = _crop_bgr(bgr, box)
    if crop.size == 0:
        return 1.0
    height = crop.shape[0]
    bottom = crop[int(height * 0.82) :, :]
    if bottom.size == 0:
        return 0.0
    gray = cv2.cvtColor(bottom, cv2.COLOR_BGR2GRAY)
    dark_ratio = float(np.mean(gray < 45))
    return min(0.6, dark_ratio * 0.9)


def _axis_fits(groups: list[float], expected_count: int, axis_length: int) -> list[AxisFit]:
    groups = sorted(groups)
    if len(groups) < max(4, expected_count // 2):
        return []

    candidates: list[AxisFit] = []
    min_cell = max(14.0, axis_length * 0.025)
    max_cell = axis_length * 0.16
    for left_group_index, left_group in enumerate(groups):
        for right_group in groups[left_group_index + 1 :]:
            gap = right_group - left_group
            for step_count in range(1, expected_count):
                cell = gap / step_count
                if cell < min_cell or cell > max_cell:
                    continue
                tolerance = max(5.0, cell * 0.14)
                for left_grid_index in range(0, expected_count - step_count):
                    start = left_group - left_grid_index * cell
                    end = start + (expected_count - 1) * cell
                    if start < -cell * 0.45 or end > axis_length + cell * 0.45:
                        continue
                    expected = [start + index * cell for index in range(expected_count)]
                    distances = [min(abs(group - point) for group in groups) for point in expected]
                    hit_distances = [distance for distance in distances if distance <= tolerance]
                    hits = len(hit_distances)
                    if hits < max(5, int(expected_count * 0.58)):
                        continue
                    mean_error = float(np.mean(hit_distances)) if hit_distances else float(np.mean(distances))
                    in_bounds = sum(1 for point in expected if 0 <= point <= axis_length)
                    edge_penalty = 0.0
                    if start < 0:
                        edge_penalty += abs(start) / cell
                    if end > axis_length:
                        edge_penalty += (end - axis_length) / cell
                    score = hits * 10.0 + in_bounds * 0.5 - mean_error - edge_penalty * 2.0
                    candidates.append(
                        AxisFit(
                            start=start,
                            end=end,
                            cell=cell,
                            hits=hits,
                            mean_error=mean_error,
                            score=score,
                        )
                    )

    candidates.sort(key=lambda fit: fit.score, reverse=True)
    deduped: list[AxisFit] = []
    for candidate in candidates:
        if any(
            abs(candidate.start - existing.start) <= max(4.0, candidate.cell * 0.08)
            and abs(candidate.cell - existing.cell) <= max(2.0, candidate.cell * 0.05)
            for existing in deduped
        ):
            continue
        deduped.append(candidate)
        if len(deduped) >= 24:
            break
    return deduped


def _warm_score(image_bgr: np.ndarray, x_fit: AxisFit, y_fit: AxisFit) -> float:
    height, width = image_bgr.shape[:2]
    pad_x = x_fit.cell * 0.65
    pad_y = y_fit.cell * 0.75
    left = max(0, int(round(x_fit.start - pad_x)))
    top = max(0, int(round(y_fit.start - pad_y)))
    right = min(width, int(round(x_fit.end + pad_x)))
    bottom = min(height, int(round(y_fit.end + pad_y)))
    crop = image_bgr[top:bottom, left:right]
    if crop.size == 0:
        return 0.0

    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    hue = hsv[:, :, 0]
    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]
    warm_ratio = np.mean((hue >= 8) & (hue <= 36) & (saturation > 25) & (value > 95))
    bright_ratio = np.mean(value > 120)
    return float(min(1.0, warm_ratio * 1.25 + bright_ratio * 0.15))


def _warm_rect_candidates(bgr: np.ndarray) -> list[tuple[int, int, int, int]]:
    height, width = bgr.shape[:2]
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    hue = hsv[:, :, 0]
    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]

    mask = ((hue >= 8) & (hue <= 36) & (saturation > 24) & (value > 90)).astype(np.uint8) * 255
    kernel = np.ones((17, 17), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

    contours, _hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates: list[tuple[int, int, int, int]] = []
    min_area = max(30_000, int(width * height * 0.02))
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w * h < min_area or w < 250 or h < 250:
            continue
        aspect = w / max(1, h)
        if 0.55 <= aspect <= 1.25:
            candidates.append((x, y, w, h))

    candidates.sort(key=lambda item: item[2] * item[3], reverse=True)
    return candidates[:12]


def _board_score(crop_bgr: np.ndarray) -> float:
    height, width = crop_bgr.shape[:2]
    if width < 250 or height < 250:
        return 0.0

    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    edges = cv2.Canny(gray, 45, 140)
    lines = cv2.HoughLinesP(
        edges,
        1,
        np.pi / 180,
        threshold=max(35, int(min(width, height) * 0.06)),
        minLineLength=int(min(width, height) * 0.18),
        maxLineGap=max(8, int(min(width, height) * 0.03)),
    )
    if lines is None:
        return 0.0

    vertical = 0
    horizontal = 0
    for line in lines[:, 0, :]:
        x1, y1, x2, y2 = [int(value) for value in line]
        dx = abs(x2 - x1)
        dy = abs(y2 - y1)
        if dy > height * 0.20 and dx < width * 0.05:
            vertical += 1
        elif dx > width * 0.20 and dy < height * 0.05:
            horizontal += 1

    line_score = min(1.0, (vertical + horizontal) / 28.0)
    aspect_score = 1.0 - min(1.0, abs(width / max(1, height) - 0.90))
    size_score = min(1.0, (width * height) / 500_000)
    return line_score * 0.65 + aspect_score * 0.20 + size_score * 0.15


def _group_positions(positions: list[float], tolerance: float) -> list[float]:
    if not positions:
        return []

    positions = sorted(positions)
    groups: list[list[float]] = [[positions[0]]]
    for position in positions[1:]:
        if abs(position - float(np.mean(groups[-1]))) <= tolerance:
            groups[-1].append(position)
        else:
            groups.append([position])
    return [float(np.mean(group)) for group in groups]


def _expand_box(
    box: tuple[int, int, int, int],
    image_width: int,
    image_height: int,
    ratio: float,
) -> tuple[int, int, int, int]:
    x, y, w, h = box
    pad = int(max(w, h) * ratio)
    x2 = min(image_width, x + w + pad)
    y2 = min(image_height, y + h + pad)
    x = max(0, x - pad)
    y = max(0, y - pad)
    return x, y, x2 - x, y2 - y


def _draw_preview(image: Image.Image, box: tuple[int, int, int, int], score: float) -> Image.Image:
    preview = np.array(image.convert("RGB"))
    preview = cv2.cvtColor(preview, cv2.COLOR_RGB2BGR)
    x, y, w, h = box
    cv2.rectangle(preview, (x, y), (x + w, y + h), (0, 0, 255), 3)
    cv2.putText(
        preview,
        f"board score={score:.2f}",
        (x, max(25, y - 8)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 0, 255),
        2,
        cv2.LINE_AA,
    )
    preview = cv2.cvtColor(preview, cv2.COLOR_BGR2RGB)
    return Image.fromarray(preview)
