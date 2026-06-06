from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

Board = list[list[str | None]]

PIECE_NAMES = [
    "red_king",
    "red_advisor",
    "red_bishop",
    "red_rook",
    "red_knight",
    "red_cannon",
    "red_pawn",
    "black_king",
    "black_advisor",
    "black_bishop",
    "black_rook",
    "black_knight",
    "black_cannon",
    "black_pawn",
]

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


@dataclass(frozen=True)
class PieceMatch:
    piece: str | None
    score: float
    template_path: str | None = None


@dataclass(frozen=True)
class TemplateImage:
    bgr: np.ndarray
    gray: np.ndarray
    ink: np.ndarray
    edge: np.ndarray
    path: Path
    descriptors: np.ndarray | None


@dataclass(frozen=True)
class BoardGeometry:
    left: float
    top: float
    right: float
    bottom: float
    cell_w: float
    cell_h: float
    crop_radius: int

    def point(self, row: int, col: int) -> tuple[int, int]:
        return (
            int(round(self.left + col * self.cell_w)),
            int(round(self.top + row * self.cell_h)),
        )


@dataclass(frozen=True)
class AxisCandidate:
    start: float
    end: float
    cell: float
    hits: int
    mean_error: float
    score: float


@dataclass(frozen=True)
class RecognitionResult:
    screen_board: Board
    point_matches: list[list[PieceMatch]]
    loaded_template_count: int
    missing_piece_templates: list[str]
    geometry: BoardGeometry


class BoardRecognizer:
    def __init__(
        self,
        templates_dir: str | Path,
        threshold: float = 0.48,
        grid_padding_ratio: dict[str, float] | None = None,
    ) -> None:
        self.templates_dir = Path(templates_dir)
        self.threshold = float(threshold)
        self.grid_padding_ratio = grid_padding_ratio
        self.sift = cv2.SIFT_create()
        self.templates: dict[str, list[TemplateImage]] = {}
        self.missing_piece_templates: list[str] = []
        self._load_templates()

    def recognize(self, board_image: Image.Image) -> RecognitionResult:
        image = np.array(board_image.convert("RGB"))
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        height, width = image.shape[:2]

        if width < 90 or height < 100:
            raise ValueError("棋盘截图区域过小，无法按 9×10 交叉点识别。")

        geometry = estimate_board_geometry(image, self.grid_padding_ratio)

        board: Board = []
        matches: list[list[PieceMatch]] = []
        for row in range(10):
            board_row: list[str | None] = []
            match_row: list[PieceMatch] = []
            for col in range(9):
                center_x, center_y = geometry.point(row, col)
                crop = self._crop_point(image, center_x, center_y, geometry.crop_radius)
                match = self._match_piece(crop, row, col)
                board_row.append(match.piece if match.score >= self.threshold else None)
                match_row.append(match if match.score >= self.threshold else PieceMatch(None, match.score))
            board.append(board_row)
            matches.append(match_row)

        return RecognitionResult(
            screen_board=board,
            point_matches=matches,
            loaded_template_count=sum(len(items) for items in self.templates.values()),
            missing_piece_templates=self.missing_piece_templates[:],
            geometry=geometry,
        )

    def _load_templates(self) -> None:
        self.templates.clear()
        self.missing_piece_templates.clear()

        if not self.templates_dir.exists():
            self.missing_piece_templates = PIECE_NAMES[:]
            return

        for piece_name in PIECE_NAMES:
            image_paths = self._find_template_paths(piece_name)
            loaded: list[TemplateImage] = []
            for image_path in image_paths:
                image = _read_color_image(image_path)
                if image is None:
                    continue
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                piece_color = "red" if piece_name.startswith("red_") else "black"
                ink = _make_ink_mask(image, piece_color)
                loaded.append(
                    TemplateImage(
                        bgr=image,
                        gray=self._preprocess_gray(gray),
                        ink=ink,
                        edge=_make_edge_image(gray),
                        path=image_path,
                        descriptors=self._sift_descriptors(ink),
                    )
                )
            if loaded:
                self.templates[piece_name] = loaded
            else:
                self.missing_piece_templates.append(piece_name)

    def _find_template_paths(self, piece_name: str) -> list[Path]:
        paths: list[Path] = []
        for ext in IMAGE_EXTENSIONS:
            paths.extend(self.templates_dir.glob(f"{piece_name}{ext}"))

        piece_dir = self.templates_dir / piece_name
        if piece_dir.exists():
            for path in piece_dir.iterdir():
                if path.suffix.lower() in IMAGE_EXTENSIONS:
                    paths.append(path)

        return sorted(set(paths))

    def _match_piece(self, crop_bgr: np.ndarray, row: int | None = None, col: int | None = None) -> PieceMatch:
        if not self.templates:
            return PieceMatch(None, 0.0)

        crop_gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
        crop_gray = self._preprocess_gray(crop_gray)
        piece_color = _detect_piece_color(crop_bgr)
        if piece_color is None:
            return PieceMatch(None, 0.0)
        crop_ink = _make_ink_mask(crop_bgr, piece_color) if piece_color else None
        crop_edge = _make_edge_image(cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY))
        crop_descriptors = self._sift_descriptors(crop_ink) if crop_ink is not None else None

        best_piece: str | None = None
        best_score = -1.0
        best_template_path: str | None = None
        per_piece_best: dict[str, tuple[float, str | None, float, float]] = {}
        allowed_pieces = _allowed_pieces_for_position(piece_color, row, col)
        for piece_name, templates in self.templates.items():
            if piece_color and not piece_name.startswith(f"{piece_color}_"):
                continue
            if allowed_pieces is not None and piece_name not in allowed_pieces:
                continue
            for template in templates:
                photo_score = self._aligned_score(crop_gray, template.gray)
                edge_score = self._aligned_score(crop_edge, template.edge)
                if crop_ink is not None:
                    ink_score = self._aligned_score(crop_ink, template.ink)
                    local_ink_score = self._match_score(crop_ink, template.ink)
                    sift_score = self._sift_score(crop_descriptors, template.descriptors)
                    score = (
                        sift_score * 0.70
                        + photo_score * 0.12
                        + edge_score * 0.06
                        + ink_score * 0.08
                        + local_ink_score * 0.04
                    )
                else:
                    score = photo_score * 0.70 + edge_score * 0.30
                if score > best_score:
                    best_piece = piece_name
                    best_score = score
                    best_template_path = str(template.path)
                if score > per_piece_best.get(piece_name, (-1.0, None, 0.0, 0.0))[0]:
                    per_piece_best[piece_name] = (
                        score,
                        str(template.path),
                        sift_score if crop_ink is not None else 0.0,
                        local_ink_score if crop_ink is not None else 0.0,
                    )

        best_piece, best_score, best_template_path = self._resolve_close_knight_pawn_match(
            piece_color,
            best_piece,
            best_score,
            best_template_path,
            per_piece_best,
        )
        return PieceMatch(best_piece, max(0.0, best_score), best_template_path)

    @staticmethod
    def _resolve_close_knight_pawn_match(
        piece_color: str,
        best_piece: str | None,
        best_score: float,
        best_template_path: str | None,
        per_piece_best: dict[str, tuple[float, str | None, float, float]],
    ) -> tuple[str | None, float, str | None]:
        knight = f"{piece_color}_knight"
        pawn = f"{piece_color}_pawn"
        if best_piece not in {knight, pawn}:
            return best_piece, best_score, best_template_path
        if knight not in per_piece_best or pawn not in per_piece_best:
            return best_piece, best_score, best_template_path

        knight_score, knight_path, knight_sift, knight_local = per_piece_best[knight]
        pawn_score, pawn_path, pawn_sift, pawn_local = per_piece_best[pawn]
        if abs(knight_score - pawn_score) > 0.035:
            return best_piece, best_score, best_template_path

        if knight_sift >= pawn_sift + 0.03 and knight_local >= pawn_local + 0.05:
            return knight, max(best_score, knight_score), knight_path
        if pawn_sift >= knight_sift + 0.03 and pawn_local >= knight_local + 0.05:
            return pawn, max(best_score, pawn_score), pawn_path

        return best_piece, best_score, best_template_path

    def _sift_descriptors(self, image: np.ndarray) -> np.ndarray | None:
        _keypoints, descriptors = self.sift.detectAndCompute(image, None)
        return descriptors

    @staticmethod
    def _sift_score(
        crop_descriptors: np.ndarray | None,
        template_descriptors: np.ndarray | None,
    ) -> float:
        if crop_descriptors is None or template_descriptors is None:
            return 0.0
        if len(crop_descriptors) < 2 or len(template_descriptors) < 2:
            return 0.0

        matcher = cv2.BFMatcher()
        matches = matcher.knnMatch(crop_descriptors, template_descriptors, k=2)
        good = 0
        for pair in matches:
            if len(pair) != 2:
                continue
            first, second = pair
            if first.distance < 0.75 * second.distance:
                good += 1
        return min(1.0, good / 18.0)

    @staticmethod
    def _match_score(crop_gray: np.ndarray, template_gray: np.ndarray) -> float:
        crop_h, crop_w = crop_gray.shape[:2]
        tpl_h, tpl_w = template_gray.shape[:2]

        if tpl_h < 3 or tpl_w < 3 or crop_h < 3 or crop_w < 3:
            return 0.0

        best_score = 0.0
        max_scale = min(crop_w / tpl_w, crop_h / tpl_h)
        for scale in (0.65, 0.75, 0.85, 0.95, 1.0, 1.08, 1.18, 1.3):
            if scale > max_scale:
                continue

            resized_w = max(3, int(round(tpl_w * scale)))
            resized_h = max(3, int(round(tpl_h * scale)))
            if resized_w > crop_w or resized_h > crop_h:
                continue

            if resized_w == tpl_w and resized_h == tpl_h:
                candidate = template_gray
            else:
                interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC
                candidate = cv2.resize(template_gray, (resized_w, resized_h), interpolation=interpolation)

            result = cv2.matchTemplate(crop_gray, candidate, cv2.TM_CCOEFF_NORMED)
            _min_val, max_val, _min_loc, _max_loc = cv2.minMaxLoc(result)
            if not np.isnan(max_val):
                best_score = max(best_score, float(max_val))

        return best_score

    @staticmethod
    def _aligned_score(crop_gray: np.ndarray, template_gray: np.ndarray) -> float:
        if crop_gray.size == 0 or template_gray.size == 0:
            return 0.0
        crop_norm = _normalize_piece_image(crop_gray)
        template_norm = _normalize_piece_image(template_gray)
        result = cv2.matchTemplate(crop_norm, template_norm, cv2.TM_CCOEFF_NORMED)
        score = float(result[0, 0])
        if np.isnan(score):
            return 0.0
        return max(0.0, score)

    @staticmethod
    def _crop_point(image: np.ndarray, center_x: int, center_y: int, radius: int) -> np.ndarray:
        height, width = image.shape[:2]
        left = center_x - radius
        right = center_x + radius + 1
        top = center_y - radius
        bottom = center_y + radius + 1

        src_left = max(0, left)
        src_right = min(width, right)
        src_top = max(0, top)
        src_bottom = min(height, bottom)
        crop = image[src_top:src_bottom, src_left:src_right]

        pad_left = src_left - left
        pad_right = right - src_right
        pad_top = src_top - top
        pad_bottom = bottom - src_bottom
        if pad_left or pad_right or pad_top or pad_bottom:
            crop = cv2.copyMakeBorder(
                crop,
                pad_top,
                pad_bottom,
                pad_left,
                pad_right,
                cv2.BORDER_REPLICATE,
            )
        return crop

    @staticmethod
    def _preprocess_gray(image: np.ndarray) -> np.ndarray:
        blurred = cv2.GaussianBlur(image, (3, 3), 0)
        return cv2.equalizeHist(blurred)


def estimate_board_geometry(
    image_bgr: np.ndarray,
    grid_padding_ratio: dict[str, float] | None = None,
) -> BoardGeometry:
    height, width = image_bgr.shape[:2]

    if grid_padding_ratio:
        return _geometry_from_padding(width, height, grid_padding_ratio)

    detected = _detect_geometry_from_grid_lines(image_bgr)
    if detected:
        return detected

    return _geometry_from_padding(
        width,
        height,
        {"left": 0.065, "right": 0.065, "top": 0.055, "bottom": 0.055},
    )


def draw_geometry_preview(image: Image.Image, geometry: BoardGeometry) -> Image.Image:
    preview = np.array(image.convert("RGB"))
    preview = cv2.cvtColor(preview, cv2.COLOR_RGB2BGR)
    for row in range(10):
        for col in range(9):
            x, y = geometry.point(row, col)
            cv2.circle(preview, (x, y), 5, (0, 0, 255), 2)
    cv2.rectangle(
        preview,
        (int(round(geometry.left)), int(round(geometry.top))),
        (int(round(geometry.right)), int(round(geometry.bottom))),
        (255, 0, 0),
        2,
    )
    preview = cv2.cvtColor(preview, cv2.COLOR_BGR2RGB)
    return Image.fromarray(preview)


def board_to_debug_text(board: Board) -> str:
    rows: list[str] = []
    for row in board:
        rows.append(" ".join(piece or "." for piece in row))
    return "\n".join(rows)


def _read_color_image(path: Path) -> np.ndarray | None:
    """Read an image from paths that may contain non-ASCII characters."""
    try:
        data = np.fromfile(path, dtype=np.uint8)
    except OSError:
        return None
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def _make_edge_image(gray: np.ndarray) -> np.ndarray:
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    return cv2.Canny(blurred, 45, 135)


def _normalize_piece_image(image: np.ndarray, size: int = 96) -> np.ndarray:
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.resize(image, (size, size), interpolation=cv2.INTER_AREA)


def _detect_piece_color(crop_bgr: np.ndarray) -> str | None:
    crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    red = crop_rgb[:, :, 0].astype(np.int16)
    green = crop_rgb[:, :, 1].astype(np.int16)
    blue = crop_rgb[:, :, 2].astype(np.int16)

    dark_ratio = np.mean((red < 100) & (green < 100) & (blue < 100))
    red_ink_ratio = np.mean(
        (red > 110)
        & (green < 105)
        & (blue < 105)
        & (red > green + 35)
        & (red > blue + 35)
    )

    if red_ink_ratio >= 0.025:
        return "red"
    if dark_ratio >= 0.035:
        return "black"
    return None


def _make_ink_mask(image_bgr: np.ndarray, piece_color: str) -> np.ndarray:
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    red = image_rgb[:, :, 0].astype(np.int16)
    green = image_rgb[:, :, 1].astype(np.int16)
    blue = image_rgb[:, :, 2].astype(np.int16)

    if piece_color == "black":
        mask = (red < 120) & (green < 120) & (blue < 120)
    else:
        mask = (
            (red > 110)
            & (green < 105)
            & (blue < 105)
            & (red > green + 35)
            & (red > blue + 35)
        )

    mask_image = mask.astype(np.uint8) * 255
    height, width = mask_image.shape[:2]
    center_mask = np.zeros_like(mask_image)
    radius = int(min(width, height) * 0.30)
    cv2.circle(center_mask, (width // 2, height // 2), max(3, radius), 255, -1)
    mask_image = cv2.bitwise_and(mask_image, center_mask)
    return cv2.GaussianBlur(mask_image, (3, 3), 0)


def _allowed_pieces_for_position(
    piece_color: str,
    row: int | None,
    col: int | None,
) -> set[str] | None:
    # Endgames can place attacking rooks, cannons, knights, and pawns inside
    # palace files. SIFT matching is now strong enough that opening-position
    # priors do more harm than good, so this hook intentionally allows every
    # piece of the detected color.
    return None


def _geometry_from_padding(width: int, height: int, padding: dict[str, float]) -> BoardGeometry:
    left = width * float(padding.get("left", 0.065))
    right = width * (1.0 - float(padding.get("right", 0.065)))
    top = height * float(padding.get("top", 0.055))
    bottom = height * (1.0 - float(padding.get("bottom", 0.055)))
    cell_w = (right - left) / 8.0
    cell_h = (bottom - top) / 9.0
    crop_radius = max(10, int(min(cell_w, cell_h) * 0.48))
    return BoardGeometry(left, top, right, bottom, cell_w, cell_h, crop_radius)


def _detect_geometry_from_grid_lines(image_bgr: np.ndarray) -> BoardGeometry | None:
    height, width = image_bgr.shape[:2]
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    edges = cv2.Canny(gray, 50, 150)

    lines = cv2.HoughLinesP(
        edges,
        1,
        np.pi / 180,
        threshold=max(30, int(min(width, height) * 0.045)),
        minLineLength=int(min(width, height) * 0.12),
        maxLineGap=max(8, int(min(width, height) * 0.02)),
    )
    if lines is None:
        return None

    vertical_positions: list[float] = []
    horizontal_positions: list[float] = []
    max_vertical_dx = max(5, width * 0.012)
    max_horizontal_dy = max(5, height * 0.012)

    for line in lines[:, 0, :]:
        x1, y1, x2, y2 = [int(value) for value in line]
        dx = abs(x2 - x1)
        dy = abs(y2 - y1)
        if dy >= height * 0.08 and dx <= max_vertical_dx:
            vertical_positions.append((x1 + x2) / 2.0)
        elif dx >= width * 0.08 and dy <= max_horizontal_dy:
            horizontal_positions.append((y1 + y2) / 2.0)

    x_groups = _group_positions(vertical_positions, tolerance=max(5, width * 0.010))
    y_groups = _group_positions(horizontal_positions, tolerance=max(5, height * 0.010))
    x_candidates = _fit_even_axis_candidates(x_groups, expected_count=9, axis_length=width)
    y_candidates = _fit_even_axis_candidates(y_groups, expected_count=10, axis_length=height)

    best: tuple[float, AxisCandidate, AxisCandidate] | None = None
    for x_axis in x_candidates:
        for y_axis in y_candidates:
            cell_ratio = x_axis.cell / max(1.0, y_axis.cell)
            if not 0.78 <= cell_ratio <= 1.22:
                continue
            grid_aspect = (x_axis.end - x_axis.start) / max(1.0, y_axis.end - y_axis.start)
            if not 0.72 <= grid_aspect <= 1.05:
                continue
            aspect_score = 1.0 - min(1.0, abs(1.0 - cell_ratio))
            score = x_axis.score + y_axis.score + aspect_score * 2.0
            if best is None or score > best[0]:
                best = (score, x_axis, y_axis)

    if best is None:
        return None

    _score, x_axis, y_axis = best
    left, right = x_axis.start, x_axis.end
    top, bottom = y_axis.start, y_axis.end
    cell_w = (right - left) / 8.0
    cell_h = (bottom - top) / 9.0
    if cell_w < 10 or cell_h < 10:
        return None

    crop_radius = max(10, int(min(cell_w, cell_h) * 0.48))
    return BoardGeometry(left, top, right, bottom, cell_w, cell_h, crop_radius)


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


def _fit_even_axis_candidates(
    groups: list[float],
    expected_count: int,
    axis_length: int,
) -> list[AxisCandidate]:
    if len(groups) < max(4, expected_count // 2):
        return []

    groups = sorted(groups)
    candidates: list[AxisCandidate] = []
    min_cell = max(14.0, axis_length * 0.035)
    max_cell = axis_length * 0.18

    for left_group_index, left_group in enumerate(groups):
        for right_group in groups[left_group_index + 1 :]:
            gap = right_group - left_group
            for step_count in range(1, expected_count):
                cell = gap / step_count
                if cell < min_cell or cell > max_cell:
                    continue
                tolerance = max(5.0, cell * 0.13)
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
                        AxisCandidate(
                            start=start,
                            end=end,
                            cell=cell,
                            hits=hits,
                            mean_error=mean_error,
                            score=score,
                        )
                    )

    candidates.sort(key=lambda item: item.score, reverse=True)
    deduped: list[AxisCandidate] = []
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
