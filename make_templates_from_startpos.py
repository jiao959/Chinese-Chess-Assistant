from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import cv2
import mss
import numpy as np
from PIL import Image

from board_recognizer import draw_geometry_preview, estimate_board_geometry

BASE_DIR = Path(__file__).resolve().parent
CONFIG_PATH = BASE_DIR / "config.json"

STARTING_BOARD = [
    [
        "black_rook",
        "black_knight",
        "black_bishop",
        "black_advisor",
        "black_king",
        "black_advisor",
        "black_bishop",
        "black_knight",
        "black_rook",
    ],
    [None, None, None, None, None, None, None, None, None],
    [None, "black_cannon", None, None, None, None, None, "black_cannon", None],
    ["black_pawn", None, "black_pawn", None, "black_pawn", None, "black_pawn", None, "black_pawn"],
    [None, None, None, None, None, None, None, None, None],
    [None, None, None, None, None, None, None, None, None],
    ["red_pawn", None, "red_pawn", None, "red_pawn", None, "red_pawn", None, "red_pawn"],
    [None, "red_cannon", None, None, None, None, None, "red_cannon", None],
    [None, None, None, None, None, None, None, None, None],
    [
        "red_rook",
        "red_knight",
        "red_bishop",
        "red_advisor",
        "red_king",
        "red_advisor",
        "red_bishop",
        "red_knight",
        "red_rook",
    ],
]


def main() -> int:
    args = parse_args()

    if not CONFIG_PATH.exists():
        print("找不到 config.json。请先运行 main.py 并选择棋盘区域。")
        return 1

    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    region = config.get("last_board_region")
    if not region:
        print("config.json 中没有 last_board_region。请先在 App 中点击“选择棋盘区域”。")
        return 1

    templates_dir = BASE_DIR / str(config.get("templates_dir", "templates"))
    templates_dir.mkdir(parents=True, exist_ok=True)

    if args.image:
        image = Image.open(args.image).convert("RGB")
    else:
        image = capture_region(region)
    source_path = templates_dir / "_template_source_capture.png"
    preview_path = templates_dir / "_grid_preview.png"
    image.save(source_path)

    image_bgr = cv2.cvtColor(np.array(image.convert("RGB")), cv2.COLOR_RGB2BGR)
    geometry = estimate_board_geometry(image_bgr)
    draw_geometry_preview(image, geometry).save(preview_path)
    config["grid_padding_ratio"] = geometry_to_padding_ratio(image.size, geometry)
    config["use_fixed_grid_padding"] = False
    CONFIG_PATH.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")

    counts: defaultdict[str, int] = defaultdict(int)
    first_sample: dict[str, Image.Image] = {}

    for row, pieces in enumerate(STARTING_BOARD):
        for col, piece_name in enumerate(pieces):
            if not piece_name:
                continue
            center_x, center_y = geometry.point(row, col)
            crop = crop_point(image, center_x, center_y, geometry.crop_radius)
            piece_dir = templates_dir / piece_name
            piece_dir.mkdir(parents=True, exist_ok=True)
            counts[piece_name] += 1
            crop.save(piece_dir / f"sample_{counts[piece_name]:02d}.png")
            first_sample.setdefault(piece_name, crop)

    for piece_name, crop in first_sample.items():
        crop.save(templates_dir / f"{piece_name}.png")

    print(f"已保存截图预览：{source_path}")
    print(f"已保存交叉点预览：{preview_path}")
    print(
        "交叉点估算："
        f"left={geometry.left:.1f}, top={geometry.top:.1f}, "
        f"right={geometry.right:.1f}, bottom={geometry.bottom:.1f}, "
        f"cell={geometry.cell_w:.1f}x{geometry.cell_h:.1f}"
    )
    print(f"已生成模板目录：{templates_dir}")
    for piece_name in sorted(counts):
        print(f"{piece_name}: {counts[piece_name]} 个样本")
    print("完成。请先打开 _grid_preview.png，确认红点落在棋盘交叉点上，再回到 App 点击“识别棋局”。")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="从标准开局棋盘截图生成棋子模板。")
    parser.add_argument(
        "--image",
        type=Path,
        help="可选：直接使用指定截图生成模板，而不是重新截屏。",
    )
    return parser.parse_args()


def capture_region(region: dict[str, int]) -> Image.Image:
    left = int(region["x"])
    top = int(region["y"])
    width = int(region["width"])
    height = int(region["height"])
    with mss.MSS() as sct:
        raw = sct.grab({"left": left, "top": top, "width": width, "height": height})
    return Image.frombytes("RGB", raw.size, raw.rgb)


def crop_point(image: Image.Image, center_x: int, center_y: int, radius: int) -> Image.Image:
    width, height = image.size
    return image.crop(
        (
            max(0, center_x - radius),
            max(0, center_y - radius),
            min(width, center_x + radius + 1),
            min(height, center_y + radius + 1),
        )
    )


def geometry_to_padding_ratio(size: tuple[int, int], geometry) -> dict[str, float]:
    width, height = size
    return {
        "left": geometry.left / width,
        "right": 1.0 - geometry.right / width,
        "top": geometry.top / height,
        "bottom": 1.0 - geometry.bottom / height,
    }


if __name__ == "__main__":
    raise SystemExit(main())
