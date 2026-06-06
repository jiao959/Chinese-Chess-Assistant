from __future__ import annotations

import json
import os
import sys
import importlib.util
import time
from pathlib import Path
from typing import Any


def _configure_qt_runtime() -> None:
    """Make PySide6 plugin loading reliable when launched from IDEs.

    PyCharm can run conda Python without fully activating the conda environment.
    In that case Qt may find PySide6 but fail to initialize the Windows platform
    plugin. Set the plugin paths before importing Qt modules.
    """
    pyside_spec = importlib.util.find_spec("PySide6")
    if not pyside_spec or not pyside_spec.submodule_search_locations:
        return

    pyside_dir = Path(next(iter(pyside_spec.submodule_search_locations))).resolve()
    plugins_dir = pyside_dir / "plugins"
    platforms_dir = plugins_dir / "platforms"

    if plugins_dir.exists():
        os.environ.setdefault("QT_PLUGIN_PATH", str(plugins_dir))
    if platforms_dir.exists():
        os.environ.setdefault("QT_QPA_PLATFORM_PLUGIN_PATH", str(platforms_dir))

    dll_dirs = [
        pyside_dir,
        Path(sys.prefix) / "Library" / "bin",
    ]

    shiboken_spec = importlib.util.find_spec("shiboken6")
    if shiboken_spec and shiboken_spec.submodule_search_locations:
        dll_dirs.append(Path(next(iter(shiboken_spec.submodule_search_locations))).resolve())

    add_dll_directory = getattr(os, "add_dll_directory", None)
    if add_dll_directory:
        for dll_dir in dll_dirs:
            if dll_dir.exists():
                add_dll_directory(str(dll_dir))


_configure_qt_runtime()

from PySide6.QtCore import Qt
from PySide6.QtGui import QFont, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from board_cropper import auto_crop_board
from board_recognizer import BoardRecognizer, board_to_debug_text, draw_geometry_preview
from board_recognizer import PIECE_NAMES
from engine_client import EngineError, PikafishClient
from fen_converter import (
    SelfColor,
    bestmove_to_screen_points,
    detect_self_color,
    screen_board_to_fen,
)
from move_overlay import draw_bestmove_arrow
from move_notation import bestmove_to_chinese, extract_bestmove, move_to_points
from screen_capture import ScreenCapture, ScreenRegion, select_board_region

BASE_DIR = Path(__file__).resolve().parent
CONFIG_PATH = BASE_DIR / "config.json"
DEBUG_OUTPUT_DIR = BASE_DIR / "debug_outputs"

DEFAULT_CONFIG: dict[str, Any] = {
    "pikafish_path": "engines/pikafish.exe",
    "pikafish_eval_file": "engines/pikafish.nnue",
    "templates_dir": "templates",
    "last_board_region": None,
    "analysis_movetime_ms": 1000,
    "analysis_mode": "movetime",
    "analysis_depth": 12,
    "engine_threads": 4,
    "engine_hash_mb": 256,
    "template_match_threshold": 0.38,
    "use_fixed_grid_padding": False,
    "grid_padding_ratio": None,
}


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("中国象棋辅助分析 MVP")
        self.resize(860, 680)

        self.config = load_config()
        self.capture = ScreenCapture()
        self.screen_board = None
        self.canonical_board = None
        self.last_recognition_image = None
        self.last_recognition_geometry = None
        self.current_fen = ""
        self.self_color: SelfColor = "unknown"
        self.engine_client: PikafishClient | None = None

        self._build_ui()
        self._load_region_from_config()
        self._refresh_self_color_ui()

    def closeEvent(self, event) -> None:  # type: ignore[override]
        if self.engine_client:
            self.engine_client.close()
        super().closeEvent(event)

    def _build_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)

        root_layout = QVBoxLayout(central)
        root_layout.setContentsMargins(16, 16, 16, 16)
        root_layout.setSpacing(12)

        button_layout = QHBoxLayout()
        button_layout.setSpacing(8)
        root_layout.addLayout(button_layout)

        self.analyze_button = QPushButton("分析最佳走法")
        self.select_region_button = QPushButton("手动选择区域")

        button_layout.addWidget(self.analyze_button)
        button_layout.addWidget(self.select_region_button)
        button_layout.addStretch(1)

        self.select_region_button.clicked.connect(self.on_select_region)
        self.analyze_button.clicked.connect(self.on_analyze_bestmove)

        info_grid = QGridLayout()
        info_grid.setHorizontalSpacing(12)
        info_grid.setVerticalSpacing(8)
        root_layout.addLayout(info_grid)

        self.region_label = QLabel("未选择")
        self.self_color_label = QLabel("无法判断")
        self.analysis_side_label = QLabel("己方（无法判断）")
        self.manual_color_combo = QComboBox()
        self.manual_color_combo.addItem("自动判断", "auto")
        self.manual_color_combo.addItem("红方", "red")
        self.manual_color_combo.addItem("黑方", "black")
        self.manual_color_combo.currentIndexChanged.connect(self.on_manual_color_changed)
        self.elapsed_label = QLabel("-")
        self.result_label = QLabel("中文走法：-")
        self.result_label.setWordWrap(True)

        info_grid.addWidget(QLabel("己方颜色显示："), 0, 0)
        info_grid.addWidget(self.self_color_label, 0, 1)
        info_grid.addWidget(QLabel("分析方："), 0, 2)
        info_grid.addWidget(self.analysis_side_label, 0, 3)
        info_grid.addWidget(QLabel("手动选择己方颜色："), 1, 0)
        info_grid.addWidget(self.manual_color_combo, 1, 1, 1, 3)

        self.analysis_mode_combo = QComboBox()
        self.analysis_mode_combo.addItem("按时间", "movetime")
        self.analysis_mode_combo.addItem("按深度", "depth")
        mode_index = self.analysis_mode_combo.findData(
            str(self.config.get("analysis_mode", "movetime"))
        )
        self.analysis_mode_combo.setCurrentIndex(max(0, mode_index))

        self.movetime_spin = QSpinBox()
        self.movetime_spin.setRange(100, 30000)
        self.movetime_spin.setSingleStep(500)
        self.movetime_spin.setSuffix(" ms")
        self.movetime_spin.setValue(int(self.config.get("analysis_movetime_ms", 1000)))

        self.depth_spin = QSpinBox()
        self.depth_spin.setRange(1, 40)
        self.depth_spin.setValue(int(self.config.get("analysis_depth", 12)))

        self.threads_spin = QSpinBox()
        self.threads_spin.setRange(1, 64)
        self.threads_spin.setValue(int(self.config.get("engine_threads", 4)))

        self.hash_spin = QSpinBox()
        self.hash_spin.setRange(16, 4096)
        self.hash_spin.setSingleStep(16)
        self.hash_spin.setSuffix(" MB")
        self.hash_spin.setValue(int(self.config.get("engine_hash_mb", 256)))

        self.analysis_mode_combo.currentIndexChanged.connect(self.on_engine_settings_changed)
        self.movetime_spin.valueChanged.connect(self.on_engine_settings_changed)
        self.depth_spin.valueChanged.connect(self.on_engine_settings_changed)
        self.threads_spin.valueChanged.connect(self.on_engine_settings_changed)
        self.hash_spin.valueChanged.connect(self.on_engine_settings_changed)

        info_grid.addWidget(QLabel("分析模式："), 2, 0)
        info_grid.addWidget(self.analysis_mode_combo, 2, 1)
        info_grid.addWidget(QLabel("思考时间："), 2, 2)
        info_grid.addWidget(self.movetime_spin, 2, 3)
        info_grid.addWidget(QLabel("搜索深度："), 3, 0)
        info_grid.addWidget(self.depth_spin, 3, 1)
        info_grid.addWidget(QLabel("线程："), 3, 2)
        info_grid.addWidget(self.threads_spin, 3, 3)
        info_grid.addWidget(QLabel("Hash："), 4, 0)
        info_grid.addWidget(self.hash_spin, 4, 1)
        info_grid.addWidget(self.result_label, 5, 0, 1, 4)

        self.preview_label = QLabel("棋盘预览图")
        self.preview_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview_label.setMinimumHeight(430)
        self.preview_label.setStyleSheet(
            "QLabel { background: #f7f7f7; border: 1px solid #d0d0d0; color: #666; }"
        )
        root_layout.addWidget(self.preview_label, stretch=1)

        mono = QFont("Consolas")
        mono.setStyleHint(QFont.StyleHint.Monospace)

        self.fen_text = self._make_text_edit(mono, min_height=72)
        self.bestmove_raw_text = self._make_text_edit(mono, min_height=60)
        self.bestmove_text = self._make_text_edit(QFont(), min_height=60)
        self.error_text = self._make_text_edit(QFont(), min_height=88)
        self.board_debug_text = self._make_text_edit(mono, min_height=140)
        self.board_debug_text.setReadOnly(False)
        self.status_label = QLabel("")
        self.status_label.setWordWrap(True)
        root_layout.addWidget(self.status_label)

    @staticmethod
    def _make_text_edit(font: QFont, min_height: int) -> QTextEdit:
        edit = QTextEdit()
        edit.setReadOnly(True)
        edit.setMinimumHeight(min_height)
        if font.family():
            edit.setFont(font)
        return edit

    def _load_region_from_config(self) -> None:
        region = ScreenRegion.from_dict(self.config.get("last_board_region"))
        if region:
            self._set_region(region, persist=False)
        else:
            self.region_label.setText("未选择")

    def on_select_region(self) -> None:
        self._set_error("")
        self.hide()
        QApplication.processEvents()

        def selected(region: ScreenRegion) -> None:
            self.show()
            self._set_region(region, persist=True)
            if self._recognize_manual_region(region):
                QApplication.processEvents()
                self._analyze_current_fen()

        def cancelled() -> None:
            self.show()
            self._set_error("已取消选择棋盘区域。")

        select_board_region(selected, cancelled)

    def _recognize_manual_region(self, region: ScreenRegion) -> bool:
        self._set_error("")
        self.result_label.setText("中文走法：-")
        templates_dir = resolve_path(str(self.config.get("templates_dir", "templates")))
        threshold = float(self.config.get("template_match_threshold", 0.75))
        grid_padding_ratio = (
            self.config.get("grid_padding_ratio")
            if self.config.get("use_fixed_grid_padding", False)
            else None
        )
        manual_region_debug_path = DEBUG_OUTPUT_DIR / "manual_region_capture.png"
        crop_preview_path = DEBUG_OUTPUT_DIR / "manual_crop_preview.png"
        capture_debug_path = DEBUG_OUTPUT_DIR / "cropped_board.png"
        grid_debug_path = DEBUG_OUTPUT_DIR / "last_recognition_grid_preview.png"
        matrix_debug_path = DEBUG_OUTPUT_DIR / "recognized_board.txt"

        try:
            DEBUG_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
            clear_debug_outputs()
            selected_image = self.capture.capture_region(region)
            selected_image.save(manual_region_debug_path)
            crop_result = auto_crop_board(selected_image)
            crop_result.preview_image.save(crop_preview_path)
            board_image = crop_result.board_image
            board_image.save(capture_debug_path)
            crop_x, crop_y, crop_w, crop_h = crop_result.box
            detected_region = ScreenRegion(
                x=region.x + crop_x,
                y=region.y + crop_y,
                width=crop_w,
                height=crop_h,
            )
            self._set_region(detected_region, persist=True)
            recognizer = BoardRecognizer(
                templates_dir,
                threshold=threshold,
                grid_padding_ratio=grid_padding_ratio,
            )
            result = recognizer.recognize(board_image)
            draw_geometry_preview(board_image, result.geometry).save(grid_debug_path)
            self._set_preview_image(capture_debug_path)
        except Exception as exc:
            self._set_error(f"手动区域识别失败：{exc}")
            return False

        self._apply_recognition_result(
            result=result,
            board_image=board_image,
            matrix_debug_path=matrix_debug_path,
            status_lines=[
                f"手动选择原始区域：x={region.x}, y={region.y}, width={region.width}, height={region.height}",
                f"裁剪后的棋盘区域：x={detected_region.x}, y={detected_region.y}, width={detected_region.width}, height={detected_region.height}",
                f"手动原始区域截图：{manual_region_debug_path}",
                f"手动区域裁剪预览：{crop_preview_path}",
                f"裁剪得到的棋盘图片：{capture_debug_path}",
                f"本次交叉点预览：{grid_debug_path}",
                f"识别出的棋盘矩阵：{matrix_debug_path}",
                f"自动裁剪得分：{crop_result.score:.2f}",
                f"模板目录：{templates_dir}",
                f"已加载模板数量：{result.loaded_template_count}",
            ],
        )
        return True

    def on_recognize_board(self) -> bool:
        self._set_error("")
        self.result_label.setText("中文走法：-")
        templates_dir = resolve_path(str(self.config.get("templates_dir", "templates")))
        threshold = float(self.config.get("template_match_threshold", 0.75))
        grid_padding_ratio = (
            self.config.get("grid_padding_ratio")
            if self.config.get("use_fixed_grid_padding", False)
            else None
        )
        full_screen_debug_path = DEBUG_OUTPUT_DIR / "full_screen_capture.png"
        crop_preview_path = DEBUG_OUTPUT_DIR / "auto_crop_preview.png"
        capture_debug_path = DEBUG_OUTPUT_DIR / "cropped_board.png"
        grid_debug_path = DEBUG_OUTPUT_DIR / "last_recognition_grid_preview.png"
        matrix_debug_path = DEBUG_OUTPUT_DIR / "recognized_board.txt"

        try:
            DEBUG_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
            clear_debug_outputs()
            self.hide()
            QApplication.processEvents()
            time.sleep(0.2)
            full_screen_image, full_screen_region = self.capture.capture_full_screen()
            self.show()
            full_screen_image.save(full_screen_debug_path)
            crop_result = auto_crop_board(full_screen_image)
            crop_result.preview_image.save(crop_preview_path)
            board_image = crop_result.board_image
            board_image.save(capture_debug_path)
            crop_x, crop_y, crop_w, crop_h = crop_result.box
            detected_region = ScreenRegion(
                x=full_screen_region.x + crop_x,
                y=full_screen_region.y + crop_y,
                width=crop_w,
                height=crop_h,
            )
            self._set_region(detected_region, persist=True)
            recognizer = BoardRecognizer(
                templates_dir,
                threshold=threshold,
                grid_padding_ratio=grid_padding_ratio,
            )
            result = recognizer.recognize(board_image)
            draw_geometry_preview(board_image, result.geometry).save(grid_debug_path)
            self._set_preview_image(capture_debug_path)
        except Exception as exc:
            self.show()
            message = f"识别棋局失败：{exc}"
            if capture_debug_path.exists():
                message += f"\n本次捕捉区域截图：{capture_debug_path}"
            self._set_error(message)
            return False

        warnings: list[str] = []
        warnings.append(f"全屏截图：{full_screen_debug_path}")
        warnings.append(f"自动裁剪预览：{crop_preview_path}")
        warnings.append(f"裁剪得到的棋盘图片：{capture_debug_path}")
        warnings.append(f"识别出的棋盘矩阵：{matrix_debug_path}")
        warnings.append(f"本次交叉点预览：{grid_debug_path}")
        warnings.append(f"自动裁剪得分：{crop_result.score:.2f}")
        warnings.append(f"模板目录：{templates_dir}")
        warnings.append(f"已加载模板数量：{result.loaded_template_count}")
        warnings.append(
            "交叉点估算："
            f"left={result.geometry.left:.1f}, top={result.geometry.top:.1f}, "
            f"right={result.geometry.right:.1f}, bottom={result.geometry.bottom:.1f}, "
            f"cell={result.geometry.cell_w:.1f}×{result.geometry.cell_h:.1f}"
        )

        self._apply_recognition_result(
            result=result,
            board_image=board_image,
            matrix_debug_path=matrix_debug_path,
            status_lines=warnings,
        )
        return True

    def _apply_recognition_result(
        self,
        result,
        board_image,
        matrix_debug_path: Path,
        status_lines: list[str],
    ) -> None:
        self.screen_board = result.screen_board
        self.last_recognition_image = board_image
        self.last_recognition_geometry = result.geometry
        board_text = board_to_debug_text(result.screen_board)
        matrix_debug_path.write_text(board_text, encoding="utf-8")
        match_details_path = matrix_debug_path.with_name("match_details.txt")
        match_details_path.write_text(match_details_to_text(result.point_matches), encoding="utf-8")
        self.board_debug_text.setPlainText(board_text)
        self.self_color = detect_self_color(result.screen_board)
        if self.manual_color_combo.currentData() in {"red", "black"}:
            self.self_color = self.manual_color_combo.currentData()
        self._refresh_self_color_ui()

        templates_dir = resolve_path(str(self.config.get("templates_dir", "templates")))
        if not templates_dir.exists():
            status_lines.append(f"templates 目录不存在：{templates_dir}")
        if result.loaded_template_count == 0:
            status_lines.append("未加载到任何棋子模板，识别结果会全部为空。")
        elif result.missing_piece_templates:
            status_lines.append("部分棋子模板缺失：" + ", ".join(result.missing_piece_templates))
        if self.self_color == "unknown":
            status_lines.append("无法判断己方颜色，请在“手动选择己方颜色”中选择红方或黑方。")
        status_lines.append(f"模板匹配详情：{match_details_path}")

        self._update_fen_from_board()
        self._set_error("\n".join(status_lines))

    def on_manual_color_changed(self) -> None:
        selected = self.manual_color_combo.currentData()
        if selected in {"red", "black"}:
            self.self_color = selected
        elif self.screen_board is not None:
            self.self_color = detect_self_color(self.screen_board)
        else:
            self.self_color = "unknown"
        self._refresh_self_color_ui()
        self._update_fen_from_board()

    def on_engine_settings_changed(self) -> None:
        self._save_engine_settings_from_ui()
        if self.engine_client:
            self.engine_client.close()
            self.engine_client = None

    def _save_engine_settings_from_ui(self) -> None:
        if not hasattr(self, "analysis_mode_combo"):
            return
        self.config["analysis_mode"] = self.analysis_mode_combo.currentData() or "movetime"
        self.config["analysis_movetime_ms"] = int(self.movetime_spin.value())
        self.config["analysis_depth"] = int(self.depth_spin.value())
        self.config["engine_threads"] = int(self.threads_spin.value())
        self.config["engine_hash_mb"] = int(self.hash_spin.value())
        save_config(self.config)

    def on_apply_board_text(self) -> None:
        try:
            board = parse_board_text(self.board_debug_text.toPlainText())
        except ValueError as exc:
            self._set_error(f"矩阵格式错误：{exc}")
            return

        self.screen_board = board
        if self.manual_color_combo.currentData() in {"red", "black"}:
            self.self_color = self.manual_color_combo.currentData()
        else:
            self.self_color = detect_self_color(board)
        self._refresh_self_color_ui()
        self._update_fen_from_board()
        self._set_error("已应用矩阵修正，并重新生成 FEN。")

    def on_save_template_samples(self) -> None:
        if self.last_recognition_image is None or self.last_recognition_geometry is None:
            self._set_error("请先点击“识别棋局”，再补充模板样本。")
            return

        try:
            board = parse_board_text(self.board_debug_text.toPlainText())
        except ValueError as exc:
            self._set_error(f"矩阵格式错误：{exc}")
            return

        templates_dir = resolve_path(str(self.config.get("templates_dir", "templates")))
        templates_dir.mkdir(parents=True, exist_ok=True)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        saved_count = 0

        for row_index, row in enumerate(board):
            for col_index, piece_name in enumerate(row):
                if not piece_name:
                    continue
                center_x, center_y = self.last_recognition_geometry.point(row_index, col_index)
                crop = crop_pil_point(
                    self.last_recognition_image,
                    center_x,
                    center_y,
                    self.last_recognition_geometry.crop_radius,
                )
                piece_dir = templates_dir / piece_name
                piece_dir.mkdir(parents=True, exist_ok=True)
                crop.save(piece_dir / f"user_{timestamp}_r{row_index}_c{col_index}.png")
                saved_count += 1

        self._set_error(f"已补充模板样本 {saved_count} 个。请重新点击“识别棋局”验证。")

    def on_analyze_bestmove(self) -> None:
        self.analyze_button.setEnabled(False)
        self.select_region_button.setEnabled(False)
        try:
            self._save_engine_settings_from_ui()
            self._set_error("正在自动捕捉并识别棋局...")
            QApplication.processEvents()
            if not self.on_recognize_board():
                return
            QApplication.processEvents()
            self._analyze_current_fen()
        finally:
            self.analyze_button.setEnabled(True)
            self.select_region_button.setEnabled(True)

    def _analyze_current_fen(self) -> None:
        if not self.current_fen:
            self._set_error("请先识别棋局，并确保已确定己方颜色后生成 FEN。")
            return
        if self.self_color not in {"red", "black"}:
            self._set_error("无法判断己方颜色，请先手动选择红方或黑方。")
            return

        engine_path = str(self.config.get("pikafish_path", "engines/pikafish.exe"))
        eval_file_path = str(self.config.get("pikafish_eval_file", "engines/pikafish.nnue"))
        movetime_ms = int(self.config.get("analysis_movetime_ms", 1000))
        analysis_mode = str(self.config.get("analysis_mode", "movetime"))
        analysis_depth = int(self.config.get("analysis_depth", 12))
        engine_threads = int(self.config.get("engine_threads", 4))
        engine_hash_mb = int(self.config.get("engine_hash_mb", 256))

        try:
            if self.engine_client:
                self.engine_client.close()
                self.engine_client = None
            self.engine_client = PikafishClient(
                engine_path,
                cwd=BASE_DIR,
                eval_file_path=eval_file_path,
                threads=engine_threads,
                hash_mb=engine_hash_mb,
            )
            bestmove_line, elapsed = self.engine_client.analyze(
                self.current_fen,
                movetime_ms=movetime_ms,
                depth=analysis_depth,
                mode=analysis_mode,
            )
        except EngineError as exc:
            self._set_error(str(exc))
            return
        except Exception as exc:
            self._set_error(f"分析失败：{exc}")
            return

        self.elapsed_label.setText(f"{elapsed:.2f} 秒")
        self.bestmove_raw_text.setPlainText(bestmove_line)
        bestmove_text = self._format_bestmove_text(bestmove_line)
        self.bestmove_text.setPlainText(bestmove_text)
        DEBUG_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        (DEBUG_OUTPUT_DIR / "bestmove_chinese.txt").write_text(bestmove_text, encoding="utf-8")
        display_lines = [line for line in bestmove_text.splitlines() if line.startswith(("最佳走法", "屏幕坐标"))]
        self.result_label.setText("；".join(display_lines) if display_lines else "最佳走法：-")
        if self._bestmove_has_valid_origin(bestmove_line):
            self._update_bestmove_preview(bestmove_line)
            self._set_error(f"分析完成，用时 {elapsed:.2f} 秒。中文走法已输出到：{DEBUG_OUTPUT_DIR / 'bestmove_chinese.txt'}")
        else:
            current_board_preview = DEBUG_OUTPUT_DIR / "cropped_board.png"
            if current_board_preview.exists():
                self._set_preview_image(current_board_preview)
            self._set_error(
                "引擎返回的 bestmove 与当前识别棋盘不一致：起点不是己方棋子。\n"
                f"本次 bestmove：{extract_bestmove(bestmove_line)}\n"
                "已取消箭头绘制。请重新识别，或降低深度后再分析。"
            )

    def on_copy_fen(self) -> None:
        if not self.current_fen:
            self._set_error("当前没有可复制的 FEN。")
            return
        QApplication.clipboard().setText(self.current_fen)
        self._set_error("FEN 已复制到剪贴板。")

    def _set_region(self, region: ScreenRegion, persist: bool) -> None:
        self.region_label.setText(
            f"x={region.x}, y={region.y}, width={region.width}, height={region.height}"
        )
        if persist:
            self.config["last_board_region"] = region.to_dict()
            save_config(self.config)

    def _refresh_self_color_ui(self) -> None:
        if self.self_color == "red":
            self.self_color_label.setText("红方")
            self.analysis_side_label.setText("己方（红方）")
        elif self.self_color == "black":
            self.self_color_label.setText("黑方")
            self.analysis_side_label.setText("己方（黑方）")
        else:
            self.self_color_label.setText("无法判断")
            self.analysis_side_label.setText("己方（无法判断）")

    def _update_fen_from_board(self) -> None:
        self.current_fen = ""
        self.canonical_board = None
        self.fen_text.clear()
        if self.screen_board is None:
            return
        if self.self_color not in {"red", "black"}:
            return

        try:
            fen, canonical_board = screen_board_to_fen(self.screen_board, self.self_color)
        except Exception as exc:
            self._set_error(f"生成 FEN 失败：{exc}")
            return

        self.current_fen = fen
        self.canonical_board = canonical_board
        self.fen_text.setPlainText(fen)

    def _format_bestmove_text(self, bestmove_line: str) -> str:
        move = extract_bestmove(bestmove_line)
        chinese_move = None
        if self.canonical_board is not None and self.self_color in {"red", "black"}:
            chinese_move = bestmove_to_chinese(bestmove_line, self.canonical_board, self.self_color)

        if chinese_move:
            text = f"最佳走法：{chinese_move}"
            text += f"\n原始 bestmove：{move}"
        else:
            text = f"最佳走法：{move or bestmove_line.strip()}"
        if self.self_color in {"red", "black"}:
            points = bestmove_to_screen_points(move, self.self_color)
            if points:
                (from_row, from_col), (to_row, to_col) = points
                text += f"\n屏幕坐标：第 {from_row + 1} 行第 {from_col + 1} 列 -> 第 {to_row + 1} 行第 {to_col + 1} 列"
        return text

    def _bestmove_has_valid_origin(self, bestmove_line: str) -> bool:
        if self.canonical_board is None or self.self_color not in {"red", "black"}:
            return False
        points = move_to_points(extract_bestmove(bestmove_line))
        if not points:
            return False
        from_row, from_col, _to_row, _to_col = points
        piece = self.canonical_board[from_row][from_col]
        return bool(piece and piece.startswith(f"{self.self_color}_"))

    def _update_bestmove_preview(self, bestmove_line: str) -> None:
        if (
            self.last_recognition_image is None
            or self.last_recognition_geometry is None
            or self.self_color not in {"red", "black"}
        ):
            return

        move = extract_bestmove(bestmove_line)
        points = bestmove_to_screen_points(move, self.self_color)
        if not points:
            return

        output_path = DEBUG_OUTPUT_DIR / "bestmove_preview.png"
        draw_bestmove_arrow(
            self.last_recognition_image,
            self.last_recognition_geometry,
            points[0],
            points[1],
            output_path,
        )
        self._set_preview_image(output_path)

    def _set_error(self, message: str) -> None:
        self.error_text.setPlainText(message)
        self.status_label.setText(message)

    def _set_preview_image(self, image_path: Path) -> None:
        pixmap = QPixmap(str(image_path))
        if pixmap.isNull():
            self.preview_label.setText("棋盘预览图加载失败")
            return

        scaled = pixmap.scaled(
            self.preview_label.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.preview_label.setPixmap(scaled)


def load_config() -> dict[str, Any]:
    config = DEFAULT_CONFIG.copy()
    if CONFIG_PATH.exists():
        try:
            with CONFIG_PATH.open("r", encoding="utf-8") as file:
                user_config = json.load(file)
            if isinstance(user_config, dict):
                config.update(user_config)
        except json.JSONDecodeError:
            pass
    else:
        save_config(config)
    return config


def save_config(config: dict[str, Any]) -> None:
    merged = DEFAULT_CONFIG.copy()
    merged.update(config)
    with CONFIG_PATH.open("w", encoding="utf-8") as file:
        json.dump(merged, file, ensure_ascii=False, indent=2)


def resolve_path(path_text: str) -> Path:
    path = Path(path_text)
    if path.is_absolute():
        return path
    return (BASE_DIR / path).resolve()


def clear_debug_outputs() -> None:
    DEBUG_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in DEBUG_OUTPUT_DIR.iterdir():
        if path.is_file():
            path.unlink()


def match_details_to_text(point_matches) -> str:
    rows: list[str] = []
    for row_index, row in enumerate(point_matches):
        for col_index, match in enumerate(row):
            if not match.piece:
                continue
            rows.append(
                f"row={row_index}, col={col_index}, piece={match.piece}, "
                f"score={match.score:.4f}, template={match.template_path or '-'}"
            )
    return "\n".join(rows)


def parse_board_text(text: str) -> list[list[str | None]]:
    rows: list[list[str | None]] = []
    valid_pieces = set(PIECE_NAMES)
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        items = line.split()
        if len(items) != 9:
            raise ValueError("每一行必须正好有 9 个内容，空位用 . 表示。")

        row: list[str | None] = []
        for item in items:
            if item == ".":
                row.append(None)
            elif item in valid_pieces:
                row.append(item)
            else:
                raise ValueError(f"未知棋子名称：{item}")
        rows.append(row)

    if len(rows) != 10:
        raise ValueError("必须正好有 10 行。")
    return rows


def crop_pil_point(image, center_x: int, center_y: int, radius: int):
    width, height = image.size
    left = center_x - radius
    right = center_x + radius + 1
    top = center_y - radius
    bottom = center_y + radius + 1

    src_left = max(0, left)
    src_right = min(width, right)
    src_top = max(0, top)
    src_bottom = min(height, bottom)
    crop = image.crop((src_left, src_top, src_right, src_bottom))

    target_size = (radius * 2 + 1, radius * 2 + 1)
    if crop.size == target_size:
        return crop

    padded = image.new("RGB", target_size)
    paste_x = src_left - left
    paste_y = src_top - top
    edge_color = crop.getpixel((0, 0)) if crop.size[0] and crop.size[1] else (0, 0, 0)
    padded.paste(edge_color, (0, 0, target_size[0], target_size[1]))
    padded.paste(crop, (paste_x, paste_y))
    return padded


def main() -> int:
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
