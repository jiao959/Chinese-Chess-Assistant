from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import mss
from PIL import Image
from PySide6.QtCore import QPoint, QRect, Qt, Signal
from PySide6.QtGui import QColor, QGuiApplication, QPainter, QPen
from PySide6.QtWidgets import QApplication, QWidget


@dataclass(frozen=True)
class ScreenRegion:
    x: int
    y: int
    width: int
    height: int

    def normalized(self) -> "ScreenRegion":
        x = self.x
        y = self.y
        width = self.width
        height = self.height
        if width < 0:
            x += width
            width = abs(width)
        if height < 0:
            y += height
            height = abs(height)
        return ScreenRegion(x=x, y=y, width=width, height=height)

    def to_dict(self) -> dict[str, int]:
        return {"x": self.x, "y": self.y, "width": self.width, "height": self.height}

    @staticmethod
    def from_dict(data: dict[str, int] | None) -> "ScreenRegion | None":
        if not data:
            return None
        try:
            region = ScreenRegion(
                x=int(data["x"]),
                y=int(data["y"]),
                width=int(data["width"]),
                height=int(data["height"]),
            ).normalized()
        except (KeyError, TypeError, ValueError):
            return None
        if region.width <= 0 or region.height <= 0:
            return None
        return region


class ScreenCapture:
    def capture_full_screen(self) -> tuple[Image.Image, ScreenRegion]:
        with mss.mss() as sct:
            monitor = sct.monitors[0]
            raw = sct.grab(monitor)
            image = Image.frombytes("RGB", raw.size, raw.rgb)
            region = ScreenRegion(
                x=int(monitor["left"]),
                y=int(monitor["top"]),
                width=int(monitor["width"]),
                height=int(monitor["height"]),
            )
            return image, region

    def capture_region(self, region: ScreenRegion) -> Image.Image:
        region = region.normalized()
        if region.width <= 0 or region.height <= 0:
            raise ValueError("截图区域无效。")

        with mss.mss() as sct:
            raw = sct.grab(
                {
                    "left": region.x,
                    "top": region.y,
                    "width": region.width,
                    "height": region.height,
                }
            )
            return Image.frombytes("RGB", raw.size, raw.rgb)


class RegionSelectionOverlay(QWidget):
    region_selected = Signal(object)
    selection_cancelled = Signal()

    def __init__(self) -> None:
        super().__init__(None)
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.Tool
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setCursor(Qt.CursorShape.CrossCursor)
        self._start: QPoint | None = None
        self._end: QPoint | None = None

        geometry = self._virtual_desktop_geometry()
        self.setGeometry(geometry)

    def paintEvent(self, _event) -> None:  # type: ignore[override]
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor(0, 0, 0, 90))

        if self._start is not None and self._end is not None:
            rect = QRect(self._start, self._end).normalized()
            painter.fillRect(rect, QColor(255, 255, 255, 30))
            pen = QPen(QColor(255, 80, 80), 2)
            painter.setPen(pen)
            painter.drawRect(rect)

    def mousePressEvent(self, event) -> None:  # type: ignore[override]
        if event.button() != Qt.MouseButton.LeftButton:
            return
        self._start = event.position().toPoint()
        self._end = self._start
        self.update()

    def mouseMoveEvent(self, event) -> None:  # type: ignore[override]
        if self._start is None:
            return
        self._end = event.position().toPoint()
        self.update()

    def mouseReleaseEvent(self, event) -> None:  # type: ignore[override]
        if event.button() != Qt.MouseButton.LeftButton or self._start is None:
            return
        self._end = event.position().toPoint()
        rect = QRect(self._start, self._end).normalized()
        self.hide()

        if rect.width() < 20 or rect.height() < 20:
            self.selection_cancelled.emit()
            self.close()
            return

        top_left = self.mapToGlobal(rect.topLeft())
        region = ScreenRegion(
            x=top_left.x(),
            y=top_left.y(),
            width=rect.width(),
            height=rect.height(),
        ).normalized()
        self.region_selected.emit(region)
        self.close()

    def keyPressEvent(self, event) -> None:  # type: ignore[override]
        if event.key() == Qt.Key.Key_Escape:
            self.selection_cancelled.emit()
            self.close()

    @staticmethod
    def _virtual_desktop_geometry() -> QRect:
        screens = QGuiApplication.screens()
        if not screens:
            return QRect(0, 0, 1024, 768)
        geometry = screens[0].geometry()
        for screen in screens[1:]:
            geometry = geometry.united(screen.geometry())
        return geometry


def select_board_region(on_selected: Callable[[ScreenRegion], None], on_cancelled: Callable[[], None]) -> None:
    overlay = RegionSelectionOverlay()
    overlay.region_selected.connect(on_selected)
    overlay.selection_cancelled.connect(on_cancelled)
    overlay.show()
    overlay.activateWindow()

    app = QApplication.instance()
    if app is not None:
        setattr(app, "_active_region_overlay", overlay)
