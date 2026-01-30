"""
LuminaRetouch - Main Window
Professional Portrait Retouching & AI Enhancement Desktop App
"""

from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QGraphicsView,
    QGraphicsScene, QGraphicsPixmapItem, QSlider, QLabel, QPushButton,
    QFileDialog, QGroupBox, QComboBox, QProgressBar, QListWidget,
    QListWidgetItem, QSplitter, QScrollArea, QFrame, QToolBar,
    QStatusBar, QMessageBox, QApplication, QSizePolicy, QToolButton,
    QMenu, QInputDialog, QSpacerItem, QButtonGroup, QRadioButton
)
from PySide6.QtCore import Qt, Signal, Slot, QRectF, QPointF, QThread, QTimer, QSize
from PySide6.QtGui import (
    QPixmap, QImage, QPainter, QPen, QColor, QBrush, QAction,
    QWheelEvent, QMouseEvent, QKeySequence, QFont, QIcon, QLinearGradient,
    QFontDatabase
)
import numpy as np
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, fields
import time

from processors import ProcessingSettings, RetouchPipeline
from templates import TemplateManager, RetouchTemplate


# Professional color palette - refined and muted
COLORS = {
    'bg_darkest': '#0a0a0a',
    'bg_dark': '#121212',
    'bg_medium': '#1a1a1a',
    'bg_light': '#222222',
    'bg_elevated': '#2a2a2a',
    'accent': '#c9a227',        # Warm gold - professional, refined
    'accent_muted': '#a68521',
    'accent_subtle': 'rgba(201, 162, 39, 0.15)',
    'success': '#4a9f6e',
    'error': '#c45c5c',
    'text_primary': '#e8e8e8',
    'text_secondary': '#999999',
    'text_muted': '#666666',
    'text_dim': '#444444',
    'border': '#2a2a2a',
    'border_subtle': '#1f1f1f',
}

# Typography
FONTS = {
    'heading': 'SF Pro Display, Segoe UI, system-ui',
    'body': 'SF Pro Text, Segoe UI, system-ui',
    'mono': 'SF Mono, Consolas, monospace',
}


@dataclass
class ImageState:
    """Holds the current image state."""
    original: Optional[np.ndarray] = None
    processed: Optional[np.ndarray] = None
    original_pixmap: Optional[QPixmap] = None
    processed_pixmap: Optional[QPixmap] = None
    file_path: Optional[Path] = None


class SplitViewCanvas(QGraphicsView):
    """Split-view canvas for before/after comparison."""

    zoom_changed = Signal(float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)

        self._before_item: Optional[QGraphicsPixmapItem] = None
        self._after_item: Optional[QGraphicsPixmapItem] = None
        self._split_position = 0.5
        self._is_dragging_slider = False
        self._slider_handle_width = 40
        self._zoom_factor = 1.0
        self._is_panning = False
        self._last_pan_point = QPointF()

        self.setRenderHints(QPainter.RenderHint.Antialiasing | QPainter.RenderHint.SmoothPixmapTransform)
        self.setViewportUpdateMode(QGraphicsView.ViewportUpdateMode.FullViewportUpdate)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setDragMode(QGraphicsView.DragMode.NoDrag)
        self.setMouseTracking(True)
        self.setStyleSheet(f"border: none; background-color: {COLORS['bg_darkest']};")

    def set_images(self, before: QPixmap, after: Optional[QPixmap] = None):
        if self._before_item:
            self._scene.removeItem(self._before_item)
        if self._after_item:
            self._scene.removeItem(self._after_item)

        self._before_item = QGraphicsPixmapItem(before)
        self._scene.addItem(self._before_item)

        if after:
            self._after_item = QGraphicsPixmapItem(after)
            self._scene.addItem(self._after_item)
        else:
            self._after_item = None

        self._scene.setSceneRect(before.rect().toRectF())
        self.fit_in_view()

    def paintEvent(self, event):
        if not self._before_item or not self._after_item:
            super().paintEvent(event)
            return

        painter = QPainter(self.viewport())
        painter.setRenderHints(QPainter.RenderHint.Antialiasing | QPainter.RenderHint.SmoothPixmapTransform)

        transform = self.viewportTransform()
        scene_rect = self._scene.sceneRect()
        split_x = scene_rect.width() * self._split_position
        split_point = transform.map(QPointF(split_x, 0))
        viewport_split_x = split_point.x()

        # Draw before (left)
        painter.save()
        painter.setClipRect(0, 0, viewport_split_x, self.viewport().height())
        painter.setTransform(transform)
        painter.drawPixmap(self._before_item.pos().toPoint(), self._before_item.pixmap())
        painter.restore()

        # Draw after (right)
        painter.save()
        painter.setClipRect(viewport_split_x, 0, self.viewport().width() - viewport_split_x, self.viewport().height())
        painter.setTransform(transform)
        painter.drawPixmap(self._after_item.pos().toPoint(), self._after_item.pixmap())
        painter.restore()

        # Divider line
        painter.setPen(QPen(QColor(COLORS['accent']), 1))
        painter.drawLine(int(viewport_split_x), 0, int(viewport_split_x), self.viewport().height())

        # Handle
        handle_y = self.viewport().height() // 2
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QBrush(QColor(COLORS['bg_elevated'])))
        painter.drawRoundedRect(int(viewport_split_x) - 14, handle_y - 20, 28, 40, 4, 4)

        # Handle accent line
        painter.setPen(QPen(QColor(COLORS['accent']), 2))
        painter.drawLine(int(viewport_split_x), handle_y - 12, int(viewport_split_x), handle_y + 12)

        # Labels
        painter.setFont(QFont(FONTS['mono'], 9))
        painter.setPen(QPen(QColor(255, 255, 255, 120)))
        painter.drawText(12, 22, "ORIGINAL")
        painter.drawText(self.viewport().width() - 70, 22, "RESULT")

        painter.end()

    def drawBackground(self, painter: QPainter, rect: QRectF):
        painter.fillRect(rect, QColor(COLORS['bg_darkest']))

    def set_split_position(self, position: float):
        self._split_position = max(0.0, min(1.0, position))
        self.viewport().update()

    def _is_over_slider(self, pos: QPointF) -> bool:
        if not self._before_item:
            return False
        scene_rect = self._scene.sceneRect()
        split_x = scene_rect.width() * self._split_position
        split_point = self.viewportTransform().map(QPointF(split_x, 0))
        return abs(pos.x() - split_point.x()) < self._slider_handle_width

    def mousePressEvent(self, event: QMouseEvent):
        pos = event.position()
        if event.button() == Qt.MouseButton.LeftButton and self._after_item and self._is_over_slider(pos):
            self._is_dragging_slider = True
            self.setCursor(Qt.CursorShape.SplitHCursor)
            return
        if event.button() == Qt.MouseButton.MiddleButton:
            self._is_panning = True
            self._last_pan_point = pos
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent):
        pos = event.position()
        if self._is_dragging_slider:
            scene_pos = self.mapToScene(pos.toPoint())
            self.set_split_position(scene_pos.x() / self._scene.sceneRect().width())
            return
        if self._is_panning:
            delta = pos - self._last_pan_point
            self._last_pan_point = pos
            self.horizontalScrollBar().setValue(int(self.horizontalScrollBar().value() - delta.x()))
            self.verticalScrollBar().setValue(int(self.verticalScrollBar().value() - delta.y()))
            return
        if self._after_item and self._is_over_slider(pos):
            self.setCursor(Qt.CursorShape.SplitHCursor)
        else:
            self.setCursor(Qt.CursorShape.ArrowCursor)
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent):
        self._is_dragging_slider = False
        self._is_panning = False
        self.setCursor(Qt.CursorShape.ArrowCursor)
        super().mouseReleaseEvent(event)

    def wheelEvent(self, event: QWheelEvent):
        factor = 1.15 if event.angleDelta().y() > 0 else 1 / 1.15
        new_zoom = self._zoom_factor * factor
        if 0.1 <= new_zoom <= 10.0:
            self._zoom_factor = new_zoom
            self.scale(factor, factor)
            self.zoom_changed.emit(self._zoom_factor)

    def fit_in_view(self):
        if self._before_item:
            self.fitInView(self._scene.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)
            self._zoom_factor = self.transform().m11()
            self.zoom_changed.emit(self._zoom_factor)

    def reset_zoom(self):
        self.resetTransform()
        self._zoom_factor = 1.0
        self.zoom_changed.emit(self._zoom_factor)


class SliderControl(QWidget):
    """Refined slider with inline value display."""

    valueChanged = Signal(int)

    def __init__(self, label: str, min_val: int = 0, max_val: int = 100, default: int = 0, parent=None):
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 6, 0, 6)
        layout.setSpacing(12)

        self._label = QLabel(label)
        self._label.setFixedWidth(120)
        self._label.setStyleSheet(f"""
            color: {COLORS['text_secondary']};
            font-size: 11px;
            font-family: {FONTS['body']};
        """)
        layout.addWidget(self._label)

        self._slider = QSlider(Qt.Orientation.Horizontal)
        self._slider.setMinimum(min_val)
        self._slider.setMaximum(max_val)
        self._slider.setValue(default)
        self._slider.setStyleSheet(f"""
            QSlider::groove:horizontal {{
                height: 3px;
                background: {COLORS['bg_darkest']};
                border-radius: 1px;
            }}
            QSlider::handle:horizontal {{
                background: {COLORS['text_secondary']};
                width: 12px;
                height: 12px;
                margin: -5px 0;
                border-radius: 6px;
            }}
            QSlider::handle:horizontal:hover {{
                background: {COLORS['accent']};
            }}
            QSlider::sub-page:horizontal {{
                background: {COLORS['accent_muted']};
                border-radius: 1px;
            }}
        """)
        self._slider.valueChanged.connect(self._on_value_changed)
        layout.addWidget(self._slider, 1)

        self._value_label = QLabel(str(default))
        self._value_label.setFixedWidth(28)
        self._value_label.setAlignment(Qt.AlignmentFlag.AlignRight)
        self._value_label.setStyleSheet(f"""
            color: {COLORS['text_muted']};
            font-size: 10px;
            font-family: {FONTS['mono']};
        """)
        layout.addWidget(self._value_label)

    def _on_value_changed(self, value: int):
        self._value_label.setText(str(value))
        if value > 0:
            self._value_label.setStyleSheet(f"color: {COLORS['accent']}; font-size: 10px; font-family: {FONTS['mono']};")
        else:
            self._value_label.setStyleSheet(f"color: {COLORS['text_muted']}; font-size: 10px; font-family: {FONTS['mono']};")
        self.valueChanged.emit(value)

    def value(self) -> int:
        return self._slider.value()

    def setValue(self, value: int):
        self._slider.setValue(value)


class SectionHeader(QWidget):
    """Collapsible section header."""

    def __init__(self, title: str, parent=None):
        super().__init__(parent)
        self._is_collapsed = False
        self._title = title

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Header button
        self._header = QPushButton(title.upper())
        self._header.setStyleSheet(f"""
            QPushButton {{
                background: transparent;
                border: none;
                border-bottom: 1px solid {COLORS['border_subtle']};
                padding: 12px 0 8px 0;
                text-align: left;
                font-size: 10px;
                font-weight: 600;
                font-family: {FONTS['body']};
                color: {COLORS['text_muted']};
                letter-spacing: 1.5px;
            }}
            QPushButton:hover {{
                color: {COLORS['text_secondary']};
            }}
        """)
        self._header.clicked.connect(self._toggle)
        layout.addWidget(self._header)

        # Content container
        self._content = QWidget()
        self._content_layout = QVBoxLayout(self._content)
        self._content_layout.setContentsMargins(0, 8, 0, 16)
        self._content_layout.setSpacing(0)
        layout.addWidget(self._content)

    def _toggle(self):
        self._is_collapsed = not self._is_collapsed
        self._content.setVisible(not self._is_collapsed)

    def add_widget(self, widget):
        self._content_layout.addWidget(widget)

    def add_slider(self, label: str, min_val: int = 0, max_val: int = 100, default: int = 0) -> SliderControl:
        slider = SliderControl(label, min_val, max_val, default)
        self._content_layout.addWidget(slider)
        return slider


class ProgressIndicator(QWidget):
    """Minimal progress indicator."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setVisible(False)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 16, 0, 16)
        layout.setSpacing(8)

        # Status text
        self._status = QLabel("Processing...")
        self._status.setStyleSheet(f"""
            color: {COLORS['text_secondary']};
            font-size: 11px;
            font-family: {FONTS['body']};
        """)
        layout.addWidget(self._status)

        # Progress bar
        self._progress = QProgressBar()
        self._progress.setTextVisible(False)
        self._progress.setFixedHeight(2)
        self._progress.setStyleSheet(f"""
            QProgressBar {{
                border: none;
                background: {COLORS['bg_darkest']};
            }}
            QProgressBar::chunk {{
                background: {COLORS['accent']};
            }}
        """)
        layout.addWidget(self._progress)

        # Time elapsed
        self._time = QLabel("")
        self._time.setStyleSheet(f"""
            color: {COLORS['text_dim']};
            font-size: 10px;
            font-family: {FONTS['mono']};
        """)
        layout.addWidget(self._time)

        self._timer = QTimer(self)
        self._timer.timeout.connect(self._update_time)
        self._start_time = 0

    def start(self):
        self._start_time = time.time()
        self._timer.start(100)
        self._progress.setValue(0)
        self.setVisible(True)

    def update(self, message: str, percent: int):
        self._status.setText(message)
        self._progress.setValue(percent)

    def _update_time(self):
        elapsed = time.time() - self._start_time
        self._time.setText(f"{elapsed:.1f}s")

    def finish(self, success: bool = True):
        self._timer.stop()
        elapsed = time.time() - self._start_time
        if success:
            self._status.setText("Complete")
            self._status.setStyleSheet(f"color: {COLORS['success']}; font-size: 11px;")
        else:
            self._status.setText("Failed")
            self._status.setStyleSheet(f"color: {COLORS['error']}; font-size: 11px;")
        self._time.setText(f"{elapsed:.1f}s")
        self._progress.setValue(100 if success else 0)
        QTimer.singleShot(2500, lambda: self.setVisible(False))


class ControlPanel(QScrollArea):
    """Professional control panel."""

    settings_changed = Signal()
    process_requested = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWidgetResizable(True)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setFixedWidth(340)
        self.setStyleSheet(f"""
            QScrollArea {{
                background: {COLORS['bg_dark']};
                border: none;
                border-right: 1px solid {COLORS['border_subtle']};
            }}
            QScrollBar:vertical {{
                background: transparent;
                width: 6px;
            }}
            QScrollBar::handle:vertical {{
                background: {COLORS['border']};
                border-radius: 3px;
                min-height: 40px;
            }}
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0; }}
        """)

        self._sliders = {}
        self._template_manager = TemplateManager()
        self._setup_ui()

    def _setup_ui(self):
        container = QWidget()
        container.setStyleSheet(f"background: {COLORS['bg_dark']};")
        layout = QVBoxLayout(container)
        layout.setContentsMargins(20, 24, 20, 24)
        layout.setSpacing(0)

        # Logo
        logo = QLabel("LUMINA")
        logo.setStyleSheet(f"""
            font-size: 13px;
            font-weight: 700;
            font-family: {FONTS['heading']};
            color: {COLORS['text_primary']};
            letter-spacing: 4px;
        """)
        layout.addWidget(logo)

        tagline = QLabel("Portrait Retouching")
        tagline.setStyleSheet(f"""
            font-size: 10px;
            font-family: {FONTS['body']};
            color: {COLORS['text_dim']};
            margin-bottom: 24px;
        """)
        layout.addWidget(tagline)

        # Processing Mode Selection
        mode_section = SectionHeader("Processing Mode")

        mode_container = QWidget()
        mode_layout = QHBoxLayout(mode_container)
        mode_layout.setContentsMargins(0, 4, 0, 8)
        mode_layout.setSpacing(8)

        self._mode_group = QButtonGroup(self)

        self._fast_mode = QRadioButton("Fast")
        self._fast_mode.setChecked(True)
        self._fast_mode.setStyleSheet(self._get_radio_style())
        self._mode_group.addButton(self._fast_mode, 0)
        mode_layout.addWidget(self._fast_mode)

        self._quality_mode = QRadioButton("Quality")
        self._quality_mode.setStyleSheet(self._get_radio_style())
        self._mode_group.addButton(self._quality_mode, 1)
        mode_layout.addWidget(self._quality_mode)

        mode_layout.addStretch()
        mode_section.add_widget(mode_container)

        mode_hint = QLabel("Quality mode uses AI enhancement (requires setup)")
        mode_hint.setStyleSheet(f"""
            color: {COLORS['text_dim']};
            font-size: 9px;
            font-family: {FONTS['body']};
            padding-bottom: 8px;
        """)
        mode_section.add_widget(mode_hint)

        layout.addWidget(mode_section)

        # Templates
        template_section = SectionHeader("Templates")

        self.template_combo = QComboBox()
        self.template_combo.setStyleSheet(f"""
            QComboBox {{
                background: {COLORS['bg_medium']};
                border: 1px solid {COLORS['border']};
                border-radius: 4px;
                padding: 8px 12px;
                color: {COLORS['text_secondary']};
                font-size: 11px;
                font-family: {FONTS['body']};
            }}
            QComboBox:hover {{ border-color: {COLORS['text_dim']}; }}
            QComboBox::drop-down {{ border: none; width: 20px; }}
            QComboBox QAbstractItemView {{
                background: {COLORS['bg_medium']};
                border: 1px solid {COLORS['border']};
                selection-background-color: {COLORS['accent_subtle']};
                color: {COLORS['text_secondary']};
            }}
        """)
        self._populate_templates()
        self.template_combo.currentTextChanged.connect(self._on_template_selected)
        template_section.add_widget(self.template_combo)

        # Template buttons
        btn_row = QWidget()
        btn_layout = QHBoxLayout(btn_row)
        btn_layout.setContentsMargins(0, 8, 0, 0)
        btn_layout.setSpacing(8)

        save_btn = QPushButton("Save")
        save_btn.setStyleSheet(self._get_secondary_btn_style())
        save_btn.clicked.connect(self._save_template)
        btn_layout.addWidget(save_btn)

        reset_btn = QPushButton("Reset")
        reset_btn.setStyleSheet(self._get_secondary_btn_style())
        reset_btn.clicked.connect(self._reset_sliders)
        btn_layout.addWidget(reset_btn)

        btn_layout.addStretch()
        template_section.add_widget(btn_row)

        layout.addWidget(template_section)

        # Skin
        skin_section = SectionHeader("Skin")
        self._sliders['skin_smoothness'] = skin_section.add_slider("Smoothing", 0, 100, 0)
        self._sliders['blemish_removal'] = skin_section.add_slider("Blemish Removal", 0, 100, 0)
        layout.addWidget(skin_section)

        # Eyes
        eyes_section = SectionHeader("Eyes")
        self._sliders['eye_brightness'] = eyes_section.add_slider("Brightness", 0, 100, 0)
        self._sliders['eye_size'] = eyes_section.add_slider("Enlargement", 0, 100, 0)
        self._sliders['dark_circle_removal'] = eyes_section.add_slider("Dark Circles", 0, 100, 0)
        layout.addWidget(eyes_section)

        # Mouth
        mouth_section = SectionHeader("Mouth")
        self._sliders['teeth_whitening'] = mouth_section.add_slider("Teeth Whitening", 0, 100, 0)
        self._sliders['lip_saturation'] = mouth_section.add_slider("Lip Color", 0, 100, 0)
        self._sliders['smile_enhancement'] = mouth_section.add_slider("Smile", 0, 100, 0)
        layout.addWidget(mouth_section)

        # Face Shape
        shape_section = SectionHeader("Face Shape")
        self._sliders['face_slimming'] = shape_section.add_slider("Slimming", 0, 100, 0)
        self._sliders['nose_slimming'] = shape_section.add_slider("Nose", 0, 100, 0)
        self._sliders['chin_adjustment'] = shape_section.add_slider("Chin", 0, 100, 0)
        self._sliders['jawline_sharpen'] = shape_section.add_slider("Jawline", 0, 100, 0)
        layout.addWidget(shape_section)

        # Enhancement
        enhance_section = SectionHeader("Enhancement")
        self._sliders['face_enhancement'] = enhance_section.add_slider("Detail", 0, 100, 0)
        layout.addWidget(enhance_section)

        # Output
        output_section = SectionHeader("Output")

        upscale_row = QWidget()
        upscale_layout = QHBoxLayout(upscale_row)
        upscale_layout.setContentsMargins(0, 4, 0, 0)
        upscale_layout.setSpacing(12)

        upscale_label = QLabel("Upscale")
        upscale_label.setFixedWidth(120)
        upscale_label.setStyleSheet(f"color: {COLORS['text_secondary']}; font-size: 11px;")
        upscale_layout.addWidget(upscale_label)

        self.upscale_combo = QComboBox()
        self.upscale_combo.addItems(["None", "2x", "4x"])
        self.upscale_combo.setStyleSheet(f"""
            QComboBox {{
                background: {COLORS['bg_medium']};
                border: 1px solid {COLORS['border']};
                border-radius: 4px;
                padding: 6px 10px;
                color: {COLORS['text_secondary']};
                font-size: 11px;
                min-width: 60px;
            }}
            QComboBox:hover {{ border-color: {COLORS['text_dim']}; }}
            QComboBox::drop-down {{ border: none; }}
        """)
        upscale_layout.addWidget(self.upscale_combo)
        upscale_layout.addStretch()

        output_section.add_widget(upscale_row)
        layout.addWidget(output_section)

        # Progress
        self.progress = ProgressIndicator()
        layout.addWidget(self.progress)

        layout.addStretch()

        # Process button
        self.process_btn = QPushButton("Process")
        self.process_btn.setMinimumHeight(44)
        self.process_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.process_btn.setStyleSheet(f"""
            QPushButton {{
                background: {COLORS['accent']};
                border: none;
                border-radius: 4px;
                font-size: 12px;
                font-weight: 600;
                font-family: {FONTS['body']};
                color: {COLORS['bg_darkest']};
                letter-spacing: 0.5px;
            }}
            QPushButton:hover {{
                background: {COLORS['accent_muted']};
            }}
            QPushButton:disabled {{
                background: {COLORS['bg_elevated']};
                color: {COLORS['text_dim']};
            }}
        """)
        self.process_btn.clicked.connect(self.process_requested.emit)
        layout.addWidget(self.process_btn)

        # Connect sliders
        for slider in self._sliders.values():
            slider.valueChanged.connect(lambda _: self.settings_changed.emit())
        self.upscale_combo.currentIndexChanged.connect(lambda _: self.settings_changed.emit())

        self.setWidget(container)

    def _get_radio_style(self):
        return f"""
            QRadioButton {{
                color: {COLORS['text_secondary']};
                font-size: 11px;
                font-family: {FONTS['body']};
                spacing: 6px;
            }}
            QRadioButton::indicator {{
                width: 14px;
                height: 14px;
                border-radius: 7px;
                border: 1px solid {COLORS['text_dim']};
                background: transparent;
            }}
            QRadioButton::indicator:checked {{
                background: {COLORS['accent']};
                border-color: {COLORS['accent']};
            }}
            QRadioButton::indicator:hover {{
                border-color: {COLORS['text_muted']};
            }}
        """

    def _get_secondary_btn_style(self):
        return f"""
            QPushButton {{
                background: transparent;
                border: 1px solid {COLORS['border']};
                border-radius: 4px;
                padding: 6px 14px;
                color: {COLORS['text_muted']};
                font-size: 10px;
                font-family: {FONTS['body']};
            }}
            QPushButton:hover {{
                border-color: {COLORS['text_dim']};
                color: {COLORS['text_secondary']};
            }}
        """

    def _populate_templates(self):
        self.template_combo.clear()
        self.template_combo.addItem("Select template...")

        for category in self._template_manager.get_categories():
            self.template_combo.addItem(f"— {category} —")
            idx = self.template_combo.count() - 1
            self.template_combo.model().item(idx).setEnabled(False)

            for template in self._template_manager.get_templates_by_category(category):
                self.template_combo.addItem(f"  {template.name}")

    def _on_template_selected(self, text: str):
        name = text.strip()
        template = self._template_manager.get_template(name)
        if template:
            self._apply_template(template)

    def _apply_template(self, template: RetouchTemplate):
        mapping = {
            'face_enhancement': 'face_enhancement',
            'skin_smoothness': 'skin_smoothness',
            'blemish_removal': 'blemish_removal',
            'eye_brightness': 'eye_brightness',
            'eye_size': 'eye_size',
            'dark_circle_removal': 'dark_circle_removal',
            'teeth_whitening': 'teeth_whitening',
            'lip_saturation': 'lip_saturation',
            'smile_enhancement': 'smile_enhancement',
            'face_slimming': 'face_slimming',
            'nose_slimming': 'nose_slimming',
            'chin_adjustment': 'chin_adjustment',
            'jawline_sharpen': 'jawline_sharpen',
        }
        for slider_key, template_key in mapping.items():
            if slider_key in self._sliders:
                value = getattr(template, template_key, 0)
                self._sliders[slider_key].setValue(int(value * 100))

    def _save_template(self):
        name, ok = QInputDialog.getText(self, "Save Template", "Name:")
        if ok and name:
            settings = self.get_settings()
            template = RetouchTemplate(
                name=name, category="Custom", description="",
                face_enhancement=settings.face_enhancement,
                skin_smoothness=settings.skin_smoothness,
                blemish_removal=settings.blemish_removal,
                eye_brightness=settings.eye_brightness,
                eye_size=settings.eye_size,
                dark_circle_removal=settings.dark_circle_removal,
                teeth_whitening=settings.teeth_whitening,
                lip_saturation=settings.lip_saturation,
                smile_enhancement=settings.smile_enhancement,
                face_slimming=settings.face_slimming,
                nose_slimming=settings.nose_slimming,
                chin_adjustment=settings.chin_adjustment,
                jawline_sharpen=settings.jawline_sharpen,
            )
            self._template_manager.save_custom_template(template)
            self._populate_templates()

    def _reset_sliders(self):
        for slider in self._sliders.values():
            slider.setValue(0)
        self.upscale_combo.setCurrentIndex(0)

    def get_settings(self) -> ProcessingSettings:
        upscale_map = {"None": 1, "2x": 2, "4x": 4}
        return ProcessingSettings(
            face_enhancement=self._sliders['face_enhancement'].value() / 100.0,
            skin_smoothness=self._sliders['skin_smoothness'].value() / 100.0,
            blemish_removal=self._sliders['blemish_removal'].value() / 100.0,
            eye_brightness=self._sliders['eye_brightness'].value() / 100.0,
            eye_size=self._sliders['eye_size'].value() / 100.0,
            dark_circle_removal=self._sliders['dark_circle_removal'].value() / 100.0,
            teeth_whitening=self._sliders['teeth_whitening'].value() / 100.0,
            lip_saturation=self._sliders['lip_saturation'].value() / 100.0,
            smile_enhancement=self._sliders['smile_enhancement'].value() / 100.0,
            face_slimming=self._sliders['face_slimming'].value() / 100.0,
            nose_slimming=self._sliders['nose_slimming'].value() / 100.0,
            chin_adjustment=self._sliders['chin_adjustment'].value() / 100.0,
            jawline_sharpen=self._sliders['jawline_sharpen'].value() / 100.0,
            upscale_factor=upscale_map.get(self.upscale_combo.currentText(), 1),
        )

    def use_quality_mode(self) -> bool:
        return self._quality_mode.isChecked()

    def set_processing(self, is_processing: bool):
        self.process_btn.setEnabled(not is_processing)
        self.process_btn.setText("Processing..." if is_processing else "Process")


class ProcessingWorker(QThread):
    progress = Signal(str, int)
    finished = Signal(np.ndarray)
    error = Signal(str)

    def __init__(self, image: np.ndarray, settings: ProcessingSettings, use_ai: bool = False):
        super().__init__()
        self.image = image
        self.settings = settings
        self.use_ai = use_ai
        self.pipeline = RetouchPipeline()

    def run(self):
        try:
            result = self.pipeline.process(
                self.image, self.settings,
                progress_callback=lambda msg, pct: self.progress.emit(msg, pct),
                use_ai=self.use_ai
            )
            self.finished.emit(result)
        except Exception as e:
            self.error.emit(str(e))


class SaveWorker(QThread):
    progress = Signal(str, int)
    finished = Signal(str)
    error = Signal(str)

    def __init__(self, pixmap: QPixmap, file_path: str):
        super().__init__()
        self.pixmap = pixmap
        self.file_path = file_path

    def run(self):
        try:
            self.progress.emit("Saving...", 50)
            self.pixmap.save(self.file_path)
            self.finished.emit(self.file_path)
        except Exception as e:
            self.error.emit(str(e))


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self._image_state = ImageState()
        self._worker: Optional[ProcessingWorker] = None
        self._save_worker: Optional[SaveWorker] = None

        self.setWindowTitle("Lumina Retouch")
        self.setWindowIcon(QIcon('assets/icon.png')) 
        self.setMinimumSize(1400, 900)
        self._apply_theme()
        self._setup_ui()
        self._setup_toolbar()
        self._setup_statusbar()

    def _apply_theme(self):
        self.setStyleSheet(f"""
            QMainWindow {{
                background: {COLORS['bg_darkest']};
            }}
            QWidget {{
                color: {COLORS['text_primary']};
                font-family: {FONTS['body']};
            }}
            QToolTip {{
                background: {COLORS['bg_elevated']};
                color: {COLORS['text_primary']};
                border: 1px solid {COLORS['border']};
                padding: 4px 8px;
                font-size: 11px;
            }}
        """)

    def _setup_ui(self):
        central = QWidget()
        self.setCentralWidget(central)

        layout = QHBoxLayout(central)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self.control_panel = ControlPanel()
        self.control_panel.process_requested.connect(self._process_image)

        self.canvas = SplitViewCanvas()
        self.canvas.zoom_changed.connect(self._on_zoom_changed)

        layout.addWidget(self.control_panel)
        layout.addWidget(self.canvas, 1)

    def _setup_toolbar(self):
        toolbar = QToolBar()
        toolbar.setMovable(False)
        toolbar.setStyleSheet(f"""
            QToolBar {{
                background: {COLORS['bg_dark']};
                border: none;
                border-bottom: 1px solid {COLORS['border_subtle']};
                padding: 6px 16px;
                spacing: 2px;
            }}
            QToolButton {{
                background: transparent;
                border: none;
                border-radius: 4px;
                padding: 8px 14px;
                color: {COLORS['text_muted']};
                font-size: 11px;
                font-family: {FONTS['body']};
            }}
            QToolButton:hover {{
                background: {COLORS['bg_elevated']};
                color: {COLORS['text_secondary']};
            }}
        """)
        self.addToolBar(toolbar)

        open_action = QAction("Open", self)
        open_action.setShortcut(QKeySequence.StandardKey.Open)
        open_action.triggered.connect(self._open_image)
        toolbar.addAction(open_action)

        save_action = QAction("Save", self)
        save_action.setShortcut(QKeySequence.StandardKey.Save)
        save_action.triggered.connect(self._save_image)
        toolbar.addAction(save_action)

        toolbar.addSeparator()

        fit_action = QAction("Fit", self)
        fit_action.triggered.connect(self.canvas.fit_in_view)
        toolbar.addAction(fit_action)

        actual_action = QAction("100%", self)
        actual_action.triggered.connect(self.canvas.reset_zoom)
        toolbar.addAction(actual_action)

        toolbar.addSeparator()

        compare_label = QLabel(" Compare ")
        compare_label.setStyleSheet(f"color: {COLORS['text_dim']}; font-size: 11px;")
        toolbar.addWidget(compare_label)

        self.split_slider = QSlider(Qt.Orientation.Horizontal)
        self.split_slider.setFixedWidth(120)
        self.split_slider.setRange(0, 100)
        self.split_slider.setValue(50)
        self.split_slider.setStyleSheet(f"""
            QSlider::groove:horizontal {{ height: 2px; background: {COLORS['border']}; }}
            QSlider::handle:horizontal {{
                background: {COLORS['text_muted']};
                width: 10px; height: 10px;
                margin: -4px 0;
                border-radius: 5px;
            }}
            QSlider::handle:horizontal:hover {{ background: {COLORS['accent']}; }}
        """)
        self.split_slider.valueChanged.connect(lambda v: self.canvas.set_split_position(v / 100.0))
        toolbar.addWidget(self.split_slider)

        spacer = QWidget()
        spacer.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        toolbar.addWidget(spacer)

        self.zoom_label = QLabel("100%")
        self.zoom_label.setStyleSheet(f"color: {COLORS['text_dim']}; font-size: 10px; font-family: {FONTS['mono']};")
        toolbar.addWidget(self.zoom_label)

    def _setup_statusbar(self):
        self.statusbar = QStatusBar()
        self.statusbar.setStyleSheet(f"""
            QStatusBar {{
                background: {COLORS['bg_dark']};
                border-top: 1px solid {COLORS['border_subtle']};
                padding: 4px 16px;
                font-size: 10px;
                color: {COLORS['text_dim']};
            }}
        """)
        self.setStatusBar(self.statusbar)

        self.status_text = QLabel("Ready")
        self.statusbar.addWidget(self.status_text, 1)

        self.image_info = QLabel("")
        self.image_info.setStyleSheet(f"color: {COLORS['text_dim']}; font-family: {FONTS['mono']};")
        self.statusbar.addPermanentWidget(self.image_info)

    def _on_zoom_changed(self, zoom: float):
        self.zoom_label.setText(f"{int(zoom * 100)}%")

    def _open_image(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Image", "", "Images (*.png *.jpg *.jpeg *.bmp *.tiff *.webp)"
        )
        if path:
            self._load_image(path)

    def _load_image(self, path: str):
        pixmap = QPixmap(path)
        if pixmap.isNull():
            QMessageBox.warning(self, "Error", f"Failed to load: {path}")
            return

        image = pixmap.toImage().convertToFormat(QImage.Format.Format_RGB888)
        w, h = image.width(), image.height()
        ptr = image.bits()
        arr = np.array(ptr).reshape((h, image.bytesPerLine()))
        arr = arr[:, :w * 3].reshape((h, w, 3)).copy()

        self._image_state = ImageState(
            original=arr, original_pixmap=pixmap, file_path=Path(path)
        )
        self.canvas.set_images(pixmap)
        self.image_info.setText(f"{Path(path).name}  {w}×{h}")
        self.status_text.setText("Image loaded")

    def _save_image(self):
        if self._image_state.processed_pixmap is None:
            QMessageBox.warning(self, "Warning", "No processed image to save.")
            return

        default = (self._image_state.file_path.stem + "_edit.png"
                   if self._image_state.file_path else "edit.png")

        path, _ = QFileDialog.getSaveFileName(
            self, "Save Image", default, "PNG (*.png);;JPEG (*.jpg)"
        )
        if path:
            self.status_text.setText("Saving...")
            self._save_worker = SaveWorker(self._image_state.processed_pixmap, path)
            self._save_worker.finished.connect(self._on_save_finished)
            self._save_worker.error.connect(self._on_save_error)
            self._save_worker.start()

    def _on_save_finished(self, path: str):
        self.status_text.setText(f"Saved: {Path(path).name}")

    def _on_save_error(self, msg: str):
        self.status_text.setText("Save failed")
        QMessageBox.critical(self, "Error", msg)

    def _process_image(self):
        if self._image_state.original is None:
            QMessageBox.warning(self, "Warning", "No image loaded.")
            return

        settings = self.control_panel.get_settings()
        if settings.is_empty() and settings.upscale_factor == 1:
            QMessageBox.information(self, "Info", "Adjust settings first.")
            return

        self.control_panel.set_processing(True)
        self.control_panel.progress.start()
        self.status_text.setText("Processing...")

        use_ai = self.control_panel.use_quality_mode()
        self._worker = ProcessingWorker(self._image_state.original.copy(), settings, use_ai)
        self._worker.progress.connect(self._on_progress)
        self._worker.finished.connect(self._on_finished)
        self._worker.error.connect(self._on_error)
        self._worker.start()

    def _on_progress(self, msg: str, pct: int):
        self.control_panel.progress.update(msg, pct)

    def _on_finished(self, result: np.ndarray):
        self.control_panel.set_processing(False)
        self.control_panel.progress.finish(True)

        h, w, c = result.shape
        qimg = QImage(result.data.tobytes(), w, h, c * w, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg)

        self._image_state.processed = result
        self._image_state.processed_pixmap = pixmap
        self.canvas.set_images(self._image_state.original_pixmap, pixmap)
        self.status_text.setText("Complete")

        if self._image_state.file_path:
            self.image_info.setText(f"{self._image_state.file_path.name}  {w}×{h}")

    def _on_error(self, msg: str):
        self.control_panel.set_processing(False)
        self.control_panel.progress.finish(False)
        self.status_text.setText("Failed")
        QMessageBox.critical(self, "Error", msg)


def main():
    import sys
    app = QApplication(sys.argv)

    # Set default font with fallback
    font = QFont()
    font.setFamily("Segoe UI")
    font.setPointSize(10)
    app.setFont(font)

    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
