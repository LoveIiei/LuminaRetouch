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
    QMenu, QInputDialog
)
from PySide6.QtCore import Qt, Signal, Slot, QRectF, QPointF, QThread, QTimer
from PySide6.QtGui import (
    QPixmap, QImage, QPainter, QPen, QColor, QBrush, QAction,
    QWheelEvent, QMouseEvent, QKeySequence
)
import numpy as np
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, fields

from processors import ProcessingSettings, RetouchPipeline
from templates import TemplateManager, RetouchTemplate


@dataclass
class ImageState:
    """Holds the current image state."""
    original: Optional[np.ndarray] = None
    processed: Optional[np.ndarray] = None
    original_pixmap: Optional[QPixmap] = None
    processed_pixmap: Optional[QPixmap] = None
    file_path: Optional[Path] = None


class SplitViewCanvas(QGraphicsView):
    """Custom QGraphicsView with real-time split-view slider."""

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

        # Configure view
        self.setRenderHints(QPainter.RenderHint.Antialiasing | QPainter.RenderHint.SmoothPixmapTransform)
        self.setViewportUpdateMode(QGraphicsView.ViewportUpdateMode.FullViewportUpdate)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setDragMode(QGraphicsView.DragMode.NoDrag)
        self.setMouseTracking(True)

    def set_images(self, before: QPixmap, after: Optional[QPixmap] = None):
        """Set before/after images."""
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
        """Custom paint for split-view."""
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

        # Draw slider line
        pen = QPen(QColor(255, 255, 255, 220))
        pen.setWidth(2)
        painter.setPen(pen)
        painter.drawLine(int(viewport_split_x), 0, int(viewport_split_x), self.viewport().height())

        # Draw handle
        handle_y = self.viewport().height() // 2
        painter.setBrush(QBrush(QColor(60, 60, 60, 200)))
        painter.drawEllipse(QPointF(viewport_split_x, handle_y), 15, 15)

        # Arrows
        painter.setPen(QPen(QColor(255, 255, 255), 2))
        painter.drawText(int(viewport_split_x) - 8, handle_y + 5, "<>")

        painter.end()

    def drawBackground(self, painter: QPainter, rect: QRectF):
        painter.fillRect(rect, QColor(30, 30, 30))

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


class CollapsibleSection(QWidget):
    """Collapsible section widget."""

    def __init__(self, title: str, parent=None):
        super().__init__(parent)
        self._is_collapsed = False

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Header
        self._header = QPushButton(f"▼ {title}")
        self._header.setStyleSheet("""
            QPushButton {
                background-color: #2a2a2a;
                border: none;
                border-radius: 4px;
                padding: 8px;
                text-align: left;
                font-weight: bold;
            }
            QPushButton:hover { background-color: #353535; }
        """)
        self._header.clicked.connect(self._toggle)
        layout.addWidget(self._header)

        # Content
        self._content = QWidget()
        self._content_layout = QVBoxLayout(self._content)
        self._content_layout.setContentsMargins(5, 5, 5, 5)
        self._content_layout.setSpacing(8)
        layout.addWidget(self._content)

        self._title = title

    def _toggle(self):
        self._is_collapsed = not self._is_collapsed
        self._content.setVisible(not self._is_collapsed)
        arrow = "▶" if self._is_collapsed else "▼"
        self._header.setText(f"{arrow} {self._title}")

    def add_widget(self, widget):
        self._content_layout.addWidget(widget)

    def add_slider(self, label: str, min_val: int = 0, max_val: int = 100,
                   default: int = 0) -> QSlider:
        """Add a labeled slider."""
        lbl = QLabel(label)
        self._content_layout.addWidget(lbl)

        slider = QSlider(Qt.Orientation.Horizontal)
        slider.setMinimum(min_val)
        slider.setMaximum(max_val)
        slider.setValue(default)
        slider.setStyleSheet("""
            QSlider::groove:horizontal { height: 6px; background: #404040; border-radius: 3px; }
            QSlider::handle:horizontal { background: #4a9eff; width: 14px; margin: -4px 0; border-radius: 7px; }
            QSlider::sub-page:horizontal { background: #4a9eff; border-radius: 3px; }
        """)
        self._content_layout.addWidget(slider)
        return slider


class ControlPanel(QScrollArea):
    """Sidebar control panel with all retouching sliders."""

    settings_changed = Signal()
    process_requested = Signal()
    template_selected = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWidgetResizable(True)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setFixedWidth(300)

        self._sliders = {}
        self._template_manager = TemplateManager()
        self._setup_ui()

    def _setup_ui(self):
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)

        # Templates Section
        template_section = CollapsibleSection("Templates")

        self.template_combo = QComboBox()
        self._populate_templates()
        self.template_combo.currentTextChanged.connect(self._on_template_selected)
        template_section.add_widget(self.template_combo)

        btn_layout = QHBoxLayout()
        save_btn = QPushButton("Save Current")
        save_btn.clicked.connect(self._save_template)
        btn_layout.addWidget(save_btn)

        reset_btn = QPushButton("Reset All")
        reset_btn.clicked.connect(self._reset_sliders)
        btn_layout.addWidget(reset_btn)

        btn_widget = QWidget()
        btn_widget.setLayout(btn_layout)
        template_section.add_widget(btn_widget)
        layout.addWidget(template_section)

        # Skin Section
        skin_section = CollapsibleSection("Skin")
        self._sliders['skin_smoothness'] = skin_section.add_slider("Skin Smoothing", 0, 100, 0)
        self._sliders['blemish_removal'] = skin_section.add_slider("Blemish Removal", 0, 100, 0)
        layout.addWidget(skin_section)

        # Eyes Section
        eyes_section = CollapsibleSection("Eyes")
        self._sliders['eye_brightness'] = eyes_section.add_slider("Eye Brightness", 0, 100, 0)
        self._sliders['eye_size'] = eyes_section.add_slider("Eye Size", 0, 100, 0)
        self._sliders['dark_circle_removal'] = eyes_section.add_slider("Dark Circle Removal", 0, 100, 0)
        layout.addWidget(eyes_section)

        # Mouth Section
        mouth_section = CollapsibleSection("Mouth")
        self._sliders['teeth_whitening'] = mouth_section.add_slider("Teeth Whitening", 0, 100, 0)
        self._sliders['lip_saturation'] = mouth_section.add_slider("Lip Color", 0, 100, 0)
        self._sliders['smile_enhancement'] = mouth_section.add_slider("Smile Enhancement", 0, 100, 0)
        layout.addWidget(mouth_section)

        # Face Shape Section
        shape_section = CollapsibleSection("Face Shape")
        self._sliders['face_slimming'] = shape_section.add_slider("Face Slimming", 0, 100, 0)
        self._sliders['nose_slimming'] = shape_section.add_slider("Nose Slimming", 0, 100, 0)
        self._sliders['chin_adjustment'] = shape_section.add_slider("Chin Adjustment", 0, 100, 0)
        self._sliders['jawline_sharpen'] = shape_section.add_slider("Jawline Definition", 0, 100, 0)
        layout.addWidget(shape_section)

        # Enhancement Section
        enhance_section = CollapsibleSection("Enhancement")
        self._sliders['face_enhancement'] = enhance_section.add_slider("Detail Enhancement", 0, 100, 0)
        layout.addWidget(enhance_section)

        # Upscaling Section
        upscale_section = CollapsibleSection("Upscaling")
        upscale_section.add_widget(QLabel("Scale Factor"))
        self.upscale_combo = QComboBox()
        self.upscale_combo.addItems(["None", "2x", "4x"])
        upscale_section.add_widget(self.upscale_combo)
        layout.addWidget(upscale_section)

        # Process Button
        self.process_btn = QPushButton("Process Image")
        self.process_btn.setMinimumHeight(45)
        self.process_btn.setStyleSheet("""
            QPushButton {
                background-color: #4a9eff; border: none; border-radius: 5px;
                font-weight: bold; font-size: 14px;
            }
            QPushButton:hover { background-color: #3a8eef; }
            QPushButton:disabled { background-color: #555555; }
        """)
        self.process_btn.clicked.connect(self.process_requested.emit)
        layout.addWidget(self.process_btn)

        # Progress
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)

        self.progress_label = QLabel("")
        self.progress_label.setVisible(False)
        layout.addWidget(self.progress_label)

        layout.addStretch()

        # Connect all sliders
        for slider in self._sliders.values():
            slider.valueChanged.connect(self.settings_changed.emit)
        self.upscale_combo.currentIndexChanged.connect(self.settings_changed.emit)

        self.setWidget(container)

    def _populate_templates(self):
        self.template_combo.clear()
        self.template_combo.addItem("-- Select Template --")

        for category in self._template_manager.get_categories():
            self.template_combo.addItem(f"── {category} ──")
            # Make category headers non-selectable
            idx = self.template_combo.count() - 1
            self.template_combo.model().item(idx).setEnabled(False)

            for template in self._template_manager.get_templates_by_category(category):
                self.template_combo.addItem(f"    {template.name}")

    def _on_template_selected(self, text: str):
        name = text.strip()
        template = self._template_manager.get_template(name)
        if template:
            self._apply_template(template)

    def _apply_template(self, template: RetouchTemplate):
        """Apply template settings to sliders."""
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
        name, ok = QInputDialog.getText(self, "Save Template", "Template name:")
        if ok and name:
            settings = self.get_settings()
            template = RetouchTemplate(
                name=name,
                category="Custom",
                description="Custom template",
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
            QMessageBox.information(self, "Saved", f"Template '{name}' saved!")

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
            upscale_factor=upscale_map[self.upscale_combo.currentText()],
        )

    def set_processing(self, is_processing: bool):
        self.process_btn.setEnabled(not is_processing)
        self.progress_bar.setVisible(is_processing)
        self.progress_label.setVisible(is_processing)

    def update_progress(self, message: str, percent: int):
        self.progress_label.setText(message)
        if percent >= 0:
            self.progress_bar.setRange(0, 100)
            self.progress_bar.setValue(percent)
        else:
            self.progress_bar.setRange(0, 0)


class ProcessingWorker(QThread):
    """Background worker for image processing."""

    progress = Signal(str, int)
    finished = Signal(np.ndarray)
    error = Signal(str)

    def __init__(self, image: np.ndarray, settings: ProcessingSettings):
        super().__init__()
        self.image = image
        self.settings = settings
        self.pipeline = RetouchPipeline()

    def run(self):
        try:
            result = self.pipeline.process(
                self.image,
                self.settings,
                progress_callback=lambda msg, pct: self.progress.emit(msg, pct)
            )
            self.finished.emit(result)
        except Exception as e:
            self.error.emit(str(e))


class MainWindow(QMainWindow):
    """Main application window."""

    def __init__(self):
        super().__init__()
        self._image_state = ImageState()
        self._worker: Optional[ProcessingWorker] = None

        self.setWindowTitle("LuminaRetouch - Professional Portrait Retouching")
        self.setMinimumSize(1280, 800)
        self._apply_dark_theme()
        self._setup_ui()
        self._setup_toolbar()
        self._setup_statusbar()

    def _apply_dark_theme(self):
        self.setStyleSheet("""
            QMainWindow, QWidget { background-color: #1e1e1e; color: #e0e0e0; }
            QGroupBox { border: 1px solid #404040; border-radius: 5px; margin-top: 10px; padding-top: 10px; }
            QPushButton { background-color: #353535; border: 1px solid #454545; border-radius: 4px; padding: 6px 12px; }
            QPushButton:hover { background-color: #404040; }
            QComboBox { background-color: #353535; border: 1px solid #454545; border-radius: 4px; padding: 5px; }
            QScrollArea { border: none; }
            QScrollBar:vertical { background: #2a2a2a; width: 10px; }
            QScrollBar::handle:vertical { background: #505050; border-radius: 5px; min-height: 20px; }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }
            QProgressBar { border: 1px solid #404040; border-radius: 4px; text-align: center; }
            QProgressBar::chunk { background-color: #4a9eff; }
            QToolBar { background-color: #252525; border: none; spacing: 5px; padding: 5px; }
            QStatusBar { background-color: #252525; }
            QLabel { background-color: transparent; }
        """)

    def _setup_ui(self):
        central = QWidget()
        self.setCentralWidget(central)

        main_layout = QHBoxLayout(central)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # Control panel (left)
        self.control_panel = ControlPanel()
        self.control_panel.process_requested.connect(self._process_image)

        # Canvas (center)
        self.canvas = SplitViewCanvas()
        self.canvas.zoom_changed.connect(self._on_zoom_changed)

        # Splitter
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(self.control_panel)
        splitter.addWidget(self.canvas)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)

        main_layout.addWidget(splitter)

    def _setup_toolbar(self):
        toolbar = QToolBar("Main Toolbar")
        toolbar.setMovable(False)
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

        zoom_action = QAction("100%", self)
        zoom_action.triggered.connect(self.canvas.reset_zoom)
        toolbar.addAction(zoom_action)

        toolbar.addSeparator()

        toolbar.addWidget(QLabel(" Split: "))
        self.split_slider = QSlider(Qt.Orientation.Horizontal)
        self.split_slider.setFixedWidth(150)
        self.split_slider.setRange(0, 100)
        self.split_slider.setValue(50)
        self.split_slider.valueChanged.connect(lambda v: self.canvas.set_split_position(v / 100.0))
        toolbar.addWidget(self.split_slider)

    def _setup_statusbar(self):
        self.statusbar = QStatusBar()
        self.setStatusBar(self.statusbar)

        self.zoom_label = QLabel("100%")
        self.zoom_label.setFixedWidth(60)
        self.statusbar.addPermanentWidget(self.zoom_label)

        self.info_label = QLabel("No image loaded")
        self.statusbar.addWidget(self.info_label)

    def _on_zoom_changed(self, zoom: float):
        self.zoom_label.setText(f"{int(zoom * 100)}%")

    def _open_image(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Open Image", "", "Images (*.png *.jpg *.jpeg *.bmp *.tiff *.webp)"
        )
        if file_path:
            self._load_image(file_path)

    def _load_image(self, file_path: str):
        pixmap = QPixmap(file_path)
        if pixmap.isNull():
            QMessageBox.warning(self, "Error", f"Failed to load: {file_path}")
            return

        # Convert to numpy
        image = pixmap.toImage().convertToFormat(QImage.Format.Format_RGB888)
        w, h = image.width(), image.height()
        ptr = image.bits()
        arr = np.array(ptr).reshape((h, image.bytesPerLine()))
        arr = arr[:, :w * 3].reshape((h, w, 3)).copy()

        self._image_state = ImageState(
            original=arr,
            original_pixmap=pixmap,
            file_path=Path(file_path)
        )

        self.canvas.set_images(pixmap)
        self.info_label.setText(f"{Path(file_path).name} - {w}x{h}")

    def _save_image(self):
        if self._image_state.processed_pixmap is None:
            QMessageBox.warning(self, "Warning", "No processed image to save.")
            return

        default_name = (self._image_state.file_path.stem + "_retouched.png"
                       if self._image_state.file_path else "retouched.png")

        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Image", default_name, "PNG (*.png);;JPEG (*.jpg)"
        )
        if file_path:
            self._image_state.processed_pixmap.save(file_path)
            self.statusbar.showMessage(f"Saved: {file_path}", 3000)

    def _process_image(self):
        if self._image_state.original is None:
            QMessageBox.warning(self, "Warning", "No image loaded.")
            return

        settings = self.control_panel.get_settings()

        if settings.is_empty() and settings.upscale_factor == 1:
            QMessageBox.information(self, "Info", "All sliders are at zero. Adjust settings first.")
            return

        self.control_panel.set_processing(True)

        self._worker = ProcessingWorker(self._image_state.original.copy(), settings)
        self._worker.progress.connect(self.control_panel.update_progress)
        self._worker.finished.connect(self._on_processing_finished)
        self._worker.error.connect(self._on_processing_error)
        self._worker.start()

    def _on_processing_finished(self, result: np.ndarray):
        self.control_panel.set_processing(False)

        h, w, c = result.shape
        qimage = QImage(result.data.tobytes(), w, h, c * w, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(qimage)

        self._image_state.processed = result
        self._image_state.processed_pixmap = pixmap

        self.canvas.set_images(self._image_state.original_pixmap, pixmap)
        self.statusbar.showMessage("Processing complete!", 3000)

    def _on_processing_error(self, error_msg: str):
        self.control_panel.set_processing(False)
        QMessageBox.critical(self, "Error", error_msg)


def main():
    import sys
    app = QApplication(sys.argv)
    app.setApplicationName("LuminaRetouch")
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
