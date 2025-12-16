# pages/ui_viewer.py

from __future__ import annotations

import numpy as np
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QSlider
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QPixmap, QImage

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

# =========================
# 2D 뷰어 (기존 그대로)
# =========================
class CT2DViewer(QWidget):
    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self._image_data: np.ndarray | None = None
        self._build_ui()

    def _build_ui(self):
        root = QVBoxLayout()
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(8)

        self.image_label = QLabel("이미지가 없습니다.")
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setMinimumSize(512, 512)
        self.image_label.setStyleSheet("""
            background-color: black;
            border: 2px solid #2dc9c8;
            border-radius: 5px;
        """)
        root.addWidget(self.image_label, stretch=1)

        bottom = QHBoxLayout()
        bottom.setSpacing(10)

        self.lbl_info = QLabel("슬라이스: - / -")
        self.lbl_info.setStyleSheet("font-weight: bold; font-size: 12px;")
        bottom.addWidget(self.lbl_info)

        self.slice_slider = QSlider(Qt.Orientation.Horizontal)
        self.slice_slider.setMinimum(0)
        self.slice_slider.setMaximum(0)
        self.slice_slider.setValue(0)
        self.slice_slider.setEnabled(False)
        self.slice_slider.valueChanged.connect(self._on_slice_changed)
        bottom.addWidget(self.slice_slider, stretch=1)

        root.addLayout(bottom)
        self.setLayout(root)

    def set_image(self, volume: np.ndarray):
        if volume is None:
            self._image_data = None
            self.image_label.setText("이미지가 없습니다.")
            self.slice_slider.setEnabled(False)
            self.lbl_info.setText("슬라이스: - / -")
            return

        arr = np.asarray(volume)
        if arr.ndim == 2:
            arr = arr[np.newaxis, ...]
        if arr.ndim != 3:
            raise ValueError("CT2DViewer.set_image() 는 (z, h, w) 형태만 받을 수 있습니다.")

        self._image_data = arr
        z = arr.shape[0]
        self.slice_slider.setMaximum(z - 1)
        self.slice_slider.setValue(z // 2)
        self.slice_slider.setEnabled(True)
        self._show_slice(self.slice_slider.value())

    def _on_slice_changed(self, idx: int):
        self._show_slice(idx)

    def _show_slice(self, idx: int):
        if self._image_data is None:
            return

        z, h, w = self._image_data.shape
        idx = max(0, min(idx, z - 1))
        self.lbl_info.setText(f"슬라이스: {idx} / {z - 1}")

        slice_img = self._image_data[idx]
        slice_img = np.clip(slice_img, -1000, 400)
        norm = (slice_img + 1000) / 1400.0
        norm = (norm * 255).astype(np.uint8)

        height, width = norm.shape
        bytes_per_line = width
        q_img = QImage(
            norm.data,
            width,
            height,
            bytes_per_line,
            QImage.Format.Format_Grayscale8,
        )

        pixmap = QPixmap.fromImage(q_img)
        scaled = pixmap.scaled(
            512,
            512,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.image_label.setPixmap(scaled)


# =========================
# 3D 뷰어 (휠 줌 추가)
# =========================
class Lung3DViewer(QWidget):
    """
    matplotlib 3D를 PyQt 위젯으로 싸서 보여주는 간단한 뷰어.
    - 마우스 휠 → 확대/축소 (matplotlib scroll_event 사용)
    """

    def __init__(self, parent=None):
        super().__init__(parent)

        self.fig = Figure(figsize=(5, 5))
        self.canvas = FigureCanvas(self.fig)
        self.ax = self.fig.add_subplot(111, projection="3d")

        layout = QVBoxLayout()
        layout.addWidget(self.canvas)
        self.setLayout(layout)

        # 현재 축 범위 저장
        self._xlim = None
        self._ylim = None
        self._zlim = None

        # 휠 스텝당 확대/축소 비율
        self._zoom_in_factor = 0.9
        self._zoom_out_factor = 1.1

        # matplotlib 이벤트로 스크롤 받기
        self.canvas.mpl_connect("scroll_event", self._on_scroll)

    def update_plot(self, verts, predictions=None, title="3D Viewer", accuracy=None):
        self.ax.clear()
        self.ax.set_title(title)

        if verts is None or len(verts) == 0:
            self.ax.text(0.5, 0.5, 0.5, "데이터 없음", color="gray")
            self.canvas.draw()
            self._xlim = self._ylim = self._zlim = None
            return

        verts = np.asarray(verts)

        # 기본 구조
        self.ax.scatter(
            verts[:, 0], verts[:, 1], verts[:, 2],
            c="gray", s=1, alpha=0.2, label="Structure"
        )

        # 예측만 별도로
        if predictions is not None and np.any(predictions):
            mask = predictions.astype(bool)
            pos = verts[mask]
            self.ax.scatter(
                pos[:, 0], pos[:, 1], pos[:, 2],
                c="red", s=6, alpha=0.6, label="Prediction"
            )

        # 정확도 텍스트
        if accuracy is not None:
            self.ax.text2D(
                0.02, 0.95,
                f"F1: {accuracy:.3f}" if accuracy <= 1.0 else f"Score: {accuracy:.3f}",
                transform=self.ax.transAxes,
                color="black",
                fontsize=9,
            )

        self.ax.legend(loc="upper right")
        self.ax.set_xlabel("X")
        self.ax.set_ylabel("Y")
        self.ax.set_zlabel("Z")
        self.ax.view_init(elev=30, azim=45)

        # 현재 축 범위를 저장
        self._xlim = self.ax.get_xlim3d()
        self._ylim = self.ax.get_ylim3d()
        self._zlim = self.ax.get_zlim3d()

        self.canvas.draw()

    # -------------------------------------------------
    # matplotlib scroll_event 핸들러
    # -------------------------------------------------
    def _on_scroll(self, event):
        """
        event.step 이 +1이면 위로(보통 확대), -1이면 아래로(보통 축소)
        """
        if self._xlim is None:
            return

        # matplotlib에서는 event.button이 'up'/'down'으로도 들어올 수 있음
        if getattr(event, "button", None) == "up" or event.step > 0:
            factor = self._zoom_in_factor
        else:
            factor = self._zoom_out_factor

        def _scale(lim, f):
            center = (lim[0] + lim[1]) / 2.0
            half = (lim[1] - lim[0]) / 2.0 * f
            return (center - half, center + half)

        self._xlim = _scale(self._xlim, factor)
        self._ylim = _scale(self._ylim, factor)
        self._zlim = _scale(self._zlim, factor)

        self.ax.set_xlim3d(self._xlim)
        self.ax.set_ylim3d(self._ylim)
        self.ax.set_zlim3d(self._zlim)
        self.canvas.draw()
