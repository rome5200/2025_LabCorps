# utils/viewer_2d.py

from __future__ import annotations

import numpy as np
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QSlider
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QPixmap, QImage


class CT2DViewer(QWidget):
    """
    3D CT 볼륨 (z, h, w) 을 받아서 슬라이더로 슬라이스를 바꿔가며 보여주는 위젯.
    - set_image(volume) 으로 데이터 주입
    - HU 범위는 기본 (-1000, 400) 로 윈도잉해서 8bit로 보여줌
    """

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self._image_data: np.ndarray | None = None

        self._build_ui()

    # -----------------------------------------------------------
    # UI 구성
    # -----------------------------------------------------------
    def _build_ui(self):
        root = QVBoxLayout()
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(8)

        # 메인 이미지
        self.image_label = QLabel("이미지가 없습니다.")
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setMinimumSize(512, 512)
        self.image_label.setStyleSheet("""
            background-color: black;
            border: 2px solid #2dc9c8;
            border-radius: 5px;
        """)
        root.addWidget(self.image_label, stretch=1)

        # 슬라이더 + 정보
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
        self.slice_slider.setStyleSheet("""
            QSlider::groove:horizontal {
                border: 1px solid #999999;
                height: 8px;
                background: #e9fcff;
                border-radius: 4px;
            }
            QSlider::handle:horizontal {
                background: #2dc9c8;
                border: 1px solid #2dc9c8;
                width: 18px;
                margin: -5px 0;
                border-radius: 9px;
            }
        """)
        bottom.addWidget(self.slice_slider, stretch=1)

        root.addLayout(bottom)

        self.setLayout(root)

    # -----------------------------------------------------------
    # 외부에서 호출하는 메서드
    # -----------------------------------------------------------
    def set_image(self, volume: np.ndarray):
        """
        CT 볼륨을 설정한다.
        volume: (z, h, w) or (h, w) ndarray
        """
        if volume is None:
            self._image_data = None
            self.image_label.setText("이미지가 없습니다.")
            self.slice_slider.setEnabled(False)
            self.lbl_info.setText("슬라이스: - / -")
            return

        arr = np.asarray(volume)
        if arr.ndim == 2:
            # (h, w) 만 들어오면 z 차원을 1로 처리
            arr = arr[np.newaxis, ...]

        if arr.ndim != 3:
            raise ValueError("CT2DViewer.set_image() 는 (z, h, w) 형태의 배열만 받을 수 있습니다.")

        self._image_data = arr

        # 슬라이더 설정
        z = arr.shape[0]
        self.slice_slider.setMaximum(z - 1)
        self.slice_slider.setValue(z // 2)
        self.slice_slider.setEnabled(True)

        # 현재 슬라이스 표시
        self._show_slice(self.slice_slider.value())

    # -----------------------------------------------------------
    # 내부 동작
    # -----------------------------------------------------------
    def _on_slice_changed(self, idx: int):
        self._show_slice(idx)

    def _show_slice(self, idx: int):
        if self._image_data is None:
            return

        z, h, w = self._image_data.shape
        idx = max(0, min(idx, z - 1))

        self.lbl_info.setText(f"슬라이스: {idx} / {z - 1}")

        slice_img = self._image_data[idx]

        # HU 윈도잉 (-1000 ~ 400) → 0~255
        slice_img = np.clip(slice_img, -1000, 400)
        norm = (slice_img + 1000) / 1400.0  # 0~1
        norm = (norm * 255).astype(np.uint8)

        # QImage로 변환
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
