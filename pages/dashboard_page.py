# pages/dashboard_page.py

import os
from pathlib import Path
from typing import Union, Optional

import numpy as np
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QFileDialog, QProgressBar, QTextEdit, QComboBox, QGroupBox,
    QTabWidget
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal

from models.model_manager import ModelManager
from pipelines.common_pipeline import BaseCTPipeline

from utils.viewer_3d import Lung3DViewer   # 공용 3D 뷰어
from utils.viewer_2d import CT2DViewer     # 공용 2D 뷰어


# ───────────────────────────────────────────────────────────────
# 백그라운드 스레드
# ───────────────────────────────────────────────────────────────
class ProcessingThread(QThread):
    progress = pyqtSignal(int, str)
    finished = pyqtSignal(dict)
    error = pyqtSignal(str)

    def __init__(
        self,
        folder_path: Union[str, Path],
        model_manager: ModelManager,
        organ: str,
    ):
        super().__init__()
        self.folder_path = Path(folder_path)
        self.model_manager = model_manager
        self.organ = organ  # "lung" or "liver"
        self.pipeline = BaseCTPipeline(organ)

    def _progress_cb(self, pct: int, msg: str) -> None:
        self.progress.emit(pct, msg)

    def run(self):
        try:
            result = self.pipeline.run(
                str(self.folder_path),
                self.model_manager,
                progress_cb=self._progress_cb,
            )
            # 혹시 파이프라인이 organ을 안 넣어줬다면 여기서라도 넣어줌
            result.setdefault("organ", self.organ)
            self.finished.emit(result)
        except Exception as e:
            self.error.emit(f"처리 중 오류 발생: {e}")


# ───────────────────────────────────────────────────────────────
# 3D 탭
# ───────────────────────────────────────────────────────────────
class Viewer3DTabWidget(QWidget):
    def __init__(self, data_store: dict):
        super().__init__()
        self.data_store = data_store

        layout = QVBoxLayout()
        self.viewer_3d = Lung3DViewer()
        layout.addWidget(self.viewer_3d)
        self.setLayout(layout)

    def update_viewer(self):
        verts = self.data_store.get("verts")
        preds = self.data_store.get("predictions")
        labels = self.data_store.get("labels")
        organ = self.data_store.get("organ", "lung")

        if verts is None:
            self.viewer_3d.update_plot(np.zeros((0, 3)), title="데이터 없음")
            return

        if organ == "liver":
            title = "3D 간 구조 및 결절 시각화"
        else:
            title = "3D 폐 구조 및 결절 시각화"

        self.viewer_3d.update_plot(
            verts,
            predictions=preds,
            ground_truth_mask=labels,
            title=title,
        )


# ───────────────────────────────────────────────────────────────
# 2D 탭
# ───────────────────────────────────────────────────────────────
class Viewer2DWidget(QWidget):
    def __init__(self):
        super().__init__()
        layout = QVBoxLayout()
        self.viewer_2d = CT2DViewer()
        layout.addWidget(self.viewer_2d)
        self.setLayout(layout)

    def update_data(self, image: np.ndarray):
        self.viewer_2d.set_image(image)


# ───────────────────────────────────────────────────────────────
# 대시보드 페이지
# ───────────────────────────────────────────────────────────────
class DashboardPage(QWidget):
    """
    왼쪽: 업로드 + 결과 요약
    오른쪽: 결과 뷰어
    위: 프로젝트 이름
    """
    processing_completed = pyqtSignal()

    def __init__(self, model_manager: ModelManager, data_store: dict):
        super().__init__()
        self.model_manager = model_manager
        self.data_store = data_store
        self.processing_thread: Optional[ProcessingThread] = None
        self._init_ui()

    def _init_ui(self):
        root = QVBoxLayout()
        root.setContentsMargins(15, 15, 15, 15)
        root.setSpacing(15)

        # 헤더
        header = QLabel("L-POT : Lung / Liver Prediction Tool")
        header.setAlignment(Qt.AlignmentFlag.AlignCenter)
        header.setStyleSheet("""
            background-color: #ffffff;
            border: 2px solid #000;
            border-radius: 5px;
            padding: 15px;
            font-size: 18px;
            font-weight: bold;
        """)
        root.addWidget(header)

        # 본문: 좌/우
        body_layout = QHBoxLayout()
        body_layout.setSpacing(15)
        root.addLayout(body_layout, stretch=1)

        # ── 왼쪽 ───────────────────────
        left_col = QVBoxLayout()
        left_col.setSpacing(15)
        body_layout.addLayout(left_col, stretch=0)

        # 업로드 박스
        upload_box = QGroupBox("업로드")
        upload_box.setStyleSheet("QGroupBox { font-weight:bold; }")
        ub_layout = QVBoxLayout()
        ub_layout.setSpacing(10)

        info = QLabel("분석할 DICOM 폴더를 선택하세요.")
        info.setWordWrap(True)
        ub_layout.addWidget(info)

        # 모드 선택
        mode_label = QLabel("모드 선택")
        self.mode_combo = QComboBox()
        self.mode_combo.addItem("폐 결절 탐지 (lung)", userData="lung")
        self.mode_combo.addItem("간 결절 탐지 (liver)", userData="liver")
        ub_layout.addWidget(mode_label)
        ub_layout.addWidget(self.mode_combo)

        self.btn_select = QPushButton("DICOM 폴더 선택")
        self.btn_select.clicked.connect(self._select_folder)
        ub_layout.addWidget(self.btn_select)

        self.lbl_selected = QLabel("선택된 폴더: -")
        ub_layout.addWidget(self.lbl_selected)

        self.progress = QProgressBar()
        self.progress.setVisible(False)
        ub_layout.addWidget(self.progress)

        self.lbl_status = QLabel("")
        ub_layout.addWidget(self.lbl_status)

        self.log_area = QTextEdit()
        self.log_area.setReadOnly(True)
        self.log_area.setMaximumHeight(120)
        self.log_area.setVisible(False)
        ub_layout.addWidget(self.log_area)

        upload_box.setLayout(ub_layout)
        left_col.addWidget(upload_box)

        # 결과 요약
        result_box = QGroupBox("결과 요약")
        rb_layout = QVBoxLayout()
        self.lbl_accuracy = QLabel("모델 정확도: -")
        self.lbl_nodule = QLabel("예측 결절 수: -")
        rb_layout.addWidget(self.lbl_accuracy)
        rb_layout.addWidget(self.lbl_nodule)
        rb_layout.addStretch()
        result_box.setLayout(rb_layout)
        left_col.addWidget(result_box)

        left_col.addStretch()

        # ── 오른쪽 ───────────────────────
        right_box = QGroupBox("결과")
        right_layout = QVBoxLayout()
        self.tab = QTabWidget()

        self.viewer_3d_tab = Viewer3DTabWidget(self.data_store)
        self.tab.addTab(self.viewer_3d_tab, "3D 뷰어")

        self.viewer_2d_tab = Viewer2DWidget()
        self.tab.addTab(self.viewer_2d_tab, "2D 뷰어")

        right_layout.addWidget(self.tab)
        right_box.setLayout(right_layout)
        body_layout.addWidget(right_box, stretch=1)

        self.setLayout(root)

    # ── 업로드 흐름 ───────────────────────────
    def _select_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "DICOM 폴더 선택", "")
        if not folder:
            return

        self.lbl_selected.setText(f"선택된 폴더: {os.path.basename(folder)}")
        self._start_processing(folder)

    def _start_processing(self, folder_path: str):
        # UI 잠시 잠그기
        self.btn_select.setEnabled(False)
        self.progress.setVisible(True)
        self.progress.setValue(0)
        self.log_area.setVisible(True)
        self.log_area.clear()

        # 결과 dict 초기화
        self.data_store.clear()
        self.data_store["selected_folder"] = Path(folder_path).name

        organ = self.mode_combo.currentData() or "lung"

        # 백그라운드 스레드 시작
        self.processing_thread = ProcessingThread(folder_path, self.model_manager, organ=organ)
        self.processing_thread.progress.connect(self._on_progress)
        self.processing_thread.finished.connect(self._on_finished)
        self.processing_thread.error.connect(self._on_error)
        self.processing_thread.start()

    def _on_progress(self, val: int, msg: str):
        self.progress.setValue(val)
        self.lbl_status.setText(msg)
        self.log_area.append(f"[{val}%] {msg}")

    def _on_finished(self, result: dict):
        # 파이프라인 결과 저장
        self.data_store.update(result)

        # 요약 업데이트
        acc = self.data_store.get("model_accuracy")
        if acc is not None:
            try:
                self.lbl_accuracy.setText(f"모델 정확도: {acc:.2%}")
            except Exception:
                self.lbl_accuracy.setText("모델 정확도: -")
        else:
            self.lbl_accuracy.setText("모델 정확도: -")

        preds = self.data_store.get("predictions")
        if isinstance(preds, np.ndarray):
            self.lbl_nodule.setText(f"예측 결절 수: {int(preds.sum())}")
        else:
            self.lbl_nodule.setText("예측 결절 수: -")

        # 뷰어 갱신
        self.viewer_3d_tab.update_viewer()
        if self.data_store.get("image") is not None:
            self.viewer_2d_tab.update_data(self.data_store["image"])

        # UI 복구
        self.lbl_status.setText("처리 완료!")
        self.log_area.append("\n처리가 완료되었습니다.")
        self.btn_select.setEnabled(True)

        self.processing_completed.emit()

    def _on_error(self, msg: str):
        self.lbl_status.setText("오류 발생")
        self.log_area.append(f"\n오류: {msg}")
        self.btn_select.setEnabled(True)
