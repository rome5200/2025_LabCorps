# pages/dashboard_page.py

import os
from pathlib import Path
from typing import Union, Optional
import numpy as np

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QFileDialog, QProgressBar, QTextEdit, QGroupBox,
    QTabWidget, QRadioButton, QSizePolicy, QApplication
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal

from models.model_manager import ModelManager
from pipelines.common_pipeline import OrganCTPipeline
from pages.ui_viewer import CT2DViewer, Lung3DViewer


# ──────────────────────────────────────────────
# 백그라운드 스레드
# ──────────────────────────────────────────────
class ProcessingThread(QThread):
    progress = pyqtSignal(int, str)
    finished = pyqtSignal(dict)
    error = pyqtSignal(str)

    def __init__(self, input_path: Union[str, Path], model_manager: ModelManager, organ: str):
        super().__init__()
        self.input_path = Path(input_path)
        self.model_manager = model_manager
        self.organ = organ
        self.pipeline = OrganCTPipeline(organ)
        self._canceled = False

    def cancel(self):
        self._canceled = True

    def _progress_cb(self, pct: int, msg: str):
        self.progress.emit(pct, msg)
        if self._canceled:
            raise RuntimeError("사용자가 취소했습니다.")

    def run(self):
        try:
            result = self.pipeline.run(
                str(self.input_path),
                self.model_manager,
                progress_cb=self._progress_cb,
            )
            result.setdefault("organ", self.organ)
            if not self._canceled:
                self.finished.emit(result)
        except Exception as e:
            if self._canceled:
                self.error.emit("사용자에 의해 취소되었습니다.")
            else:
                self.error.emit(f"처리 중 오류 발생: {e}")


# ──────────────────────────────────────────────
# 뷰어 위젯
# ──────────────────────────────────────────────
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
        organ = self.data_store.get("organ", "lung")
        acc = (
            self.data_store.get("prediction_f1")
            or self.data_store.get("model_accuracy")
        )

        if verts is None:
            self.viewer_3d.update_plot(np.zeros((0, 3)), title="데이터 없음")
            return

        title = "3D 간 구조 및 결절 시각화" if organ == "liver" else "3D 폐 구조 및 결절 시각화"
        self.viewer_3d.update_plot(
            verts,
            predictions=preds,
            title=title,
            accuracy=acc,
        )

    def clear_viewer(self):
        self.viewer_3d.update_plot(np.zeros((0, 3)), title="데이터 없음")


class Viewer2DWidget(QWidget):
    def __init__(self):
        super().__init__()
        layout = QVBoxLayout()
        self.viewer_2d = CT2DViewer()
        layout.addWidget(self.viewer_2d)
        self.setLayout(layout)

    def update_data(self, image: np.ndarray):
        self.viewer_2d.set_image(image)

    def clear_data(self):
        self.viewer_2d.set_image(None)


# ──────────────────────────────────────────────
# 대시보드 페이지
# ──────────────────────────────────────────────
class DashboardPage(QWidget):
    processing_completed = pyqtSignal()

    def __init__(self, model_manager: ModelManager, data_store: dict):
        super().__init__()
        self.model_manager = model_manager
        self.data_store = data_store
        self.processing_thread: Optional[ProcessingThread] = None
        self._init_ui()

    def _init_ui(self):
        root = QVBoxLayout()

        # 헤더
        header = QLabel("L-POT")
        header.setAlignment(Qt.AlignmentFlag.AlignCenter)
        header.setStyleSheet("font-size: 20px; font-weight: bold; padding: 10px;")
        root.addWidget(header)

        body_layout = QHBoxLayout()
        root.addLayout(body_layout)

        # ── 왼쪽 패널 ────────────────────────────
        left_col = QVBoxLayout()
        body_layout.addLayout(left_col, stretch=0)

        # 1️⃣ 사용 방법 (stretch = 2)
        usage_box = QGroupBox("사용 방법")
        usage_layout = QVBoxLayout()
        usage_label = QLabel(
            "1. 폐/간 모드를 선택합니다.\n"
            "2. ZIP 파일을 선택합니다.\n"
            "3. 분석 결과는 우측 뷰어에서 확인합니다.\n"
            "4. 처리 중에는 '취소'로 중단할 수 있습니다.\n"
            "5. 3D 뷰어는 마우스로 회전, 휠로 확대/축소 가능합니다."
        )
        usage_label.setWordWrap(True)
        usage_layout.addWidget(usage_label)
        usage_box.setLayout(usage_layout)
        left_col.addWidget(usage_box, stretch=2)

        # 2️⃣ 모드 선택 (stretch = 1)
        mode_box = QGroupBox("모드 선택")
        mode_layout = QVBoxLayout()
        self.radio_lung = QRadioButton("폐 결절 탐지")
        self.radio_liver = QRadioButton("간 결절 탐지")
        self.radio_lung.setChecked(True)
        mode_layout.addWidget(self.radio_lung)
        mode_layout.addWidget(self.radio_liver)
        mode_box.setLayout(mode_layout)
        left_col.addWidget(mode_box, stretch=1)

        # 3️⃣ 업로드 (stretch = 4)
        upload_box = QGroupBox("업로드")
        ub_layout = QVBoxLayout()
        info = QLabel("DICOM ZIP 파일을 선택하세요.")
        ub_layout.addWidget(info)

        btn_row = QHBoxLayout()
        self.btn_select = QPushButton("ZIP 파일 선택")
        self.btn_select.clicked.connect(self._select_zip)
        btn_row.addWidget(self.btn_select)
        self.btn_cancel = QPushButton("취소")
        self.btn_cancel.setEnabled(False)
        self.btn_cancel.clicked.connect(self._cancel_processing)
        btn_row.addWidget(self.btn_cancel)
        ub_layout.addLayout(btn_row)

        self.lbl_selected = QLabel("선택된 파일: -")
        ub_layout.addWidget(self.lbl_selected)
        self.progress = QProgressBar()
        self.progress.setVisible(False)
        ub_layout.addWidget(self.progress)
        self.lbl_status = QLabel("")
        ub_layout.addWidget(self.lbl_status)
        self.log_area = QTextEdit()
        self.log_area.setReadOnly(True)
        self.log_area.setVisible(False)
        ub_layout.addWidget(self.log_area)
        upload_box.setLayout(ub_layout)
        left_col.addWidget(upload_box, stretch=4)

        # 4️⃣ 결과 요약 (stretch = 3)
        result_box = QGroupBox("결과 요약")
        rb_layout = QVBoxLayout()
        self.lbl_accuracy = QLabel("예측 성능 지수 : -")
        self.lbl_nodule = QLabel("예측 결절 수 : -")
        self.lbl_nodule_len = QLabel("예측 결절 최대 직경(축단면) : -")

        rb_layout.addWidget(self.lbl_accuracy)
        rb_layout.addWidget(self.lbl_nodule)
        rb_layout.addWidget(self.lbl_nodule_len)

        desc = QLabel("예측 성능 지수는 모델의 정확도를 종합적으로 표현한 값입니다.\n"
                      "값이 1.0에 가까울수록 예측이 실제 결과와 잘 일치합니다.")
        desc.setWordWrap(True)
        desc.setStyleSheet("color: light gray; font-size: 10px;")  # 글자 색 + 크기 조정
        rb_layout.addWidget(desc)

        self.lbl_notice = QLabel("※ 본 시스템은 임상 판독을 대체하지 않습니다.")
        self.lbl_notice.setWordWrap(True)
        self.lbl_notice.setStyleSheet("color: red; font-size: 11px; font-weight: bold;")
        self.lbl_notice.setVisible(False)
        rb_layout.addWidget(self.lbl_notice)

        result_box.setLayout(rb_layout)
        left_col.addWidget(result_box, stretch=3)

        # ── 오른쪽 (뷰어) ───────────────────────
        right_box = QGroupBox("결과 뷰어")
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

    # ── 파일 선택 및 처리 흐름 ─────────────────
    def _select_zip(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "DICOM ZIP 선택", "", "ZIP Files (*.zip);;All Files (*)",
        )
        if not file_path:
            return
        self.lbl_selected.setText(f"선택된 파일: {os.path.basename(file_path)}")
        self._start_processing(file_path)

    def _get_selected_organ(self) -> str:
        return "liver" if self.radio_liver.isChecked() else "lung"

    def _start_processing(self, input_path: str):
        self.btn_select.setEnabled(False)
        self.btn_cancel.setEnabled(True)
        self.progress.setVisible(True)
        self.progress.setValue(0)
        self.log_area.setVisible(True)
        self.log_area.clear()
        self.data_store.clear()

        organ = self._get_selected_organ()
        self.processing_thread = ProcessingThread(input_path, self.model_manager, organ)
        self.processing_thread.progress.connect(self._on_progress)
        self.processing_thread.finished.connect(self._on_finished)
        self.processing_thread.error.connect(self._on_error)
        self.processing_thread.start()

    def _cancel_processing(self):
        if self.processing_thread and self.processing_thread.isRunning():
            self.processing_thread.cancel()
        self._reset_ui_after_action("취소되었습니다.")

    def _reset_ui_after_action(self, msg=""):
        self.btn_select.setEnabled(True)
        self.btn_cancel.setEnabled(False)
        self.progress.setVisible(False)
        self.lbl_status.setText(msg)
        self.lbl_selected.setText("선택된 파일: -")
        self.lbl_accuracy.setText("예측 성능 지수 : -")
        self.lbl_nodule.setText("예측 결절 수 : -")
        self.lbl_nodule_len.setText("예측 결절 최대 직경(축단면) : -")
        self.lbl_nodule_len.setVisible(True)
        self.lbl_notice.setVisible(False)
        self.viewer_3d_tab.clear_viewer()
        self.viewer_2d_tab.clear_data()
        self.data_store.clear()

    def _on_progress(self, val: int, msg: str):
        self.progress.setValue(val)
        self.lbl_status.setText(msg)
        self.log_area.append(f"[{val}%] {msg}")
        QApplication.processEvents()

    def _on_finished(self, result: dict):
        self.data_store.update(result)
        organ = self.data_store.get("organ", "lung")

        f1 = self.data_store.get("prediction_f1")
        model_acc = self.data_store.get("model_accuracy")
        preds = self.data_store.get("predictions")
        nodule_len = self.data_store.get("nodule_length_mm")

        if f1 is not None:
            self.lbl_accuracy.setText(f"예측 성능 지수 : {f1:.3f}")
        elif model_acc is not None:
            self.lbl_accuracy.setText(f"예측 성능 지수 : (참고) {model_acc:.2%}")
        else:
            self.lbl_accuracy.setText("예측 성능 지수 : -")

        if isinstance(preds, np.ndarray):
            self.lbl_nodule.setText(f"예측 결절 수 : {int(preds.sum())}")
        else:
            self.lbl_nodule.setText("예측 결절 수 : -")

        # 간 모드에서는 최대 직경 항목 숨김
        if organ == "liver":
            self.lbl_nodule_len.setVisible(False)
        else:
            self.lbl_nodule_len.setVisible(True)
            if nodule_len is not None:
                self.lbl_nodule_len.setText(f"예측 결절 최대 직경(축단면) : {nodule_len:.1f} mm")
            else:
                self.lbl_nodule_len.setText("예측 결절 최대 직경(축단면) : -")

        self.lbl_notice.setVisible(True)
        self.viewer_3d_tab.update_viewer()
        if self.data_store.get("image") is not None:
            self.viewer_2d_tab.update_data(self.data_store["image"])

        self.lbl_status.setText("처리 완료!")
        self.btn_select.setEnabled(True)
        self.btn_cancel.setEnabled(False)
        self.processing_thread = None
        self.processing_completed.emit()

    def _on_error(self, msg: str):
        self.log_area.append(f"\n{msg}")
        self._reset_ui_after_action("오류 / 취소")
        self.processing_thread = None
