# pages/dashboard_page.py

import os
from pathlib import Path
from typing import Union, Optional
import numpy as np

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QFileDialog, QProgressBar, QTextEdit, QGroupBox,
    QTabWidget, QRadioButton, QApplication
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal

from models.model_manager import ModelManager
from pipelines.common_pipeline import OrganCTPipeline
from pages.ui_viewer import CT2DViewer, Lung3DViewer


# ──────────────────────────────────────────────
# 백그라운드 처리 스레드
# ──────────────────────────────────────────────
class ProcessingThread(QThread):
    progress = pyqtSignal(int, str)
    finished = pyqtSignal(dict)
    error = pyqtSignal(str)

    def __init__(self, input_folder: Union[str, Path], model_manager: ModelManager):
        super().__init__()
        self.input_folder = Path(input_folder)
        self.model_manager = model_manager

        # lung-only 파이프라인
        self.pipeline = OrganCTPipeline("lung")
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
                str(self.input_folder),
                self.model_manager,
                progress_cb=self._progress_cb,
            )
            result["organ"] = "lung"
            if not self._canceled:
                self.finished.emit(result)
        except Exception as e:
            if self._canceled:
                self.error.emit("사용자에 의해 취소되었습니다.")
            else:
                self.error.emit(f"처리 중 오류 발생: {e}")


# ──────────────────────────────────────────────
# 3D Viewer 탭
# ──────────────────────────────────────────────
class Viewer3DTabWidget(QWidget):
    def __init__(self, data_store: dict):
        super().__init__()
        self.data_store = data_store
        self.display_mode = "pred"  # pred | gt | both

        layout = QVBoxLayout()
        self.viewer_3d = Lung3DViewer()
        layout.addWidget(self.viewer_3d)
        self.setLayout(layout)

    def set_display_mode(self, mode: str):
        if mode in ("pred", "gt", "both"):
            self.display_mode = mode
        else:
            self.display_mode = "pred"

    def _get_ground_truth(self):
        return (
            self.data_store.get("ground_truth")
            or self.data_store.get("labels")
        )

    def update_viewer(self):
        verts = self.data_store.get("verts")
        preds = self.data_store.get("predictions")
        gt = self._get_ground_truth()
        acc = (
            self.data_store.get("prediction_f1")
            or self.data_store.get("model_accuracy")
        )

        if verts is None:
            self.viewer_3d.update_plot(np.zeros((0, 3)), title="데이터 없음")
            return

        base_title = "3D 폐 구조 및 결절 시각화"

        if self.display_mode == "pred":
            self.viewer_3d.update_plot(
                verts,
                predictions=preds,
                title=f"{base_title} (예측)",
                accuracy=acc,
            )
        elif self.display_mode == "gt":
            self.viewer_3d.update_plot(
                verts,
                predictions=gt,
                title=f"{base_title} (실제)",
                accuracy=acc,
            )
        else:  # both
            self.viewer_3d.update_plot(
                verts,
                predictions=preds,
                title=f"{base_title} (예측 + 실제)",
                accuracy=acc,
            )

    def clear_viewer(self):
        self.viewer_3d.update_plot(np.zeros((0, 3)), title="데이터 없음")


# ──────────────────────────────────────────────
# 2D Viewer 탭
# ──────────────────────────────────────────────
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

    # ───────────────── UI 구성 ─────────────────
    def _init_ui(self):
        root = QVBoxLayout()

        header = QLabel("L-POT")
        header.setAlignment(Qt.AlignmentFlag.AlignCenter)
        header.setStyleSheet("font-size: 20px; font-weight: bold; padding: 10px;")
        root.addWidget(header)

        body = QHBoxLayout()
        root.addLayout(body)

        # ── 좌측 패널 ───────────────────────────
        left = QVBoxLayout()
        body.addLayout(left, stretch=0)

        # 사용 방법
        usage_box = QGroupBox("사용 방법")
        usage_layout = QVBoxLayout()
        usage_layout.addWidget(QLabel(
            "1. 표시 모드를 선택합니다.\n"
            "2. DICOM 폴더를 선택합니다.\n"
            "3. 결과는 우측 뷰어에서 확인합니다.\n"
            "4. 처리 중에는 '취소'로 중단할 수 있습니다."
        ))
        usage_box.setLayout(usage_layout)
        left.addWidget(usage_box, stretch=2)

        # 표시 모드
        mode_box = QGroupBox("표시 모드 선택")
        mode_layout = QVBoxLayout()
        self.radio_pred = QRadioButton("예측 결과만 보기")
        self.radio_gt = QRadioButton("실제 결과만 보기")
        self.radio_both = QRadioButton("예측 + 실제 결과만 보기")
        self.radio_pred.setChecked(True)

        for r in (self.radio_pred, self.radio_gt, self.radio_both):
            r.toggled.connect(self._on_display_mode_changed)
            mode_layout.addWidget(r)

        mode_box.setLayout(mode_layout)
        left.addWidget(mode_box, stretch=1)

        # 업로드
        upload_box = QGroupBox("업로드")
        up = QVBoxLayout()
        up.addWidget(QLabel("DICOM 폴더를 선택하세요. (ZIP 미지원)"))

        btn_row = QHBoxLayout()
        self.btn_select = QPushButton("DICOM 폴더 선택")
        self.btn_select.clicked.connect(self._select_folder)
        btn_row.addWidget(self.btn_select)

        self.btn_cancel = QPushButton("취소")
        self.btn_cancel.setEnabled(False)
        self.btn_cancel.clicked.connect(self._cancel_processing)
        btn_row.addWidget(self.btn_cancel)

        up.addLayout(btn_row)

        self.lbl_selected = QLabel("선택된 폴더: -")
        up.addWidget(self.lbl_selected)

        self.progress = QProgressBar()
        self.progress.setVisible(False)
        up.addWidget(self.progress)

        self.lbl_status = QLabel("")
        up.addWidget(self.lbl_status)

        self.log_area = QTextEdit()
        self.log_area.setReadOnly(True)
        self.log_area.setVisible(False)
        up.addWidget(self.log_area)

        upload_box.setLayout(up)
        left.addWidget(upload_box, stretch=4)

        # 결과 요약
        result_box = QGroupBox("결과 요약")
        rb = QVBoxLayout()
        self.lbl_accuracy = QLabel("예측 성능 지수 : -")
        self.lbl_nodule = QLabel("예측 결절 수 : -")
        self.lbl_nodule_len = QLabel("예측 결절 최대 직경(축단면) : -")
        rb.addWidget(self.lbl_accuracy)
        rb.addWidget(self.lbl_nodule)
        rb.addWidget(self.lbl_nodule_len)

        self.lbl_notice = QLabel("※ 본 시스템은 임상 판독을 대체하지 않습니다.")
        self.lbl_notice.setStyleSheet("color: red; font-size: 11px; font-weight: bold;")
        self.lbl_notice.setVisible(False)
        rb.addWidget(self.lbl_notice)

        result_box.setLayout(rb)
        left.addWidget(result_box, stretch=3)

        # ── 우측 뷰어 ───────────────────────────
        right = QGroupBox("결과 뷰어")
        rv = QVBoxLayout()
        self.tab = QTabWidget()
        self.viewer_3d_tab = Viewer3DTabWidget(self.data_store)
        self.viewer_2d_tab = Viewer2DWidget()
        self.tab.addTab(self.viewer_3d_tab, "3D 뷰어")
        self.tab.addTab(self.viewer_2d_tab, "2D 뷰어")
        rv.addWidget(self.tab)
        right.setLayout(rv)
        body.addWidget(right, stretch=1)

        self.setLayout(root)

    # ───────────────── 표시 모드 ─────────────────
    def _get_display_mode(self) -> str:
        if self.radio_gt.isChecked():
            return "gt"
        if self.radio_both.isChecked():
            return "both"
        return "pred"

    def _on_display_mode_changed(self):
        self.viewer_3d_tab.set_display_mode(self._get_display_mode())
        if self.data_store.get("verts") is not None:
            self.viewer_3d_tab.update_viewer()

    # ───────────────── 처리 흐름 ─────────────────
    def _select_folder(self):
        folder = QFileDialog.getExistingDirectory(
            self, "DICOM 폴더 선택", "", QFileDialog.Option.ShowDirsOnly
        )
        if not folder:
            return

        self.lbl_selected.setText(f"선택된 폴더: {os.path.basename(folder)}")
        self._start_processing(folder)

    def _start_processing(self, folder: str):
        self.btn_select.setEnabled(False)
        self.btn_cancel.setEnabled(True)
        self.progress.setVisible(True)
        self.progress.setValue(0)
        self.log_area.setVisible(True)
        self.log_area.clear()
        self.data_store.clear()

        self.processing_thread = ProcessingThread(folder, self.model_manager)
        self.processing_thread.progress.connect(self._on_progress)
        self.processing_thread.finished.connect(self._on_finished)
        self.processing_thread.error.connect(self._on_error)
        self.processing_thread.start()

    def _cancel_processing(self):
        if self.processing_thread and self.processing_thread.isRunning():
            self.processing_thread.cancel()
        self._reset_ui("취소되었습니다.")

    def _on_progress(self, val: int, msg: str):
        self.progress.setValue(val)
        self.lbl_status.setText(msg)
        self.log_area.append(f"[{val}%] {msg}")
        QApplication.processEvents()

    def _on_finished(self, result: dict):
        self.data_store.update(result)

        preds = self.data_store.get("predictions")
        f1 = self.data_store.get("prediction_f1")
        acc = self.data_store.get("model_accuracy")
        nlen = self.data_store.get("nodule_length_mm")

        if f1 is not None:
            self.lbl_accuracy.setText(f"예측 성능 지수 : {f1:.3f}")
        elif acc is not None:
            self.lbl_accuracy.setText(f"예측 성능 지수 : (참고) {acc:.2%}")

        if isinstance(preds, np.ndarray):
            self.lbl_nodule.setText(f"예측 결절 수 : {int(preds.sum())}")

        if nlen is not None:
            self.lbl_nodule_len.setText(
                f"예측 결절 최대 직경(축단면) : {nlen:.1f} mm"
            )

        self.lbl_notice.setVisible(True)
        self.viewer_3d_tab.update_viewer()

        if self.data_store.get("image") is not None:
            self.viewer_2d_tab.update_data(self.data_store["image"])

        self.btn_select.setEnabled(True)
        self.btn_cancel.setEnabled(False)
        self.lbl_status.setText("처리 완료!")
        self.processing_thread = None
        self.processing_completed.emit()

    def _on_error(self, msg: str):
        self.log_area.append(f"\n{msg}")
        self._reset_ui("오류 발생")

    def _reset_ui(self, msg: str):
        self.btn_select.setEnabled(True)
        self.btn_cancel.setEnabled(False)
        self.progress.setVisible(False)
        self.lbl_status.setText(msg)
        self.lbl_selected.setText("선택된 폴더: -")
        self.lbl_accuracy.setText("예측 성능 지수 : -")
        self.lbl_nodule.setText("예측 결절 수 : -")
        self.lbl_nodule_len.setText("예측 결절 최대 직경(축단면) : -")
        self.lbl_notice.setVisible(False)
        self.viewer_3d_tab.clear_viewer()
        self.viewer_2d_tab.clear_data()
        self.data_store.clear()
