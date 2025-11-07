# utils/lung_3d_viewer.py
import numpy as np
from PyQt6.QtWidgets import QWidget, QVBoxLayout
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


class Lung3DViewer(QWidget):
    """간단한 3D 폐/결절 시각화용 위젯"""

    def __init__(self, parent=None):
        super().__init__(parent)

        self.fig = Figure(figsize=(5, 5))
        self.canvas = FigureCanvas(self.fig)
        self.ax = self.fig.add_subplot(111, projection="3d")

        layout = QVBoxLayout()
        layout.addWidget(self.canvas)
        self.setLayout(layout)

    def update_plot(self, verts, predictions=None, ground_truth_mask=None, title="3D Viewer"):
        """verts (N, 3): 정점 좌표
        predictions: 예측 결과 (True/False mask)
        ground_truth_mask: 실제 라벨 (True/False mask)
        """

        self.ax.clear()
        self.ax.set_title(title)

        if verts is None or len(verts) == 0:
            self.ax.text(0.5, 0.5, 0.5, "데이터 없음", color="gray")
            self.canvas.draw()
            return

        # 기본점: 전체 구조
        self.ax.scatter(
            verts[:, 0], verts[:, 1], verts[:, 2],
            c="gray", s=1, alpha=0.2, label="Structure"
        )

        # 예측 결과 (결절)
        if predictions is not None and np.any(predictions):
            pos = verts[predictions.astype(bool)]
            self.ax.scatter(pos[:, 0], pos[:, 1], pos[:, 2],
                            c="red", s=6, alpha=0.6, label="Prediction")

        # 실제 결과 (ground truth)
        if ground_truth_mask is not None and np.any(ground_truth_mask):
            gt = verts[ground_truth_mask.astype(bool)]
            self.ax.scatter(gt[:, 0], gt[:, 1], gt[:, 2],
                            c="cyan", s=6, alpha=0.6, label="Ground Truth")

        self.ax.legend(loc="upper right")
        self.ax.set_xlabel("X")
        self.ax.set_ylabel("Y")
        self.ax.set_zlabel("Z")
        self.ax.view_init(elev=30, azim=45)
        self.canvas.draw()
