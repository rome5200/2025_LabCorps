# dashboard_main.py
import logging
import matplotlib as mpl

# 윈도우에서 있는 한글 폰트로 덮어쓰기
mpl.rcParams["font.family"] = ["Malgun Gothic", "NanumGothic", "sans-serif"]
mpl.rcParams["axes.unicode_minus"] = False

# AppleGothic 못 찾았다는 경고 안 보이게
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)

import sys
import os
from pathlib import Path

from PyQt6.QtWidgets import QApplication, QMainWindow

# ① 대시보드 페이지
from pages.dashboard_page import DashboardPage

# ② 모델 관련
from models.model_loader import ModelLoader   # 실제 모델/스파이럴 로더
from models.model_manager import ModelManager


def load_model_manager() -> ModelManager:
    """
    모델을 로드해서 ModelManager에 넘겨준다.
    models/model_loader.py가 lung/liver 둘 다 로드하도록 되어 있으므로
    실패 fallback도 그 키 구조에 맞춰준다.
    """
    base_path = Path(__file__).resolve().parent

    try:
        loader = ModelLoader(base_path=base_path)
        loaded_dict = loader.load()
        return ModelManager(loaded_dict)
    except Exception as e:
        # 모델 파일이 없거나 torch가 없거나 경로가 안 맞는 경우
        print(f"[dashboard_main] 모델 로드 실패: {e}")

        fallback = {
            "device": None,
            "expected_vertex_count": None,
            "spiral_idx": None,
            "lung_model": None,
            "threshold": 0.5,
            "model_accuracy": None,
        }
        return ModelManager(fallback)


class DashboardWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        # 실제 모델 로드
        model_manager = load_model_manager()

        # 파이프라인/뷰어와 공유할 저장소
        data_store = {}

        # 대시보드 페이지 붙이기
        dashboard = DashboardPage(model_manager=model_manager, data_store=data_store)
        self.setCentralWidget(dashboard)

        self.setWindowTitle("CT-DICOM 기반 3D 재구성 및 폐 결절 탐지 연구")
        self.resize(1400, 800)


def main():
    # 일부 환경에서 MKL 충돌 때문에 필요할 수 있음
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

    app = QApplication(sys.argv)

    win = DashboardWindow()
    win.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
