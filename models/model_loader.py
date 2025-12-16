# models/model_loader.py
from __future__ import annotations

from pathlib import Path
import numpy as np
from utils.helper import resource_path  # 프로젝트 루트 기준 경로 가져오기 :contentReference[oaicite:5]{index=5}

try:
    from torch.serialization import add_safe_globals
    import torch
    from models.spiralnet import (
        HybridSpiralNetPointNetTransformer,
        ImprovedSpiralNet,
    )
    TORCH_AVAILABLE = True
except ImportError as e:
    print(f"PyTorch 임포트 실패: {e}")
    TORCH_AVAILABLE = False


class ModelLoader:
    """
    모델과 필요한 부가 리소스(spiral 인덱스, threshold, accuracy)를 로드해서
    models/model_manager.py 가 바로 쓸 수 있는 dict로 돌려준다. :contentReference[oaicite:6]{index=6}
    """

    def __init__(self, base_path: Path | None = None):
        self.torch_available = TORCH_AVAILABLE
        self.base_path = (
            Path(base_path)
            if base_path is not None
            else Path(__file__).resolve().parents[1]
        )

        self.SPIRAL_PATH = resource_path("models/spiral_9.npy")
        self.LUNG_WEIGHT_PATH = resource_path("models/new_2.pth")

    def _build_backbone(self, spiral_idx_list, device):
        num_blocks = len(spiral_idx_list)

        spiralnet = ImprovedSpiralNet(
            in_channels=7,
            hidden_channels=256,
            out_channels=1,
            num_blocks=num_blocks,
            spiral_indices=spiral_idx_list,
            dropout=0.3,
            use_se=True,
        ).to(device)

        model = HybridSpiralNetPointNetTransformer(
            spiralnet=spiralnet,
            in_channels=7,
            hidden_dim=256,
            out_dim=1,
            dropout=0.3,
            transformer_heads=4,
            transformer_blocks=1,
        ).to(device)

        return model

    def _load_weight(self, model, weight_path: Path, device):
        add_safe_globals([np.core.multiarray.scalar, np.dtype])
        ckpt = torch.load(weight_path, map_location=device, weights_only=False)
        state = ckpt.get("model_state_dict", ckpt)
        model.load_state_dict(state, strict=True)

        threshold = 0.5
        accuracy = None
        if isinstance(ckpt, dict):
            threshold = float(ckpt.get("threshold", 0.5))
            acc = ckpt.get("accuracy")
            if acc is not None:
                try:
                    accuracy = float(acc)
                except (TypeError, ValueError):
                    accuracy = None

        model.eval()
        return model, threshold, accuracy

    def load(self):
        if not self.torch_available:
            print("PyTorch가 사용할 수 없습니다. 더미 예측만 반환합니다.")
            return {
                "device": None,
                "spiral_idx": None,
                "spiral_idx_lung": spiral_idx_list,
                "expected_vertex_count": None,
                "lung_model": None,
                "threshold": 0.5,
                "model_accuracy": None,
            }

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if not self.SPIRAL_PATH.exists():
            raise FileNotFoundError(f"Spiral index file not found: {self.SPIRAL_PATH}")

        spiral_all = np.load(self.SPIRAL_PATH, allow_pickle=True)
        expected_vertex_count = int(spiral_all.shape[0])

        # spiral 리스트 만들기
        num_blocks = 4
        spiral_idx_list = [
            torch.from_numpy(spiral_all).long().to(device) for _ in range(num_blocks)
        ]

        # 1) 폐 모델
        if not self.LUNG_WEIGHT_PATH.exists():
            raise FileNotFoundError(f"Lung model weight file not found: {self.LUNG_WEIGHT_PATH}")
        lung_model = self._build_backbone(spiral_idx_list, device)
        lung_model, lung_thr, lung_acc = self._load_weight(
            lung_model, self.LUNG_WEIGHT_PATH, device
        )

        print(f"모델 로드 완료 (device={device})")

        return {
            "device": device,
            "expected_vertex_count": expected_vertex_count,
            "spiral_idx": spiral_idx_list,
            "spiral_idx_lung": spiral_idx_list,
            "lung_model": lung_model,
            "threshold": lung_thr,
            "model_accuracy": lung_acc,
        }
