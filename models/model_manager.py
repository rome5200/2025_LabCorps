# models/model_manager.py
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Union, List, Tuple

import numpy as np

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class ModelManager:
    def __init__(self, loaded: Dict[str, Any]):
        """
        Parameters
        ----------
        loaded: dict
            models.model_loader.ModelLoader.load() 가 돌려준 dict
            이 dict 안에 아래 키들이 있을 수 있음:
            - "model": 기본 모델 (구버전 호환)
            - "lung_model": 폐용 모델 (권장)
            - "spiral_idx": 공통 spiral 인덱스 (구버전 호환)
            - "spiral_idx_lung": 폐 전용 spiral 인덱스 (권장)
            - "device": torch device
            - "threshold": segmentation threshold (default: 0.5)
            - "model_accuracy": optional float
            - "expected_vertex_count": optional int
        """
        # 공통 / 기본 메타
        self.device = loaded.get("device")
        self.threshold: float = float(loaded.get("threshold", 0.5))
        self.model_accuracy: Optional[float] = loaded.get("model_accuracy")
        self.expected_vertex_count: Optional[int] = loaded.get("expected_vertex_count")

        # 모델 (lung-only)
        base_model = loaded.get("model")  # 구버전 호환
        self.lung_model = loaded.get("lung_model", base_model)

        # spiral 인덱스 (lung-only)
        common_spiral = loaded.get("spiral_idx")  # 구버전 호환
        self.spiral_idx_lung = loaded.get("spiral_idx_lung", common_spiral)

        # torch 사용 가능 여부: lung 모델과 device가 있어야 함
        self.torch_available = (
            TORCH_AVAILABLE
            and self.device is not None
            and self.lung_model is not None
        )

    # ------------------------------------------------------------------
    # 내부 유틸: lung 모델/spiral 선택
    # ------------------------------------------------------------------
    def _select_model_and_spiral(self) -> Tuple[Any, Optional[List[Any]]]:
        """
        lung-only: (lung_model, spiral_idx_lung) 를 반환한다.
        """
        model = self.lung_model
        spiral_idx = self.spiral_idx_lung

        if model is None:
            raise RuntimeError("폐(lung) 모델이 로드되지 않았습니다.")
        if spiral_idx is None:
            raise RuntimeError("spiral 인덱스가 로드되지 않았습니다.")

        return model, spiral_idx

    # ------------------------------------------------------------------
    # 내부: 확률 예측 (lung-only)
    # ------------------------------------------------------------------
    def _infer_probabilities(self, features: np.ndarray) -> np.ndarray:
        """
        lung 모델로 확률을 예측한다.
        """
        if not self.torch_available:
            raise RuntimeError("모델이 로드되지 않았거나 PyTorch를 사용할 수 없습니다.")

        model, spiral_idx = self._select_model_and_spiral()
        model.eval()

        # (1, N, C) 형태로 맞춰서 모델에 입력
        with torch.no_grad():
            x = (
                torch.tensor(features, dtype=torch.float32)
                .unsqueeze(0)  # batch dimension
                .to(self.device)
            )

            # spiral_idx 는 list[Tensor] 형태를 가정
            spiral_idx_batch = [idx.unsqueeze(0) for idx in spiral_idx]

            logits = model(x, spiral_idx_batch)
            logits = logits.detach().cpu().numpy()

        # (1, N) -> (N,)
        if logits.ndim == 2:
            logits = logits[0]

        # sigmoid
        probs = 1.0 / (1.0 + np.exp(-logits))
        return probs.astype(np.float32, copy=False)

    # ------------------------------------------------------------------
    # 외부: organ 인자를 받더라도 lung-only로 처리
    # ------------------------------------------------------------------
    def predict_for_organ(
        self,
        organ: str,
        features: np.ndarray,
        *,
        return_probabilities: bool = False,
    ):
        """
        lung-only: 호환성을 위해 organ 인자를 받지만, 무시하고 폐 모델로 예측한다.
        """
        _ = organ  # intentionally ignored (lung-only)
        probs = self._infer_probabilities(features)
        preds = probs > self.threshold

        if return_probabilities:
            return preds, probs
        return preds

    # ------------------------------------------------------------------
    # 외부: 기존 코드 호환용 (장기 미지정 → 폐로 취급)
    # ------------------------------------------------------------------
    def predict_segmentation(
        self,
        features: np.ndarray,
        *,
        return_probabilities: bool = False,
    ):
        probs = self._infer_probabilities(features)
        preds = probs > self.threshold

        if return_probabilities:
            return preds, probs
        return preds

    # ------------------------------------------------------------------
    # feature 파일에서 바로 예측
    # ------------------------------------------------------------------
    def predict_from_feature_file(
        self,
        feature_file: Union[Path, str],
        organ: str = "lung",
    ) -> Dict[str, Any]:
        """
        사전 계산된 feature npy 파일을 읽어서 예측까지 한 번에 수행한다.
        lung-only: organ 인자를 받더라도 결과는 lung 기준이며, 반환값의 organ도 "lung"로 고정한다.
        """
        feature_path = Path(feature_file)
        if not feature_path.exists():
            raise FileNotFoundError("Feature file not found: %s" % feature_path)

        try:
            features = np.load(feature_path)
        except Exception as exc:
            raise ValueError("Failed to load feature file: %s" % exc)

        if (
            not isinstance(features, np.ndarray)
            or features.ndim != 2
            or features.shape[1] < 3
        ):
            raise ValueError("Feature array must be 2D ndarray with at least 3 columns.")

        features = np.asarray(features, dtype=np.float32)

        preds, probs = self.predict_segmentation(features, return_probabilities=True)
        verts = features[:, :3].astype(np.float32, copy=False)

        return {
            "features": features,
            "verts": verts,
            "predictions": preds,
            "probabilities": probs,
            "accuracy": self.model_accuracy,
            "organ": "lung",
        }
