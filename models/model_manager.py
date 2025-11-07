# models/model_manager.py
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Union, List

import numpy as np

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class ModelManager:
    """
    ModelLoader가 로드해 준 모델/메타데이터를 받아서
    실제 추론을 담당하는 매니저.

    - 장기별(lung / liver ...) 모델 선택
    - segmentation 예측
    - feature 파일을 바로 넣어서 예측
    """

    def __init__(self, loaded: Dict[str, Any]):
        """
        Parameters
        ----------
        loaded: dict
            models.model_loader.ModelLoader.load() 가 돌려준 dict
            이 dict 안에 아래 키들이 있을 수 있음:
            - "model": 기본 모델 (lung에 쓸 수도 있음)
            - "lung_model": 폐용 모델
            - "liver_model": 간용 모델
            - "spiral_idx": 공통 spiral 인덱스
            - "spiral_idx_lung": 폐 전용 spiral 인덱스
            - "spiral_idx_liver": 간 전용 spiral 인덱스
            - "device": torch device
        """
        # 공통 / 기본
        self.device = loaded.get("device")
        self.threshold: float = loaded.get("threshold", 0.5)
        self.model_accuracy: Optional[float] = loaded.get("model_accuracy")
        self.expected_vertex_count: Optional[int] = loaded.get("expected_vertex_count")

        # 모델들
        base_model = loaded.get("model")  # 예전 구조 대비
        self.lung_model = loaded.get("lung_model", base_model)
        self.liver_model = loaded.get("liver_model")  # 없을 수 있음

        # spiral 인덱스도 장기별로 다를 수 있으니 둘 다 받아둔다
        common_spiral = loaded.get("spiral_idx")
        self.spiral_idx_lung = loaded.get("spiral_idx_lung", common_spiral)
        self.spiral_idx_liver = loaded.get("spiral_idx_liver", common_spiral)

        # torch 사용 가능 여부는 "어떤 모델이든 하나라도 있고 device도 있을 때"만 True
        self.torch_available = (
            TORCH_AVAILABLE
            and self.device is not None
            and (
                self.lung_model is not None
                or self.liver_model is not None
                or base_model is not None
            )
        )

    # ------------------------------------------------------------------
    # 내부 유틸: 장기 이름으로 모델/spiral을 고른다
    # ------------------------------------------------------------------
    def _select_model_and_spiral(
        self,
        organ: str,
    ) -> (Any, Optional[List[Any]]):
        """
        organ 에 맞는 (model, spiral_idx_list) 를 돌려준다.
        organ 이 "liver"인데 간 모델이 없으면 RuntimeError.
        organ 이 없거나 잘못되면 lung을 기본으로 본다.
        """
        organ = (organ or "lung").lower()

        if organ == "liver":
            model = self.liver_model
            spiral_idx = self.spiral_idx_liver
            if model is None:
                raise RuntimeError("간(liver) 모델이 로드되지 않았습니다.")
        else:  # default = lung
            model = self.lung_model
            spiral_idx = self.spiral_idx_lung
            if model is None:
                raise RuntimeError("폐(lung) 모델이 로드되지 않았습니다.")

        return model, spiral_idx

    # ------------------------------------------------------------------
    # 내부: 확률 예측 (장기별 모델로)
    # ------------------------------------------------------------------
    def _infer_probabilities_for_organ(
        self,
        organ: str,
        features: np.ndarray,
    ) -> np.ndarray:
        """
        organ 에 맞는 모델로 확률을 예측한다.
        """
        if not self.torch_available:
            raise RuntimeError("모델이 로드되지 않았거나 PyTorch를 사용할 수 없습니다.")

        model, spiral_idx = self._select_model_and_spiral(organ)

        model.eval()

        # (1, N, C)로 맞춰서 모델에 넣음
        with torch.no_grad():
            x = (
                torch.tensor(features, dtype=torch.float32)
                .unsqueeze(0)  # batch 차원
                .to(self.device)
            )

            # spiral_idx 는 원래 list[Tensor] 형태일 걸 가정
            if spiral_idx is None:
                raise RuntimeError("spiral 인덱스가 로드되지 않았습니다.")

            spiral_idx_batch = [idx.unsqueeze(0) for idx in spiral_idx]

            logits = model(x, spiral_idx_batch)
            logits = logits.detach().cpu().numpy()

        # (1, N) 형태면 (N,)로 펴기
        if logits.ndim == 2:
            logits = logits[0]

        # 시그모이드로 확률화
        probs = 1.0 / (1.0 + np.exp(-logits))
        return probs.astype(np.float32, copy=False)

    # ------------------------------------------------------------------
    # 외부: 장기 지정 세그멘테이션
    # ------------------------------------------------------------------
    def predict_for_organ(
        self,
        organ: str,
        features: np.ndarray,
        *,
        return_probabilities: bool = False,
    ):
        """
        organ 에 맞는 모델로 세그멘테이션 예측을 수행한다.
        organ: "lung" | "liver" ...
        """
        probs = self._infer_probabilities_for_organ(organ, features)
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
        """
        예전 코드 호환용. 기본적으로 폐(lung) 모델로 예측한다.
        """
        return self.predict_for_organ(
            "lung", features, return_probabilities=return_probabilities
        )

    # ------------------------------------------------------------------
    # feature 파일에서 바로 예측
    # ------------------------------------------------------------------
    def predict_from_feature_file(
        self,
        feature_file: Union[Path, str],
        organ: str = "lung",
    ) -> Dict[str, Any]:
        """
        사전 계산된 feature npy 파일을 바로 읽어서 예측까지 한 번에 수행한다.
        organ 을 지정할 수 있게 해 둔다.
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
            raise ValueError(
                "Feature array must be 2D ndarray with at least 3 columns."
            )

        features = np.asarray(features, dtype=np.float32)

        preds, probs = self.predict_for_organ(
            organ, features, return_probabilities=True
        )
        verts = features[:, :3].astype(np.float32, copy=False)

        return {
            "features": features,
            "verts": verts,
            "predictions": preds,
            "probabilities": probs,
            "accuracy": self.model_accuracy,
            "organ": organ,
        }
