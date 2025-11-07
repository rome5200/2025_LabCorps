# pipelines/common_pipeline.py

from __future__ import annotations

from pathlib import Path
from typing import Optional, Dict, Any, Callable, Tuple

import numpy as np

from utils.config import FEATURES_DIR, LABELS_DIR
from utils.dicom_processor import DICOMProcessor
from utils.mesh_processor import MeshProcessor

# 진행상황 콜백 타입
ProgressCB = Optional[Callable[[int, str], None]]


class BaseCTPipeline:
    """
    CT/DICOM을 읽고 공통 경로를 만들어주는 베이스 클래스.
    여기까지는 '장기' 개념이 없음.
    """
    features_dir: Path = FEATURES_DIR
    labels_dir: Path = LABELS_DIR

    def _progress(self, cb: ProgressCB, pct: int, msg: str) -> None:
        if cb is not None:
            cb(pct, msg)

    def _extract_id_from_folder(self, folder: Path) -> str:
        name = folder.name
        if "-" in name:
            return name.split("-")[-1]
        return name

    def _get_feature_path(self, pid: str) -> Optional[Path]:
        cand = self.features_dir / f"{pid}_features.npy"
        return cand if cand.exists() else None

    def _get_label_path(self, pid: str) -> Optional[Path]:
        cand = self.labels_dir / f"{pid}_vertex_labels.npy"
        return cand if cand.exists() else None

    def prepare_paths(self, folder: Path) -> Tuple[str, Optional[Path], Optional[Path]]:
        """
        폴더명에서 환자 ID를 뽑고, 해당 ID로 feature/label 경로를 찾아준다.
        """
        pid = self._extract_id_from_folder(folder)
        feature_path = self._get_feature_path(pid)
        label_path = self._get_label_path(pid)
        return pid, feature_path, label_path

    def load_ct(self, folder: Path, progress_cb: ProgressCB = None) -> Dict[str, Any]:
        """
        DICOM 폴더를 읽어서 image, mask, spacing, slices 를 돌려준다.
        """
        self._progress(progress_cb, 10, "DICOM/CT 데이터 불러오는 중...")
        dcm = DICOMProcessor()
        result = dcm.load_and_process(str(folder))
        self._progress(progress_cb, 20, "DICOM/CT 로드 완료")
        return result

    def make_result_dict(
        self,
        *,
        image: np.ndarray,
        verts: Optional[np.ndarray],
        predictions: Optional[np.ndarray],
        probabilities: Optional[np.ndarray],
        labels: Optional[np.ndarray],
        mesh_path: Optional[str],
        model_accuracy: Optional[float],
        prediction_accuracy: Optional[float],
        selected_folder: str,
        feature_file: Optional[str],
        organ: Optional[str] = None,
    ) -> Dict[str, Any]:
        return {
            "image": image,
            "verts": verts,
            "predictions": predictions,
            "probabilities": probabilities,
            "labels": labels,
            "mesh_path": mesh_path,
            "model_accuracy": model_accuracy,
            "prediction_accuracy": prediction_accuracy,
            "selected_folder": selected_folder,
            "feature_file": feature_file,
            "organ": organ,
        }


class OrganCTPipeline(BaseCTPipeline):
    """
    장기 이름만 다르고 나머지 흐름이 같은 파이프라인.
    Dashboard나 Thread에서는 그냥 OrganCTPipeline("lung") / ("liver") 이런 식으로 쓰면 됨.
    """

    def __init__(self, organ: str):
        super().__init__()
        self.organ = organ  # "lung" or "liver" ...

    def run(
        self,
        folder_path: str,
        model_manager,
        progress_cb: ProgressCB = None,
    ) -> Dict[str, Any]:
        folder = Path(folder_path)

        # 1) 경로 준비
        pid, feature_path, label_path = self.prepare_paths(folder)

        # 2) CT 로드
        ct_result = self.load_ct(folder, progress_cb)
        image = ct_result["image"]
        slices = ct_result.get("slices")

        # 3) 사전 feature가 있는 경우
        if feature_path is not None and feature_path.exists():
            self._progress(progress_cb, 30, f"사전 계산 feature 발견: {feature_path.name}")
            features = np.load(str(feature_path)).astype(np.float32, copy=False)

            if features.ndim != 2 or features.shape[1] < 3:
                raise ValueError(f"feature 데이터 형태가 잘못되었습니다: {features.shape!r}")

            verts = features[:, :3].astype(np.float32, copy=False)

            # organ에 맞는 모델로 예측
            self._progress(progress_cb, 50, "모델 예측 중...")
            preds, probs = model_manager.predict_for_organ(
                self.organ,
                features,
                return_probabilities=True,
            )

            # 라벨이 있으면 길이가 맞을 때만
            labels = self._load_labels_if_match(label_path, verts.shape[0], progress_cb)

            prediction_accuracy = None
            if labels is not None:
                prediction_accuracy = float((preds.astype(int) == labels.astype(int)).mean())

            self._progress(progress_cb, 100, "처리 완료")

            return self.make_result_dict(
                image=image,
                verts=verts,
                predictions=preds,
                probabilities=probs,
                labels=labels,
                mesh_path=None,
                model_accuracy=getattr(model_manager, "model_accuracy", None),
                prediction_accuracy=prediction_accuracy,
                selected_folder=folder.name,
                feature_file=str(feature_path),
                organ=self.organ,
            )

        # 4) 사전 feature가 없으면 → 메쉬 생성 후 추출
        self._progress(progress_cb, 40, "사전 feature가 없어 메쉬를 생성합니다...")
        mesh_proc = MeshProcessor()
        mesh, mesh_path = mesh_proc.create_mesh(image, slices, tmpdir=None)

        self._progress(progress_cb, 60, "정점 특징 추출 중...")
        verts, feat, transform, dists = mesh_proc.extract_features(mesh)

        self._progress(progress_cb, 70, "모델 예측 중...")
        preds, probs = model_manager.predict_for_organ(
            self.organ,
            feat,
            return_probabilities=True,
        )

        labels = self._load_labels_if_match(label_path, verts.shape[0], progress_cb)

        prediction_accuracy = None
        if labels is not None and labels.size == preds.size:
            prediction_accuracy = float((preds.astype(int) == labels.astype(int)).mean())

        self._progress(progress_cb, 100, "처리 완료")

        return self.make_result_dict(
            image=image,
            verts=verts,
            predictions=preds,
            probabilities=probs,
            labels=labels,
            mesh_path=mesh_path,
            model_accuracy=getattr(model_manager, "model_accuracy", None),
            prediction_accuracy=prediction_accuracy,
            selected_folder=folder.name,
            feature_file=None,
            organ=self.organ,
        )

    # ------------------------------------------------------------------
    # 내부: 라벨 로드
    # ------------------------------------------------------------------
    def _load_labels_if_match(
        self,
        label_path: Optional[Path],
        n_verts: int,
        progress_cb: ProgressCB = None,
    ) -> Optional[np.ndarray]:
        if label_path is None or not label_path.exists():
            return None

        lab = np.load(str(label_path)).astype(bool)
        if lab.size == n_verts:
            self._progress(progress_cb, 65, f"라벨 로드 완료: {label_path.name}")
            return lab
        else:
            self._progress(
                progress_cb,
                65,
                f"라벨 길이가 맞지 않아 무시: {lab.size} vs {n_verts}",
            )
            return None
