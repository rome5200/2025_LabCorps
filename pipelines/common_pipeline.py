# pipelines/common_pipeline.py
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Callable, Tuple
import tempfile

import numpy as np

from utils import DICOMProcessor, MeshProcessor
from utils.config import FEATURES_DIR, FEATURES_DIRS, LABELS_DIR, LABELS_DIRS

ProgressCB = Optional[Callable[[int, str], None]]


class BaseCTPipeline:
    """
    DICOM → CT 정보 → feature / label 경로 준비까지 공통 파이프라인.
    장기(lung, liver) 구분은 하위 클래스에서 한다.
    """

    def __init__(self) -> None:
        self.features_dir: Path = FEATURES_DIR
        self.labels_dir: Path = LABELS_DIR

    # 공통 progress 헬퍼
    def _progress(self, cb: ProgressCB, pct: int, msg: str) -> None:
        if cb is not None:
            cb(int(pct), msg)

    # ---------------- ID/경로 처리 ----------------
    def _extract_id_from_folder(self, folder: Path) -> str:
        """
        폴더/zip 이름에서 케이스 ID를 만든다.
        예) 'case001.zip' -> 'case001'
        """
        name = folder.name
        if name.lower().endswith(".zip"):
            name = name[:-4]
        return name

    def _feature_path_for(self, pid: str) -> Path:
        return self.features_dir / f"{pid}_features.npy"

    def _label_path_for(self, pid: str) -> Path:
        return self.labels_dir / f"{pid}_vertex_labels.npy"

    def prepare_paths(self, folder: Path) -> Tuple[str, Path, Path]:
        pid = self._extract_id_from_folder(folder)
        return pid, self._feature_path_for(pid), self._label_path_for(pid)

    # ---------------- CT 로드 ----------------
    def load_ct(self, folder: Path, progress_cb: ProgressCB = None) -> Dict[str, Any]:
        # 1단계
        self._progress(progress_cb, 30, "압축/로드 중...")
        dcm = DICOMProcessor()
        ct = dcm.load_and_process(str(folder))
        return ct

    # ---------------- 라벨 로드 ----------------
    def _load_labels_if_valid(
        self,
        label_path: Path,
        n_verts: int,
        progress_cb: ProgressCB = None,
    ) -> Optional[np.ndarray]:
        if not label_path.exists():
            return None
        labels = np.load(str(label_path)).astype(bool)
        if labels.size == n_verts:
            self._progress(progress_cb, 95, "라벨 로드 완료")
            return labels
        self._progress(progress_cb, 95, "라벨 길이가 안 맞아 무시합니다.")
        return None

    # ---------------- 성능: F1 ----------------
    def _compute_f1(self, preds: np.ndarray, labels: np.ndarray) -> Optional[float]:
        """
        모델 학습에서 F1을 썼다면 여기서도 F1로 계산해 보여준다.
        단, 라벨이 전부 0이거나 전부 1이면 F1은 의미가 없으므로 None 반환.
        """
        if preds.shape != labels.shape:
            return None

        y_true = labels.astype(int)
        y_pred = preds.astype(int)

        pos_mask = y_true == 1
        neg_mask = y_true == 0
        n_pos = int(pos_mask.sum())
        n_neg = int(neg_mask.sum())

        # 한쪽 클래스만 있으면 F1 무의미
        if n_pos == 0 or n_neg == 0:
            return None

        tp = int(((y_pred == 1) & (y_true == 1)).sum())
        fp = int(((y_pred == 1) & (y_true == 0)).sum())
        fn = int(((y_pred == 0) & (y_true == 1)).sum())

        # 양성 예측이 하나도 맞지 않은 경우
        if tp == 0 and (fp > 0 or fn > 0):
            return 0.0

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

        if precision == 0.0 and recall == 0.0:
            return 0.0

        f1 = 2 * precision * recall / (precision + recall)
        return float(f1)

    # ---------------- 축단면 최대 직경 ----------------
    def _estimate_axial_max_diameter_mm(
        self,
        verts: Optional[np.ndarray],
        preds: Optional[np.ndarray],
        spacing: Optional[np.ndarray],
        z_tol_mm: float = 1.5,
    ) -> Optional[float]:
        """
        병원에서 보통 쓰는 표현에 가까운 값:
        - 예측된 결절점들 중
        - 같은 z(축단면)라고 볼 수 있는 점들을 묶어서
        - 그 단면에서의 x-y 평면 최대 거리(mm)를 구하고
        - 그중 가장 큰 값을 돌려준다.

        spacing: (z, y, x) 순서로 들어온다고 보고 x,y,z 에 맞춰서 곱해준다.
        z_tol_mm: z축이 이 값(mm) 안에 있으면 같은 슬라이스로 본다.
        """
        if verts is None or preds is None:
            return None

        mask = preds.astype(bool)
        if not mask.any():
            return None

        pts = verts[mask]  # (N, 3)  -> (x, y, z) 라고 가정
        if pts.shape[0] < 2:
            return None

        # spacing 처리
        if spacing is not None and len(spacing) == 3:
            sx = float(spacing[2])
            sy = float(spacing[1])
            sz = float(spacing[0])
        else:
            sx = sy = sz = 1.0

        # z 기준 정렬
        pts_sorted = pts[np.argsort(pts[:, 2])]
        n = pts_sorted.shape[0]
        max_diameter_mm = 0.0
        i = 0

        while i < n:
            z0 = pts_sorted[i, 2]
            same_idx = [i]
            j = i + 1
            # 같은 축단면으로 볼 수 있는 점들 모으기
            while j < n and abs(pts_sorted[j, 2] - z0) * sz <= z_tol_mm:
                same_idx.append(j)
                j += 1

            slice_pts = pts_sorted[same_idx]  # (k, 3)
            # x-y 평면 투영 + mm 변환
            xy = slice_pts[:, :2].copy()
            xy[:, 0] *= sx
            xy[:, 1] *= sy

            # 이 단면 내에서 최대 거리
            if xy.shape[0] >= 2:
                local_max = 0.0
                for a in range(xy.shape[0]):
                    diff = xy[a + 1:] - xy[a]
                    if diff.size == 0:
                        continue
                    dist = np.sqrt((diff ** 2).sum(axis=1))
                    local_max = max(local_max, float(dist.max()))
                max_diameter_mm = max(max_diameter_mm, local_max)

            i = j

        return max_diameter_mm if max_diameter_mm > 0 else None

    # ---------------- 결과 만들기 ----------------
    def make_result(
        self,
        *,
        image: np.ndarray,
        verts: Optional[np.ndarray],
        predictions: Optional[np.ndarray],
        probabilities: Optional[np.ndarray],
        labels: Optional[np.ndarray],
        mesh_path: Optional[str],
        model_accuracy: Optional[float],
        prediction_f1: Optional[float],
        selected_folder: str,
        feature_file: Optional[str],
        organ: Optional[str] = None,
        nodule_length_mm: Optional[float] = None,
    ) -> Dict[str, Any]:
        return {
            "image": image,
            "verts": verts,
            "predictions": predictions,
            "probabilities": probabilities,
            "labels": labels,
            "mesh_path": mesh_path,
            "model_accuracy": model_accuracy,
            "prediction_f1": prediction_f1,
            "selected_folder": selected_folder,
            "feature_file": feature_file,
            "organ": organ,
            "nodule_length_mm": nodule_length_mm,
        }


class OrganCTPipeline(BaseCTPipeline):
    """
    장기 이름만 다르고 나머지는 동일한 CT 파이프라인.
    """

    def __init__(self, organ: str):
        super().__init__()
        self.organ = organ
        # 장기별 디렉터리 적용
        self.features_dir = FEATURES_DIRS[organ]
        self.labels_dir = LABELS_DIRS[organ]

    def run(
        self,
        folder_path: str,
        model_manager,
        progress_cb: ProgressCB = None,
    ) -> Dict[str, Any]:
        folder = Path(folder_path)

        # 1) ID 및 feature/label 경로
        pid, feature_path, label_path = self.prepare_paths(folder)

        # 2) CT 로드
        ct = self.load_ct(folder, progress_cb)
        image = ct["image"]
        slices = ct["slices"]
        spacing = ct.get("spacing", None)

        # ── 사전 feature 있는 경우 ─────────────────────────
        if feature_path.exists():
            self._progress(progress_cb, 60, "features 추출 중...")
            features = np.load(str(feature_path)).astype(np.float32, copy=False)
            verts = features[:, :3].astype(np.float32, copy=False)

            self._progress(progress_cb, 90, "예측 중...")
            preds, probs = model_manager.predict_for_organ(
                self.organ,
                features,
                return_probabilities=True,
            )

            labels = self._load_labels_if_valid(label_path, verts.shape[0], progress_cb)

            f1 = None
            if labels is not None:
                f1 = self._compute_f1(preds, labels)

            # 병원식 축단면 최대 직경
            axial_len = self._estimate_axial_max_diameter_mm(verts, preds, spacing)

            self._progress(progress_cb, 100, "완료")

            return self.make_result(
                image=image,
                verts=verts,
                predictions=preds,
                probabilities=probs,
                labels=labels,
                mesh_path=None,
                model_accuracy=getattr(model_manager, "model_accuracy", None),
                prediction_f1=f1,
                selected_folder=pid,
                feature_file=str(feature_path),
                organ=self.organ,
                nodule_length_mm=axial_len,
            )

        # ── 사전 feature 없으면 메쉬부터 ───────────────────
        self._progress(progress_cb, 40, "메쉬 생성 중...")
        mesh_proc = MeshProcessor()
        with tempfile.TemporaryDirectory() as tmpdir:
            mesh, mesh_path = mesh_proc.create_mesh(image, slices, tmpdir=tmpdir)

        self._progress(progress_cb, 60, "features 추출 중...")
        verts, feat, transform, dists = mesh_proc.extract_features(mesh)

        self._progress(progress_cb, 90, "예측 중...")
        preds, probs = model_manager.predict_for_organ(
            self.organ,
            feat,
            return_probabilities=True,
        )

        labels = self._load_labels_if_valid(label_path, verts.shape[0], progress_cb)

        f1 = None
        if labels is not None and labels.size == preds.size:
            f1 = self._compute_f1(preds, labels)

        axial_len = self._estimate_axial_max_diameter_mm(verts, preds, spacing)

        self._progress(progress_cb, 100, "완료")

        return self.make_result(
            image=image,
            verts=verts,
            predictions=preds,
            probabilities=probs,
            labels=labels,
            mesh_path=mesh_path,
            model_accuracy=getattr(model_manager, "model_accuracy", None),
            prediction_f1=f1,
            selected_folder=pid,
            feature_file=None,
            organ=self.organ,
            nodule_length_mm=axial_len,
        )
