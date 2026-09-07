"""
Face detection and primary subject selection using InsightFace SCRFD.
Supports landmark extraction, quality gating, and automatic primary face selection.
"""

import contextlib
from dataclasses import dataclass
import io
import logging
from pathlib import Path
from typing import List, Optional, Tuple, Union
import warnings

import cv2
from insightface.app import FaceAnalysis
import numpy as np
import onnxruntime

# Filter third-party deprecation warning originating inside insightface's use of scikit-image
warnings.filterwarnings("ignore", category=FutureWarning, module="insightface")
warnings.filterwarnings("ignore", message=".*estimate is deprecated.*")

from src.face.quality import check_face_quality

logger = logging.getLogger(__name__)


@dataclass
class DetectedFace:
    """Represents a detected face with localized landmarks, confidence, and embedding."""
    face_index: int
    bbox: Tuple[int, int, int, int]
    landmarks: np.ndarray
    confidence: float
    embedding: np.ndarray
    sharpness: float
    passes_quality: bool
    quality_reason: str

    @property
    def area(self) -> int:
        x1, y1, x2, y2 = self.bbox
        return max(0, x2 - x1) * max(0, y2 - y1)


class FaceDetector:
    """Wrapper around InsightFace buffalo_l detector & feature extractor."""

    _instance: Optional["FaceDetector"] = None

    def __init__(
        self,
        model_name: str = "buffalo_l",
        ctx_id: int = -1,
        det_size: Tuple[int, int] = (640, 640),
        silent: bool = True,
    ):
        # Prioritize DirectML (DirectX 12 on NVIDIA/AMD/Intel) and CUDA before CPU
        available_providers = onnxruntime.get_available_providers()
        if "DmlExecutionProvider" in available_providers:
            providers = ["DmlExecutionProvider", "CPUExecutionProvider"]
            resolved_ctx_id = 0 if ctx_id == -1 else ctx_id
        elif "CUDAExecutionProvider" in available_providers:
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
            resolved_ctx_id = 0 if ctx_id == -1 else ctx_id
        else:
            providers = ["CPUExecutionProvider"]
            resolved_ctx_id = -1

        # Prune unused neural networks (3D landmark, 2D landmark, gender/age)
        # We only need detection (det_10g) and recognition (ArcFace w600k_r50)
        allowed_modules = ["detection", "recognition"]

        # Suppress InsightFace's hardcoded stdout print statements during model loading
        if silent:
            with contextlib.redirect_stdout(io.StringIO()):
                self.app = FaceAnalysis(
                    name=model_name,
                    providers=providers,
                    allowed_modules=allowed_modules,
                )
                self.app.prepare(ctx_id=resolved_ctx_id, det_size=det_size)
        else:
            self.app = FaceAnalysis(
                name=model_name,
                providers=providers,
                allowed_modules=allowed_modules,
            )
            self.app.prepare(ctx_id=resolved_ctx_id, det_size=det_size)

    @classmethod
    def get_shared_instance(cls) -> "FaceDetector":
        """Singleton accessor to prevent reloading heavy models into memory across runs."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def detect(self, image_input: Union[str, Path, np.ndarray]) -> List[DetectedFace]:
        """
        Detects all faces in an image and extracts normalized ArcFace embeddings.
        Accepts a file path or a BGR numpy array.
        """
        if isinstance(image_input, (str, Path)):
            path = Path(image_input)
            if not path.exists():
                raise FileNotFoundError(f"Image not found at {path}")
            img_bgr = cv2.imread(str(path))
            if img_bgr is None:
                raise ValueError(f"Could not decode image at {path}")
        else:
            img_bgr = image_input

        raw_faces = self.app.get(img_bgr)
        detected: List[DetectedFace] = []

        for idx, f in enumerate(raw_faces):
            bbox = tuple(int(v) for v in f.bbox[:4])
            conf = float(f.det_score)
            emb = f.embedding
            # Ensure strict L2 normalization
            norm = np.linalg.norm(emb)
            if norm > 0:
                emb = emb / norm

            passes, reason, sharpness = check_face_quality(bbox, img_bgr, confidence=conf)

            detected.append(DetectedFace(
                face_index=idx,
                bbox=bbox,
                landmarks=f.kps,
                confidence=conf,
                embedding=emb,
                sharpness=sharpness,
                passes_quality=passes,
                quality_reason=reason,
            ))

        return detected

    def select_primary_face(
        self,
        faces: List[DetectedFace],
        face_index_override: Optional[int] = None,
    ) -> DetectedFace:
        """
        Selects the target face for matching.
        If face_index_override is specified, returns that index.
        Otherwise auto-selects the primary face by maximum confidence-weighted bounding box area.
        """
        if not faces:
            raise ValueError("No faces detected in the provided image.")

        if face_index_override is not None:
            for f in faces:
                if f.face_index == face_index_override:
                    return f
            raise IndexError(f"Face index {face_index_override} out of range (detected {len(faces)} faces).")

        # Select by highest confidence-weighted area
        return max(faces, key=lambda f: f.area * f.confidence)

    get_primary_face = select_primary_face

    def crop_face(
        self,
        image_input: Union[str, Path, np.ndarray],
        bbox: Tuple[int, int, int, int],
        padding_ratio: float = 0.2,
    ) -> Optional[np.ndarray]:
        """Crops a detected face region with optional padding."""
        if isinstance(image_input, (str, Path)):
            img_bgr = cv2.imread(str(image_input))
            if img_bgr is None:
                return None
        else:
            img_bgr = image_input

        h, w = img_bgr.shape[:2]
        x1, y1, x2, y2 = bbox
        bw, bh = x2 - x1, y2 - y1
        pad_x = int(bw * padding_ratio)
        pad_y = int(bh * padding_ratio)

        cx1 = max(0, x1 - pad_x)
        cy1 = max(0, y1 - pad_y)
        cx2 = min(w, x2 + pad_x)
        cy2 = min(h, y2 + pad_y)

        crop = img_bgr[cy1:cy2, cx1:cx2]
        return crop if crop.size > 0 else None

    def crop_face_b64(
        self,
        image_input: Union[str, Path, np.ndarray],
        bbox: Tuple[int, int, int, int],
        padding_ratio: float = 0.2,
    ) -> Optional[str]:
        """Crops a face and encodes it to a base64 JPEG data URI."""
        crop = self.crop_face(image_input, bbox, padding_ratio=padding_ratio)
        if crop is None:
            return None
        import base64
        success, buf = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 90])
        if not success:
            return None
        b64 = base64.b64encode(buf.tobytes()).decode("utf-8")
        return f"data:image/jpeg;base64,{b64}"
