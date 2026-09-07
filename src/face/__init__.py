"""Face detection, recognition, quality filtering, and matching package."""

from src.face.detector import DetectedFace, FaceDetector
from src.face.embedder import compute_cosine_similarity, normalize_embedding
from src.face.matcher import (
    CandidateFaceScore,
    CandidateVerificationResult,
    FaceMatcher,
    MatchVerdict,
)
from src.face.quality import calculate_sharpness, check_face_quality

__all__ = [
    "DetectedFace",
    "FaceDetector",
    "compute_cosine_similarity",
    "normalize_embedding",
    "CandidateFaceScore",
    "CandidateVerificationResult",
    "FaceMatcher",
    "MatchVerdict",
    "calculate_sharpness",
    "check_face_quality",
]
