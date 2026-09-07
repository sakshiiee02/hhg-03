"""
Unit tests for face detection, embedding normalization, cosine similarity,
quality gating, and matching on sample public figures.
"""

from pathlib import Path
import cv2
import numpy as np
import pytest

from src.face.detector import FaceDetector
from src.face.embedder import compute_cosine_similarity, normalize_embedding
from src.face.matcher import FaceMatcher, MatchVerdict
from src.face.quality import calculate_sharpness, check_face_quality


def test_cosine_similarity_properties():
    # Identical vectors
    v1 = np.random.randn(512)
    assert pytest.approx(compute_cosine_similarity(v1, v1), 1e-5) == 1.0

    # Opposite vectors
    assert pytest.approx(compute_cosine_similarity(v1, -v1), 1e-5) == -1.0

    # Orthogonal vectors
    v2 = np.zeros(512)
    v2[0] = 1.0
    v3 = np.zeros(512)
    v3[1] = 1.0
    assert pytest.approx(compute_cosine_similarity(v2, v3), 1e-5) == 0.0


def test_quality_gate_blur_and_resolution():
    # Blurred synthetic image
    sharp_img = np.zeros((100, 100, 3), dtype=np.uint8)
    cv2.circle(sharp_img, (50, 50), 30, (255, 255, 255), -1)
    blurred_img = cv2.GaussianBlur(sharp_img, (25, 25), 0)

    sharpness_sharp = calculate_sharpness(sharp_img)
    sharpness_blur = calculate_sharpness(blurred_img)
    assert sharpness_sharp > sharpness_blur

    # Too small face bbox
    passes, reason, _ = check_face_quality([0, 0, 20, 20], sharp_img, min_size=40)
    assert passes is False
    assert "smaller than minimum" in reason


def test_detector_on_jensen_huang():
    image_path = Path("examples/jensen_huang_portrait.jpg")
    if not image_path.exists():
        pytest.skip("Example image not present")

    detector = FaceDetector.get_shared_instance()
    faces = detector.detect(image_path)
    assert len(faces) >= 1

    primary = detector.select_primary_face(faces)
    assert primary.confidence >= 0.80
    assert primary.passes_quality is True
    assert len(primary.embedding) == 512
    # Verify unit norm
    assert pytest.approx(np.linalg.norm(primary.embedding), 1e-4) == 1.0


def test_cross_person_matching_discrimination():
    """Verify that Jensen Huang and negative control exhibit low similarity (< 0.60)."""
    p_jensen = Path("examples/jensen_huang_portrait.jpg")
    p_negative = Path("examples/negative_control.jpg")
    if not p_jensen.exists() or not p_negative.exists():
        pytest.skip("Example images not present")

    detector = FaceDetector.get_shared_instance()
    faces_j = detector.detect(p_jensen)
    faces_n = detector.detect(p_negative)

    emb_jensen = detector.select_primary_face(faces_j).embedding
    emb_negative = detector.select_primary_face(faces_n).embedding

    similarity = compute_cosine_similarity(emb_jensen, emb_negative)
    # Impostor pairs should be significantly below 0.60
    assert similarity < 0.60, f"Impostor pair similarity unexpectedly high: {similarity:.4f}"
