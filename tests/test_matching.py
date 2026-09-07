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


def test_detector_on_sam_altman():
    image_path = Path("examples/sam_altman_portrait.jpg")
    if not image_path.exists():
        pytest.skip("Example image not present")

    detector = FaceDetector.get_shared_instance()
    faces = detector.detect(image_path)
    assert len(faces) == 1

    primary = detector.select_primary_face(faces)
    assert primary.confidence >= 0.80
    assert primary.passes_quality is True
    assert len(primary.embedding) == 512
    assert pytest.approx(np.linalg.norm(primary.embedding), 1e-4) == 1.0


def test_detector_on_multiple_faces_group(tmp_path):
    image_path = Path("examples/multiple_faces_group.jpg")
    if not image_path.exists():
        pytest.skip("Example image not present")

    detector = FaceDetector.get_shared_instance()
    faces = detector.detect(image_path)
    assert len(faces) >= 2
    assert len(faces) == 3

    # Verify that each detected face can be cropped and saved to disk for individual reverse search
    for f in faces:
        dest_crop = tmp_path / f"crop_face_{f.face_index}.jpg"
        saved = detector.save_face_crop(image_path, f.bbox, dest_crop, padding_ratio=0.35)
        assert saved is True
        assert dest_crop.exists()
        assert dest_crop.stat().st_size > 1000  # valid image file
        # Verify saved crop can be loaded and has valid dimensions
        crop_img = cv2.imread(str(dest_crop))
        assert crop_img is not None
        assert crop_img.shape[0] >= 200
        assert crop_img.shape[1] >= 150


def test_detector_on_low_quality_blur_face():
    image_path = Path("examples/low_quality_blur_face.jpg")
    if not image_path.exists():
        pytest.skip("Example image not present")

    detector = FaceDetector.get_shared_instance()
    faces = detector.detect(image_path)
    assert len(faces) >= 1
    # Quality filter must reject high blur
    primary = detector.select_primary_face(faces)
    assert primary.passes_quality is False
    assert "sharpness" in primary.quality_reason.lower() or "blur" in primary.quality_reason.lower()


def test_detector_on_no_face_landscape():
    image_path = Path("examples/no_face_landscape.jpg")
    if not image_path.exists():
        pytest.skip("Example image not present")

    detector = FaceDetector.get_shared_instance()
    faces = detector.detect(image_path)
    assert len(faces) == 0


def test_cross_person_matching_discrimination():
    """Verify that Sam Altman and an impostor face exhibit low similarity (< 0.60)."""
    p_sam = Path("examples/sam_altman_portrait.jpg")
    p_multi = Path("examples/multiple_faces_group.jpg")
    if not p_sam.exists() or not p_multi.exists():
        pytest.skip("Example images not present")

    detector = FaceDetector.get_shared_instance()
    faces_sam = detector.detect(p_sam)
    faces_multi = detector.detect(p_multi)

    emb_sam = detector.select_primary_face(faces_sam).embedding
    emb_impostor = detector.select_primary_face(faces_multi).embedding

    similarity = compute_cosine_similarity(emb_sam, emb_impostor)
    assert similarity < 0.60, f"Impostor pair similarity unexpectedly high: {similarity:.4f}"
