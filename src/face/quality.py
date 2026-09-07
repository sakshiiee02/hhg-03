"""
Quality gating and image sanity checks for detected face crops.
Prevents low-resolution, occluded, or motion-blurred imagery from corrupting identity matching.
"""

from typing import Sequence, Tuple, Union
import cv2
import numpy as np


def calculate_sharpness(image_bgr: np.ndarray) -> float:
    """
    Calculates the sharpness of an image crop using the variance of the Laplacian.
    Higher values correspond to sharper edges; low values signify motion blur or defocus.
    """
    if image_bgr is None or image_bgr.size == 0:
        return 0.0
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY) if len(image_bgr.shape) == 3 else image_bgr
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def check_face_quality(
    bbox: Union[Sequence[int], Sequence[float], np.ndarray],
    image_bgr: np.ndarray,
    min_size: int = 40,
    min_sharpness: float = 35.0,
    confidence: float = 1.0,
    min_confidence: float = 0.60,
) -> Tuple[bool, str, float]:
    """
    Applies quality gates to a detected face bounding box.

    Returns:
        (passes_gate, reason, sharpness_score)
    """
    if confidence < min_confidence:
        return False, f"Detection confidence {confidence:.2f} below threshold {min_confidence:.2f}", 0.0

    x1, y1, x2, y2 = [int(v) for v in bbox[:4]]
    w = max(0, x2 - x1)
    h = max(0, y2 - y1)

    if w < min_size or h < min_size:
        return False, f"Face dimensions ({w}x{h}) smaller than minimum ({min_size}x{min_size})", 0.0

    h_img, w_img = image_bgr.shape[:2]
    # Bound crop coordinates safely within image dimensions
    cx1, cy1 = max(0, x1), max(0, y1)
    cx2, cy2 = min(w_img, x2), min(h_img, y2)

    if cx2 <= cx1 or cy2 <= cy1:
        return False, "Invalid face bounding box boundaries", 0.0

    crop = image_bgr[cy1:cy2, cx1:cx2]
    sharpness = calculate_sharpness(crop)

    if sharpness < min_sharpness:
        return False, f"Laplacian sharpness {sharpness:.1f} below threshold {min_sharpness:.1f} (motion blur)", sharpness

    return True, "Quality gate passed", sharpness
