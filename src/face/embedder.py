"""
ArcFace 512-D feature embedder and normalization utilities.
"""

from typing import Union
import numpy as np


def normalize_embedding(vector: np.ndarray) -> np.ndarray:
    """Strictly projects feature vector onto the 512-D unit hypersphere (L2-norm = 1.0)."""
    norm = np.linalg.norm(vector)
    if norm == 0:
        return vector
    return vector / norm


def compute_cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    """
    Computes cosine similarity between two feature vectors:
    S_C = (a . b) / (||a|| * ||b||)
    Returns a scalar float in [-1.0, 1.0].
    """
    a_norm = normalize_embedding(vec_a)
    b_norm = normalize_embedding(vec_b)
    return float(np.dot(a_norm, b_norm))
