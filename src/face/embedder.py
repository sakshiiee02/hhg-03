"""
ArcFace 512-D feature embedder and normalization utilities.
Includes scalar and vectorized matrix cosine similarity computation.
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


def compute_cosine_similarity_matrix(target_vec: np.ndarray, candidate_matrix: np.ndarray) -> np.ndarray:
    """
    Vectorized cosine similarity of one target vector (shape [D]) against
    a matrix of candidate embeddings (shape [N, D]).
    Returns a 1D float array of shape [N].
    """
    mat = np.asarray(candidate_matrix, dtype=np.float32)
    if mat.size == 0:
        return np.zeros((0,), dtype=np.float32)
    if mat.ndim == 1:
        mat = mat.reshape(1, -1)

    tgt = normalize_embedding(target_vec)
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    normed_mat = mat / norms
    return (normed_mat @ tgt).ravel()
