"""
Biometric face matching, multi-face candidate scoring, and margin evaluation.
Implements the calibrated acceptance thresholds and runner-up margin checks.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Tuple

import numpy as np

from src.face.detector import DetectedFace, FaceDetector
from src.face.embedder import compute_cosine_similarity


class MatchVerdict(str, Enum):
    HIGH_CONFIDENCE = "HIGH_CONFIDENCE_MATCH"
    REVIEW_UNCERTAIN = "REVIEW_UNCERTAIN"
    REJECTED = "REJECTED"


@dataclass
class CandidateFaceScore:
    face_index: int
    similarity: float
    bbox: Tuple[int, int, int, int]
    confidence: float
    sharpness: float
    passes_quality: bool
    quality_reason: str


@dataclass
class CandidateVerificationResult:
    candidate_id: str
    image_url: str
    page_url: str
    source_domain: str
    title: Optional[str]
    num_faces_detected: int
    best_similarity: float
    best_face_index: int
    best_bbox: Optional[Tuple[int, int, int, int]]
    all_face_scores: List[CandidateFaceScore] = field(default_factory=list)
    verdict: MatchVerdict = MatchVerdict.REJECTED
    error: Optional[str] = None


class FaceMatcher:
    """Evaluates biometric similarity between a target embedding and candidate web imagery."""

    def __init__(
        self,
        similarity_threshold: float = 0.80,
        margin_threshold: float = 0.10,
        early_stop_similarity: float = 0.85,
    ):
        self.similarity_threshold = similarity_threshold
        self.margin_threshold = margin_threshold
        self.early_stop_similarity = early_stop_similarity

    def verify_candidate(
        self,
        target_embedding: np.ndarray,
        candidate_image_bgr: np.ndarray,
        candidate_id: str,
        image_url: str,
        page_url: str,
        source_domain: str,
        title: Optional[str] = None,
        detector: Optional[FaceDetector] = None,
    ) -> CandidateVerificationResult:
        """
        Detects faces in candidate image and computes maximum similarity against target embedding.
        """
        if detector is None:
            detector = FaceDetector.get_shared_instance()

        try:
            faces = detector.detect(candidate_image_bgr)
        except Exception as e:
            return CandidateVerificationResult(
                candidate_id=candidate_id,
                image_url=image_url,
                page_url=page_url,
                source_domain=source_domain,
                title=title,
                num_faces_detected=0,
                best_similarity=-1.0,
                best_face_index=-1,
                best_bbox=None,
                verdict=MatchVerdict.REJECTED,
                error=str(e),
            )

        return self.score_candidate_faces(
            target_embedding=target_embedding,
            faces=faces,
            candidate_id=candidate_id,
            image_url=image_url,
            page_url=page_url,
            source_domain=source_domain,
            title=title,
        )

    def score_candidate_faces(
        self,
        target_embedding: np.ndarray,
        faces: List[DetectedFace],
        candidate_id: str,
        image_url: str,
        page_url: str,
        source_domain: str,
        title: Optional[str] = None,
    ) -> CandidateVerificationResult:
        """
        Computes maximum similarity against target embedding from pre-detected faces.
        Avoids re-running face detection on the candidate image across multiple probe faces.
        """
        if not faces:
            return CandidateVerificationResult(
                candidate_id=candidate_id,
                image_url=image_url,
                page_url=page_url,
                source_domain=source_domain,
                title=title,
                num_faces_detected=0,
                best_similarity=-1.0,
                best_face_index=-1,
                best_bbox=None,
                verdict=MatchVerdict.REJECTED,
                error="No faces detected in candidate image",
            )

        face_scores: List[CandidateFaceScore] = []
        best_sim = -1.0
        best_idx = -1
        best_box = None

        for f in faces:
            sim = compute_cosine_similarity(target_embedding, f.embedding)
            face_scores.append(CandidateFaceScore(
                face_index=f.face_index,
                similarity=sim,
                bbox=f.bbox,
                confidence=f.confidence,
                sharpness=f.sharpness,
                passes_quality=f.passes_quality,
                quality_reason=f.quality_reason,
            ))

            if f.passes_quality and sim > best_sim:
                best_sim = sim
                best_idx = f.face_index
                best_box = f.bbox

        # If no face passed quality gates, fall back to best overall face score with uncertain verdict
        if best_sim < 0.0 and face_scores:
            best_unfiltered = max(face_scores, key=lambda s: s.similarity)
            best_sim = best_unfiltered.similarity
            best_idx = best_unfiltered.face_index
            best_box = best_unfiltered.bbox

        verdict = (
            MatchVerdict.HIGH_CONFIDENCE if best_sim >= self.similarity_threshold
            else MatchVerdict.REVIEW_UNCERTAIN if best_sim >= (self.similarity_threshold - 0.10)
            else MatchVerdict.REJECTED
        )

        return CandidateVerificationResult(
            candidate_id=candidate_id,
            image_url=image_url,
            page_url=page_url,
            source_domain=source_domain,
            title=title,
            num_faces_detected=len(faces),
            best_similarity=best_sim,
            best_face_index=best_idx,
            best_bbox=best_box,
            all_face_scores=face_scores,
            verdict=verdict,
        )

    def rank_and_evaluate_margin(
        self,
        results: List[CandidateVerificationResult],
    ) -> Tuple[Optional[CandidateVerificationResult], float, bool]:
        """
        Sorts candidates by similarity and calculates margin over runner-up.

        Returns:
            (best_match, margin_over_runner_up, should_early_stop)
        """
        valid_results = [r for r in results if r.best_similarity > 0.0]
        if not valid_results:
            return None, 0.0, False

        sorted_results = sorted(valid_results, key=lambda r: r.best_similarity, reverse=True)
        best = sorted_results[0]

        if len(sorted_results) > 1:
            runner_up = sorted_results[1]
            margin = best.best_similarity - runner_up.best_similarity
        else:
            margin = best.best_similarity  # Single candidate

        # Check early stopping: high-confidence match with clear margin
        should_early_stop = (
            best.best_similarity >= self.early_stop_similarity
            and margin >= self.margin_threshold
        )

        # Update verdict based on margin
        if best.best_similarity >= self.similarity_threshold and margin >= self.margin_threshold:
            best.verdict = MatchVerdict.HIGH_CONFIDENCE
        elif best.best_similarity >= (self.similarity_threshold - 0.10):
            best.verdict = MatchVerdict.REVIEW_UNCERTAIN
        else:
            best.verdict = MatchVerdict.REJECTED

        return best, margin, should_early_stop
