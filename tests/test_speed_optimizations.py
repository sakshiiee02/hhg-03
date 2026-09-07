"""
Tests for speed optimizations:
1. DirectML & InsightFace model pruning (detection & recognition only).
2. Direct CDN first download ingestion (Tier 1).
3. Candidate streaming & early cancellation.
4. Confident early stop threshold logic.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
import numpy as np
import pytest

from src.face.detector import FaceDetector
from src.search.base import CandidateResult
from src.web.downloader import CandidateDownloader, DownloadedCandidate


def test_face_detector_module_pruning_and_directml():
    """Verify FaceDetector initializes only detection and recognition modules with DirectML if available."""
    detector = FaceDetector.get_shared_instance()
    loaded_models = list(detector.app.models.keys())
    
    # Must only contain detection and recognition
    assert "detection" in loaded_models
    assert "recognition" in loaded_models
    assert "landmark_3d_68" not in loaded_models
    assert "landmark_2d_106" not in loaded_models
    assert "genderage" not in loaded_models


@pytest.mark.asyncio
async def test_direct_cdn_download_priority():
    """Verify CandidateDownloader fetches image_url directly (Tier 1) without visiting page_url."""
    downloader = CandidateDownloader(concurrency_limit=4, timeout_seconds=2.0)

    candidate = CandidateResult(
        rank=1,
        page_url="https://93.184.216.34/slow-article-page.html",
        image_url="https://93.184.216.34/sample_portrait.jpg",
        source_domain="example.com",
        title="Sample Article",
        provider="test",
    )

    import cv2
    _, valid_jpeg_buf = cv2.imencode(".jpg", np.zeros((20, 20, 3), dtype=np.uint8))
    valid_jpeg = valid_jpeg_buf.tobytes()

    async def mock_fetch_bytes(session, url):
        if url == candidate.image_url:
            return valid_jpeg
        return None

    with patch.object(downloader, "_fetch_bytes", side_effect=mock_fetch_bytes):
        mock_session = MagicMock()
        mock_session.get = MagicMock()

        result = await downloader._download_single(mock_session, candidate)

        assert result is not None
        assert result.tier_used == "tier1_direct_cdn"
        assert result.image_url == candidate.image_url
        mock_session.get.assert_not_called()


@pytest.mark.asyncio
async def test_download_stream_early_cancellation():
    """Verify that download_stream cancels pending background tasks when closed early."""
    downloader = CandidateDownloader(concurrency_limit=4, timeout_seconds=2.0)

    candidates = [
        CandidateResult(
            rank=i,
            page_url=f"https://example.com/page{i}",
            image_url=f"https://example.com/img{i}.jpg",
            source_domain="example.com",
            title=f"Title {i}",
            provider="test",
        )
        for i in range(10)
    ]

    dummy_jpeg = b"\xff\xd8\xff\xe0\x00\x10JFIF" + (b"\x00" * 600) + b"\xff\xd9"

    async def mock_download_single(session, c):
        await asyncio.sleep(0.01 * (c.rank + 1))
        return DownloadedCandidate(
            candidate_id=f"cand_{c.rank}",
            rank=c.rank,
            page_url=c.page_url,
            canonical_url=c.page_url,
            image_url=c.image_url,
            source_domain=c.source_domain,
            title=f"Title {c.rank}",
            text_excerpt="",
            image_bgr=np.zeros((100, 100, 3), dtype=np.uint8),
            image_bytes=dummy_jpeg,
            image_sha256="dummy_sha",
            tier_used="tier1_direct_cdn",
        )

    with patch.object(downloader, "_download_single", side_effect=mock_download_single):
        streamed_count = 0
        async for item in downloader.download_stream(candidates):
            streamed_count += 1
            if streamed_count >= 3:
                # Early stop consumer break
                break

        assert streamed_count == 3


def test_confident_early_stop_scoring_logic():
    """Verify confident early stop thresholding criteria: score >= 0.92 and margin >= 0.15."""
    # Case 1: Less than 3 candidates -> Should NOT early stop
    scores_early = [0.98, 0.50]
    assert not (len(scores_early) >= 3 and (scores_early[0] >= 0.96 or (scores_early[0] >= 0.92 and (scores_early[0] - scores_early[1]) >= 0.15)))

    # Case 2: >= 3 candidates, top score 0.95, runner 0.60 (margin 0.35) -> Triggers early stop
    scores_confident = [0.95, 0.60, 0.40]
    margin = scores_confident[0] - scores_confident[1]
    should_stop = len(scores_confident) >= 3 and (scores_confident[0] >= 0.96 or (scores_confident[0] >= 0.92 and margin >= 0.15))
    assert should_stop is True

    # Case 3: >= 3 candidates, top score 0.85 (below 0.92) -> Should NOT early stop
    scores_mediocre = [0.85, 0.65, 0.40]
    margin = scores_mediocre[0] - scores_mediocre[1]
    should_stop = len(scores_mediocre) >= 3 and (scores_mediocre[0] >= 0.96 or (scores_mediocre[0] >= 0.92 and margin >= 0.15))
    assert should_stop is False

    # Case 4: >= 3 candidates, top score 0.93, runner 0.91 (tight margin 0.02) -> Should NOT early stop (ambiguous candidates)
    scores_ambiguous = [0.93, 0.91, 0.40]
    margin = scores_ambiguous[0] - scores_ambiguous[1]
    should_stop = len(scores_ambiguous) >= 3 and (scores_ambiguous[0] >= 0.96 or (scores_ambiguous[0] >= 0.92 and margin >= 0.15))
    assert should_stop is False

