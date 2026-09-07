"""
Test suite validating the high-value features adopted from competitive audit:
1. Google Lens /goto redirect unwrapping
2. Social canonical key generation and tracking token stripping
3. Server-Side Request Forgery (SSRF) URL security validation
4. Persistent disk search cache
5. Vectorized cosine similarity matrix matching
6. Forensic evidence bundle persistence and standalone on-chain re-verification (exit codes 0 & 6)
"""

import json
from pathlib import Path
import shutil
import tempfile
import numpy as np
import pytest

from src.blockchain.contract import MockBlockchainClient
from src.blockchain.hash import build_canonical_record, hash_bytes, hash_canonical_record
from src.face.embedder import (
    compute_cosine_similarity,
    compute_cosine_similarity_matrix,
    normalize_embedding,
)
from src.pipeline.evidence import EvidenceLedger, verify_evidence_bundle
from src.search.base import (
    CandidateResult,
    canonical_page_key,
    is_safe_public_url,
    normalize_url,
    resolve_google_goto,
)
from src.search.cache import PersistentSearchCache


def test_google_goto_resolution():
    # 1. Google goto with encoded destination
    url1 = "https://www.google.com/goto?url=https%3A%2F%2Fgithub.com%2Ftorvalds"
    assert resolve_google_goto(url1) == "https://github.com/torvalds"

    # 2. Google url with q param
    url2 = "https://www.google.com/url?q=https%3A%2F%2Fx.com%2Fsama&source=images"
    assert resolve_google_goto(url2) == "https://x.com/sama"

    # 3. Standard direct URL untouched
    url3 = "https://linkedin.com/in/williamhgates"
    assert resolve_google_goto(url3) == "https://linkedin.com/in/williamhgates"


def test_canonical_page_key_social_stripping():
    # 1. Instagram with tracking params
    ig_raw = "https://www.instagram.com/p/C-xyz123/?igsh=MWQ1Z3==&utm_source=qr"
    assert canonical_page_key(ig_raw) == "instagram.com/p/c-xyz123"

    # 2. X / Twitter with tracking parameters
    x_raw = "https://x.com/sama/status/123456789?s=20&t=abcdef123"
    assert canonical_page_key(x_raw) == "x.com/sama/status/123456789"

    # 3. LinkedIn with tracking parameters
    li_raw = "https://www.linkedin.com/in/satyanadella/?trk=feed_post_details"
    assert canonical_page_key(li_raw) == "linkedin.com/in/satyanadella"

    # 4. Redundant slashes and case normalization
    dirty_url = "https://github.com//torvalds//linux/"
    assert canonical_page_key(dirty_url) == "github.com/torvalds/linux"


def test_ssrf_protection():
    # Loopback addresses
    assert not is_safe_public_url("http://127.0.0.1:8080/probe.jpg")
    assert not is_safe_public_url("http://localhost/secret.png")

    # Private networks
    assert not is_safe_public_url("http://10.0.0.1/intranet.jpg")
    assert not is_safe_public_url("http://192.168.1.100/admin.png")
    assert not is_safe_public_url("http://172.16.0.1/photo.jpg")

    # Cloud metadata endpoint
    assert not is_safe_public_url("http://169.254.169.254/latest/meta-data/")

    # Non-HTTP schemes
    assert not is_safe_public_url("ftp://files.example.com/photo.jpg")
    assert not is_safe_public_url("file:///etc/passwd")

    # Safe public domains
    assert is_safe_public_url("https://upload.wikimedia.org/wikipedia/commons/avatar.jpg")
    assert is_safe_public_url("https://github.com/torvalds.png")


def test_persistent_search_cache():
    temp_dir = Path(tempfile.mkdtemp(prefix="test_search_cache_"))
    try:
        cache = PersistentSearchCache(cache_dir=temp_dir)
        dummy_bytes = b"sample_probe_image_data_12345"

        # Initially miss
        assert cache.get(dummy_bytes) is None
        assert cache.stats()["misses"] == 1

        # Store candidate results
        candidates = [
            CandidateResult(
                rank=1,
                page_url="https://github.com/torvalds",
                image_url="https://github.com/torvalds.png",
                source_domain="github.com",
                title="Linus Torvalds",
                provider="SerpApi (Google Lens)",
            )
        ]
        cache.put(dummy_bytes, candidates)
        assert cache.stats()["writes"] == 1

        # Hit
        retrieved = cache.get(dummy_bytes)
        assert retrieved is not None
        assert len(retrieved) == 1
        assert retrieved[0].page_url == "https://github.com/torvalds"
        assert retrieved[0].source_domain == "github.com"
        assert cache.stats()["hits"] == 1

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_vectorized_cosine_similarity_matrix():
    np.random.seed(42)
    target = normalize_embedding(np.random.randn(512).astype(np.float32))
    
    # 5 random candidate embeddings
    candidates = [normalize_embedding(np.random.randn(512).astype(np.float32)) for _ in range(5)]
    cand_matrix = np.stack(candidates)

    # Vectorized
    matrix_sims = compute_cosine_similarity_matrix(target, cand_matrix)

    # Scalar loop
    scalar_sims = [compute_cosine_similarity(target, c) for c in candidates]

    # Verify identical results
    assert len(matrix_sims) == 5
    for m, s in zip(matrix_sims, scalar_sims):
        assert abs(m - s) < 1e-5


def test_evidence_bundle_and_standalone_reverification():
    run_id = f"test_run_{tempfile.mktemp().replace(tempfile.gettempdir(), '').replace(r'\\', '').replace('/', '')[:8]}"
    ledger = EvidenceLedger(run_id=run_id)
    mock_chain = MockBlockchainClient()

    fake_image_bytes = b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x00test_image_bytes"
    canon_record = build_canonical_record(
        canonical_url="https://github.com/torvalds",
        source_domain="github.com",
        title="Linus Torvalds Profile",
        image_sha256=hash_bytes(fake_image_bytes),
        discovered_at="2026-09-07T20:00:00+00:00",
        text_excerpt="Creator of Linux kernel",
        identified_name="Linus Torvalds",
        social_profiles={"github": "https://github.com/torvalds"},
    )

    rec_sha = hash_canonical_record(canon_record)
    cnt_sha = hash_bytes(fake_image_bytes)

    receipt = mock_chain.attest(cnt_sha, rec_sha)

    # Save evidence bundle
    bundle_dir = ledger.save_evidence_bundle(
        matched_image_bytes=fake_image_bytes,
        canonical_record=canon_record,
        attestation_receipt=receipt,
    )

    assert (bundle_dir / "metadata.json").is_file()
    assert (bundle_dir / "matched_image.bin").is_file()

    # 1. Normal Verification Check -> Expected exit code 0
    report = verify_evidence_bundle(bundle_dir, blockchain_client=mock_chain)
    assert report["verified"] is True
    assert report["tampered"] is False
    assert report["exit_code"] == 0
    assert report["metadata_hash_match"] is True
    assert report["image_hash_match"] is True

    # 2. Tamper Check: Mutate image bytes on disk -> Expected exit code 6
    bin_file = bundle_dir / "matched_image.bin"
    tampered_bytes = b"TAMPERED_MODIFIED_BYTES" + fake_image_bytes
    bin_file.write_bytes(tampered_bytes)

    tamper_report = verify_evidence_bundle(bundle_dir, blockchain_client=mock_chain)
    assert tamper_report["verified"] is False
    assert tamper_report["tampered"] is True
    assert tamper_report["exit_code"] == 6
    assert tamper_report["image_hash_match"] is False

    # Cleanup test run directory
    shutil.rmtree(ledger.run_dir, ignore_errors=True)
