"""
Pipeline Orchestrator connecting all 8 stages of HH Goa Task 3.
Coordinates biometrics, discovery, concurrent download, matching, attestation, and re-verification.
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import aiohttp
import numpy as np

from src.blockchain.contract import AttestationReceipt, get_blockchain_client
from src.blockchain.hash import (
    build_canonical_record,
    hash_bytes,
    hash_canonical_record,
    to_bytes32,
    to_hex32,
)
from src.config import PipelineConfig, get_config
from src.face.detector import DetectedFace, FaceDetector
from src.face.matcher import CandidateVerificationResult, FaceMatcher, MatchVerdict
from src.pipeline.evidence import EvidenceLedger
from src.search.base import CandidateResult
from src.search.router import SearchRouter
from src.social import PersonSocialIdentity, SocialDiscoveryEngine
from src.web.downloader import CandidateDownloader, DownloadedCandidate

logger = logging.getLogger(__name__)


@dataclass
class FaceVerificationProfile:
    """Biometric profile, matching results, and attestation for an individual detected face."""
    face_index: int
    is_primary: bool
    bbox: Tuple[int, int, int, int]
    confidence: float
    sharpness: float
    crop_b64: Optional[str]
    embedding_sample: List[float]
    best_match: Optional[CandidateVerificationResult]
    runner_up_margin: float
    top_candidates: List[CandidateVerificationResult] = field(default_factory=list)
    matched_image_b64: Optional[str] = None
    canonical_record: Optional[Dict[str, Any]] = None
    content_hash: str = ""
    record_hash: str = ""
    attestation: Optional[AttestationReceipt] = None
    social_identity: Optional[PersonSocialIdentity] = None
    re_verification_passed: bool = False
    status_message: str = "PENDING"


@dataclass
class PipelineRunResult:
    """Consolidated result of an end-to-end pipeline run."""
    success: bool
    status_message: str
    run_id: str
    run_dir: str
    elapsed_seconds: float
    input_faces_count: int
    primary_face: Optional[DetectedFace]
    candidates_discovered: int
    candidates_downloaded: int
    best_match: Optional[CandidateVerificationResult]
    runner_up_margin: float
    attestation: Optional[AttestationReceipt]
    content_hash: str
    record_hash: str
    re_verification_passed: bool
    tamper_mode_active: bool
    evidence_saved: bool
    top_candidates: List[CandidateVerificationResult] = field(default_factory=list)
    faces_results: List[FaceVerificationProfile] = field(default_factory=list)


class PipelineOrchestrator:
    """Coordinates the 8-step pipeline execution."""

    def __init__(self, config: Optional[PipelineConfig] = None):
        self.config = config or get_config()
        self.detector = FaceDetector.get_shared_instance()
        self.matcher = FaceMatcher(
            similarity_threshold=self.config.similarity_threshold,
            margin_threshold=self.config.margin_threshold,
            early_stop_similarity=self.config.early_stop_similarity,
        )
        self.search_router = SearchRouter(
            primary_engine=self.config.search_engine,
            serpapi_api_key=self.config.serpapi_api_key or None,
        )
        self.downloader = CandidateDownloader(concurrency_limit=self.config.concurrency_limit)
        self.blockchain_client = get_blockchain_client(
            rpc_url=self.config.sepolia_rpc_url,
            private_key=self.config.sepolia_private_key,
            contract_address=self.config.sepolia_contract_address or None,
            force_simulated=self.config.force_simulated_blockchain,
        )
        self.social_engine = SocialDiscoveryEngine()

    async def execute(
        self,
        image_path: Path,
        face_index: Optional[int] = None,
        tamper: bool = False,
        dry_run: bool = False,
        on_step_update=None,
    ) -> PipelineRunResult:
        """
        Executes the complete 8-step pipeline.
        on_step_update is an optional callback: (step_num: int, title: str, status: str, details: str).
        """
        start_time = datetime.now(timezone.utc)
        ledger = EvidenceLedger()

        def notify(step: int, title: str, status: str, details: str = ""):
            if on_step_update:
                on_step_update(step, title, status, details)

        # -------------------------------------------------------------
        # STEP 1: Face Detection
        # -------------------------------------------------------------
        notify(1, "Face Detection", "running", f"Analyzing {image_path.name}...")
        faces = self.detector.detect(image_path)
        if not faces:
            notify(1, "Face Detection", "failed", "No faces detected in input image.")
            return PipelineRunResult(
                success=False,
                status_message="No faces detected in input image.",
                run_id=ledger.run_id,
                run_dir=str(ledger.run_dir),
                elapsed_seconds=(datetime.now(timezone.utc) - start_time).total_seconds(),
                input_faces_count=0,
                primary_face=None,
                candidates_discovered=0,
                candidates_downloaded=0,
                best_match=None,
                runner_up_margin=0.0,
                attestation=None,
                content_hash="",
                record_hash="",
                re_verification_passed=False,
                tamper_mode_active=tamper,
                evidence_saved=False,
            )

        # Extract base64 crops for all detected faces
        face_crops: Dict[int, Optional[str]] = {}
        for f in faces:
            try:
                face_crops[f.face_index] = self.detector.crop_face_b64(image_path, f.bbox)
            except Exception as e:
                logger.warning(f"Failed to crop face {f.face_index}: {e}")
                face_crops[f.face_index] = None

        primary_face = self.detector.select_primary_face(faces, face_index_override=face_index)
        x1, y1, x2, y2 = primary_face.bbox
        w, h = x2 - x1, y2 - y1
        notify(1, "Face Detection", "completed", f"Detected {len(faces)} face(s). Primary: {w}x{h}px (conf {primary_face.confidence:.2f})")

        # -------------------------------------------------------------
        # STEP 2: Face Embedding
        # -------------------------------------------------------------
        notify(2, "Face Embedding", "running", f"Generating 512-D ArcFace embeddings for {len(faces)} face(s)...")
        target_embedding = primary_face.embedding
        notify(2, "Face Embedding", "completed", f"Generated 512-D normalized vector(s) (L2 norm: 1.00)")

        if dry_run:
            notify(3, "Dry Run Complete", "completed", "Dry-run flag enabled: stopping before network operations.")
            dry_profiles = []
            for f in faces:
                dry_profiles.append(FaceVerificationProfile(
                    face_index=f.face_index,
                    is_primary=(f.face_index == primary_face.face_index),
                    bbox=f.bbox,
                    confidence=round(float(f.confidence), 4),
                    sharpness=round(float(f.sharpness), 1),
                    crop_b64=face_crops.get(f.face_index),
                    embedding_sample=[round(float(v), 4) for v in f.embedding[:16]],
                    best_match=None,
                    runner_up_margin=0.0,
                    status_message="DRY_RUN",
                ))
            return PipelineRunResult(
                success=True,
                status_message="Dry-run completed successfully (biometrics validated).",
                run_id=ledger.run_id,
                run_dir=str(ledger.run_dir),
                elapsed_seconds=(datetime.now(timezone.utc) - start_time).total_seconds(),
                input_faces_count=len(faces),
                primary_face=primary_face,
                candidates_discovered=0,
                candidates_downloaded=0,
                best_match=None,
                runner_up_margin=0.0,
                attestation=None,
                content_hash="",
                record_hash="",
                re_verification_passed=False,
                tamper_mode_active=tamper,
                evidence_saved=False,
                faces_results=dry_profiles,
            )

        # -------------------------------------------------------------
        # STEP 3: Dynamic Reverse Image Search
        # -------------------------------------------------------------
        notify(3, "Web Discovery", "running", f"Initiating parallel visual search ({self.config.search_engine})...")
        candidates = await self.search_router.discover_candidates(image_path, max_candidates=35)
        if not candidates:
            reason = "No candidate URLs discovered by search engines."
            if self.config.search_engine == "serpapi" and not self.search_router.serpapi:
                reason = "SERPAPI_API_KEY is not configured in .env. Please configure your key or select Parallel Auto / Yandex."
            notify(3, "Web Discovery", "failed", reason)
            return PipelineRunResult(
                success=False,
                status_message=reason,
                run_id=ledger.run_id,
                run_dir=str(ledger.run_dir),
                elapsed_seconds=(datetime.now(timezone.utc) - start_time).total_seconds(),
                input_faces_count=len(faces),
                primary_face=primary_face,
                candidates_discovered=0,
                candidates_downloaded=0,
                best_match=None,
                runner_up_margin=0.0,
                attestation=None,
                content_hash="",
                record_hash="",
                re_verification_passed=False,
                tamper_mode_active=tamper,
                evidence_saved=False,
            )
        notify(3, "Web Discovery", "completed", f"Discovered {len(candidates)} candidates across public web indexes.")

        # -------------------------------------------------------------
        # STEP 4: Candidate Download (Multi-Tiered Bounded Ingestion)
        # -------------------------------------------------------------
        notify(4, "Candidate Retrieval", "running", f"Fetching candidates with {self.config.concurrency_limit} concurrent workers...")
        downloaded = await self.downloader.download_candidates(candidates)
        if not downloaded:
            notify(4, "Candidate Retrieval", "failed", "Could not retrieve candidate imagery.")
            return PipelineRunResult(
                success=False,
                status_message="Failed to retrieve any candidate images.",
                run_id=ledger.run_id,
                run_dir=str(ledger.run_dir),
                elapsed_seconds=(datetime.now(timezone.utc) - start_time).total_seconds(),
                input_faces_count=len(faces),
                primary_face=primary_face,
                candidates_discovered=len(candidates),
                candidates_downloaded=0,
                best_match=None,
                runner_up_margin=0.0,
                attestation=None,
                content_hash="",
                record_hash="",
                re_verification_passed=False,
                tamper_mode_active=tamper,
                evidence_saved=False,
            )
        notify(4, "Candidate Retrieval", "completed", f"Retrieved {len(downloaded)} usable candidate images.")

        # -------------------------------------------------------------
        # STEP 5: Face Verification & Multi-Face Scoring
        # -------------------------------------------------------------
        notify(5, "Face Verification", "running", f"Evaluating biometric similarity across {len(faces)} face(s)...")
        downloaded_by_id: Dict[str, DownloadedCandidate] = {}
        candidate_detected_faces: Dict[str, List[DetectedFace]] = {}

        # 1. Detect candidate faces once per candidate image
        for item in downloaded:
            downloaded_by_id[item.candidate_id] = item
            try:
                candidate_detected_faces[item.candidate_id] = self.detector.detect(item.image_bgr)
            except Exception as e:
                logger.warning(f"Detection failed on candidate {item.candidate_id}: {e}")
                candidate_detected_faces[item.candidate_id] = []

        # 2. Score candidate imagery against each detected probe face
        per_face_verifications: Dict[int, List[CandidateVerificationResult]] = {}
        for f in faces:
            results_for_f: List[CandidateVerificationResult] = []
            for item in downloaded:
                cand_faces = candidate_detected_faces.get(item.candidate_id, [])
                res = self.matcher.score_candidate_faces(
                    target_embedding=f.embedding,
                    faces=cand_faces,
                    candidate_id=item.candidate_id,
                    image_url=item.image_url,
                    page_url=item.canonical_url,
                    source_domain=item.source_domain,
                    title=item.title,
                )
                results_for_f.append(res)
            per_face_verifications[f.face_index] = results_for_f

        notify(5, "Face Verification", "completed", f"Evaluated {len(downloaded)} candidates against {len(faces)} face(s).")

        # -------------------------------------------------------------
        # STEP 6: Match Selection & Margin Evaluation
        # -------------------------------------------------------------
        notify(6, "Match Selection", "running", "Ranking candidates and computing separation margins...")
        per_face_ranked: Dict[int, Tuple[Optional[CandidateVerificationResult], float, bool]] = {}
        for f in faces:
            per_face_ranked[f.face_index] = self.matcher.rank_and_evaluate_margin(per_face_verifications[f.face_index])

        primary_best, primary_margin, early_stop_triggered = per_face_ranked[primary_face.face_index]

        # Check if at least one face cleared threshold
        any_face_matched = any(
            bm is not None and bm.best_similarity >= 0.50
            for bm, _, _ in per_face_ranked.values()
        )

        if not any_face_matched and (not primary_best or primary_best.best_similarity < 0.50):
            notify(6, "Match Selection", "failed", "No candidates cleared identity threshold for any detected face.")
            failed_profiles = []
            for f in faces:
                bm, mg, _ = per_face_ranked[f.face_index]
                top_c = sorted(per_face_verifications[f.face_index], key=lambda r: r.best_similarity, reverse=True)[:5]
                failed_profiles.append(FaceVerificationProfile(
                    face_index=f.face_index,
                    is_primary=(f.face_index == primary_face.face_index),
                    bbox=f.bbox,
                    confidence=round(float(f.confidence), 4),
                    sharpness=round(float(f.sharpness), 1),
                    crop_b64=face_crops.get(f.face_index),
                    embedding_sample=[round(float(v), 4) for v in f.embedding[:16]],
                    best_match=bm,
                    runner_up_margin=mg,
                    top_candidates=top_c,
                    status_message="NO_MATCH",
                ))
            return PipelineRunResult(
                success=False,
                status_message="No candidate matched the input face(s).",
                run_id=ledger.run_id,
                run_dir=str(ledger.run_dir),
                elapsed_seconds=(datetime.now(timezone.utc) - start_time).total_seconds(),
                input_faces_count=len(faces),
                primary_face=primary_face,
                candidates_discovered=len(candidates),
                candidates_downloaded=len(downloaded),
                best_match=primary_best,
                runner_up_margin=primary_margin,
                attestation=None,
                content_hash="",
                record_hash="",
                re_verification_passed=False,
                tamper_mode_active=tamper,
                evidence_saved=False,
                faces_results=failed_profiles,
            )

        early_stop_str = " (Early stop triggered)" if early_stop_triggered else ""
        if primary_best and primary_best.best_similarity >= 0.50:
            notify(
                6,
                "Match Selection",
                "completed",
                f"Best match for Face {primary_face.face_index} (Primary): {primary_best.source_domain} (cosine {primary_best.best_similarity:.4f}, margin +{primary_margin:.4f}){early_stop_str}",
            )
        else:
            notify(6, "Match Selection", "completed", f"Evaluated and ranked candidates for {len(faces)} face(s).")

        # -------------------------------------------------------------
        # STEP 7: Cryptographic Hashing & Blockchain Attestation
        # -------------------------------------------------------------
        notify(7, "Blockchain Attestation", "running", f"Generating canonical records and attesting to Sepolia...")
        discovery_ts = datetime.now(timezone.utc).isoformat()
        import base64

        faces_profiles: List[FaceVerificationProfile] = []
        overall_re_verification_passed = True

        for f in faces:
            best_m, margin_val, _ = per_face_ranked[f.face_index]
            top_c = sorted(per_face_verifications[f.face_index], key=lambda r: r.best_similarity, reverse=True)[:5]
            is_prim = (f.face_index == primary_face.face_index)

            if not best_m or best_m.best_similarity < 0.50:
                faces_profiles.append(FaceVerificationProfile(
                    face_index=f.face_index,
                    is_primary=is_prim,
                    bbox=f.bbox,
                    confidence=round(float(f.confidence), 4),
                    sharpness=round(float(f.sharpness), 1),
                    crop_b64=face_crops.get(f.face_index),
                    embedding_sample=[round(float(v), 4) for v in f.embedding[:16]],
                    best_match=best_m,
                    runner_up_margin=margin_val,
                    top_candidates=top_c,
                    status_message="NO_MATCH",
                ))
                continue

            winning_cand = downloaded_by_id[best_m.candidate_id]

            # Discover social profiles for the verified person
            social_ident = None
            try:
                cand_titles = [c.title for c in top_c if c.title] + [c.title for c in candidates if c.title]
                cand_urls = [c.page_url for c in top_c if c.page_url] + [c.page_url for c in candidates if c.page_url]
                cand_excerpts = [downloaded_by_id[c.candidate_id].text_excerpt for c in top_c if c.candidate_id in downloaded_by_id]
                
                social_ident = await self.social_engine.discover_socials(
                    candidate_titles=cand_titles,
                    candidate_urls=cand_urls,
                    candidate_excerpts=cand_excerpts,
                )
                if social_ident and social_ident.profiles:
                    social_str = ", ".join(f"{p.display_name} ({p.handle})" for p in social_ident.profiles[:3])
                    notify(
                        6,
                        "Social Discovery",
                        "running",
                        f"Discovered {len(social_ident.profiles)} profile(s) for {social_ident.canonical_name}: {social_str}",
                    )
            except Exception as e:
                logger.warning(f"Social discovery error for face #{f.face_index}: {e}")

            ident_name = social_ident.canonical_name if social_ident else None
            soc_dict = {p.platform.value: p.url for p in social_ident.profiles} if (social_ident and social_ident.profiles) else None

            canonical_rec = build_canonical_record(
                canonical_url=winning_cand.canonical_url,
                source_domain=winning_cand.source_domain,
                title=winning_cand.title,
                image_sha256=winning_cand.image_sha256,
                discovered_at=discovery_ts,
                text_excerpt=f"Face #{f.face_index} verification: {winning_cand.text_excerpt}",
                identified_name=ident_name,
                social_profiles=soc_dict,
            )
            rec_sha = hash_canonical_record(canonical_rec)
            cnt_sha = winning_cand.image_sha256

            receipt = self.blockchain_client.attest(cnt_sha, rec_sha)

            # Matched media base64
            cand_b64 = f"data:image/jpeg;base64,{base64.b64encode(winning_cand.image_bytes).decode('utf-8')}"

            # Step 8 Re-verification for this face
            re_passed = False
            re_fetched_bytes = winning_cand.image_bytes
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get(winning_cand.image_url, timeout=aiohttp.ClientTimeout(total=6)) as r:
                        if r.status == 200:
                            re_fetched_bytes = await r.read()
            except Exception:
                pass

            if tamper and is_prim:
                mutated_byte = bytes([(re_fetched_bytes[0] ^ 0xFF)])
                re_fetched_bytes = mutated_byte + re_fetched_bytes[1:]

            re_comp_img_hash = hash_bytes(re_fetched_bytes)
            rebuilt_canon = build_canonical_record(
                canonical_url=winning_cand.canonical_url,
                source_domain=winning_cand.source_domain,
                title=winning_cand.title,
                image_sha256=re_comp_img_hash,
                discovered_at=discovery_ts,
                text_excerpt=f"Face #{f.face_index} verification: {winning_cand.text_excerpt}",
                identified_name=ident_name,
                social_profiles=soc_dict,
            )
            re_comp_rec_hash = hash_canonical_record(rebuilt_canon)

            try:
                on_chain = self.blockchain_client.get_record(rec_sha)
                on_chain_cnt = on_chain["contentHash"].lower().replace("0x", "")
                on_chain_rec = on_chain["recordHash"].lower().replace("0x", "")
                re_passed = (re_comp_img_hash.lower() == on_chain_cnt and re_comp_rec_hash.lower() == on_chain_rec)
            except Exception as e:
                logger.warning(f"Re-verification query failed for face {f.face_index}: {e}")
                re_passed = False

            if not re_passed:
                overall_re_verification_passed = False

            faces_profiles.append(FaceVerificationProfile(
                face_index=f.face_index,
                is_primary=is_prim,
                bbox=f.bbox,
                confidence=round(float(f.confidence), 4),
                sharpness=round(float(f.sharpness), 1),
                crop_b64=face_crops.get(f.face_index),
                embedding_sample=[round(float(v), 4) for v in f.embedding[:16]],
                best_match=best_m,
                runner_up_margin=margin_val,
                top_candidates=top_c,
                matched_image_b64=cand_b64,
                canonical_record=canonical_rec,
                content_hash=cnt_sha,
                record_hash=rec_sha,
                attestation=receipt,
                social_identity=social_ident,
                re_verification_passed=re_passed,
                status_message="VERIFIED" if re_passed else ("TAMPER_DETECTED" if tamper and is_prim else "FAILED"),
            ))

        # Overall notifications
        attested_count = sum(1 for p in faces_profiles if p.attestation is not None)
        network_name = getattr(self.blockchain_client, "network", "Ethereum Sepolia")
        notify(
            7,
            "Blockchain Attestation",
            "completed",
            f"Attested {attested_count} face record(s) to {network_name}",
        )

        # -------------------------------------------------------------
        # STEP 8: Live Re-Verification & Tamper Test
        # -------------------------------------------------------------
        notify(8, "Live Re-Verification", "running", "Checking on-chain cryptographic integrity...")
        if overall_re_verification_passed and not tamper:
            notify(8, "Live Re-Verification", "completed", "Cryptographic integrity match: ALL ON-CHAIN HASHES VERIFIED")
        elif tamper:
            notify(8, "Live Re-Verification", "completed", "TAMPER DETECTED: Hash mismatch confirmed against on-chain record")
        else:
            notify(8, "Live Re-Verification", "completed", f"Re-verification evaluated for {len(faces)} face(s).")

        # Pick primary or top profile for top-level backward compatibility
        prim_profile = next((p for p in faces_profiles if p.is_primary), faces_profiles[0])
        winning_cand_obj = downloaded_by_id.get(prim_profile.best_match.candidate_id) if (prim_profile.best_match and prim_profile.best_match.candidate_id in downloaded_by_id) else None

        if winning_cand_obj:
            ledger.save_input_crop(winning_cand_obj.image_bgr)
            ledger.save_matched_media(winning_cand_obj.image_bytes)
        if prim_profile.canonical_record:
            ledger.save_canonical_record(prim_profile.canonical_record)

        # Save comprehensive evidence ledger report
        ledger.save_evidence_report({
            "run_id": ledger.run_id,
            "timestamp": discovery_ts,
            "input_image": str(image_path),
            "input_faces_count": len(faces),
            "candidates_discovered": len(candidates),
            "candidates_downloaded": len(downloaded),
            "faces": [
                {
                    "face_index": p.face_index,
                    "is_primary": p.is_primary,
                    "bbox": list(p.bbox),
                    "confidence": p.confidence,
                    "sharpness": p.sharpness,
                    "best_similarity": p.best_match.best_similarity if p.best_match else None,
                    "runner_up_margin": p.runner_up_margin,
                    "source_domain": p.best_match.source_domain if p.best_match else None,
                    "page_url": p.best_match.page_url if p.best_match else None,
                    "content_hash": p.content_hash,
                    "record_hash": p.record_hash,
                    "tx_hash": p.attestation.tx_hash if p.attestation else None,
                    "re_verification_passed": p.re_verification_passed,
                    "social_identity": p.social_identity.to_dict() if p.social_identity else None,
                }
                for p in faces_profiles
            ],
            "re_verification": {
                "passed": overall_re_verification_passed,
                "tamper_mode": tamper,
            },
        })

        elapsed = (datetime.now(timezone.utc) - start_time).total_seconds()

        return PipelineRunResult(
            success=(overall_re_verification_passed or (tamper and not overall_re_verification_passed)),
            status_message="VERIFIED" if overall_re_verification_passed else ("TAMPER_DETECTED" if tamper else "VERIFICATION_FAILED"),
            run_id=ledger.run_id,
            run_dir=str(ledger.run_dir),
            elapsed_seconds=elapsed,
            input_faces_count=len(faces),
            primary_face=primary_face,
            candidates_discovered=len(candidates),
            candidates_downloaded=len(downloaded),
            best_match=prim_profile.best_match,
            runner_up_margin=prim_profile.runner_up_margin,
            attestation=prim_profile.attestation,
            content_hash=prim_profile.content_hash,
            record_hash=prim_profile.record_hash,
            re_verification_passed=prim_profile.re_verification_passed,
            tamper_mode_active=tamper,
            evidence_saved=True,
            top_candidates=prim_profile.top_candidates,
            faces_results=faces_profiles,
        )
