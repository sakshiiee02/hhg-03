"""
Lightweight async Web server and API for HH Goa 2026 Task 3.
Serves the Monochrome Bento Grid GUI and exposes WebSocket + REST execution endpoints.
Supports instant face analysis on upload, drag & drop, and large files.
"""

import asyncio
import base64
import json
import logging
import re
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from aiohttp import web

from src.config import get_config
from src.face.detector import FaceDetector
from src.pipeline.evidence import verify_evidence_bundle
from src.pipeline.orchestrator import PipelineOrchestrator, PipelineRunResult

logger = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).resolve().parent / "static"
EXAMPLES_DIR = Path(__file__).resolve().parent.parent / "examples"
RUNS_DIR = Path(__file__).resolve().parent.parent / "runs"
UPLOADS_DIR = RUNS_DIR / "uploads"
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)


def file_to_base64(path: Path) -> Optional[str]:
    if not path.exists():
        return None
    data = path.read_bytes()
    ext = path.suffix.lower().replace(".", "")
    mime = f"image/{ext}" if ext != "jpg" else "image/jpeg"
    return f"data:{mime};base64,{base64.b64encode(data).decode('utf-8')}"


def bytes_to_base64(data: bytes, mime: str = "image/jpeg") -> str:
    return f"data:{mime};base64,{base64.b64encode(data).decode('utf-8')}"


async def handle_index(request: web.Request) -> web.Response:
    index_file = STATIC_DIR / "index.html"
    if not index_file.exists():
        return web.Response(text="index.html not found", status=404)
    return web.Response(text=index_file.read_text(encoding="utf-8"), content_type="text/html")


async def handle_get_examples(request: web.Request) -> web.Response:
    files = []
    if EXAMPLES_DIR.exists():
        for f in EXAMPLES_DIR.glob("*.jpg"):
            files.append({
                "filename": f.name,
                "size_bytes": f.stat().st_size,
                "preview_b64": file_to_base64(f),
            })
    return web.json_response({"examples": files})


async def handle_upload(request: web.Request) -> web.Response:
    """Handle multipart file upload or JSON payload with instant biometric analysis."""
    try:
        content = None
        filename = "uploaded_probe.jpg"

        if request.content_type.startswith("multipart/"):
            reader = await request.multipart()
            while True:
                part = await reader.next()
                if part is None:
                    break
                if part.name in ("image", "file"):
                    filename = part.filename or filename
                    content = await part.read()
                    break
        else:
            try:
                data = await request.json()
                b64_str = data.get("image_b64", "")
                filename = data.get("filename", filename)
                if "," in b64_str:
                    _, raw = b64_str.split(",", 1)
                else:
                    raw = b64_str
                content = base64.b64decode(raw)
            except Exception:
                pass

        if not content:
            return web.json_response({"success": False, "error": "No image payload received"}, status=400)

        ext = Path(filename).suffix.lower()
        if not ext or ext not in (".jpg", ".jpeg", ".png", ".webp", ".bmp"):
            ext = ".jpg"

        upload_id = f"upload_{uuid.uuid4().hex[:10]}{ext}"
        saved_path = UPLOADS_DIR / upload_id
        saved_path.write_bytes(content)

        # Run instant face detection & biometric profiling
        detector = FaceDetector.get_shared_instance()
        detected_faces = detector.detect(saved_path)

        faces_count = len(detected_faces)
        primary = detector.get_primary_face(detected_faces) if faces_count > 0 else None
        primary_face_info = None
        embedding_sample = []
        all_faces_info = []

        for idx, f in enumerate(detected_faces):
            is_prim = (primary is not None and f.face_index == primary.face_index)
            crop_b64 = detector.crop_face_b64(saved_path, f.bbox)
            face_dict = {
                "face_index": f.face_index,
                "is_primary": is_prim,
                "bbox": [int(x) for x in f.bbox],
                "confidence": round(float(f.confidence), 4),
                "sharpness": round(float(f.sharpness), 1),
                "crop_b64": crop_b64,
                "embedding_sample": [round(float(v), 4) for v in f.embedding[:16]],
            }
            all_faces_info.append(face_dict)
            if is_prim:
                primary_face_info = face_dict
                embedding_sample = face_dict["embedding_sample"]

        return web.json_response({
            "success": True,
            "upload_id": upload_id,
            "filename": filename,
            "size_bytes": len(content),
            "preview_b64": file_to_base64(saved_path),
            "faces_count": faces_count,
            "primary_face": primary_face_info,
            "all_faces": all_faces_info,
            "embedding_sample": embedding_sample,
            "message": f"Analyzed {filename}: {faces_count} face(s) detected.",
        })
    except Exception as e:
        logger.error(f"Upload error: {e}", exc_info=True)
        return web.json_response({"success": False, "error": str(e)}, status=500)


async def handle_get_ledger(request: web.Request) -> web.Response:
    run_id = request.match_info.get("run_id", "")
    ledger_path = RUNS_DIR / run_id / "run_ledger.json"
    if not ledger_path.exists():
        return web.Response(text="Ledger not found", status=404)
    return web.Response(
        text=ledger_path.read_text(encoding="utf-8"),
        content_type="application/json",
        headers={"Content-Disposition": f'attachment; filename="ledger_{run_id}.json"'},
    )


async def handle_get_runs(request: web.Request) -> web.Response:
    """Returns list of past pipeline execution runs for the audit ledger."""
    runs = []
    if RUNS_DIR.exists():
        for run_path in sorted(RUNS_DIR.glob("run_*"), key=lambda p: p.stat().st_mtime, reverse=True)[:30]:
            ev_file = run_path / "evidence_report.json"
            ledger_file = run_path / "run_ledger.json"
            data = {}
            if ev_file.exists():
                try:
                    data = json.loads(ev_file.read_text(encoding="utf-8"))
                except Exception:
                    pass
            elif ledger_file.exists():
                try:
                    data = json.loads(ledger_file.read_text(encoding="utf-8"))
                except Exception:
                    pass

            faces = data.get("faces", [])
            primary_name = None
            if faces:
                first_face = faces[0]
                soc = first_face.get("social_identity") or {}
                primary_name = soc.get("canonical_name")

            runs.append({
                "run_id": run_path.name,
                "timestamp": data.get("timestamp", datetime.fromtimestamp(run_path.stat().st_mtime, tz=timezone.utc).isoformat()),
                "faces_count": data.get("input_faces_count", len(faces) or 1),
                "candidates_discovered": data.get("candidates_discovered", 0),
                "re_verification_passed": data.get("re_verification", {}).get("passed", False),
                "tamper_mode": data.get("re_verification", {}).get("tamper_mode", False),
                "identified_name": primary_name or "Unknown Subject",
                "top_domain": faces[0].get("source_domain") if faces else None,
            })
    return web.json_response({"runs": runs})


async def handle_reverify(request: web.Request) -> web.Response:
    """Independent on-chain re-verification of evidence bundles."""
    try:
        data = {}
        if request.can_read_body:
            try:
                data = await request.json()
            except Exception:
                pass

        target = data.get("run_id") or data.get("evidence_id") or data.get("bundle_path")
        if not target:
            target = request.query.get("run_id") or request.query.get("evidence_id")

        if not target:
            runs = sorted(RUNS_DIR.glob("run_*"), key=lambda p: p.stat().st_mtime, reverse=True)
            for r in runs:
                if (r / "evidence" / "metadata.json").exists() or (r / "metadata.json").exists():
                    target = str(r)
                    break
            if not target and runs:
                target = str(runs[0])
            elif not target:
                return web.json_response({"success": False, "error": "No evidence bundle or past run found to re-verify."}, status=400)

        report = await asyncio.to_thread(verify_evidence_bundle, target)
        return web.json_response(report)
    except Exception as e:
        logger.error(f"Re-verification error: {e}", exc_info=True)
        return web.json_response({"success": False, "error": str(e)}, status=500)


async def handle_ws(request: web.Request) -> web.WebSocketResponse:
    ws = web.WebSocketResponse(max_msg_size=32 * 1024 * 1024)
    await ws.prepare(request)

    async for msg in ws:
        if msg.type == web.WSMsgType.TEXT:
            try:
                data = json.loads(msg.data)
                action = data.get("action")

                if action == "reverify":
                    target = data.get("run_id") or data.get("evidence_id") or data.get("bundle_path")
                    if not target:
                        runs = sorted(RUNS_DIR.glob("run_*"), key=lambda p: p.stat().st_mtime, reverse=True)
                        for r in runs:
                            if (r / "evidence" / "metadata.json").exists() or (r / "metadata.json").exists():
                                target = str(r)
                                break
                    if target:
                        report = await asyncio.to_thread(verify_evidence_bundle, target)
                        await ws.send_json({
                            "type": "reverify_result",
                            **report,
                        })
                    else:
                        await ws.send_json({
                            "type": "error",
                            "message": "No evidence bundle or past run found to re-verify.",
                        })

                elif action == "preview":
                    preset = data.get("preset", "jensen_huang_portrait.jpg")
                    image_path = EXAMPLES_DIR / preset
                    if image_path.exists():
                        detector = FaceDetector.get_shared_instance()
                        faces = detector.detect(image_path)
                        primary = detector.get_primary_face(faces) if faces else None
                        primary_info = None
                        sample = []
                        all_faces = []
                        for f in faces:
                            is_prim = (primary is not None and f.face_index == primary.face_index)
                            crop_b64 = detector.crop_face_b64(image_path, f.bbox)
                            face_dict = {
                                "face_index": f.face_index,
                                "is_primary": is_prim,
                                "bbox": [int(x) for x in f.bbox],
                                "confidence": round(float(f.confidence), 4),
                                "sharpness": round(float(f.sharpness), 1),
                                "crop_b64": crop_b64,
                                "embedding_sample": [round(float(v), 4) for v in f.embedding[:16]],
                            }
                            all_faces.append(face_dict)
                            if is_prim:
                                primary_info = face_dict
                                sample = face_dict["embedding_sample"]

                        await ws.send_json({
                            "type": "preview_ready",
                            "preset": preset,
                            "image_b64": file_to_base64(image_path),
                            "faces_count": len(faces),
                            "primary_face": primary_info,
                            "all_faces": all_faces,
                            "embedding_sample": sample,
                        })

                elif action == "run":
                    upload_id = data.get("upload_id")
                    custom_b64 = data.get("custom_image_b64")
                    preset = data.get("preset")
                    tamper = bool(data.get("tamper", False))
                    dry_run = bool(data.get("dry_run", False))
                    engine = data.get("engine", "auto")
                    face_index = data.get("face_index")
                    if face_index is not None:
                        face_index = int(face_index)

                    if upload_id:
                        image_path = UPLOADS_DIR / upload_id
                        if not image_path.exists():
                            await ws.send_json({
                                "type": "error",
                                "message": f"Uploaded image '{upload_id}' not found on server. Please re-upload.",
                            })
                            continue
                    elif custom_b64:
                        if "," in custom_b64:
                            _, raw_data = custom_b64.split(",", 1)
                        else:
                            raw_data = custom_b64
                        file_bytes = base64.b64decode(raw_data)
                        upload_filename = f"upload_{uuid.uuid4().hex[:8]}.jpg"
                        image_path = UPLOADS_DIR / upload_filename
                        image_path.write_bytes(file_bytes)
                    elif preset:
                        image_path = EXAMPLES_DIR / preset
                        if not image_path.exists():
                            await ws.send_json({
                                "type": "error",
                                "message": f"Preset image '{preset}' not found on server.",
                            })
                            continue
                    else:
                        await ws.send_json({
                            "type": "error",
                            "message": "No probe target specified. Please select a preset or upload an image before running.",
                        })
                        continue

                    similarity_thresh = float(data.get("similarity_threshold", 0.80))

                    config = get_config()
                    if engine:
                        object.__setattr__(config, "search_engine", engine)
                    object.__setattr__(config, "similarity_threshold", similarity_thresh)
                    object.__setattr__(config, "early_stop_similarity", min(0.95, similarity_thresh + 0.05))

                    orchestrator = PipelineOrchestrator(config=config)

                    def on_step(step_num: int, title: str, status: str, details: str):
                        timestamp = time.strftime("%H:%M:%S")
                        asyncio.create_task(ws.send_json({
                            "type": "step_update",
                            "step": step_num,
                            "title": title,
                            "status": status,
                            "details": details,
                            "timestamp": timestamp,
                        }))

                    result = await orchestrator.execute(
                        image_path=image_path,
                        face_index=face_index,
                        tamper=tamper,
                        dry_run=dry_run,
                        on_step_update=on_step,
                    )

                    input_b64 = file_to_base64(image_path)
                    matched_b64 = None
                    matched_media_path = Path(result.run_dir) / "matched_media.jpg"
                    if matched_media_path.exists():
                        matched_b64 = file_to_base64(matched_media_path)

                    leaderboard_data = []
                    for idx, c in enumerate(result.top_candidates, 1):
                        leaderboard_data.append({
                            "rank": idx,
                            "domain": c.source_domain,
                            "title": c.title or c.source_domain,
                            "similarity": round(c.best_similarity, 4),
                            "page_url": c.page_url,
                            "image_url": c.image_url,
                            "status": "MATCH" if idx == 1 and c.best_similarity >= similarity_thresh else ("RUNNER" if idx == 2 else "REJECT"),
                        })

                    attestation_data = None
                    if result.attestation:
                        attestation_data = {
                            "content_hash": result.content_hash,
                            "record_hash": result.record_hash,
                            "tx_hash": result.attestation.tx_hash,
                            "network": result.attestation.network,
                            "submitter": result.attestation.submitter,
                            "block_number": result.attestation.block_number,
                            "is_simulated": result.attestation.is_simulated,
                            "explorer_url": result.attestation.explorer_url,
                        }

                    primary_face_data = None
                    embedding_sample = []
                    if result.primary_face:
                        primary_face_data = {
                            "bbox": list(result.primary_face.bbox),
                            "confidence": round(result.primary_face.confidence, 4),
                            "sharpness": round(result.primary_face.sharpness, 1),
                        }
                        embedding_sample = [round(float(v), 4) for v in result.primary_face.embedding[:16]]

                    # Multi-face profiles serialization
                    faces_results_data = []
                    for p in result.faces_results:
                        p_leaderboard = []
                        for idx, c in enumerate(p.top_candidates, 1):
                            p_leaderboard.append({
                                "rank": idx,
                                "domain": c.source_domain,
                                "title": c.title or c.source_domain,
                                "similarity": round(c.best_similarity, 4),
                                "page_url": c.page_url,
                                "image_url": c.image_url,
                                "status": "MATCH" if idx == 1 and c.best_similarity >= similarity_thresh else ("RUNNER" if idx == 2 else "REJECT"),
                            })

                        p_attestation = None
                        if p.attestation:
                            p_attestation = {
                                "content_hash": p.content_hash,
                                "record_hash": p.record_hash,
                                "tx_hash": p.attestation.tx_hash,
                                "network": p.attestation.network,
                                "submitter": p.attestation.submitter,
                                "block_number": p.attestation.block_number,
                                "is_simulated": p.attestation.is_simulated,
                                "explorer_url": p.attestation.explorer_url,
                            }

                        faces_results_data.append({
                            "face_index": p.face_index,
                            "is_primary": p.is_primary,
                            "bbox": list(p.bbox),
                            "confidence": p.confidence,
                            "sharpness": p.sharpness,
                            "crop_b64": p.crop_b64,
                            "embedding_sample": p.embedding_sample,
                            "matched_image_b64": p.matched_image_b64,
                            "best_match": {
                                "domain": p.best_match.source_domain,
                                "title": p.best_match.title or p.best_match.source_domain,
                                "similarity": round(p.best_match.best_similarity, 4),
                                "page_url": p.best_match.page_url,
                                "image_url": p.best_match.image_url,
                            } if p.best_match else None,
                            "runner_up_margin": round(p.runner_up_margin, 4),
                            "leaderboard": p_leaderboard,
                            "attestation": p_attestation,
                            "social_identity": p.social_identity.to_dict() if p.social_identity else None,
                            "re_verification_passed": p.re_verification_passed,
                            "status_message": p.status_message,
                        })

                    payload = {
                        "type": "run_complete",
                        "success": result.success,
                        "status_message": result.status_message,
                        "elapsed_seconds": round(result.elapsed_seconds, 2),
                        "run_id": result.run_id,
                        "input_image_b64": input_b64,
                        "matched_image_b64": matched_b64,
                        "primary_face": primary_face_data,
                        "embedding_sample": embedding_sample,
                        "candidates_discovered": result.candidates_discovered,
                        "candidates_downloaded": result.candidates_downloaded,
                        "runner_up_margin": round(result.runner_up_margin, 4),
                        "leaderboard": leaderboard_data,
                        "attestation": attestation_data,
                        "social_identity": faces_results_data[0].get("social_identity") if faces_results_data else None,
                        "re_verification_passed": result.re_verification_passed,
                        "tamper_mode_active": result.tamper_mode_active,
                        "faces_results": faces_results_data,
                    }

                    await ws.send_json(payload)

            except Exception as e:
                logger.error(f"WS error: {e}", exc_info=True)
                await ws.send_json({"type": "error", "message": str(e)})

    return ws


def create_app() -> web.Application:
    # Set client_max_size to 50MB to support high-resolution custom portrait uploads
    app = web.Application(client_max_size=50 * 1024 * 1024)
    app.router.add_get("/", handle_index)
    app.router.add_get("/api/examples", handle_get_examples)
    app.router.add_get("/api/runs", handle_get_runs)
    app.router.add_post("/api/upload", handle_upload)
    app.router.add_post("/api/reverify", handle_reverify)
    app.router.add_get("/api/reverify", handle_reverify)
    app.router.add_get("/api/ledger/{run_id}", handle_get_ledger)
    app.router.add_get("/ws", handle_ws)
    return app


def run_server(host: str = "127.0.0.1", port: int = 8000):
    app = create_app()
    print("=" * 64)
    print("HH GOA TASK 3 — MONOCHROME BENTO GRID DASHBOARD")
    print(f"Local URL: http://{host}:{port}")
    print("=" * 64)
    web.run_app(app, host=host, port=port, access_log=None)


if __name__ == "__main__":
    run_server()
