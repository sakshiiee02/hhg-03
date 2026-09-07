# Aegis — Enterprise Facial Intelligence & Cryptographic Attestation Platform

[![Python 3.11+](https://img.shields.io/badge/python-3.11%20%7C%203.12%20%7C%203.13-blue.svg)](https://www.python.org/)
[![Hardware Acceleration](https://img.shields.io/badge/GPU%20Accel-DirectX%2012%20DirectML-orange.svg)](https://onnxruntime.ai/)
[![InsightFace](https://img.shields.io/badge/Biometrics-SCRFD%20%2B%20ArcFace-green.svg)](https://github.com/deepinsight/insightface)
[![Blockchain](https://img.shields.io/badge/Blockchain-Ethereum%20Sepolia%20%7C%20Simulated%20EVM-purple.svg)](https://sepolia.etherscan.io/)
[![Interface](https://img.shields.io/badge/GUI-Enterprise%20SaaS%20%28Linear%20Design%29-white.svg)](#interactive-enterprise-saas-gui)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> **HH Goa 2026 — Task 3 Master Build Specification & Implementation**
> Aegis is a state-of-the-art, high-performance facial intelligence system. Given an input facial portrait or group image, Aegis discovers authentic online copies across the open web via genuine multi-engine reverse search, independently verifies biometric identities locally with SCRFD and 512-D ArcFace, extracts verified social media profiles and canonical entity data, and seals immutable cryptographic attestation records directly onto Ethereum Sepolia.

---

## 1. System Architecture

Aegis executes an 8-stage asynchronous pipeline coordinating local GPU neural inference, concurrent web intelligence, and blockchain settlement:

```
                                  PROBE IMAGE
                                       │
                    ┌──────────────────┴──────────────────┐
                    ▼                                     ▼
          [1/8] Face Detection                  Quality Gate Validation
          • SCRFD-10G Landmark Align            • Laplacian Blur Var (>= 35.0)
          • Single vs Multi-Face Detect         • Resolution Gate (>= 40px)
          • 0-Face Strict Rejection             • Bounding Box Normalization
                    │                                     │
                    └──────────────────┬──────────────────┘
                                       │
                                       ▼
                             [2/8] Face Embedding
                             • 512-D ArcFace L2-Normalized Vector
                             • DirectML GPU Accel (31.9ms / image)
                             • Model Pruned (Det + Rec Only)
                                       │
                    ┌──────────────────┴──────────────────┐
                    ▼                                     ▼
          Single Portrait Mode                  Multi-Face Scene Mode
          • Send Full Image Query               • Adaptive Per-Subject Crops
                                                • Context Padding Ratio: 0.35
                                                • Lanczos-4 Upscaling (>= 250px)
                    │                                     │
                    └──────────────────┬──────────────────┘
                                       │
                                       ▼
                       [3/8] Parallel Web Discovery
                       • SerpApi (Google Lens 2-Step Protocol)
                       • Yandex Images (Playwright Stealth)
                       • Round-Robin Interleaving & Quality Scoring
                       • Persistent SHA-256 Search Disk Cache
                       • Google Lens /goto Redirect Resolution
                       • Canonical Social URL Normalization
                                       │
                                       ▼
                       [4/8] Streaming Candidate Fetch
                       • Tier 1: Direct Search Engine CDN (Fast-path)
                       • Tier 2: Page DOM Extraction (Fallback)
                       • SSRF Protection (Private & Cloud Metadata blocked)
                       • Bounded Async Producer-Consumer Queue
                                       │
                                       ▼
                       [5/8] Real-Time Biometric Scoring
                       • Vectorized Batch Cosine Similarity Matrix
                       • Parallel Overlapping of Network I/O & GPU
                       • Confident Early Stopping (Cosine >= 0.92)
                                       │
                                       ▼
                       [6/8] Match Selection & Attribution
                       • Multi-Face Resolution & Ranking
                       • Automated Social Media Discovery Engine
                       • Wikidata Entity & Anchor Extraction
                       • Verified Platform Badges (X, LinkedIn, Wiki, GitHub)
                                       │
                                       ▼
                       [7/8] Cryptographic Attestation
                       • RFC-8785 Deterministic Canonical JSON
                       • Dual SHA-256 Hashes: Content & Canonical Record
                       • EVM bytes32 Smart Contract Settlement
                       • Ethereum Sepolia Testnet + Simulated EVM
                       • Forensic Evidence Bundle Generation
                                       │
                                       ▼
                       [8/8] Live Re-Verification & Tamper
                       • Live HTTP Re-fetch & Re-hash Validation
                       • On-Chain Immutable Hash Comparison
                       • --tamper Mutated Byte Attack Detection
                       • Standalone --reverify Forensic Audit Mode
```

---

## 2. Core Capabilities & Architecture Breakdown

### 2.1 Biometric Identification & Local GPU Acceleration
- **SCRFD-10G Face Detector**: High-efficiency, sub-millisecond face detector with 5-point facial landmark alignment.
- **512-D ArcFace Deep Embeddings**: Deep representation mapping faces to a unit hypersphere ($L_2$-norm = 1.0) where identity comparison is strictly equivalent to cosine distance.
- **DirectX 12 DirectML Hardware Acceleration**: Leverages Microsoft DirectML (`onnxruntime-directml`) natively utilizing Windows DirectX 12 on dedicated NVIDIA GPUs (e.g. NVIDIA GeForce RTX 2050). Cuts inference latency from $\approx 250\,\text{ms}$ on CPU down to **31.9 ms per image** ($8\times$ speedup) without requiring external CUDA/cuDNN installations.
- **Model Pruning**: Pruned InsightFace `FaceAnalysis` to `allowed_modules=['detection', 'recognition']`, eliminating 3 unused neural networks (`1k3d68`, `2d106det`, `genderage`) and saving 60–120 redundant forward passes per search.
- **Biometric Quality Gate**: Laplacian variance blur filter (rejecting motion blur $< 35.0$), resolution bounds ($\ge 40\times 40\,\text{px}$), and confidence cutoffs ($\ge 0.60$).
- **Strict 0-Face Rejection Gate**: Non-human or scenery images with 0 detected faces are immediately rejected across the UI, WebSocket server, and CLI before initiating any network operations.

### 2.2 Adaptive Query Dispatch & Multi-Face Crop Search
- **Single-Face Portraits**: Full probe image is transmitted to visual search engines to preserve comprehensive photographic context.
- **Multi-Face Group Photos**: When `len(faces) > 1` (e.g. group photos), Aegis automatically activates **adaptive per-subject crop search**:
  - `FaceDetector.save_face_crop()` extracts each subject with `padding_ratio=0.35` (capturing face, forehead/hair, and collar).
  - Automatically upscales small crops to $\ge 250\times 250\,\text{px}$ via Lanczos-4 interpolation for optimal visual search ingestion.
  - Queries visual engines for each subject in parallel bounded by an `asyncio.Semaphore(2)` to protect against rate limits.
  - Groups discovered candidates by face index (`candidates_by_face[f.face_index]`), preventing single-source group dominance and giving every individual their own distinct gallery, leaderboard, and social identity.

### 2.3 Parallel Multi-Engine Web Discovery
- **Parallel Visual Engine Execution**:
  - **SerpApi Multi-Engine**: Uses the official 2-step Image Upload API and Google Lens reverse-search protocol, querying Google Lens, Bing Visual Search, and Google Reverse Image.
  - **Yandex Images**: Playwright Stealth engine with anti-bot evasions and programmatic file change dispatch.
- **Round-Robin Interleaving & Deduplication**: Merges multi-engine result sets in quality-weighted round-robin order, scoring candidates on CDN availability, direct destination fidelity, and title quality.
- **Google Lens `/goto` Unwrapping**: Resolves redirect wrappers (`google.com/goto?url=...` and `google.com/url?q=...`) to expose authentic third-party publisher URLs.
- **Canonical URL Normalization**: Normalizes social handles, removes language subdomains (`m.`, `mobile.`, `en.`), and strips tracking/noise parameters (`utm_*`, `ref`, `s`, `t`, `fbclid`).
- **Persistent SHA-256 Search Disk Cache**: Automatically hashes query image bytes and stores visual search results in `.cache/search_cache.json`. Repeated investigations or re-verifications load in **0ms** without consuming API credits or triggering network requests.

### 2.4 High-Throughput Streaming Retrieval & SSRF Protection
- **Direct CDN Ingestion (Tier 1)**: Directly downloads indexed search engine CDN previews in 100–200ms for over 95% of discovered results.
- **Page DOM Fallback (Tier 2)**: Only visits third-party HTML pages if the CDN image URL is missing or corrupted.
- **SSRF Hardening**: Validates all candidate URLs before opening connections, blocking loopback (`127.0.0.1`), private RFC-1918 subnets (`10.0.0.0/8`, `172.16.0.0/12`, `192.168.0.0/16`), and cloud metadata IP addresses (`169.254.169.254`).
- **Producer-Consumer Streaming Queue**: Uses `download_stream()` to overlap HTTP image downloads with GPU neural detection and embedding. Network latency is completely hidden behind inference time.
- **Confident Early Stopping**: If a candidate achieves $\text{cosine} \ge 0.96$ or ($\text{cosine} \ge 0.92$ with runner-up margin $\ge 0.15$), the orchestrator cancels remaining in-flight downloads immediately, cutting real-world runtime from 34.3s to **19.8s**.

### 2.5 Automated Social Media & Identity Attribution Engine
- **Live Entity Resolution**: Discovers genuine subject identities from OpenGraph metadata, page titles, URL slugs, and Wikidata claim graphs without hardcoding hints.
- **Verified Platform Badges**: Automatically resolves authentic profile URLs across 6 major platforms:
  - **X (Twitter)**: `@handle` format
  - **LinkedIn**: `in/handle` format
  - **GitHub**: `github.com/handle`
  - **Wikipedia**: Direct biographical article links
  - **Instagram**: `@handle` profile links
  - **YouTube**: `youtube.com/@channel`
- **Canonical Binding**: Social media profile links and canonical names are bound immutably into the RFC-8785 canonical record prior to hashing.

### 2.6 Cryptographic Attestation & Sepolia Smart Contract
- **RFC-8785 Canonical JSON**: Formats off-chain intelligence bundles with order-invariant deterministic serialization:
  - Lexicographically sorted dictionary keys.
  - No extraneous whitespace (`separators=(',', ':')`).
  - Unescaped UTF-8 characters (`ensure_ascii=False`).
- **Dual SHA-256 Fingerprints**:
  - `content_hash = SHA256(matched_image_bytes)`
  - `record_hash = SHA256(canonical_json_bytes)`
- **Smart Contract (`FaceVerificationRegistry.sol`)**:
  - Solidity contract maintaining an on-chain ledger mapping `recordHash` to `(contentHash, timestamp, submitter)`.
  - Emits `RecordAttested(recordHash, contentHash, submitter, timestamp)` events.
  - Implements `getRecord(recordHash)` and `verifyRecord(recordHash, contentHash)` for public cryptographic verification.
- **Simulated EVM Mode**: Fallback mode allowing full end-to-end attestation testing without gas or network connectivity.

### 2.7 Standalone Forensic Re-Verification & Evidence Bundling
- **Forensic Evidence Bundle**: Exports full audit bundles to `runs/<run_id>/evidence/`:
  - `metadata.json`: Machine-readable audit ledger containing input parameters, face bounding boxes, hashes, Sepolia transaction receipt, and ranked candidates.
  - `matched_image.bin`: Raw byte capture of the winning matched image.
- **Standalone `--reverify` Command**:
  - Accepts a run directory, `metadata.json` path, or run ID.
  - Verifies local SHA-256 hashes against Ethereum Sepolia on-chain storage.
  - Re-evaluates 512-D ArcFace biometrics locally.
  - Exits with `code 0` on cryptographic validity; exits with `code 6` on tamper detection.

---

## 3. Standardized Probe Examples

The `examples/` directory contains 4 standardized test archetypes:

| Preset Name | File | Description | Expected Pipeline Behavior |
| :--- | :--- | :--- | :--- |
| **1 · Famous Person** | `sam_altman_portrait.jpg` | Single high-resolution portrait (sharpness 195.8, confidence 0.901). | Passes quality gate; executes full-image search; identifies Sam Altman; resolves verified X, LinkedIn, GitHub, and Wikipedia profiles; attests to Sepolia. |
| **2 · Multi-Person Group** | `multiple_faces_group.jpg` | Group photograph containing 6 distinct faces. | Detects 6 faces; generates interactive selector chips; executes individual face crop searches ($\ge 250\text{px}$) in parallel; provides distinct results per person. |
| **3 · Low Quality Blur Face** | `low_quality_blur_face.jpg` | High motion blur (Laplacian sharpness score 2.6). | Flags `(LOW QUALITY BLUR)` alert; fails quality gate threshold (35.0). |
| **4 · No Face Image** | `no_face_landscape.jpg` | Scenery landscape image with 0 human faces. | **Strictly rejected**: Execution button disabled (`REJECTED: NO FACE DETECTED`); WebSocket halts execution; CLI aborts with exit code 1. |

---

## 4. Interactive Enterprise SaaS GUI

The interactive web application is designed with an enterprise SaaS aesthetic (Linear and Palantir Foundry style) featuring a flat monochrome palette (`#090a0c` canvas, `#111317` surface, `#1a1d24` cards, `#1f2329` borders), zero emojis, and zero gradients.

Launch the GUI:
```powershell
python run_gui.py
```
*Options: `--port 8000`, `--host 127.0.0.1`, `--no-browser`.*

### Key Interface Features:
- **Sticky Execution Action Bar**: The `EXECUTE IDENTIFICATION` button and the 8-step stepper progress indicator are permanently pinned to the bottom of the left control pane (`position: sticky; bottom: 0;`), accessible without scrolling.
- **Photos-First Candidate Visual Gallery**: Displays discovered candidates as rich visual cards with high-resolution thumbnails, biometric similarity progress meters, rank badges (`#1`, `#2`, etc.), status tags (`MATCH`, `RUNNER`, `REJECT`), and direct outbound source links.
- **View Mode Switcher**: Seamless 1-click toggle between **Gallery (Visual Cards)** view and compact **Table (Data Rows)** view.
- **Multi-Person Interactive Selector Chips**: For multi-face images, selector chips (`Person #1 (Primary)` through `Person #6`) allow switching between subjects to inspect per-person candidates, leaderboards, and social identities.
- **Executive Verdict Banner**: High-contrast indicator displaying verification status (`100% VERIFIED`, `CRYPTOGRAPHIC TAMPER DETECTED`, or `REJECTED: NO FACE DETECTED`).
- **Cryptographic Attestation Drawer**: Tabbed bottom panel providing Content SHA-256, Canonical Record SHA-256, Sepolia Tx hash, and a 1-click JSON ledger download button.
- **Audit Ledger Screen (`/api/runs`)**: Dedicated historical audit table listing past runs, execution times, identified subjects, candidate counts, and verification statuses.
- **On-Chain Registry Screen**: Smart contract interface parameters, contract ABI inspector, and Etherscan explorer links.
- **Engine Settings Screen**: Configure reverse-image search router preferences (Auto, SerpApi, Yandex), API key status, and concurrency limits.

---

## 5. Quickstart Guide

### 5.1 Installation

```powershell
# 1. Clone repository
git clone https://github.com/sakshiiee02/hhg-03.git
cd hhg-03

# 2. Create and activate virtual environment
python -m venv venv
.\venv\Scripts\Activate.ps1

# 3. Install dependencies and Playwright Chromium
pip install -r requirements.txt
playwright install chromium
```

### 5.2 Configure Environment Variables

```powershell
Copy-Item .env.example .env
```

Edit `.env` as needed:
```ini
# Reverse Search Engines
SEARCH_ENGINE=auto             # 'auto', 'serpapi', or 'yandex'
SERPAPI_API_KEY=your_key_here  # Optional: for Google Lens, Bing & Google Reverse search

# Blockchain Attestation
BLOCKCHAIN_MODE=simulated      # 'sepolia' for live Ethereum testnet, 'simulated' for local EVM
SEPOLIA_RPC_URL=https://rpc.sepolia.org
SEPOLIA_PRIVATE_KEY=your_private_key_here
SEPOLIA_CONTRACT_ADDRESS=0x96e46f92ed8ee02e973e73db81d6f5adbc6df4c7
```

---

## 6. Command-Line Interface (CLI) Usage

### 6.1 Standard Pipeline Execution
Execute face detection, embedding, dynamic web discovery, biometric scoring, and blockchain attestation:

```powershell
# Standard execution (Sam Altman portrait)
python -m src.main --image examples/sam_altman_portrait.jpg

# Custom similarity threshold (default: 0.80)
python -m src.main --image examples/sam_altman_portrait.jpg --threshold 0.85

# Select specific face in a group photo (0-indexed)
python -m src.main --image examples/multiple_faces_group.jpg --face-index 1

# Override search engine (auto, serpapi, or yandex)
python -m src.main --image examples/sam_altman_portrait.jpg --engine serpapi
```

### 6.2 Cryptographic Tamper Demonstration
Deliberately mutates a byte in the off-chain payload before re-verification to demonstrate cryptographic mismatch detection:

```powershell
python -m src.main --image examples/sam_altman_portrait.jpg --tamper
```
*Result: Step 8 detects mismatch between live computed SHA-256 and the immutable on-chain record, triggering a high-contrast tamper alert.*

### 6.3 Fast Dry-Run (Biometrics Only)
Validates SCRFD face detection and ArcFace 512-D embedding locally without executing network calls:

```powershell
python -m src.main --image examples/sam_altman_portrait.jpg --dry-run
```

### 6.4 Standalone Forensic Re-Verification
Verifies an existing evidence bundle or past run ID against the blockchain:

```powershell
# Re-verify by run ID
python -m src.main --reverify run_20260907_193022_a1b2c3d4

# Re-verify by evidence directory
python -m src.main --reverify runs/run_20260907_193022_a1b2c3d4/evidence

# Re-verify by metadata.json path
python -m src.main --reverify runs/run_20260907_193022_a1b2c3d4/evidence/metadata.json
```

---

## 7. Smart Contract (`FaceVerificationRegistry.sol`)

Located at `contracts/FaceVerificationRegistry.sol`:

```solidity
// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

contract FaceVerificationRegistry {
    struct VerificationRecord {
        bytes32 contentHash;    // SHA-256 of matched image media
        bytes32 recordHash;     // SHA-256 of RFC-8785 canonical JSON
        uint256 timestamp;      // Block timestamp
        address submitter;      // Attesting wallet address
    }

    mapping(bytes32 => VerificationRecord) private _records;
    
    event RecordAttested(
        bytes32 indexed recordHash,
        bytes32 indexed contentHash,
        address indexed submitter,
        uint256 timestamp
    );

    function attestRecord(bytes32 contentHash, bytes32 recordHash) external returns (bool);
    function getRecord(bytes32 recordHash) external view returns (bytes32 contentHash, bytes32 recordHashOut, uint256 timestamp, address submitter);
    function verifyRecord(bytes32 recordHash, bytes32 expectedContentHash) external view returns (bool isValid, uint256 timestamp);
}
```

Pre-compiled ABI and bytecode are maintained in `contracts/build/FaceVerificationRegistry.json`. To redeploy:
```powershell
python -m src.blockchain.deploy
```

---

## 8. Verification & Accuracy Benchmarking

### 8.1 Automated Test Suite
Run the comprehensive test suite covering biometrics, candidate parsing, hashing, smart contracts, multi-engine routing, social discovery, and speed optimizations:

```powershell
pytest tests/ -v
```
```text
tests/test_candidate_parsing.py::test_extract_domain PASSED              [  2%]
tests/test_candidate_parsing.py::test_normalize_url PASSED               [  5%]
tests/test_candidate_parsing.py::test_extract_page_metadata PASSED       [  8%]
tests/test_competitor_features.py::test_google_goto_resolution PASSED    [ 11%]
tests/test_competitor_features.py::test_canonical_page_key_social_stripping PASSED [ 14%]
tests/test_competitor_features.py::test_ssrf_protection PASSED           [ 17%]
tests/test_competitor_features.py::test_persistent_search_cache PASSED   [ 20%]
tests/test_competitor_features.py::test_vectorized_cosine_similarity_matrix PASSED [ 22%]
tests/test_competitor_features.py::test_evidence_bundle_and_standalone_reverification PASSED [ 25%]
tests/test_hashing.py::test_hash_bytes PASSED                            [ 28%]
tests/test_hashing.py::test_canonicalize_json_order_invariance PASSED    [ 31%]
tests/test_hashing.py::test_to_bytes32_and_to_hex32 PASSED               [ 34%]
tests/test_hashing.py::test_to_bytes32_invalid_length PASSED             [ 37%]
tests/test_hashing.py::test_mock_blockchain_attestation_and_retrieval PASSED [ 40%]
tests/test_hashing.py::test_mock_blockchain_tamper_detection PASSED      [ 42%]
tests/test_matching.py::test_cosine_similarity_properties PASSED         [ 45%]
tests/test_matching.py::test_quality_gate_blur_and_resolution PASSED     [ 48%]
tests/test_matching.py::test_detector_on_sam_altman PASSED               [ 51%]
tests/test_matching.py::test_detector_on_multiple_faces_group PASSED     [ 54%]
tests/test_matching.py::test_detector_on_low_quality_blur_face PASSED    [ 57%]
tests/test_matching.py::test_detector_on_no_face_landscape PASSED        [ 60%]
tests/test_matching.py::test_cross_person_matching_discrimination PASSED [ 62%]
tests/test_multi_engine_search.py::test_round_robin_interleaving_and_deduplication PASSED [ 65%]
tests/test_multi_engine_search.py::test_search_router_auto_fallback_without_serpapi PASSED [ 68%]
tests/test_multi_engine_search.py::test_search_router_parallel_with_serpapi PASSED [ 71%]
tests/test_social_discovery.py::test_extract_social_from_url PASSED      [ 74%]
tests/test_social_discovery.py::test_extract_socials_from_html PASSED    [ 77%]
tests/test_social_discovery.py::test_clean_title_candidate PASSED        [ 80%]
tests/test_social_discovery.py::test_extract_entity_name PASSED          [ 82%]
tests/test_social_discovery.py::test_extract_entity_name_with_noise_slugs PASSED [ 85%]
tests/test_social_discovery.py::test_social_discovery_engine_live PASSED [ 88%]
tests/test_speed_optimizations.py::test_face_detector_module_pruning_and_directml PASSED [ 91%]
tests/test_speed_optimizations.py::test_direct_cdn_download_priority PASSED [ 94%]
tests/test_speed_optimizations.py::test_download_stream_early_cancellation PASSED [ 97%]
tests/test_speed_optimizations.py::test_confident_early_stop_scoring_logic PASSED [100%]

============================= 35 passed in 8.54s ==============================
```

### 8.2 Section 17 Biometric Accuracy Benchmark
Run the biometric discrimination benchmark evaluating genuine pairs, impostor discrimination, and quality gates:

```powershell
python -m tests.evaluate_models
```

---

## 9. Repository Structure

```text
hhg-03/
├── README.md                           # Master architectural & user documentation
├── requirements.txt                    # Pinned Python dependencies
├── .env.example                        # Configuration environment template
├── pytest.ini                          # Pytest configuration
├── run_gui.py                          # GUI server launcher script
├── contracts/
│   ├── FaceVerificationRegistry.sol    # Solidity smart contract
│   └── build/
│       └── FaceVerificationRegistry.json# Pre-compiled ABI and bytecode
├── docs/                               # Developer and evaluation documentation
│   ├── README.md                       # Documentation map
│   ├── HH_Goa_2026_Task3_Developer_Handoff.md # Master developer handoff specification
│   ├── ARCHITECTURE.md                 # In-depth subsystem architecture
│   ├── SETUP_GUIDE.md                  # Developer & evaluator setup guide
│   ├── DEMO_SCRIPT.md                  # 00:00–01:20 video walkthrough guide
│   └── EVALUATION_GUIDE.md             # Accuracy benchmarking guide
├── src/
│   ├── main.py                         # Rich terminal CLI entry point
│   ├── server.py                       # aiohttp WebSocket & REST server
│   ├── config.py                       # Configuration loader & settings
│   ├── face/
│   │   ├── detector.py                 # SCRFD detector, primary face selector & crop engine
│   │   ├── embedder.py                 # 512-D ArcFace vector extraction & cosine similarity
│   │   ├── quality.py                  # Laplacian blur score & resolution quality gates
│   │   └── matcher.py                  # Vectorized cosine matching & runner-up separation
│   ├── search/
│   │   ├── base.py                     # CandidateResult, Google Lens redirect unwrapping & SSRF safety
│   │   ├── router.py                   # Multi-engine search router & round-robin interleaver
│   │   ├── cache.py                    # Persistent SHA-256 search disk cache
│   │   ├── yandex.py                   # Playwright Stealth Yandex Images engine
│   │   └── serpapi.py                  # 2-step official Google Lens, Bing & Reverse engine
│   ├── social/
│   │   ├── resolver.py                 # Noise slug filter, title cleaner & entity name extractor
│   │   ├── wikidata.py                 # Wikidata SPARQL claim graph query engine
│   │   └── engine.py                   # End-to-end multi-platform social discovery engine
│   ├── web/
│   │   ├── downloader.py               # Bounded streaming downloader (Direct CDN first)
│   │   └── extract.py                  # OpenGraph & HTML metadata parser
│   ├── blockchain/
│   │   ├── hash.py                     # RFC-8785 canonical JSON & SHA-256 hashing
│   │   ├── contract.py                 # Web3 Sepolia client & Simulated EVM fallback
│   │   └── deploy.py                   # Smart contract deployment utility
│   ├── pipeline/
│   │   ├── orchestrator.py             # Complete 8-step pipeline coordinator
│   │   └── evidence.py                 # Evidence bundle ledger exporter
│   └── static/
│       └── index.html                  # Enterprise SaaS UI (Linear design, sticky actions)
├── examples/                           # 4 standardized demo images
│   ├── sam_altman_portrait.jpg         # 1. Famous Person (Sam Altman)
│   ├── multiple_faces_group.jpg        # 2. Multi-Person Group (6 faces)
│   ├── low_quality_blur_face.jpg       # 3. Low Quality Blur Face (sharpness 2.6)
│   └── no_face_landscape.jpg           # 4. No Face Image (0 faces, rejected)
├── runs/                               # Immutable run evidence artifacts & audit bundles
└── tests/                              # Automated test suite (35 tests passing)
```

---
