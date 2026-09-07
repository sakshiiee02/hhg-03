# Face ID + Dynamic Web Discovery + Blockchain Verification

[![Python 3.11+](https://img.shields.io/badge/python-3.11%20%7C%203.12%20%7C%203.13-blue.svg)](https://www.python.org/)
[![InsightFace](https://img.shields.io/badge/model-InsightFace%20%2F%20ArcFace-green.svg)](https://github.com/deepinsight/insightface)
[![Network](https://img.shields.io/badge/blockchain-Ethereum%20Sepolia-purple.svg)](https://sepolia.etherscan.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> **HH Goa 2026 — Task 3 Build Specification & Implementation**
> A high-performance, demonstrable command-line pipeline that accepts a facial image, discovers candidate copies across the open web via genuine reverse-image search, independently verifies biometric identity with SCRFD + ArcFace, and anchors a tamper-evident cryptographic fingerprint on Ethereum Sepolia.

---

## Architecture Flow

```text
+---------------------------------------------------------------------------------------+
|                                    CLI PIPELINE                                       |
+---------------------------------------------------------------------------------------+
|  [1/8] Input Image Validation -> SCRFD Face Detection (Primary face / --face-index)   |
|  [2/8] ArcFace Feature Extraction -> 512-D L2-Normalized Embedding                   |
|  [3/8] Reverse Image Discovery -> Playwright Stealth (Yandex -> Google Lens -> Serp)  |
|  [4/8] Concurrent Candidate Fetch -> Bounded async download + multi-tiered fallback    |
|  [5/8] Local Face Verification -> Candidate SCRFD + ArcFace cosine similarity         |
|  [6/8] Match Scoring & Selection -> Best match + runner-up margin + early stopping     |
|  [7/8] Blockchain Attestation -> Canonical Record SHA-256 + Image SHA-256 -> Sepolia  |
|  [8/8] Live Re-Verification -> Re-fetch + re-hash + on-chain comparison (--tamper)   |
+---------------------------------------------------------------------------------------+
```

---

## Core Capabilities

1. **Biometric Face Identification**:
   - High-precision SCRFD detector with landmark alignment.
   - 512-dimensional $L_2$-normalized ArcFace deep feature representations.
   - Automatic primary-face selection for group images with `--face-index` override.
2. **Authentic Reverse-Image Discovery**:
   - Zero hardcoding: conducts real dynamic search on public web indexes.
   - Resilient multi-engine routing: Playwright stealth on Yandex Images + Google Lens, with optional SerpApi fallback.
3. **Multi-Tiered Web Ingestion**:
   - Bounded async downloader (8 concurrent workers) with strict 5s timeouts.
   - Bypasses social media login gates (Instagram, X, TikTok) by falling back to search engine direct CDN previews while retaining the authentic source URL.
4. **Adaptive Early Stopping**:
   - Halts candidate processing once a high-confidence match ($S_C \ge 0.85$ with solid margin) is verified.
   - Completes the entire pipeline in under 25 seconds for video recording.
5. **Cryptographic Attestation & Sepolia Registry**:
   - RFC-8785 deterministic canonical JSON formatting.
   - Dual SHA-256 fingerprints (`image_sha256` and `record_sha256`) mapped to Solidity `bytes32`.
   - Pure Python `web3.py` client interacting with `FaceVerificationRegistry.sol`.
   - Simulated EVM fallback mode guarantees 100% demo uptime even if testnet RPCs or faucets fail.
6. **Live Re-Verification & Tamper Protection**:
   - Step 8 re-fetches media over the network and verifies hashes against the immutable on-chain record.
   - `--tamper` CLI mode deliberately mutates data off-chain to demonstrate cryptographic mismatch detection.



## Quickstart

### 1. Environment Setup
```powershell
# Create and activate virtual environment
python -m venv venv
.\venv\Scripts\Activate.ps1

# Install dependencies and Playwright Chromium
pip install -r requirements.txt
playwright install chromium
```

### 2. Configure Environment
```powershell
Copy-Item .env.example .env
```
*(Optional)* Add your `SEPOLIA_RPC_URL` and `SEPOLIA_PRIVATE_KEY` to attest on live Ethereum Sepolia. If left unconfigured, the pipeline automatically runs in Simulated EVM Mode.

### 3. Execute the Pipeline (Command Line)
```powershell
# Standard happy path execution
python -m src.main --image examples/jensen_huang_portrait.jpg

# Cryptographic tamper demonstration
python -m src.main --image examples/jensen_huang_portrait.jpg --tamper
```

### 4. Launch the Interactive GUI (Monochrome Bento Grid)
A graphical dashboard arranged in a modular Bento Grid with zero animations, zero gradients, and a strict monochrome palette (#000000 to #ffffff).

```powershell
python run_gui.py
```
Options:
- `--port 8000`: Custom HTTP port (default: 8000).
- `--host 127.0.0.1`: Custom host binding.
- `--no-browser`: Disable automatic browser window opening.

Access the dashboard in your browser at `http://127.0.0.1:8000`. Features:
- **Card 01 (Config)**: Preset selector (Jensen Huang, Sam Altman, Linus Torvalds) + Custom image file upload.
- **Card 02 (Stepper)**: Real-time 8-step execution progression badges.
- **Card 03 (Probe Profile)**: Instant image preview, SCRFD bounding box, Laplacian blur score, and 512-D ArcFace vector sample.
- **Card 04 (Matched Media)**: Discovered online web image, top cosine similarity, and runner-up margin.
- **Card 05 (Leaderboard)**: Full candidate verification rankings and status tags (`MATCH`, `RUNNER`, `REJECT`).
- **Card 06 (Log Stream)**: Real-time STDOUT execution console feed.
- **Card 07 (Attestation)**: Canonical RFC-8785 hash, Image SHA-256, Sepolia Tx hash, and 1-click JSON ledger download.
- **Card 08 (Verdict Banner)**: High-contrast cryptographic integrity indicator (`VERIFIED` vs `TAMPER DETECTED`).

---

## Repository Structure

```text
hh-task3/
├── README.md                           # Quickstart & project overview
├── requirements.txt                    # Pinned Python dependencies
├── .env.example                        # Configuration template
├── docs/                               # Comprehensive project documentation
│   ├── README.md                       # Documentation index & guide map
│   ├── HH_Goa_2026_Task3_Developer_Handoff.md # Master handoff specification
│   ├── ARCHITECTURE.md                 # Deep technical subsystem architecture
│   ├── SETUP_GUIDE.md                  # Developer installation & setup instructions
│   ├── DEMO_SCRIPT.md                  # Evaluator screen-recording walkthrough
│   └── EVALUATION_GUIDE.md             # Biometric accuracy benchmarking guide
├── contracts/
│   ├── FaceVerificationRegistry.sol    # Smart contract source
│   └── build/                          # Pre-compiled ABI and bytecode
├── src/
│   ├── main.py                         # Rich CLI entry point
│   ├── config.py                       # Configuration & settings loader
│   ├── face/                           # SCRFD detector & ArcFace embedder
│   ├── search/                         # Yandex, Lens & SerpApi search router
│   ├── web/                            # Bounded async multi-tiered downloader
│   ├── blockchain/                     # Web3 client, deployer & hashing
│   └── pipeline/                       # 8-step orchestrator & evidence exporter
├── examples/                           # Bundled demo & test images
├── runs/                               # Immutable run evidence artifacts
└── tests/                              # Unit & accuracy evaluation tests
```

---

## Submission Checklist

- [x] Tested with fresh input image from the command line.
- [x] SCRFD face detection and 512-D ArcFace embedding generated locally.
- [x] Genuine dynamic reverse search conducted without hardcoded posts.
- [x] Multi-tiered extraction retrieves candidate images despite social login walls.
- [x] Face independently verified ($S_C \ge 0.80$ with margin over runner-up).
- [x] Deterministic canonical record and media SHA-256 computed.
- [x] Attestation anchored to Ethereum Sepolia.
- [x] Live re-fetch and re-hash confirms on-chain match.
- [x] `--tamper` flag triggers visible cryptographic mismatch.
- [x] Complete developer and evaluation documentation committed.
