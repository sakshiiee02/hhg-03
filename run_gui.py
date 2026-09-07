"""
HH Goa 2026 — Task 3
Graphical User Interface Launcher (Monochrome Bento Grid)

Usage:
    python run_gui.py
    python run_gui.py --port 8080 --no-browser
"""

import argparse
import sys
import webbrowser
from pathlib import Path

# Ensure project root is in sys.path
ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.server import run_server


def main():
    parser = argparse.ArgumentParser(
        description="HH Goa 2026 Task 3 — Monochrome Bento Grid GUI"
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Host interface to bind (default: 127.0.0.1)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to listen on (default: 8000)",
    )
    parser.add_argument(
        "--no-browser",
        action="store_true",
        help="Do not automatically open web browser on startup",
    )
    args = parser.parse_args()

    url = f"http://{args.host}:{args.port}"
    print("\n" + "=" * 68)
    print("  HH GOA 2026 — TASK 3: BIOMETRIC & BLOCKCHAIN ATTESTATION GUI")
    print(f"  Interface: {url}")
    print("  Design   : Strict Monochrome Bento Grid (0 animations, 0 gradients)")
    print("=" * 68 + "\n")

    if not args.no_browser and args.host in ("127.0.0.1", "localhost"):
        try:
            webbrowser.open(url)
        except Exception:
            pass

    run_server(host=args.host, port=args.port)


if __name__ == "__main__":
    main()
