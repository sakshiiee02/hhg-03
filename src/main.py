"""
Command-Line Interface (CLI) entry point for HH Goa 2026 Task 3.
Provides high-impact Rich terminal visualization, live step spinners, candidate leaderboards,
cryptographic panels, and deliberate tamper demonstration modes.
"""

import argparse
import asyncio
import os
import sys
from pathlib import Path

# Ensure project root is in sys.path so 'src' is always importable
_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

# Force UTF-8 on Windows terminal streams to prevent charmap/cp1252 encode errors
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from src.config import PipelineConfig, get_config
from src.pipeline.orchestrator import PipelineOrchestrator, PipelineRunResult

console = Console(legacy_windows=False)


def print_banner():
    banner = Text(
        "================================================================================\n"
        "           HH GOA TASK 3 — FACE ID + BLOCKCHAIN VERIFICATION PIPELINE           \n"
        "================================================================================",
        style="bold cyan",
    )
    console.print(banner)


def render_leaderboard(result: PipelineRunResult):
    if not result.top_candidates:
        return

    table = Table(
        title="Candidate Face Verification Leaderboard",
        box=box.ROUNDED,
        title_style="bold yellow",
        header_style="bold magenta",
        expand=False,
    )
    table.add_column("Rank", justify="center", style="cyan", width=6)
    table.add_column("Domain", style="green", width=18)
    table.add_column("Page Title", style="white", width=34)
    table.add_column("Cosine Sim", justify="right", style="bold", width=12)
    table.add_column("Status", justify="center", width=10)

    for idx, cand in enumerate(result.top_candidates, 1):
        sim = cand.best_similarity
        if idx == 1 and sim >= 0.80:
            status_style = "[bold green]MATCH[/bold green]"
            sim_style = f"[bold green]{sim:.4f}[/bold green]"
        elif idx == 2:
            status_style = "[yellow]RUNNER[/yellow]"
            sim_style = f"[yellow]{sim:.4f}[/yellow]"
        else:
            status_style = "[dim]REJECT[/dim]"
            sim_style = f"[dim]{sim:.4f}[/dim]"

        raw_title = cand.title or "N/A"
        clean_title = (raw_title[:31] + "...") if len(raw_title) > 34 else raw_title

        table.add_row(
            str(idx),
            cand.source_domain,
            clean_title,
            sim_style,
            status_style,
        )

    console.print(table)
    margin = result.runner_up_margin
    margin_text = Text()
    margin_text.append("Margin over runner-up: ", style="bold white")
    if margin >= 0.10:
        margin_text.append(f"+{margin:.4f} (STRONG SEPARATION)", style="bold green")
    elif margin >= 0.05:
        margin_text.append(f"+{margin:.4f} (MODERATE SEPARATION)", style="yellow")
    else:
        margin_text.append(f"+{margin:.4f} (NARROW MARGIN / AMBIGUOUS)", style="red")
    console.print(margin_text)
    console.print()


def render_social_profiles_panel(result: PipelineRunResult):
    profiles_found = False
    for p in result.faces_results:
        if p.social_identity and p.social_identity.profiles:
            profiles_found = True
            break
    if not profiles_found:
        return

    for p in result.faces_results:
        if not p.social_identity or not p.social_identity.profiles:
            continue

        ident = p.social_identity
        suffix = f" (Face #{p.face_index})" if len(result.faces_results) > 1 else ""
        table = Table(
            title=f"[bold cyan]Discovered Social Profiles: {ident.canonical_name}{suffix}[/bold cyan]",
            box=box.ROUNDED,
            show_header=True,
            header_style="bold magenta",
        )
        table.add_column("Platform", style="bold white", width=16)
        table.add_column("Handle / Identifier", style="cyan", width=26)
        table.add_column("Source", style="dim white", width=16)
        table.add_column("Verified URL", style="underline blue")

        for prof in ident.profiles:
            table.add_row(
                prof.display_name,
                prof.handle,
                prof.source.upper(),
                prof.url,
            )

        console.print(table)
        if ident.bio_summary:
            console.print(f"[dim italic]Summary: {ident.bio_summary}[/dim italic]")
        console.print()


def render_attestation_panel(result: PipelineRunResult):
    att = result.attestation
    if not att:
        return

    grid = Table.grid(padding=(0, 2))
    grid.add_column(style="bold cyan", justify="right")
    grid.add_column(style="white")

    grid.add_row("Content SHA-256 :", result.content_hash)
    grid.add_row("Record SHA-256  :", result.record_hash)
    grid.add_row("Network Target  :", att.network)
    grid.add_row("Submitter Wallet:", att.submitter)
    grid.add_row("Block Number    :", str(att.block_number))
    grid.add_row("Transaction Tx  :", f"[bold green]{att.tx_hash}[/bold green]")
    if att.explorer_url:
        grid.add_row("Block Explorer  :", f"[underline blue]{att.explorer_url}[/underline blue]")

    panel = Panel(
        grid,
        title="[bold yellow]Ethereum Sepolia Cryptographic Attestation[/bold yellow]",
        border_style="cyan",
        box=box.DOUBLE,
    )
    console.print(panel)


def render_final_status(result: PipelineRunResult):
    console.print()
    if result.success and not result.attestation and not result.re_verification_passed:
        # Dry-run success case
        panel = Panel(
            Text(
                f"STATUS: DRY-RUN SUCCESSFUL\n"
                f"Face detected and 512-D ArcFace embedding validated.\n"
                f"Total Elapsed Time: {result.elapsed_seconds:.1f}s",
                justify="center",
                style="bold white",
            ),
            style="bold cyan",
            box=box.HEAVY,
        )
    elif result.re_verification_passed and not result.tamper_mode_active:
        panel = Panel(
            Text(
                f"FINAL STATUS: VERIFIED\n"
                f"On-chain cryptographic match confirmed across all hashes.\n"
                f"Evidence Ledger saved to: {result.run_dir}\n"
                f"Total Elapsed Time: {result.elapsed_seconds:.1f}s",
                justify="center",
                style="bold white",
            ),
            style="bold green",
            box=box.HEAVY,
        )
    elif result.tamper_mode_active:
        panel = Panel(
            Text(
                f"[ALERT] CRYPTOGRAPHIC TAMPER DETECTED: ON-CHAIN HASH MISMATCH\n"
                f"Deliberately altered payload failed verification against immutable Sepolia registry.\n"
                f"Evidence Ledger saved to: {result.run_dir}\n"
                f"Total Elapsed Time: {result.elapsed_seconds:.1f}s",
                justify="center",
                style="bold white",
            ),
            style="bold red",
            box=box.HEAVY,
        )
    else:
        panel = Panel(
            Text(
                f"FINAL STATUS: VERIFICATION FAILED\n"
                f"Details: {result.status_message}\n"
                f"Total Elapsed Time: {result.elapsed_seconds:.1f}s",
                justify="center",
                style="bold white",
            ),
            style="bold red",
            box=box.HEAVY,
        )
    console.print(panel)


def run_reverify(target: str) -> int:
    """Executes standalone forensic re-verification of an evidence bundle against the blockchain."""
    from src.pipeline.evidence import verify_evidence_bundle
    print_banner()
    console.print(Panel("[bold yellow]STANDALONE CRYPTOGRAPHIC EVIDENCE RE-VERIFICATION[/bold yellow]", box=box.ROUNDED))
    try:
        report = verify_evidence_bundle(target)
    except Exception as e:
        console.print(f"[bold red]Failed to inspect evidence bundle: {e}[/bold red]")
        return 1

    table = Table(title="Cryptographic Hash Verification", box=box.ROUNDED)
    table.add_column("Property", style="bold white", width=26)
    table.add_column("Fingerprint / Value", style="cyan", width=50)
    table.add_column("Status", justify="center", width=12)

    # Metadata hash row
    meta_status = "[bold green]✓ MATCH[/bold green]" if report["metadata_hash_match"] else "[bold red]✗ MISMATCH[/bold red]"
    table.add_row("Stored Metadata SHA-256", report["stored_metadata_hash"] or "N/A", "")
    table.add_row("Recomputed Metadata SHA-256", report["recomputed_metadata_hash"] or "N/A", meta_status)

    # Image hash row
    img_status = "[bold green]✓ MATCH[/bold green]" if report["image_hash_match"] else "[bold red]✗ MISMATCH[/bold red]"
    table.add_row("Stored Image SHA-256", report["stored_image_hash"] or "N/A", "")
    table.add_row("Recomputed Image SHA-256", report["recomputed_image_hash"] or "N/A", img_status)

    # Blockchain
    table.add_row("Blockchain Network", str(report.get("network", "N/A")), "")
    table.add_row("Transaction Hash", str(report.get("transaction", "N/A")), "")
    table.add_row("Block Number", str(report.get("block_number", "N/A")), "")

    console.print(table)
    console.print()

    if report["verified"]:
        console.print(Panel("[bold green]✓ EVIDENCE VERIFIED — UNTAMPERED ON-CHAIN ATTESTATION CONFIRMED[/bold green]", box=box.HEAVY))
        return 0
    else:
        console.print(Panel("[bold red]✗ TAMPER DETECTED — EVIDENCE HAS BEEN ALTERED OR HASHES DIVERGED[/bold red]", box=box.HEAVY))
        return 6


def main():
    parser = argparse.ArgumentParser(
        description="HH Goa 2026 Task 3 — Face ID + Web Discovery + Blockchain Verification",
    )
    parser.add_argument(
        "--image",
        "-i",
        type=str,
        default=None,
        help="Path to the input face portrait image (e.g. examples/sam_altman_portrait.jpg)",
    )
    parser.add_argument(
        "--reverify",
        type=str,
        default=None,
        help="Path to evidence bundle directory, metadata.json, or past run ID to re-verify against blockchain",
    )
    parser.add_argument(
        "--face-index",
        type=int,
        default=None,
        help="Index of face to select if input contains multiple people (default: auto-selects primary)",
    )
    parser.add_argument(
        "--tamper",
        action="store_true",
        help="Activate deliberate tamper demonstration mode (proves on-chain hash mismatch detection)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run biometric detection and embedding only, skipping external network operations",
    )
    parser.add_argument(
        "--engine",
        type=str,
        choices=["auto", "yandex", "serpapi"],
        default=None,
        help="Override search engine provider ('auto', 'yandex', or 'serpapi')",
    )

    args = parser.parse_args()

    if args.reverify:
        sys.exit(run_reverify(args.reverify))

    if not args.image:
        parser.error("Must provide either --image <path> to run pipeline, or --reverify <path> to verify evidence.")

    image_path = Path(args.image).resolve()

    if not image_path.exists():
        console.print(f"[bold red]Error: Input image does not exist at {image_path}[/bold red]")
        sys.exit(1)

    print_banner()

    # Step callback with colored output
    step_titles = {
        1: "Face Detection      ",
        2: "Face Embedding      ",
        3: "Web Discovery       ",
        4: "Candidate Retrieval ",
        5: "Face Verification   ",
        6: "Match Selection     ",
        7: "Attestation         ",
        8: "Live Re-Verification",
    }

    def on_step_update(step: int, title: str, status: str, details: str):
        label = step_titles.get(step, f"Step [{step}/8]         ")
        if status == "running":
            console.print(f"[cyan][{step}/8] {label}[/cyan] [yellow][...][/yellow] {details}")
        elif status == "completed":
            console.print(f"[bold green][{step}/8] {label}[/bold green] [bold green][OK][/bold green] {details}")
        elif status == "failed":
            console.print(f"[bold red][{step}/8] {label}[/bold red] [bold red][X][/bold red] {details}")

    config = get_config()
    if args.engine:
        # Override search engine if specified on CLI
        object.__setattr__(config, "search_engine", args.engine)

    orchestrator = PipelineOrchestrator(config=config)

    # Run the async pipeline
    try:
        result = asyncio.run(
            orchestrator.execute(
                image_path=image_path,
                face_index=args.face_index,
                tamper=args.tamper,
                dry_run=args.dry_run,
                on_step_update=on_step_update,
            )
        )
    except KeyboardInterrupt:
        console.print("\n[yellow]Pipeline aborted by user.[/yellow]")
        sys.exit(130)
    except Exception as e:
        console.print(f"\n[bold red]Pipeline error: {e}[/bold red]")
        sys.exit(1)

    console.print()
    if not args.dry_run and result.best_match:
        render_leaderboard(result)
        render_social_profiles_panel(result)
        render_attestation_panel(result)

    render_final_status(result)

    sys.exit(0 if result.success else 1)


if __name__ == "__main__":
    main()
