#!/usr/bin/env python3
"""
CLI script for downloading chess data from various sources.

Usage:
    python scripts/download_data.py --source wikipedia --limit 10
    python scripts/download_data.py --source lichess --limit 50 --min-elo 2600
    python scripts/download_data.py --source all
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

from loguru import logger
from rich.console import Console
from rich.progress import BarColumn, Progress, SpinnerColumn, TaskProgressColumn, TextColumn
from rich.table import Table

# Add src to path
# sys.path.insert(0, str(Path(__file__).parent.parent))
from chess_agentic_rag.core.config import settings
from chess_agentic_rag.core.models import Document
from chess_agentic_rag.data.loaders import LichessLoader, PGNMentorLoader, WikipediaLoader

console = Console()


def setup_logging(verbose: bool) -> None:
    """Configure logging."""
    logger.remove()

    level = "DEBUG" if verbose else "INFO"

    logger.add(
        lambda msg: console.print(msg, style="dim"),
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
        level=level,
    )


def save_document(doc: Document, output_dir: Path) -> Path:
    """
    Save document to file.

    Args:
        doc: Document to save
        output_dir: Output directory

    Returns:
        Path to saved file
    """
    # Create source subdirectory
    source_dir = output_dir / doc.metadata["source"]
    source_dir.mkdir(parents=True, exist_ok=True)

    # Generate filename
    doc_id = doc.id.replace("/", "_").replace(" ", "_")
    filename = f"{doc_id}.json"
    filepath = source_dir / filename

    # Save as JSON
    data = {
        "id": doc.id,
        "content": doc.content,
        "doc_type": doc.doc_type.value,
        "metadata": doc.metadata,
        "created_at": doc.created_at.isoformat() if doc.created_at else None,
    }

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    return filepath


def download_wikipedia(
    limit: int | None,
    output_dir: Path,
) -> tuple[int, int]:
    """
    Download Wikipedia articles.

    Args:
        limit: Maximum articles to download
        output_dir: Output directory

    Returns:
        Tuple of (successful, failed) counts
    """
    console.print("\n[bold blue]Downloading Wikipedia articles...[/bold blue]")

    loader = WikipediaLoader()

    if not loader.validate():
        console.print("[red]✗ Wikipedia is not accessible[/red]")
        return 0, 0

    console.print("[green]✓ Wikipedia is accessible[/green]")

    titles = loader.titles[:limit] if limit else loader.titles
    successful = 0
    failed = 0

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    ) as progress:
        task = progress.add_task(
            f"Downloading {len(titles)} articles...",
            total=len(titles),
        )

        loader_limited = WikipediaLoader(titles=titles)

        for doc in loader_limited.load():
            try:
                save_document(doc, output_dir)
                successful += 1
                progress.update(
                    task,
                    advance=1,
                    description=f"Downloaded: {doc.metadata['title'][:40]}...",
                )
            except Exception as e:
                logger.error(f"Failed to save document: {e}")
                failed += 1
                progress.update(task, advance=1)

    return successful, failed


def download_lichess(
    limit: int | None,
    min_elo: int,
    output_dir: Path,
) -> tuple[int, int]:
    """
    Download Lichess games.

    Args:
        limit: Maximum games to download
        min_elo: Minimum ELO rating
        output_dir: Output directory

    Returns:
        Tuple of (successful, failed) counts
    """
    console.print("\n[bold blue]Downloading Lichess games...[/bold blue]")

    max_games = limit if limit else 500
    loader = LichessLoader(min_elo=min_elo, max_games=max_games)

    if not loader.validate():
        console.print("[red]✗ Lichess API is not accessible[/red]")
        return 0, 0

    console.print("[green]✓ Lichess API is accessible[/green]")

    successful = 0
    failed = 0

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    ) as progress:
        task = progress.add_task(
            f"Downloading up to {max_games} games (ELO ≥{min_elo})...",
            total=max_games,
        )

        for doc in loader.load():
            try:
                save_document(doc, output_dir)
                successful += 1

                white = doc.metadata.get("white", "?")
                black = doc.metadata.get("black", "?")
                progress.update(
                    task,
                    advance=1,
                    description=f"Downloaded: {white} vs {black}",
                )
            except Exception as e:
                logger.error(f"Failed to save document: {e}")
                failed += 1
                progress.update(task, advance=1)

    return successful, failed


def download_pgnmentor(
    limit: int | None,
    output_dir: Path,
    collections: list[str] | None = None,
) -> tuple[int, int]:
    """
    Download PGNMentor historical games.

    Args:
        limit: Maximum games per collection
        output_dir: Output directory
        collections: Player/event names (None = default famous players)

    Returns:
        Tuple of (successful, failed) counts
    """
    console.print("\n[bold blue]Downloading PGNMentor historical games...[/bold blue]")

    loader = PGNMentorLoader(
        collections=collections,
        collection_type="players",
        max_games_per_collection=limit,
    )

    if not loader.validate():
        console.print("[red]✗ PGNMentor is not accessible[/red]")
        return 0, 0

    console.print("[green]✓ PGNMentor is accessible[/green]")
    console.print(f"Collections to download: {', '.join(loader.collections[:5])}...")

    successful = 0
    failed = 0

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        console=console,
    ) as progress:
        task = progress.add_task(
            "Downloading historical games...",
            total=None,  # Unknown total
        )

        for doc in loader.load():
            try:
                save_document(doc, output_dir)
                successful += 1

                white = doc.metadata.get("white", "?")
                black = doc.metadata.get("black", "?")
                collection = doc.metadata.get("collection", "?")
                progress.update(
                    task,
                    description=f"[{collection}] {white} vs {black}",
                )
            except Exception as e:
                logger.error(f"Failed to save document: {e}")
                failed += 1

        progress.update(task, description=f"Downloaded {successful} games")

    return successful, failed


def generate_manifest(output_dir: Path, stats: dict) -> None:
    """Generate download manifest file."""
    manifest = {
        "version": "1.0",
        "created_at": datetime.now().isoformat(),
        "statistics": stats,
    }

    manifest_path = output_dir / "manifest_raw.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    console.print(f"\n[green]✓ Manifest saved to: {manifest_path}[/green]")


def display_summary(stats: dict) -> None:
    """Display download summary."""
    table = Table(title="\n📊 Download Summary", show_header=True, header_style="bold magenta")
    table.add_column("Source", style="cyan")
    table.add_column("Successful", justify="right", style="green")
    table.add_column("Failed", justify="right", style="red")
    table.add_column("Total", justify="right", style="bold")

    for source, data in stats.items():
        if source != "total":
            table.add_row(
                source.capitalize(),
                str(data["successful"]),
                str(data["failed"]),
                str(data["successful"] + data["failed"]),
            )

    table.add_row(
        "TOTAL",
        str(stats["total"]["successful"]),
        str(stats["total"]["failed"]),
        str(stats["total"]["successful"] + stats["total"]["failed"]),
        style="bold",
    )

    console.print(table)


def main() -> None:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Download chess data from various sources",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--source",
        choices=["wikipedia", "lichess", "pgnmentor", "all"],
        default="all",
        help="Data source to download from",
    )

    parser.add_argument(
        "--limit",
        type=int,
        help="Maximum number of documents to download per source",
    )

    parser.add_argument(
        "--min-elo",
        type=int,
        default=2500,
        help="Minimum ELO rating for Lichess games (default: 2500)",
    )

    parser.add_argument(
        "--collections",
        nargs="+",
        help="Specific PGNMentor collections to download (player names)",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/raw"),
        help="Output directory (default: data/raw)",
    )

    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    # Setup
    setup_logging(args.verbose)
    args.output.mkdir(parents=True, exist_ok=True)

    console.print("\n[bold]♟️  Chess Data Downloader[/bold]")
    console.print(f"Output directory: {args.output.absolute()}\n")

    # Statistics
    stats = {
        "wikipedia": {"successful": 0, "failed": 0},
        "lichess": {"successful": 0, "failed": 0},
        "pgnmentor": {"successful": 0, "failed": 0},
        "total": {"successful": 0, "failed": 0},
    }

    # Download based on source
    try:
        if args.source in ["wikipedia", "all"]:
            wiki_success, wiki_fail = download_wikipedia(args.limit, args.output)
            stats["wikipedia"]["successful"] = wiki_success
            stats["wikipedia"]["failed"] = wiki_fail

        if args.source in ["lichess", "all"]:
            lichess_success, lichess_fail = download_lichess(
                args.limit,
                args.min_elo,
                args.output,
            )
            stats["lichess"]["successful"] = lichess_success
            stats["lichess"]["failed"] = lichess_fail

        if args.source in ["pgnmentor", "all"]:
            pgn_success, pgn_fail = download_pgnmentor(
                args.limit,
                args.output,
                args.collections,
            )
            stats["pgnmentor"]["successful"] = pgn_success
            stats["pgnmentor"]["failed"] = pgn_fail

        # Calculate totals
        stats["total"]["successful"] = (
            stats["wikipedia"]["successful"] +
            stats["lichess"]["successful"] +
            stats["pgnmentor"]["successful"]
        )
        stats["total"]["failed"] = (
            stats["wikipedia"]["failed"] +
            stats["lichess"]["failed"] +
            stats["pgnmentor"]["failed"]
        )

        # Display summary
        display_summary(stats)

        # Generate manifest
        generate_manifest(args.output, stats)

        # Exit status
        if stats["total"]["successful"] == 0:
            console.print("\n[red]✗ No documents downloaded[/red]")
            sys.exit(1)
        else:
            console.print(f"\n[green]✓ Successfully downloaded {stats['total']['successful']} documents[/green]")
            sys.exit(0)

    except KeyboardInterrupt:
        console.print("\n\n[yellow]Download interrupted by user[/yellow]")
        sys.exit(1)
    except Exception as e:
        console.print(f"\n[red]✗ Error: {e}[/red]")
        logger.exception("Download failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
