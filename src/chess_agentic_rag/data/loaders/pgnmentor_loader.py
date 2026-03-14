import io
import time
import zipfile
from collections.abc import Iterator
from datetime import datetime

import chess.pgn
import requests
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential

from chess_agentic_rag.core.config import settings
from chess_agentic_rag.core.exceptions import LoaderError
from chess_agentic_rag.core.models import Document, DocumentType
from chess_agentic_rag.data.loaders.base import BaseLoader


class PGNMentorLoader(BaseLoader):
    """
    Loader for PGNMentor historical game collections.

    Downloads and parses curated PGN collections from PGNMentor.com,
    including games by world champions and famous tournaments.

    Attributes:
        collections: List of collections to download (player names or events)
        collection_type: Type of collection ('players' or 'events')
        max_games_per_collection: Maximum games to load per collection
        base_url: PGNMentor base URL

    Example:
        >>> loader = PGNMentorLoader(
        ...     collections=["Fischer", "Kasparov"],
        ...     collection_type="players"
        ... )
        >>> for game_doc in loader.load():
        ...     print(f"{game_doc.metadata['white']} vs {game_doc.metadata['black']}")
    """

    def __init__(
        self,
        collections: list[str] | None = None,
        collection_type: str = "players",
        max_games_per_collection: int | None = None,
        rate_limit_delay: float = 2.0,
        user_agent: str | None = None,
    ) -> None:
        """
        Initialize PGNMentor loader.

        Args:
            collections: List of collection names (player names or event names).
                If None, uses default famous players.
            collection_type: Type of collection - 'players' or 'events'
            max_games_per_collection: Max games per collection (None = unlimited)
            rate_limit_delay: Seconds between downloads (default: 2.0)
        """
        self.collections = collections or self._get_default_collections()
        self.collection_type = collection_type
        self.max_games_per_collection = max_games_per_collection
        self.rate_limit_delay = rate_limit_delay
        self.base_url = "https://www.pgnmentor.com"
        # Allow setting a descriptive User-Agent; some sites block default
        # python requests UA strings and return access-denied HTML.
        self.user_agent = (
            user_agent or "chess-agentic-rag/0.1 (+https://example.com/contact)"
        )

        logger.info(
            "PGNMentor loader initialized",
            num_collections=len(self.collections),
            collection_type=collection_type,
        )

    @property
    def source_name(self) -> str:
        """Return source identifier."""
        return "pgnmentor"

    def validate(self) -> bool:
        """
        Validate PGNMentor is accessible.

        Returns:
            True if site responds, False otherwise
        """
        try:
            response = requests.get(
                self.base_url,
                headers=self._default_headers(),
                timeout=10,
            )
            is_valid = response.status_code == 200

            if is_valid:
                logger.info("PGNMentor validation passed")
            else:
                logger.warning(
                    "PGNMentor validation failed",
                    status_code=response.status_code,
                )

            return is_valid

        except Exception as e:
            logger.error("PGNMentor validation error", error=str(e))
            return False

    def _default_headers(self) -> dict[str, str]:
        """Return default headers for requests to PGNMentor."""

        return {
            "User-Agent": self.user_agent,
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Referer": "https://www.google.com/",
        }

    def load(self) -> Iterator[Document]:
        """
        Load games from PGNMentor collections.

        Yields:
            Document objects for each game

        Raises:
            LoaderError: If loading fails for all collections
        """
        total_loaded = 0
        failed_collections = 0

        for collection_name in self.collections:
            try:
                games_loaded = 0

                for game_doc in self._load_collection(collection_name):
                    yield game_doc
                    total_loaded += 1
                    games_loaded += 1

                    if (self.max_games_per_collection and
                        games_loaded >= self.max_games_per_collection):
                        break

                logger.info(
                    "Loaded collection",
                    collection=collection_name,
                    games=games_loaded,
                )

                # Rate limiting between collections
                time.sleep(self.rate_limit_delay)

            except Exception as e:
                logger.error(
                    "Failed to load collection",
                    collection=collection_name,
                    error=str(e),
                )
                failed_collections += 1
                continue

        logger.info(
            "PGNMentor loading complete",
            total_loaded=total_loaded,
            failed_collections=failed_collections,
        )

        if total_loaded == 0:
            raise LoaderError("Failed to load any games from PGNMentor")

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
    )
    def _load_collection(self, collection_name: str) -> Iterator[Document]:
        """
        Load games from a single collection (player or event).

        Args:
            collection_name: Name of collection (e.g., 'Fischer', 'WorldCh1972')

        Yields:
            Document objects for games in collection
        """
        logger.debug("Loading collection", collection=collection_name)

        # Build download URL
        # PGNMentor structure: /players/Fischer.zip or /events/WorldCh1972.zip
        url = f"{self.base_url}/{self.collection_type}/{collection_name}.zip"

        # Download ZIP file
        logger.debug("Downloading ZIP", url=url)
        response = requests.get(url, timeout=60, headers=self._default_headers(), allow_redirects=True)

        # Some hosting/CDN setups return non-standard status codes (e.g., 465)
        # or an HTML page with "Access Denied" when requests are blocked.
        if response.status_code != 200:
            text_snippet = response.text[:512].lower() if response.text else ""
            if "access denied" in text_snippet or response.status_code == 465:
                logger.error(
                    "PGNMentor access denied",
                    status_code=response.status_code,
                    url=url,
                )
                raise LoaderError(
                    "Access denied by PGNMentor (status: %s). "
                    "Try setting a different User-Agent, using a proxy, or "
                    "checking robots policy." % response.status_code
                )

            if response.status_code == 404:
                logger.warning("Collection not found", collection=collection_name, url=url)
                return

            # For other non-200 codes, raise to trigger retry/backoff
            response.raise_for_status()

        if response.status_code == 404:
            logger.warning("Collection not found", collection=collection_name, url=url)
            return

        response.raise_for_status()

        # Parse ZIP file
        try:
            with zipfile.ZipFile(io.BytesIO(response.content)) as zf:
                # Find PGN files in ZIP
                pgn_files = [f for f in zf.namelist() if f.endswith('.pgn')]

                if not pgn_files:
                    logger.warning("No PGN files in ZIP", collection=collection_name)
                    return

                logger.debug(
                    "Found PGN files in ZIP",
                    collection=collection_name,
                    num_files=len(pgn_files),
                )

                # Parse each PGN file
                for pgn_filename in pgn_files:
                    with zf.open(pgn_filename) as pgn_file:
                        # Decode bytes to string
                        pgn_content = pgn_file.read().decode('utf-8', errors='ignore')

                        # Parse games from PGN
                        yield from self._parse_pgn_content(
                            pgn_content,
                            collection_name,
                            pgn_filename,
                        )

        except zipfile.BadZipFile as e:
            logger.error("Invalid ZIP file", collection=collection_name, error=str(e))
            raise

    def _parse_pgn_content(
        self,
        pgn_content: str,
        collection_name: str,
        pgn_filename: str,
    ) -> Iterator[Document]:
        """
        Parse PGN content and yield Document objects.

        Args:
            pgn_content: PGN file content as string
            collection_name: Name of collection
            pgn_filename: Name of PGN file

        Yields:
            Document objects for each game
        """
        pgn_io = io.StringIO(pgn_content)
        game_count = 0

        while True:
            try:
                game = chess.pgn.read_game(pgn_io)

                if game is None:
                    # End of file
                    break

                # Convert game to Document
                doc = self._game_to_document(game, collection_name, pgn_filename)

                if doc:
                    yield doc
                    game_count += 1

            except Exception as e:
                logger.warning(
                    "Failed to parse game",
                    collection=collection_name,
                    file=pgn_filename,
                    error=str(e),
                )
                continue

        logger.debug(
            "Parsed PGN file",
            collection=collection_name,
            file=pgn_filename,
            games=game_count,
        )

    def _game_to_document(
        self,
        game: chess.pgn.Game,
        collection_name: str,
        pgn_filename: str,
    ) -> Document | None:
        """
        Convert chess.pgn.Game to Document.

        Args:
            game: Parsed chess game
            collection_name: Collection name
            pgn_filename: Source PGN filename

        Returns:
            Document object or None if invalid
        """
        try:
            headers = game.headers

            # Extract basic info
            white = headers.get("White", "Unknown")
            black = headers.get("Black", "Unknown")
            result = headers.get("Result", "*")
            event = headers.get("Event", "Unknown")
            site = headers.get("Site", "Unknown")
            date = headers.get("Date", "????.??.??")
            eco = headers.get("ECO", "")
            opening = headers.get("Opening", "")

            # Parse ELO if available
            white_elo = self._parse_elo(headers.get("WhiteElo"))
            black_elo = self._parse_elo(headers.get("BlackElo"))

            # Extract year
            try:
                year = int(date.split('.')[0]) if date != "????.??.??" else None
            except (ValueError, IndexError):
                year = None

            # Get moves
            moves = []
            board = game.board()
            for move in game.mainline_moves():
                moves.append(board.san(move))
                board.push(move)

            # Extract comments/annotations
            annotations = self._extract_annotations(game)

            # Generate full PGN string (for game viewer)
            exporter = chess.pgn.StringExporter(headers=True, variations=True, comments=True)

            full_pgn = game.accept(exporter)

            # Generate unique ID
            game_id = f"{white}_{black}_{date}".replace(" ", "_").replace(".", "")

            # Build metadata
            metadata = {
                "source": "pgnmentor",
                "collection": collection_name,
                "collection_type": self.collection_type,
                "pgn_file": pgn_filename,
                "game_id": game_id,
                "white": white,
                "black": black,
                "elo_white": white_elo,
                "elo_black": black_elo,
                "opening": opening,
                "eco": eco,
                "result": result,
                "event": event,
                "site": site,
                "date": date,
                "year": year,
                "move_count": len(moves),
                "has_annotations": len(annotations) > 0,
                "pgn": full_pgn,
            }

            # Build content
            content_parts = [
                f"{white} ({white_elo or '?'}) vs {black} ({black_elo or '?'})",
                f"Event: {event}",
                f"Site: {site}, Date: {date}",
                f"Opening: {opening} ({eco})" if opening else f"ECO: {eco}",
                f"Result: {result}",
                f"Moves: {' '.join(moves[:20])}{'...' if len(moves) > 20 else ''}",
            ]

            if annotations:
                content_parts.append(f"Annotations: {len(annotations)} comments")

            content = "\n".join(content_parts)

            # Create document
            doc = Document(
                id=f"pgnmentor_{game_id}",
                content=content,
                doc_type=DocumentType.PGN,
                metadata=metadata,
                created_at=datetime.now(),
            )

            return doc

        except Exception as e:
            logger.warning("Failed to convert game to document", error=str(e))
            return None

    def _parse_elo(self, elo_str: str | None) -> int | None:
        """Parse ELO rating from string."""
        if not elo_str or elo_str == "-":
            return None
        try:
            return int(elo_str)
        except ValueError:
            return None

    def _extract_annotations(self, game: chess.pgn.Game) -> list[str]:
        """Extract comments and annotations from game."""
        annotations = []

        # Get comments from main line
        node = game
        while node.variations:
            node = node.variation(0)
            if node.comment:
                annotations.append(node.comment.strip())

        return annotations

    def get_metadata(self) -> dict[str, str | int]:
        """Get loader metadata."""
        return {
            **super().get_metadata(),
            "num_collections": len(self.collections),
            "collection_type": self.collection_type,
        }

    @staticmethod
    def _get_default_collections() -> list[str]:
        """
        Get default list of famous players.

        Returns:
            List of player names available on PGNMentor
        """
        return [
            # World Champions
            "Fischer",
            "Kasparov",
            "Karpov",
            "Tal",
            "Capablanca",
            "Alekhine",
            "Lasker",
            "Botvinnik",
            "Petrosian",
            "Spassky",
            "Carlsen",
            "Anand",
            "Kramnik",

            # Other legends
            "Morphy",
            "Steinitz",
            "Nimzowitsch",
            "Reti",
            "Rubinstein",
        ]
