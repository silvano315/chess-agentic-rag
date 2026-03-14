import time
from collections.abc import Iterator
from datetime import datetime

import requests
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential

from chess_agentic_rag.core.exceptions import LoaderError
from chess_agentic_rag.core.models import Document, DocumentType
from chess_agentic_rag.data.loaders.base import BaseLoader


class LichessLoader(BaseLoader):
    """
    Loader for Lichess chess games.

    Downloads PGN games from Lichess database using their API.
    Supports filtering by ELO rating, date range, and player names.

    Attributes:
        min_elo: Minimum ELO rating for games (default: 2500)
        max_games: Maximum number of games to download
        date_from: Start date for games (YYYY-MM-DD)
        date_to: End date for games (YYYY-MM-DD)
        players: Specific player usernames to download (optional)
        rate_limit_delay: Seconds to wait between API requests

    Example:
        >>> loader = LichessLoader(min_elo=2600, max_games=100)
        >>> for game_doc in loader.load():
        ...     print(f"{game_doc.metadata['white']} vs {game_doc.metadata['black']}")
    """

    def __init__(
        self,
        min_elo: int = 2500,
        max_games: int = 500,
        date_from: str | None = None,
        date_to: str | None = None,
        players: list[str] | None = None,
        rate_limit_delay: float = 1.0,
    ) -> None:
        """
        Initialize Lichess loader.

        Args:
            min_elo: Minimum average ELO rating
            max_games: Maximum games to download
            date_from: Start date (YYYY-MM-DD format)
            date_to: End date (YYYY-MM-DD format)
            players: List of Lichess usernames (optional)
            rate_limit_delay: Seconds between API requests
        """
        self.min_elo = min_elo
        self.max_games = max_games
        self.date_from = date_from
        self.date_to = date_to
        self.players = players or self._get_default_players()
        self.rate_limit_delay = rate_limit_delay
        self.base_url = "https://lichess.org/api"

        logger.info(
            "Lichess loader initialized",
            min_elo=min_elo,
            max_games=max_games,
            num_players=len(self.players),
        )

    @property
    def source_name(self) -> str:
        """Return source identifier."""
        return "lichess"

    def validate(self) -> bool:
        """
        Validate Lichess API is accessible.

        Returns:
            True if API responds, False otherwise
        """
        try:
            response = requests.get(
                f"{self.base_url}/user/{self.players[0]}",
                timeout=10,
            )
            is_valid = response.status_code in [200, 404]  # 404 is ok, means API works

            if is_valid:
                logger.info("Lichess API validation passed")
            else:
                logger.warning(
                    "Lichess API validation failed",
                    status_code=response.status_code,
                )

            return is_valid

        except Exception as e:
            logger.error("Lichess API validation error", error=str(e))
            return False

    def load(self) -> Iterator[Document]:
        """
        Load games from Lichess.

        Yields:
            Document objects for each game

        Raises:
            LoaderError: If loading fails completely
        """
        total_loaded = 0
        failed = 0

        for player in self.players:
            if total_loaded >= self.max_games:
                break

            try:
                games_loaded = 0

                for game_doc in self._load_player_games(player):
                    if total_loaded >= self.max_games:
                        break

                    yield game_doc
                    total_loaded += 1
                    games_loaded += 1

                logger.info(
                    "Loaded games for player",
                    player=player,
                    games=games_loaded,
                )

                # Rate limiting between players
                time.sleep(self.rate_limit_delay)

            except Exception as e:
                logger.error(
                    "Failed to load games for player",
                    player=player,
                    error=str(e),
                )
                failed += 1
                continue

        logger.info(
            "Lichess loading complete",
            total_loaded=total_loaded,
            failed_players=failed,
        )

        if total_loaded == 0:
            raise LoaderError("Failed to load any games from Lichess")

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
    )
    def _load_player_games(self, player: str) -> Iterator[Document]:
        """
        Load games for a specific player.

        Args:
            player: Lichess username

        Yields:
            Document objects for games
        """
        logger.debug("Loading games for player", player=player)

        # Build API request
        url = f"{self.base_url}/games/user/{player}"
        params = {
            "max": min(100, self.max_games),  # API limit per request
            "rated": "true",
            "perfType": "classical,rapid,blitz",
            "opening": "true",
            "moves": "false",  # We don't need full moves for metadata
            "tags": "true",
        }

        if self.date_from:
            params["since"] = self._date_to_timestamp(self.date_from)
        if self.date_to:
            params["until"] = self._date_to_timestamp(self.date_to)

        headers = {
            "Accept": "application/x-ndjson",  # Newline-delimited JSON
        }

        # Make request
        response = requests.get(url, params=params, headers=headers, timeout=60)
        response.raise_for_status()

        # Parse games
        games_parsed = 0

        for line in response.iter_lines():
            if not line:
                continue

            try:
                import json
                game_json = json.loads(line)

                # Filter by ELO
                white_elo = game_json.get("players", {}).get("white", {}).get("rating", 0)
                black_elo = game_json.get("players", {}).get("black", {}).get("rating", 0)
                avg_elo = (white_elo + black_elo) / 2

                if avg_elo < self.min_elo:
                    continue

                # Convert to Document
                doc = self._game_json_to_document(game_json)
                if doc:
                    yield doc
                    games_parsed += 1

            except Exception as e:
                logger.warning("Failed to parse game", error=str(e))
                continue

        logger.debug(
            "Player games loaded",
            player=player,
            games=games_parsed,
        )

    def _game_json_to_document(self, game_json: dict) -> Document | None:
        """
        Convert Lichess API JSON to Document.

        Args:
            game_json: Game data from Lichess API

        Returns:
            Document object or None if invalid
        """
        try:
            # Extract players
            white = game_json["players"]["white"]["user"]["name"]
            black = game_json["players"]["black"]["user"]["name"]
            white_elo = game_json["players"]["white"]["rating"]
            black_elo = game_json["players"]["black"]["rating"]

            # Extract game info
            game_id = game_json["id"]
            result = game_json["status"]
            winner = game_json.get("winner", "draw")

            # Extract opening
            opening = game_json.get("opening", {})
            opening_name = opening.get("name", "Unknown")
            eco = opening.get("eco", "")

            # Extract date
            created_at = datetime.fromtimestamp(game_json["createdAt"] / 1000)

            # Build metadata
            metadata = {
                "source": "lichess",
                "game_id": game_id,
                "white": white,
                "black": black,
                "elo_white": white_elo,
                "elo_black": black_elo,
                "opening": opening_name,
                "eco": eco,
                "result": self._format_result(winner),
                "year": created_at.year,
                "event": "Lichess",
                "speed": game_json.get("speed", "unknown"),
                "url": f"https://lichess.org/{game_id}",
            }

            # Create content summary
            content = f"{white} ({white_elo}) vs {black} ({black_elo})\n"
            content += f"Opening: {opening_name} ({eco})\n"
            content += f"Result: {metadata['result']}"

            # Create document
            doc = Document(
                id=f"lichess_{game_id}",
                content=content,
                doc_type=DocumentType.PGN,
                metadata=metadata,
                created_at=created_at,
            )

            return doc

        except KeyError as e:
            logger.warning("Missing required field in game JSON", field=str(e))
            return None

    def _format_result(self, winner: str) -> str:
        """Format game result."""
        if winner == "white":
            return "1-0"
        elif winner == "black":
            return "0-1"
        else:
            return "1/2-1/2"

    def _date_to_timestamp(self, date_str: str) -> int:
        """Convert YYYY-MM-DD to millisecond timestamp."""
        dt = datetime.strptime(date_str, "%Y-%m-%d")
        return int(dt.timestamp() * 1000)

    def get_metadata(self) -> dict[str, str | int]:
        """Get loader metadata."""
        return {
            **super().get_metadata(),
            "min_elo": self.min_elo,
            "max_games": self.max_games,
            "num_players": len(self.players),
        }

    @staticmethod
    def _get_default_players() -> list[str]:
        """
        Get default list of elite players.

        Returns:
            List of Lichess usernames
        """
        return [
            "DrNykterstein",  # Magnus Carlsen
            "FabianoCaruana",
            "Hikaru",
            "DanielNaroditsky",
            "GothamChess",
            "GMHess",
            "GMBenjaminFinegold",
            "GMWSO",
            "rpragchess",  # Praggnanandhaa
            "chesswarrior7197",  # Alireza Firouzja
        ]
