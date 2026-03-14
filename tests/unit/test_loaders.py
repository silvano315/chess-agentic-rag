from datetime import datetime
from unittest.mock import MagicMock, Mock, patch

import pytest
import requests

from chess_agentic_rag.core.exceptions import LoaderError
from chess_agentic_rag.core.models import Document, DocumentType
from chess_agentic_rag.data.loaders import BaseLoader, LichessLoader, WikipediaLoader


class TestBaseLoader:
    """Test BaseLoader abstract class."""

    def test_cannot_instantiate_base_loader(self) -> None:
        """Test that BaseLoader cannot be instantiated directly."""
        with pytest.raises(TypeError):
            BaseLoader()

    def test_concrete_loader_must_implement_methods(self) -> None:
        """Test that concrete loaders must implement required methods."""

        class IncompleteLoader(BaseLoader):
            pass

        with pytest.raises(TypeError):
            IncompleteLoader()

    def test_valid_concrete_loader(self) -> None:
        """Test that a properly implemented loader works."""

        class ValidLoader(BaseLoader):
            @property
            def source_name(self) -> str:
                return "test"

            def load(self):
                yield Document(
                    id="test_1",
                    content="test content",
                    doc_type=DocumentType.ARTICLE,
                    metadata={"source": "test"},
                    created_at=datetime.now(),
                )

            def validate(self) -> bool:
                return True

        loader = ValidLoader()
        assert loader.source_name == "test"
        assert loader.validate() is True

        docs = list(loader.load())
        assert len(docs) == 1
        assert docs[0].id == "test_1"


class TestWikipediaLoader:
    """Test WikipediaLoader implementation."""

    def test_initialization_with_defaults(self) -> None:
        """Test loader initializes with default values."""
        loader = WikipediaLoader()

        assert loader.source_name == "wikipedia"
        assert loader.language == "en"
        assert len(loader.titles) > 0
        assert "Sicilian_Defence" in loader.titles

    def test_initialization_with_custom_titles(self) -> None:
        """Test loader initializes with custom titles."""
        custom_titles = ["Chess", "Checkmate"]
        loader = WikipediaLoader(titles=custom_titles)

        assert loader.titles == custom_titles

    @patch("chess_agentic_rag.data.loaders.wikipedia_loader.requests.get")
    def test_validate_success(self, mock_get: Mock) -> None:
        """Test successful validation."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_get.return_value = mock_response

        loader = WikipediaLoader()
        assert loader.validate() is True

    @patch("chess_agentic_rag.data.loaders.wikipedia_loader.requests.get")
    def test_validate_failure(self, mock_get: Mock) -> None:
        """Test validation failure."""
        mock_response = Mock()
        mock_response.status_code = 500
        mock_get.return_value = mock_response

        loader = WikipediaLoader()
        assert loader.validate() is False

    @patch("chess_agentic_rag.data.loaders.wikipedia_loader.requests.get")
    def test_validate_network_error(self, mock_get: Mock) -> None:
        """Test validation with network error."""
        mock_get.side_effect = requests.RequestException("Network error")

        loader = WikipediaLoader()
        assert loader.validate() is False

    @patch("chess_agentic_rag.data.loaders.wikipedia_loader.requests.get")
    @patch("chess_agentic_rag.data.loaders.wikipedia_loader.BeautifulSoup")
    def test_load_article_success(self, mock_soup: Mock, mock_get: Mock) -> None:
        """Test successful article loading."""
        # Mock HTML response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.content = b"<html>test</html>"
        mock_get.return_value = mock_response

        # Mock BeautifulSoup parsing
        mock_parsed = MagicMock()

        # Mock content div
        content_div = MagicMock()
        mock_paragraph = MagicMock()
        mock_paragraph.get_text.return_value = "Test paragraph content " * 10
        content_div.find_all.return_value = [mock_paragraph]

        mock_parsed.find.return_value = content_div
        mock_soup.return_value = mock_parsed

        # Load single article
        loader = WikipediaLoader(titles=["Test_Article"])
        docs = list(loader.load())

        assert len(docs) == 1
        assert docs[0].doc_type == DocumentType.ARTICLE
        assert "wikipedia" in docs[0].id
        assert docs[0].metadata["source"] == "wikipedia"

    @patch("chess_agentic_rag.data.loaders.wikipedia_loader.requests.get")
    def test_load_article_not_found(self, mock_get: Mock) -> None:
        """Test handling of 404 errors."""
        mock_response = Mock()
        mock_response.status_code = 404
        mock_get.return_value = mock_response

        loader = WikipediaLoader(titles=["NonExistent_Article"])
        docs = list(loader.load())

        assert len(docs) == 0

    @patch("chess_agentic_rag.data.loaders.wikipedia_loader.requests.get")
    def test_load_raises_error_if_all_fail(self, mock_get: Mock) -> None:
        """Test that LoaderError is raised if all articles fail."""
        mock_get.side_effect = requests.RequestException("Network error")

        loader = WikipediaLoader(titles=["Article1"])

        with pytest.raises(LoaderError):
            list(loader.load())

    def test_get_metadata(self) -> None:
        """Test metadata retrieval."""
        loader = WikipediaLoader(titles=["Test"])
        metadata = loader.get_metadata()

        assert metadata["source_name"] == "wikipedia"
        assert metadata["num_articles"] == 1
        assert metadata["language"] == "en"

    @patch("chess_agentic_rag.data.loaders.wikipedia_loader.time.sleep")
    @patch("chess_agentic_rag.data.loaders.wikipedia_loader.requests.get")
    def test_rate_limiting(self, mock_get: Mock, mock_sleep: Mock) -> None:
        """Test that rate limiting delays are applied."""
        mock_response = Mock()
        mock_response.status_code = 404
        mock_get.return_value = mock_response

        loader = WikipediaLoader(titles=["Article1", "Article2"])
        list(loader.load())

        # Should sleep between articles
        assert mock_sleep.call_count >= 1


class TestLichessLoader:
    """Test LichessLoader implementation."""

    def test_initialization_with_defaults(self) -> None:
        """Test loader initializes with default values."""
        loader = LichessLoader()

        assert loader.source_name == "lichess"
        assert loader.min_elo == 2500
        assert loader.max_games == 500
        assert len(loader.players) > 0

    def test_initialization_with_custom_values(self) -> None:
        """Test loader with custom configuration."""
        loader = LichessLoader(
            min_elo=2600,
            max_games=100,
            players=["player1", "player2"],
        )

        assert loader.min_elo == 2600
        assert loader.max_games == 100
        assert loader.players == ["player1", "player2"]

    @patch("chess_agentic_rag.data.loaders.lichess_loader.requests.get")
    def test_validate_success(self, mock_get: Mock) -> None:
        """Test successful validation."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_get.return_value = mock_response

        loader = LichessLoader()
        assert loader.validate() is True

    @patch("chess_agentic_rag.data.loaders.lichess_loader.requests.get")
    def test_validate_failure(self, mock_get: Mock) -> None:
        """Test validation failure."""
        mock_response = Mock()
        mock_response.status_code = 500
        mock_get.return_value = mock_response

        loader = LichessLoader()
        assert loader.validate() is False

    @patch("chess_agentic_rag.data.loaders.lichess_loader.requests.get")
    def test_load_games_success(self, mock_get: Mock) -> None:
        """Test successful game loading."""
        # Mock API response
        game_json = {
            "id": "test123",
            "players": {
                "white": {"user": {"name": "Player1"}, "rating": 2700},
                "black": {"user": {"name": "Player2"}, "rating": 2650},
            },
            "status": "mate",
            "winner": "white",
            "opening": {"name": "Sicilian Defense", "eco": "B20"},
            "createdAt": 1700000000000,
            "speed": "blitz",
        }

        import json
        ndjson_data = json.dumps(game_json).encode()

        mock_response = Mock()
        mock_response.iter_lines.return_value = [ndjson_data]
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response

        loader = LichessLoader(max_games=1, players=["test_player"])
        docs = list(loader.load())

        assert len(docs) == 1
        assert docs[0].doc_type == DocumentType.PGN
        assert docs[0].metadata["source"] == "lichess"
        assert docs[0].metadata["white"] == "Player1"
        assert docs[0].metadata["black"] == "Player2"

    @patch("chess_agentic_rag.data.loaders.lichess_loader.requests.get")
    def test_load_filters_by_elo(self, mock_get: Mock) -> None:
        """Test that games below min_elo are filtered out."""
        # Mock two games: one above, one below min_elo
        game_high_elo = {
            "id": "game1",
            "players": {
                "white": {"user": {"name": "GM1"}, "rating": 2700},
                "black": {"user": {"name": "GM2"}, "rating": 2650},
            },
            "status": "mate",
            "winner": "white",
            "opening": {"name": "Test", "eco": "B00"},
            "createdAt": 1700000000000,
        }

        game_low_elo = {
            "id": "game2",
            "players": {
                "white": {"user": {"name": "Player1"}, "rating": 1500},
                "black": {"user": {"name": "Player2"}, "rating": 1450},
            },
            "status": "mate",
            "winner": "white",
            "opening": {"name": "Test", "eco": "B00"},
            "createdAt": 1700000000000,
        }

        import json
        ndjson_data = (
            json.dumps(game_high_elo).encode() + b"\n" +
            json.dumps(game_low_elo).encode()
        )

        mock_response = Mock()
        mock_response.iter_lines.return_value = ndjson_data.split(b"\n")
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response

        loader = LichessLoader(min_elo=2500, max_games=10, players=["test"])
        docs = list(loader.load())

        # Should only get the high ELO game
        assert len(docs) == 1
        assert docs[0].metadata["game_id"] == "game1"

    def test_format_result(self) -> None:
        """Test result formatting."""
        loader = LichessLoader()

        assert loader._format_result("white") == "1-0"
        assert loader._format_result("black") == "0-1"
        assert loader._format_result("draw") == "1/2-1/2"

    def test_get_metadata(self) -> None:
        """Test metadata retrieval."""
        loader = LichessLoader(min_elo=2600, max_games=100)
        metadata = loader.get_metadata()

        assert metadata["source_name"] == "lichess"
        assert metadata["min_elo"] == 2600
        assert metadata["max_games"] == 100


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
