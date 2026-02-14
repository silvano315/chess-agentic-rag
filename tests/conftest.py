import pytest
from loguru import logger


def pytest_configure(config: pytest.Config) -> None:
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers",
        "slow: marks tests as slow (deselect with '-m \"not slow\"')",
    )
    config.addinivalue_line(
        "markers",
        "integration: marks tests as integration tests",
    )
    config.addinivalue_line(
        "markers",
        "unit: marks tests as unit tests",
    )
    config.addinivalue_line(
        "markers",
        "requires_ollama: marks tests that require Ollama to be running",
    )


@pytest.fixture(scope="session", autouse=True)
def setup_logging() -> None:
    """Configure logging for tests."""
    logger.remove()  # Remove default handler
    logger.add(
        lambda msg: print(msg, end=""),  # Print to stdout
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
        level="INFO",
    )


@pytest.fixture
def sample_chess_text() -> str:
    """Sample chess text for testing."""
    return """
    The Sicilian Defense is a chess opening that begins with the moves:
    1. e4 c5

    This is the most popular choice of aggressive players with the black pieces.
    The Sicilian is the most popular and best-scoring response to White's first
    move 1.e4.
    """


@pytest.fixture
def sample_pgn() -> str:
    """Sample PGN game notation for testing."""
    return """
[Event "World Championship"]
[Site "New York, NY USA"]
[Date "1972.07.11"]
[Round "6"]
[White "Fischer, Robert J."]
[Black "Spassky, Boris V."]
[Result "1-0"]
[ECO "B88"]
[WhiteElo "2785"]
[BlackElo "2660"]

1. e4 c5 2. Nf3 d6 3. d4 cxd4 4. Nxd4 Nf6 5. Nc3 a6 6. Bc4 e6 7. Bb3 b5
8. O-O Be7 9. Qf3 Qc7 10. Qg3 O-O 11. Bh6 Ne8 12. Rad1 Bd7 13. Nxe6 fxe6
14. Bxe6+ Kh8 15. Bxd7 Nxd7 16. Qh4 Nef6 17. Rxd6 Qb7 18. Rfd1 Rad8
19. Qf4 Bf6 20. Rxd7 Nxd7 21. Qh6 Rde8 22. Qxf6 Rxf6 23. Rxd7 1-0
"""
