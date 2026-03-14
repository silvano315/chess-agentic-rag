import time
from collections.abc import Iterator
from datetime import datetime
from urllib.parse import quote

import requests
from bs4 import BeautifulSoup
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential

from chess_agentic_rag.core.exceptions import LoaderError
from chess_agentic_rag.core.models import Document, DocumentType
from chess_agentic_rag.data.loaders.base import BaseLoader


class WikipediaLoader(BaseLoader):
    """
    Loader for Wikipedia chess articles.

    Downloads articles from Wikipedia, extracts clean text content,
    and creates Document objects with appropriate metadata.

    Attributes:
        titles: List of Wikipedia article titles to load
        language: Wikipedia language code (default: 'en')
        base_url: Wikipedia API base URL
        rate_limit_delay: Seconds to wait between requests

    Example:
        >>> loader = WikipediaLoader(
        ...     titles=["Sicilian_Defence", "Magnus_Carlsen"]
        ... )
        >>> for doc in loader.load():
        ...     print(f"Loaded: {doc.metadata['title']}")
    """

    def __init__(
        self,
        titles: list[str] | None = None,
        language: str = "en",
        rate_limit_delay: float = 1.0,
        user_agent: str | None = None,
    ) -> None:
        """
        Initialize Wikipedia loader.

        Args:
            titles: List of article titles. If None, uses default chess articles.
            language: Wikipedia language code (e.g., 'en', 'it')
            rate_limit_delay: Seconds to wait between requests (default: 1.0)
        """
        self.titles = titles or self._get_default_titles()
        self.language = language
        self.base_url = f"https://{language}.wikipedia.org"
        self.rate_limit_delay = rate_limit_delay
        # Respect Wikimedia's requirement to set a descriptive User-Agent
        # (see https://w.wiki/4wJS). Allow override for tests or deploy.
        self.user_agent = (
            user_agent
            or "chess-agentic-rag/0.1 (+https://example.com/contact)"
        )

        logger.info(
            "Wikipedia loader initialized",
            num_titles=len(self.titles),
            language=language,
        )

    @property
    def source_name(self) -> str:
        """Return source identifier."""
        return "wikipedia"

    def validate(self) -> bool:
        """
        Validate Wikipedia is accessible.

        Returns:
            True if Wikipedia API responds, False otherwise
        """
        try:
            response = requests.get(
                f"{self.base_url}/api/rest_v1/",
                headers=self._default_headers(),
                timeout=10,
            )
            is_valid = response.status_code == 200

            if is_valid:
                logger.info("Wikipedia validation passed")
            else:
                logger.warning(
                    "Wikipedia validation failed",
                    status_code=response.status_code,
                )

            return is_valid

        except Exception as e:
            logger.error("Wikipedia validation error", error=str(e))
            return False

    def load(self) -> Iterator[Document]:
        """
        Load Wikipedia articles.

        Yields:
            Document objects for each article

        Raises:
            LoaderError: If loading fails for all articles
        """
        successful = 0
        failed = 0
        exception_count = 0

        for title in self.titles:
            try:
                doc = self._load_article(title)
                if doc:
                    yield doc
                    successful += 1
                else:
                    failed += 1

                # Rate limiting
                time.sleep(self.rate_limit_delay)

            except Exception as e:
                logger.error(
                    "Failed to load article",
                    title=title,
                    error=str(e),
                )
                failed += 1
                exception_count += 1
                continue

        logger.info(
            "Wikipedia loading complete",
            successful=successful,
            failed=failed,
        )

        # Only raise a LoaderError when all attempts failed due to
        # exceptions (e.g. network errors). If articles are simply
        # missing (404) we treat that as non-exceptional and return
        # zero results so callers/tests can handle it.
        if successful == 0 and exception_count > 0:
            raise LoaderError("Failed to load any Wikipedia articles")

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
    )
    def _load_article(self, title: str) -> Document | None:
        """
        Load a single Wikipedia article.

        Args:
            title: Article title (e.g., 'Sicilian_Defence')

        Returns:
            Document object or None if article not found
        """
        logger.debug("Loading article", title=title)

        # Fetch article HTML
        url = f"{self.base_url}/wiki/{quote(title)}"
        response = requests.get(url, timeout=30, headers=self._default_headers())

        if response.status_code == 404:
            logger.warning("Article not found", title=title)
            return None

        response.raise_for_status()

        # Parse HTML
        soup = BeautifulSoup(response.content, "html.parser")

        # Extract content
        content = self._extract_content(soup)

        if not content or len(content) < 100:
            logger.warning("Article content too short", title=title)
            return None

        # Extract metadata
        metadata = self._extract_metadata(soup, title, url)

        # Create document
        doc = Document(
            id=f"wikipedia_{title}",
            content=content,
            doc_type=DocumentType.ARTICLE,
            metadata=metadata,
            created_at=datetime.now(),
        )

        logger.debug(
            "Article loaded",
            title=title,
            content_length=len(content),
        )

        return doc

    def _extract_content(self, soup: BeautifulSoup) -> str:
        """
        Extract clean text content from Wikipedia HTML.

        Args:
            soup: BeautifulSoup parsed HTML

        Returns:
            Clean text content
        """
        # Find main content div
        content_div = soup.find("div", {"id": "mw-content-text"})
        if not content_div:
            return ""

        # Remove unwanted elements
        for element in content_div.find_all([
            "table",  # Remove tables
            "div",    # Remove infoboxes and side content
            "sup",    # Remove citation numbers
            "span",   # Remove edit links
        ], {"class": [
            "infobox",
            "navbox",
            "vertical-navbox",
            "mw-editsection",
            "reference",
        ]}):
            element.decompose()

        # Get all paragraphs
        paragraphs = content_div.find_all("p")
        text_parts = []

        for p in paragraphs:
            text = p.get_text(strip=True)
            if text and len(text) > 50:  # Skip very short paragraphs
                text_parts.append(text)

        return "\n\n".join(text_parts)

    def _extract_metadata(
        self,
        soup: BeautifulSoup,
        title: str,
        url: str,
    ) -> dict[str, str | list[str]]:
        """
        Extract metadata from Wikipedia article.

        Args:
            soup: BeautifulSoup parsed HTML
            title: Article title
            url: Article URL

        Returns:
            Metadata dictionary
        """
        metadata: dict[str, str | list[str]] = {
            "source": "wikipedia",
            "title": title.replace("_", " "),
            "url": url,
            "language": self.language,
        }

        # Extract categories
        categories = []
        for cat_link in soup.find_all("a", href=lambda x: x and "/wiki/Category:" in x):
            cat_text = cat_link.get_text(strip=True)
            if cat_text:
                categories.append(cat_text)

        if categories:
            metadata["categories"] = categories[:10]  # Limit to 10

        # Infer topic from title
        title_lower = title.lower()
        if any(term in title_lower for term in ["opening", "defence", "defense", "gambit", "variation"]):
            metadata["topic"] = "opening"
        elif any(term in title_lower for term in ["championship", "tournament", "match"]):
            metadata["topic"] = "tournament"
        elif "tactic" in title_lower or "strategy" in title_lower:
            metadata["topic"] = "strategy"
        else:
            metadata["topic"] = "general"

        return metadata

    def get_metadata(self) -> dict[str, str | int]:
        """Get loader metadata."""
        return {
            **super().get_metadata(),
            "num_articles": len(self.titles),
            "language": self.language,
        }

    def _default_headers(self) -> dict[str, str]:
        """Return default headers for requests, including User-Agent.

        Returns:
            Headers dict to pass to ``requests`` calls.
        """

        return {
            "User-Agent": self.user_agent,
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        }

    @staticmethod
    def _get_default_titles() -> list[str]:
        """
        Get default list of important chess articles.

        Returns:
            List of article titles
        """
        return [
            # Openings
            "Sicilian_Defence",
            "Ruy_Lopez",
            "Queen's_Gambit",
            "Italian_Game",
            "French_Defence",
            "Caro-Kann_Defence",
            "King's_Indian_Defence",
            "Nimzo-Indian_Defence",
            "English_Opening",
            "Catalan_Opening",

            # Players
            "Magnus_Carlsen",
            "Garry_Kasparov",
            "Bobby_Fischer",
            "Anatoly_Karpov",
            "Mikhail_Tal",
            "José_Raúl_Capablanca",
            "Emanuel_Lasker",
            "Vishwanathan_Anand",
            "Vladimir_Kramnik",
            "Fabiano_Caruana",

            # Tournaments
            "World_Chess_Championship",
            "Candidates_Tournament",
            "Chess_Olympiad",
            "Tata_Steel_Chess_Tournament",

            # Theory
            "Chess_tactics",
            "Chess_strategy",
            "Chess_endgame",
            "Chess_opening",
            "Checkmate",
            "Castling",
        ]
