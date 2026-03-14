from abc import ABC, abstractmethod
from collections.abc import Iterator

from chess_agentic_rag.core.models import Document


class BaseLoader(ABC):
    """
    Abstract base class for all data loaders.

    All loaders must inherit from this class and implement the required methods.
    This ensures a consistent interface across different data sources.

    Attributes:
        source_name: Identifier for this loader (e.g., 'wikipedia', 'lichess')
    """

    @property
    @abstractmethod
    def source_name(self) -> str:
        """
        Get the name of this data source.

        Returns:
            Source identifier string (e.g., 'wikipedia', 'lichess')
        """
        pass

    @abstractmethod
    def load(self) -> Iterator[Document]:
        """
        Load documents from the data source.

        This method should yield Document objects one at a time rather than
        loading everything into memory at once.

        Yields:
            Document objects from this source

        Raises:
            LoaderError: If loading fails
        """
        pass

    @abstractmethod
    def validate(self) -> bool:
        """
        Validate that the data source is accessible.

        This method should check if the source can be reached (e.g., API is up,
        files exist, network connection works) before attempting to load.

        Returns:
            True if source is accessible, False otherwise
        """
        pass

    def get_metadata(self) -> dict[str, str | int | bool]:
        """
        Get metadata about this loader.

        Override this method to provide loader-specific metadata such as
        configuration, statistics, or status information.

        Returns:
            Dictionary of loader metadata
        """
        return {
            "source_name": self.source_name,
            "loader_type": self.__class__.__name__,
        }

    def __repr__(self) -> str:
        """String representation of loader."""
        return f"{self.__class__.__name__}(source='{self.source_name}')"
