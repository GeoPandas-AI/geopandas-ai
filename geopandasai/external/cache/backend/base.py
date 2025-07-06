import abc
from typing import Optional


class ACacheBackend(abc.ABC):
    """
    Base class for all cache backends.
    This class defines the interface for caching mechanisms used in GeoPandasAI.
    It provides methods to get, set, clear, and reset cache entries.
    Subclasses should implement these methods to provide specific caching functionality.
    """

    def get_cache(self, key: str) -> Optional[bytes]:
        """
        Get the cached result for the given key.
        """
        pass

    def set_cache(self, key: str, value: bytes) -> None:
        """
        Set the cached result for the given key.
        """
        pass

    def clear_cache(self, key: str) -> None:
        """
        Clear the cached result for the given key.
        """
        pass

    def reset_cache(self) -> None:
        """
        Reset the cache.
        """
        pass
