import abc
from typing import Iterable, Type

from ....shared.types import GeoOrDataFrame


class ACodeExecutor(abc.ABC):
    """
    Abstract base class for executing Python code.
    """

    @abc.abstractmethod
    def execute(self, code: str, return_type: Type, *dfs: Iterable[GeoOrDataFrame]):
        pass
