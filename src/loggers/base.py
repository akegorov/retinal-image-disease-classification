"""Logger interface. Use as a context manager so the experiment is always closed."""

import warnings
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, ClassVar, Self

from matplotlib.figure import Figure


class Logger(ABC):
    """Logs params, metrics, texts, figures and model files of one experiment."""

    _warned: ClassVar[set[tuple[str, str]]] = set()

    @abstractmethod
    def log_params(self, params: dict[str, Any]) -> None: ...

    @abstractmethod
    def log_metrics(
        self,
        metrics: dict[str, float],
        step: int | None = None,
        epoch: int | None = None,
    ) -> None: ...

    @abstractmethod
    def finish(self) -> None: ...

    def log_text(self, name: str, text: str) -> None:
        self._unsupported("log_text")

    def log_figure(self, name: str, fig: Figure, step: int | None = None) -> None:
        self._unsupported("log_figure")

    def log_model(self, name: str, path: str | Path) -> None:
        self._unsupported("log_model")

    def _unsupported(self, method: str) -> None:
        """Warn once per logger class and method; stacklevel points at the caller."""
        key = (type(self).__name__, method)
        if key not in Logger._warned:
            Logger._warned.add(key)
            warnings.warn(f"{key[0]} does not support {method}, ignored", stacklevel=3)

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.finish()
