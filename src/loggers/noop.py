"""Logger that does nothing (debugging, tests)."""

from pathlib import Path
from typing import Any

import pandas as pd
from matplotlib.figure import Figure

from src.loggers.base import Logger


class NoOpLogger(Logger):
    """Ignores everything, without warnings."""

    def log_params(self, params: dict[str, Any]) -> None:
        pass

    def log_metrics(
        self,
        metrics: dict[str, float],
        step: int | None = None,
        epoch: int | None = None,
    ) -> None:
        pass

    def log_text(self, name: str, text: str) -> None:
        pass

    def log_table(self, name: str, table: pd.DataFrame) -> None:
        pass

    def log_figure(self, name: str, fig: Figure, step: int | None = None) -> None:
        pass

    def log_model(self, name: str, path: str | Path) -> None:
        pass

    def finish(self) -> None:
        pass
