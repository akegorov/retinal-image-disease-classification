"""Logger that prints to the console without breaking tqdm progress bars."""

from typing import Any

import pandas as pd
from tqdm.auto import tqdm

from src.loggers.base import Logger


class ConsoleLogger(Logger):
    """Prints params, metrics, texts and tables with tqdm.write; no figures or models.

    Params are collected and printed as one block before the next metrics or text.
    Per-epoch metrics are printed every ``print_every`` epochs; the latest skipped epoch
    is printed before any later non-epoch output, so the last epoch is always shown.
    """

    def __init__(self, print_every: int = 10) -> None:
        self.print_every = print_every
        self._params: dict[str, Any] = {}
        self._pending_epoch: str | None = None

    def _flush(self) -> None:
        if self._params:
            lines = [f"  {k}: {v}" for k, v in self._params.items()]
            tqdm.write("params:\n" + "\n".join(lines))
            self._params = {}
        if self._pending_epoch is not None:
            tqdm.write(self._pending_epoch)
            self._pending_epoch = None

    def log_params(self, params: dict[str, Any]) -> None:
        self._params |= params

    def log_metrics(
        self,
        metrics: dict[str, float],
        step: int | None = None,
        epoch: int | None = None,
    ) -> None:
        if epoch is not None:
            self._pending_epoch = None
        self._flush()
        line = " ".join(f"{k}={v:.4g}" for k, v in metrics.items())
        if epoch is None:
            tqdm.write(line)
        elif epoch % self.print_every == 0:
            tqdm.write(f"[epoch {epoch}] {line}")
        else:
            self._pending_epoch = f"[epoch {epoch}] {line}"

    def log_text(self, name: str, text: str) -> None:
        self._flush()
        tqdm.write(f"{name}:\n{text}")

    def log_table(self, name: str, table: pd.DataFrame) -> None:
        self._flush()
        tqdm.write(f"{name}:\n{table.to_string()}")

    def finish(self) -> None:
        self._flush()
