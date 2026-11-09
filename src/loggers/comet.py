"""Comet.ml logger.

comet_ml finds the API key itself: the COMET_API_KEY environment variable or
~/.comet.config written by `comet login`. The key is never passed in code.
"""

from pathlib import Path
from typing import Any

import pandas as pd
from matplotlib.figure import Figure

from src.loggers.base import Logger


class CometLogger(Logger):
    """One Comet experiment."""

    def __init__(
        self,
        project_name: str,
        experiment_name: str | None = None,
        tags: list[str] | None = None,
        workspace: str | None = None,
    ) -> None:
        import comet_ml

        self.experiment = comet_ml.Experiment(
            project_name=project_name, workspace=workspace
        )
        if experiment_name:
            self.experiment.set_name(experiment_name)
        if tags:
            self.experiment.add_tags(tags)

    def log_params(self, params: dict[str, Any]) -> None:
        self.experiment.log_parameters(params)

    def log_metrics(
        self,
        metrics: dict[str, float],
        step: int | None = None,
        epoch: int | None = None,
    ) -> None:
        self.experiment.log_metrics(metrics, step=step, epoch=epoch)

    def log_text(self, name: str, text: str) -> None:
        self.experiment.log_text(text, metadata={"name": name})

    def log_table(self, name: str, table: pd.DataFrame) -> None:
        self.experiment.log_table(f"{name}.csv", tabular_data=table)

    def log_figure(self, name: str, fig: Figure, step: int | None = None) -> None:
        self.experiment.log_figure(figure_name=name, figure=fig, step=step)

    def log_model(self, name: str, path: str | Path) -> None:
        self.experiment.log_model(name, str(path))

    def finish(self) -> None:
        self.experiment.end()
