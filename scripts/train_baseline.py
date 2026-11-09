"""Train and compare the baselines on split_v1, test metrics at patient level.

- prior: predicts the train frequencies of the patient labels;
- single_eye: linear head on 512-d eye embeddings with eye labels, eyes -> patient;
- paired: linear head on [left, right] 1024-d embeddings with patient labels.

Needs data/processed/embeddings/resnet18/ (scripts/extract_embeddings.py). Writes
outputs/baseline/metrics.csv, outputs/baseline/predictions/{model}_{val,test}.csv
(patient_id + one probability column per class) and checkpoints/baseline/{model}.pt
(nn.Linear state dicts).

Usage: uv run python scripts/train_baseline.py [--logger none|console|comet]
"""

import argparse
from contextlib import ExitStack
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from src.data.datasets import EYE_LABELS, EmbeddingDataset
from src.data.labels import LABELS, load_eyes
from src.data.splits import load_split
from src.inference.aggregate import eyes_to_patient
from src.loggers import CometLogger, ConsoleLogger, Logger, NoOpLogger
from src.metrics.classification import compute_metrics, find_thresholds
from src.training.trainer import fit, predict
from src.utils.device import get_device
from src.utils.environment import environment
from src.utils.paths import CHECKPOINTS_DIR, EMBEDDINGS_DIR, OUTPUTS_DIR
from src.utils.seed import make_generator, set_seed

SPLIT = "split_v1"
BACKBONE = "resnet18"
SPLITS = ["train", "val", "test"]
OUT_DIR = OUTPUTS_DIR / "baseline"
PREDICTIONS_DIR = OUT_DIR / "predictions"
METRICS = ["pr_auc", "roc_auc", "f1", "precision", "recall"]
COMET_WORKSPACE = "soulqrat"
COMET_PROJECT = "retinal-image-disease-classification"
Probs = dict[str, pd.DataFrame]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and compare baselines.")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--pos-weight",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="weight positives by #neg / #pos per class (from train)",
    )
    parser.add_argument(
        "--logger", choices=["none", "console", "comet"], default="console"
    )
    return parser.parse_args()


def make_loggers(kind: str, models: list[str]) -> dict[str, Logger]:
    """Logger per baseline: a separate Comet experiment each, or one shared console."""
    if kind == "comet":
        return {
            m: CometLogger(
                COMET_PROJECT,
                m,
                tags=["baseline", m],
                workspace=COMET_WORKSPACE,
            )
            for m in models
        }
    shared = ConsoleLogger() if kind == "console" else NoOpLogger()
    return dict.fromkeys(models, shared)


def checkpoint_path(name: str) -> Path:
    """Where the head of a baseline is saved."""
    return CHECKPOINTS_DIR / "baseline" / f"{name}.pt"


def data_params(eyes: pd.DataFrame, patients: pd.DataFrame) -> dict[str, object]:
    """Patients, eyes and class counts (patient and eye labels) per split."""
    params: dict[str, object] = {}
    for s in SPLITS:
        p, e = patients[patients["split"] == s], eyes[eyes["split"] == s]
        params[f"data/{s}/patients"] = len(p)
        params[f"data/{s}/eyes"] = len(e)
        params[f"data/{s}/class_counts"] = {
            k: int(v) for k, v in p[LABELS].sum().items()
        }
        params[f"data/{s}/eye_class_counts"] = {
            k: int(v) for k, v in e[EYE_LABELS].sum().items()
        }
    return params


def load_embeddings(eyes: pd.DataFrame) -> np.ndarray:
    """Embeddings aligned to the rows of eyes, joined on (patient_id, eye)."""
    folder = EMBEDDINGS_DIR / BACKBONE
    if not (folder / "embeddings.npy").exists() or not (folder / "index.csv").exists():
        raise SystemExit(f"{folder} not found: run scripts/extract_embeddings.py first")
    embeddings = np.load(folder / "embeddings.npy")
    index = pd.read_csv(folder / "index.csv")[["patient_id", "eye"]]
    index["row"] = np.arange(len(index))
    rows = eyes[["patient_id", "eye"]].merge(index, how="left", validate="1:1")["row"]
    if rows.isna().any():
        raise SystemExit(
            f"{rows.isna().sum()} eyes have no embedding: "
            "run scripts/extract_embeddings.py first"
        )
    return embeddings[rows.to_numpy(dtype=int)]


def train_head(
    name: str,
    x: dict[str, np.ndarray],
    y: dict[str, np.ndarray],
    args: argparse.Namespace,
    logger: Logger,
) -> dict[str, np.ndarray]:
    """Fit a linear head (best epoch by val), save it, return val/test probabilities."""
    set_seed(args.seed)
    pos_weight = None
    if args.pos_weight:
        positives = y["train"].sum(axis=0)
        pos_weight = (len(y["train"]) - positives) / np.maximum(positives, 1)

    def loader(split: str, shuffle: bool = False) -> DataLoader:
        return DataLoader(
            EmbeddingDataset(x[split], y[split]),
            batch_size=args.batch_size,
            shuffle=shuffle,
            generator=make_generator(args.seed) if shuffle else None,
        )

    device = get_device()
    head = nn.Linear(x["train"].shape[1], y["train"].shape[1])
    head = fit(
        head,
        loader("train", shuffle=True),
        loader("val"),
        args.epochs,
        args.lr,
        device,
        logger,
        pos_weight=pos_weight,
    )
    path = checkpoint_path(name)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(head.state_dict(), path)
    return {s: predict(head, loader(s), device) for s in ["val", "test"]}


def run_prior(truth: dict[str, pd.DataFrame]) -> Probs:
    """Train frequency of every patient label, the same for every patient."""
    freq = truth["train"].mean().to_numpy()
    return {
        s: pd.DataFrame(np.tile(freq, (len(truth[s]), 1)), truth[s].index, LABELS)
        for s in ["val", "test"]
    }


def run_single_eye(
    eyes: pd.DataFrame, emb: np.ndarray, args: argparse.Namespace, logger: Logger
) -> Probs:
    """Head on single-eye embeddings with eye labels, then eyes_to_patient."""
    mask = {s: (eyes["split"] == s).to_numpy() for s in SPLITS}
    x = {s: emb[m] for s, m in mask.items()}
    y = {s: eyes.loc[m, EYE_LABELS].to_numpy() for s, m in mask.items()}
    eye_probs = train_head("single_eye", x, y, args, logger)
    return {s: eyes_to_patient(p, eyes[mask[s]]) for s, p in eye_probs.items()}


def run_paired(
    eyes: pd.DataFrame,
    emb: np.ndarray,
    truth: dict[str, pd.DataFrame],
    args: argparse.Namespace,
    logger: Logger,
) -> Probs:
    """Head on [left, right] embeddings of a patient with patient labels."""
    by_eye = pd.DataFrame(emb, pd.MultiIndex.from_frame(eyes[["patient_id", "eye"]]))
    left = by_eye.xs("left", level="eye")
    right = by_eye.xs("right", level="eye")
    x, y = {}, {}
    for s in SPLITS:
        ids = truth[s].index
        x[s] = np.hstack([left.loc[ids].to_numpy(), right.loc[ids].to_numpy()])
        y[s] = truth[s].to_numpy()
    probs = train_head("paired", x, y, args, logger)
    return {s: pd.DataFrame(p, truth[s].index, LABELS) for s, p in probs.items()}


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    eyes = load_eyes().merge(load_split(SPLIT), on="patient_id", validate="m:1")
    emb = load_embeddings(eyes)
    patients = eyes.drop_duplicates("patient_id").set_index("patient_id").sort_index()
    truth = {s: patients.loc[patients["split"] == s, LABELS] for s in SPLITS}

    runs = {
        "prior": lambda logger: run_prior(truth),
        "single_eye": lambda logger: run_single_eye(eyes, emb, args, logger),
        "paired": lambda logger: run_paired(eyes, emb, truth, args, logger),
    }
    run_params = (
        vars(args)
        | {"split": SPLIT, "embeddings": BACKBONE, "embeddings_dim": emb.shape[1]}
        | data_params(eyes, patients)
        | environment()
    )

    results = {}
    with ExitStack() as stack:
        loggers = make_loggers(args.logger, list(runs))
        unique_loggers = list(dict.fromkeys(loggers.values()))
        for logger in unique_loggers:
            stack.enter_context(logger)
        for name, run in runs.items():
            logger = loggers[name]
            logger.log_params(run_params | {"model": name})
            probs = run(logger)
            for s in ["val", "test"]:
                assert probs[s].index.equals(truth[s].index), (name, s)
            y_val, y_test = truth["val"].to_numpy(), truth["test"].to_numpy()
            thresholds = find_thresholds(y_val, probs["val"].to_numpy())
            metrics = compute_metrics(y_test, probs["test"].to_numpy(), thresholds)
            results[name] = metrics

            logger.log_params(
                {f"threshold/{k}": float(t) for k, t in zip(LABELS, thresholds)}
            )
            logger.log_metrics(
                {f"test_{m}_macro": metrics[f"{m}_macro"] for m in METRICS}
            )
            per_class = pd.DataFrame(
                {m: [metrics[f"{m}_{c}"] for c in LABELS] for m in METRICS},
                index=pd.Index(LABELS, name="class"),
            )
            per_class["threshold"] = thresholds
            logger.log_table("test_per_class", per_class)
            if name != "prior":
                logger.log_model(name, checkpoint_path(name))
            PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
            for s in ["val", "test"]:
                probs[s].to_csv(PREDICTIONS_DIR / f"{name}_{s}.csv")

        columns = [f"{m}_macro" for m in METRICS] + [f"pr_auc_{c}" for c in LABELS]
        table = pd.DataFrame(results).T[columns]
        table.index.name = "model"
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        table.to_csv(OUT_DIR / "metrics.csv", float_format="%.4f")

        report = (
            f"Test metrics, patient level ({SPLIT}, {len(truth['test'])} patients):\n"
            f"{table.round(3).to_string()}"
        )
        for logger in unique_loggers:
            logger.log_text("baseline_metrics", report)
        if args.logger != "console":
            print(report)
    print(f"Saved {OUT_DIR / 'metrics.csv'} and {PREDICTIONS_DIR}/")


if __name__ == "__main__":
    main()
