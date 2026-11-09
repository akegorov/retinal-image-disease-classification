"""Create a patient-level split data/splits/{name}.csv and print class counts per split.

Usage:
    uv run python scripts/make_split.py                         # split_v1, seed 42
    uv run python scripts/make_split.py --name split_v2 --seed 0
"""

import argparse

from src.data.labels import LABELS, load_eyes
from src.data.splits import make_patient_split, save_split


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a patient-level split.")
    parser.add_argument("--name", default="split_v1", help="file name without .csv")
    parser.add_argument("--val-size", type=float, default=0.15)
    parser.add_argument("--test-size", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    eyes = load_eyes()
    split = make_patient_split(eyes, args.val_size, args.test_size, args.seed)
    save_split(split, args.name)

    eyes = eyes.merge(split, on="patient_id")
    patients = eyes.drop_duplicates("patient_id")
    for title, table, cols in [
        ("patients, patient labels", patients, LABELS),
        ("eyes, eye labels", eyes, [f"eye_{label}" for label in LABELS]),
    ]:
        counts = table.groupby("split")[cols].sum()
        counts["total"] = table["split"].value_counts()
        print(f"\n{title}:\n{counts.loc[['train', 'val', 'test']].T.to_string()}")


if __name__ == "__main__":
    main()
