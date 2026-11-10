"""Extract frozen ImageNet backbone embeddings for all training images.

Writes data/processed/embeddings/{backbone}/embeddings.npy and index.csv
(patient_id, eye, image) in the same row order. A non-default --size adds _{size}
to the folder name, --limit N adds _limitN.

Usage: uv run python scripts/extract_embeddings.py [--backbone resnet18] [--limit 50]
"""

import argparse
import time

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from src.data.datasets import FundusDataset
from src.data.labels import load_eyes
from src.models.backbones import get_backbone
from src.utils.device import get_device
from src.utils.paths import EMBEDDINGS_DIR


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser(description="Extract backbone embeddings.")
    parser.add_argument("--backbone", default="resnet18", help="see get_backbone")
    parser.add_argument("--size", type=int, default=224, help="input image size")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--limit", type=int, help="first N eyes only (test run)")
    args = parser.parse_args()

    eyes = load_eyes().iloc[: args.limit]
    name = args.backbone + (f"_{args.size}" if args.size != 224 else "")
    name += f"_limit{args.limit}" if args.limit else ""

    device = get_device()
    model = get_backbone(args.backbone)[0].to(device).eval()
    loader = DataLoader(
        FundusDataset(eyes, size=args.size),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    start = time.perf_counter()
    embeddings = np.concatenate(
        [model(b["x"].to(device)).cpu().numpy() for b in tqdm(loader, desc=name)]
    )
    elapsed = time.perf_counter() - start

    out_dir = EMBEDDINGS_DIR / name
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "embeddings.npy", embeddings.astype(np.float32))
    eyes[["patient_id", "eye", "image"]].to_csv(out_dir / "index.csv", index=False)
    print(
        f"{len(eyes)} images on {device} in {elapsed:.1f} s "
        f"({elapsed / len(eyes) * 1000:.0f} ms/image) -> {out_dir}"
    )


if __name__ == "__main__":
    main()
