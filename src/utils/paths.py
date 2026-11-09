"""Project paths. Import these instead of hardcoding paths."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_RAW = PROJECT_ROOT / "data" / "raw"
ODIR_DIR = DATA_RAW / "ODIR-5K" / "ODIR-5K"
TRAIN_IMAGES_DIR = ODIR_DIR / "Training Images"
TEST_IMAGES_DIR = ODIR_DIR / "Testing Images"
ANNOTATIONS_XLSX = ODIR_DIR / "data.xlsx"

DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
SPLITS_DIR = PROJECT_ROOT / "data" / "splits"
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"
CHECKPOINTS_DIR = PROJECT_ROOT / "checkpoints"
