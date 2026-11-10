"""Project paths. Import these instead of hardcoding paths."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_RAW = PROJECT_ROOT / "data" / "raw"
ODIR_DIR = DATA_RAW / "ODIR-5K" / "ODIR-5K"
TRAIN_IMAGES_DIR = ODIR_DIR / "Training Images"
TEST_IMAGES_DIR = ODIR_DIR / "Testing Images"
ANNOTATIONS_XLSX = ODIR_DIR / "data.xlsx"

DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
EMBEDDINGS_DIR = DATA_PROCESSED / "embeddings"
IMAGE_STATS_CSV = DATA_PROCESSED / "image_stats.csv"
SPLITS_DIR = PROJECT_ROOT / "data" / "splits"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
CHECKPOINTS_DIR = PROJECT_ROOT / "checkpoints"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
