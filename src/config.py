"""
Central configuration for the trading system.
Paths and defaults used across data, training, and backtesting.
"""

from pathlib import Path
from typing import Optional

# Project root (parent of src/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Data
DATA_DIR = PROJECT_ROOT / "data"
SAMPLE_DATA_PATH = DATA_DIR / "sample_data.csv"

# Results
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
METRICS_DIR = RESULTS_DIR / "metrics"
MODELS_DIR = RESULTS_DIR / "models"

# Configs
CONFIGS_DIR = PROJECT_ROOT / "configs"
MODEL_CONFIG_PATH = CONFIGS_DIR / "model_config.yaml"
TRAINING_CONFIG_PATH = CONFIGS_DIR / "training_config.yaml"

# Defaults
DEFAULT_SEQUENCE_LENGTH = 60
DEFAULT_TRAIN_SPLIT = 0.7
DEFAULT_VAL_SPLIT = 0.15
DEFAULT_TEST_SPLIT = 0.15
DEFAULT_N_FEATURES = 45


def get_data_path(path: Optional[str] = None) -> Path:
    """Return data path; default to sample_data.csv if path is None."""
    if path is None:
        return SAMPLE_DATA_PATH
    p = Path(path)
    return p if p.is_absolute() else PROJECT_ROOT / path


def get_save_dir(save_dir: Optional[str] = None, subdir: str = "models") -> Path:
    """Return directory for saving outputs (models, metrics, etc.)."""
    if save_dir is not None:
        return Path(save_dir)
    return RESULTS_DIR / subdir
