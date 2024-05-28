from pathlib import Path
import os

PARENT_DIR = Path(__file__).parent.resolve().parent
DATA_DIR = PARENT_DIR / 'data'
RAW_DATA_DIR = PARENT_DIR / 'data' / 'raw'
TRANSFORMED_DATA_DIR = PARENT_DIR / 'data' / 'transformed'
DATA_CACHE_DIR = PARENT_DIR / 'data' / 'cache'
MODELS_DIR = PARENT_DIR / 'models'
METRICS_DIR = PARENT_DIR / 'data' / 'metrics'
RESIDUALS_DATA_DIR = PARENT_DIR / 'data' / 'residuals'
VISUALIZATIONS_DIR = PARENT_DIR / 'data' / 'visualizations'

if not Path(DATA_DIR).exists():
    os.mkdir(DATA_DIR)

if not Path(RAW_DATA_DIR).exists():
    os.mkdir(RAW_DATA_DIR)

if not Path(TRANSFORMED_DATA_DIR).exists():
    os.mkdir(TRANSFORMED_DATA_DIR)

if not Path(DATA_CACHE_DIR).exists():
    os.mkdir(DATA_CACHE_DIR)

if not Path(MODELS_DIR).exists():
    os.mkdir(MODELS_DIR)

if not Path(METRICS_DIR).exists():
    os.mkdir(METRICS_DIR)

if not Path(VISUALIZATIONS_DIR).exists():
    os.mkdir(VISUALIZATIONS_DIR)

if not Path(RESIDUALS_DATA_DIR).exists():
    os.mkdir(RESIDUALS_DATA_DIR)