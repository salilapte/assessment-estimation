'''
@File    :   config.py
@Date    :   19/06/2025
@Author  :   salil apte
@Version :   1.0
@Desc    :   Store global variables, signal processing parameters,
xgboost model parameters and training configuration
'''

from pathlib import Path

from dotenv import load_dotenv
from loguru import logger

# Load environment variables from .env file if it exists
load_dotenv()

# Paths
PROJ_ROOT = Path(__file__).resolve().parents[1]
logger.info(f"PROJ_ROOT path is: {PROJ_ROOT}")

DATA_DIR = PROJ_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXTERNAL_DATA_DIR = DATA_DIR / "external"

MODELS_DIR = PROJ_ROOT / "models"

REPORTS_DIR = PROJ_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
RESULTS_DIR = REPORTS_DIR / "results"

# Signal processing parameters
START_IDX = 500
END_IDX = 2500
LOW_PASS = 0.5 # low pass frequency in hz
HIGH_PASS = 12 # high pass frequency in hz
FILTER_ORDER = 4
SMOOTHING_WINDOWS = {'ppg': 50, 'vpg': 10, 'apg': 10, 'jpg': 10}
FS = 100 # sampling frequency in Hz

# XGBoost parameters
XGB_PARAMS = {
    "objective": "reg:squarederror",
    "eval_metric": "rmse",
    "max_depth": 5,
    "eta": 0.03,
    "subsample": 0.75,
    "colsample_bytree": 0.7,
    "reg_alpha": 3.0,
    "reg_lambda": 2.0,
    "seed": 1,
    "tree_method": "gpu_hist",  
}

# Training parameters
SEED = 1
VAL_SET_SIZE = 0.05
NUM_BOOST_ROUNDS = 400
EARLY_STOPPING_ROUNDS = 30
