# -*-coding:utf-8 -*-
'''
@File    :   features.py
@Date    :   19/06/2025
@Author  :   salil apte
@Version :   1.0
@Desc    :   Process the ppg signals from .csv data to batch extract features using
the pyppg package (https://pyppg.readthedocs.io) and save them as a .parquet file
'''

from pathlib import Path

from dotmap import DotMap
from joblib import Parallel, delayed
from loguru import logger
import numpy as np
import pandas as pd
from pyPPG import PPG, Fiducials
import pyPPG.biomarkers as BM
import pyPPG.fiducials as FP
import pyPPG.ppg_sqi as SQI
import pyPPG.preproc as PP
from tqdm import tqdm
import typer

from config import (
    END_IDX,
    FILTER_ORDER,
    FS,
    HIGH_PASS,
    LOW_PASS,
    PROCESSED_DATA_DIR,
    RAW_DATA_DIR,
    SMOOTHING_WINDOWS,
    START_IDX,
)

# CLI app using Typer
app = typer.Typer()


def load_signal_data(input_sig: np.ndarray, fs = np.nan, start_sig = 0, end_sig = -1, use_tk=True, print_flag=True):
    """
    Load raw PPG data. This function is a modified version of the original "load_data" function in the pyPPG package.

    :param input_sig: array containing the PPG signal
    :type data_path: ndarray
    :param start_sig: the first sample of the signal to be analysed
    :type start_sig: int
    :param fs: the sampling frequency of the PPG in Hz
    :type fs: int
    :param end_sig: the last sample of the signal to be analysed
    :type end_sig: int
    :param use_tk: a bool for using tkinter interface
    :type use_tk: bool
    :param print_flag: a bool for print message
    :type print_flag: bool

    :return: s: dictionary of the PPG signal:

        * s.start_sig: the first sample of the signal to be analysed
        * s.end_sig: the last sample of the signal to be analysed
        * s.v: a vector of PPG values
        * s.fs: the sampling frequency of the PPG in Hz
        * s.name: name of the record
        * s.v: 1-d array, a vector of PPG values
        * s.fs: the sampling frequency of the PPG in Hz
        * s.ppg: 1-d array, a vector of the filtered PPG values
        * s.vpg: 1-d array, a vector of the filtered PPG' values
        * s.apg: 1-d array, a vector of the filtered PPG" values
        * s.jpg: 1-d array, a vector of the filtered PPG'" values
        * s.filtering: a bool for filtering
        * s.correct: a bool for correcting fiducial points
    """

    sig = input_sig

    # Initialize the signal as a dotmap object
    s = DotMap()

    s.start_sig = start_sig
    if start_sig<end_sig:
        s.end_sig = end_sig
    else:
        s.end_sig = len(sig)

    try:
        s.v=sig[s.start_sig:s.end_sig]
    except Exception:
        raise('There is no valid PPG signal!')

    s.fs=fs
    s.name="default"

    return s

def flatten_features_as_list(bm_stats: dict):
    """
    Flatten nested biomarker statistics dictionary into feature vectors.

    Parameters:
        bm_stats (dict): A dictionary of pandas DataFrames grouped by biomarker types.

    Returns:
        Tuple[list, list]: A list of flattened biomarker values and their corresponding feature names.
    """
    flat_parts = []
    for group_name, df in bm_stats.items():
        df_named = df.copy()
        df_named.index.name = 'stat'
        df_named.columns.name = 'biomarker'
        df_named = df_named.stack()
        df_named.index = df_named.index.map(lambda x: f"{group_name}_{x[1]}_{x[0]}")
        flat_parts.append(df_named)

    flat_series = pd.concat(flat_parts)
    flat_series = flat_series.replace([np.inf, -np.inf], np.nan).fillna(0)

    values = flat_series.tolist()
    names = flat_series.index.tolist()
    return values, names

def extract_pyppg_features(ppg_signal: np.ndarray, fs: int = 100):
    """
    Extract PPG features for a single sample using vitalpy.

    Parameters:
        ppg_signal (np.ndarray): 1D array of PPG signal values.
        fs (int): Sampling frequency in Hz (default = 100).

    Returns:
        Tuple of (feature_values, feature_names) or None if extraction fails.
    """
    try:
        # Load and preprocess signal
        s = load_signal_data(
            input_sig=ppg_signal, fs=fs,
            start_sig=START_IDX, end_sig=END_IDX,
            use_tk=True, print_flag=False
        )
        s.filtering = True
        s.fL = LOW_PASS
        s.fH = HIGH_PASS
        s.order = FILTER_ORDER
        s.sm_wins = SMOOTHING_WINDOWS

        prep = PP.Preprocess(fL=s.fL, fH=s.fH, order=s.order, sm_wins=s.sm_wins)
        s.ppg, s.vpg, s.apg, s.jpg = prep.get_signals(s=s)

        # Initialise the correction for fiducial points
        corr_on = ['on', 'dn', 'dp', 'v', 'w', 'f']
        correction=pd.DataFrame()
        correction.loc[0, corr_on] = True
        s.correction=correction

        # Create a PPG class
        s = PPG(s)

        # Extract fiducial points
        fpex = FP.FpCollection(s=s)
        fiducials = fpex.get_fiducials(s=s)
        fp = Fiducials(fp=fiducials)
        
        # Init the biomarkers package
        bmex = BM.BmCollection(s=s, fp=fp)

        # Extract biomarkers
        bm_defs, bm_vals, bm_stats = bmex.get_biomarkers()

        # SQI computation
        ppg_sqi = round(np.mean(SQI.get_ppgSQI(ppg=s.ppg, fs=s.fs, annotation=fp.sp)) * 100, 2)

        # Flatten features
        values, names = flatten_features_as_list(bm_stats)
        values.append(ppg_sqi)
        names.append("sqi")
        return values, names

    except Exception as e:
        logger.warning(f"Feature extraction failed: {e}")
        return None

@app.command()
def main(
    # Set input_path and features_path according to training or test data
    input_path: Path = RAW_DATA_DIR / "test.csv",
    features_path: Path = PROCESSED_DATA_DIR / "test_features.parquet",
):
    # TODO Convert the processing steps into a function

    logger.info(f"Loading input data from {input_path}")
    df = pd.read_csv(input_path)
    # First 3000 columns represent PPG signals
    signal_data = df.iloc[:, :3000].to_numpy()

    logger.info(f"Processing {len(signal_data)} signals with pyPPG...")

    # Batch extract pyPPG features for all samples
    results = Parallel(n_jobs=-1)(
    delayed(extract_pyppg_features)(signal_data[i], fs=FS)
    for i in tqdm(range(len(signal_data)), desc="Extracting PPG features")
    )

    results = [r for r in results if r is not None]
    if not results:
        raise RuntimeError("All feature extractions failed.")

    # Collect feature values and feature names
    features = [r[0] for r in results]
    feature_names = results[0][1]

    # Transform into a pandas dataframe
    features_df = pd.DataFrame(features, columns=feature_names)
    
    # Should be (60000, 919) for training and (30000, 919) for test data
    print(features_df.shape)      
    # Check if "sqi" is at the end
    print(features_df.columns[-5:])
    # Check the values quickly
    print(features_df.head())

    # Save output features to parquet files
    features_df.to_parquet(features_path, index=False)
    
    logger.success(f"Saved pyPPG features to {features_path}")

if __name__ == "__main__":
    app()
