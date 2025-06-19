
# -*-coding:utf-8 -*-
'''
@File    :   train.py
@Date    :   19/06/2025
@Author  :   salil apte
@Version :   1.0
@Desc    :   Train a xgboost model (https://xgboost.readthedocs.io)
using the pyppg and demographic features, and save the model as a .json file
Run this script as: $ python -m src.modeling.train
'''

from pathlib import Path

from loguru import logger
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error
from sklearn.model_selection import train_test_split
import typer
import xgboost as xgb

from src.config import (
    EARLY_STOPPING_ROUNDS,
    MODELS_DIR,
    NUM_BOOST_ROUNDS,
    PROCESSED_DATA_DIR,
    RAW_DATA_DIR,
    RESULTS_DIR,
    SEED,
    VAL_SET_SIZE,
    XGB_PARAMS,
)

app = typer.Typer()


@app.command()
def main(
    demographics_path: Path = PROCESSED_DATA_DIR / "demographics.parquet",
    features_path: Path = PROCESSED_DATA_DIR / "features.parquet",
    labels_path: Path = RAW_DATA_DIR / "train_labels.csv",
    model_path: Path = MODELS_DIR / "xgb_dem_pyppg_20250620.json",
    results_path: Path = RESULTS_DIR,
):
    logger.info("Loading the data")
    # Load the labels and training data as dataframes
    labels = pd.read_csv(labels_path)
    df_demo = pd.read_parquet(demographics_path)
    df_ppg = pd.read_parquet(features_path)
    df = pd.concat([df_demo, df_ppg], axis=1)
    df.drop(columns=["id"], inplace=True)
    # The shape should be (60000,924)
    print(df.shape)
    df.head()

    logger.info("Preparing the training and validation sets")
    # Prepare features, labels, and user ids
    X = df.to_numpy()
    y = labels.to_numpy()
    unique_ids = df_demo["id"].unique()

    # Split the ids for training and testing    
    train_ids, test_ids = train_test_split(unique_ids, test_size=VAL_SET_SIZE, random_state=SEED)

    train_mask = df_demo["id"].isin(train_ids)
    test_mask = df_demo["id"].isin(test_ids)

    # Split the train and test dataset
    X_train, y_train = X[train_mask], y[train_mask]
    X_val, y_val = X[test_mask], y[test_mask]

    # Create DMatrix objects
    dtrain = xgb.DMatrix(data=X_train, label=y_train)
    dval = xgb.DMatrix(data=X_val, label=y_val)

    logger.info("Training XGBoost model")
    # Train model with early stopping
    evals = [(dtrain, "train"), (dval, "val")]

    xgb_model = xgb.train(
        params=XGB_PARAMS,
        dtrain=dtrain,
        num_boost_round=NUM_BOOST_ROUNDS,
        evals=evals,
        early_stopping_rounds=EARLY_STOPPING_ROUNDS,
        verbose_eval=False,
    )

    y_pred = xgb_model.predict(dval)
    mse = mean_squared_error(y_val, y_pred)
    mape = mean_absolute_percentage_error(y_val, y_pred)

    print(f"XGBoost MSE: {mse:0.2f}")
    print(f"XGBoost MAPE: {mape:0.2f}")
    logger.success("Modeling training complete.")

    # Save the model as a json file
    xgb_model.save_model(model_path)

    # Load the saved model and compare outputs 
    loaded_xgb_model =  xgb.Booster()
    loaded_xgb_model.load_model(model_path)

    y_loaded_model_pred = loaded_xgb_model.predict(dval)
    if np.allclose(y_pred, y_loaded_model_pred):
        print("Model saved correctly.")

if __name__ == "__main__":
    app()
