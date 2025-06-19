# -*-coding:utf-8 -*-
'''
@File    :   predict.py
@Date    :   19/06/2025
@Author  :   salil apte
@Version :   1.0
@Desc    :   Use a xgboost model trained with the pyppg and
demographic features to predict and save the tes labels
Run this script as: $ python -m src.modeling.predict
'''

from pathlib import Path

from loguru import logger
import pandas as pd
import typer
import xgboost as xgb

from src.config import MODELS_DIR, PROCESSED_DATA_DIR, RESULTS_DIR

app = typer.Typer()


@app.command()
def main(
    features_path: Path = PROCESSED_DATA_DIR / "test_features.parquet",
    demographics_path: Path = PROCESSED_DATA_DIR / "test_demographics.parquet",
    model_path: Path = MODELS_DIR / "xgb_dem_pyppg_20250620.json",
    predictions_path: Path = RESULTS_DIR / "test_predictions.csv",
):
    logger.info("Loading the data")
    # Load the labels and training data as dataframes
    df_demo = pd.read_parquet(demographics_path)
    df_ppg = pd.read_parquet(features_path)
    df = pd.concat([df_demo, df_ppg], axis=1)
    df.drop(columns=["id"], inplace=True)
    # The shape should be (30000,924)
    print(df.shape)
    df.head()

    logger.info("Preparing the test sets")
    # Prepare features
    X_test = df.to_numpy()
    dtest = xgb.DMatrix(data=X_test)

    logger.info("Performing inference for model...")

    # Load the saved model and compare outputs 
    loaded_xgb_model =  xgb.Booster()
    loaded_xgb_model.load_model(model_path)

    y_pred = loaded_xgb_model.predict(dtest)
    y_pred_df = pd.DataFrame(y_pred,columns=["target"], )
    y_pred_df.to_csv(predictions_path, index=False)
    logger.success("Inference complete.")


if __name__ == "__main__":
    app()
