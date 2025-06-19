# -*-coding:utf-8 -*-
'''
@File    :   dataset.py
@Date    :   19/06/2025
@Author  :   salil apte
@Version :   1.0
@Desc    :   Process the raw .csv data to extract unique user ids
and save them together with other demographics as a .parquet file
'''

from pathlib import Path

from loguru import logger
import pandas as pd
import typer

from config import PROCESSED_DATA_DIR, RAW_DATA_DIR

app = typer.Typer()

def add_user_id(df: pd.DataFrame):
    """
    Identify unique users and add a new column with user ids.

    Parameters:
        df (pd.DataFrame): A dataframe with demographic features

    Returns:
        df (pd.DataFrame):  A dataframe with demographic features and user ids
    """
    # Check unique combination of features (likely demographics)
    unique_users = df[["features_0", "features_1", "features_2", "features_3", "features_4"]].drop_duplicates().reset_index(drop=True)
    # Assign zero-padded person IDs
    unique_users["id"] = unique_users.index.astype(str).str.zfill(3)
    print(f"Number of unique demographic profiles: {unique_users.shape[0]}")
    # Merge back to original DataFrame
    df_with_ids = df.merge(unique_users, on=["features_0", "features_1", "features_2", "features_3", "features_4"], how="left")

    return df_with_ids

@app.command()
def main(
    train_input_path: Path = RAW_DATA_DIR / "train.csv",
    train_demographics_path: Path = PROCESSED_DATA_DIR / "demographics.parquet",
    test_input_path: Path = RAW_DATA_DIR / "test.csv",
    test_demographics_path: Path = PROCESSED_DATA_DIR / "test_demographics.parquet",
):
    # TODO Convert the processing steps into a function

    logger.info(f"Loading training input data from {train_input_path}")
    train_df = pd.read_csv(train_input_path)
    # First 3000 columns represent PPH signals and last five are likely demographics
    train_demographics = train_df.iloc[:, 3000:]
    # Process the demographic data
    logger.info("Processing training dataset...")
    train_demographics_with_ids = add_user_id(df=train_demographics)
    # Save output features to parquet files
    train_demographics_with_ids.to_parquet(train_demographics_path, index=False)
    logger.success(f"Saved training data demographics to {train_demographics_path}")

    logger.info(f"Loading test input data from {test_input_path}")
    test_df = pd.read_csv(test_input_path)
    # First 3000 columns represent PPH signals and last five are likely demographics
    test_demographics = test_df.iloc[:, 3000:]
    # Process the demographic data
    logger.info("Processing test dataset...")
    test_demographics_with_ids = add_user_id(df=test_demographics)
    # Save output features to parquet files
    test_demographics_with_ids.to_parquet(test_demographics_path, index=False)
    logger.success(f"Saved test data demographics to {test_demographics_path}")


if __name__ == "__main__":
    app()
