# common/utils/utils.py

import pandas as pd
import logging
from typing import Optional


def clean_dataframe(df: pd.DataFrame, required_columns: Optional[list] = None) -> pd.DataFrame:
    """
    Cleans the DataFrame by handling missing values and ensuring required columns are present.

    Parameters:
    - df (pd.DataFrame): DataFrame to clean.
    - required_columns (Optional[list]): List of columns that must be present.

    Returns:
    - pd.DataFrame: Cleaned DataFrame.
    """
    logger = logging.getLogger(__name__)
    logger.info("Starting DataFrame cleaning process.")

    # Drop rows with any NaN values
    initial_shape = df.shape
    df = df.dropna()
    logger.info(f"Dropped rows with NaN values: {initial_shape} -> {df.shape}")

    # Ensure required columns are present
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            logger.error(f"Missing required columns: {missing_cols}")
            raise ValueError(f"Missing required columns: {missing_cols}")

    # Additional cleaning steps can be added here

    logger.info("DataFrame cleaning completed successfully.")
    return df


def calculate_percentage_change(current: float, previous: float) -> float:
    """
    Calculates the percentage change between two values.

    Parameters:
    - current (float): Current value.
    - previous (float): Previous value.

    Returns:
    - float: Percentage change.
    """
    try:
        if previous == 0:
            return 0.0
        return ((current - previous) / previous) * 100
    except Exception as e:
        logging.getLogger(__name__).error(f"Error calculating percentage change: {e}")
        return 0.0

# Add more utility functions as needed

