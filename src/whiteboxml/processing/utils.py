"""Data processing utilities module.

This module implements utilities to process data in memory (using pandas DataFrame).
"""

####################################################################################################
# IMPORTS

import logging

import pandas as pd

####################################################################################################
# LOGGING

logger = logging.getLogger(__name__)


####################################################################################################
# FUNCTIONS

def optimize_size(df: pd.DataFrame, uniqueness_thr: float = 0.5) -> None:
    """
    This function reduces a pandas dataframe size adjusting optimum dtypes, inplace.

    Args:
        df (pd.Dataframe): dataframe to optimize.
        uniqueness_thr (float): threshold of the uniqueness of column values. If the
        ratio of unique values of the column is below this threshold, it will be
        considered of a categorical type.

    Returns:
        This function returns None, as modifies the DataFrame inplace.
    """

    start = df.memory_usage(deep=True).sum()

    # optimize numeric

    f_cols = df.select_dtypes('float').columns
    i_cols = df.select_dtypes('integer').columns

    logger.debug(f'float columns detected: {f_cols}')
    df.loc[:, f_cols] = df.loc[:, f_cols].apply(pd.to_numeric, downcast='float')

    logger.debug(f'integer columns detected: {i_cols}')
    df.loc[:, i_cols] = df.loc[:, i_cols].apply(pd.to_numeric, downcast='integer')

    # optimize object

    o_cols = df.select_dtypes('object')

    for o_col in o_cols:

        # uniqueness will be 1.0 if all elements in a column are different
        uniqueness = df.loc[:, o_col].nunique() / len(df)

        if uniqueness < uniqueness_thr:
            df.loc[:, o_col] = df.loc[:, o_col].astype('category', copy=False)

    end = df.memory_usage(deep=True).sum()

    logger.info(f'optimized df size from {start / 1e6:.2f}MB to {end / 1e6:.2f}MB')
