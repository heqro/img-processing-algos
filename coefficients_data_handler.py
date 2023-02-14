# This module encapsulates functionality relating to reading coefficients for early stoppage
import math

import pandas as pd
from pandas import DataFrame
import numpy as np


# We load data whose headers are:
# | width | height | noise_1 | noise_1_coef | noise_2 | noise_2_coef | noise_3 | noise_3_coef |

def load_data(path) -> DataFrame:
    return pd.read_csv(path, header=0)


def distance(df: DataFrame, height: float, width: float):
    """

    :param df: a DataFrame
    :param width:
    :param height:
    :return: the distances between the point (width, height) and the points (df["width"], df["height"])
    """
    return np.sqrt((df["height"] - height) ** 2 + (df["width"] - width) ** 2)


def closest_std_coefficient(df_row, noise_std: float):
    """

    :param df_row: given a single DataFrame row
    :param noise_std: a standard deviation noise value
    :return: the linear coefficient corresponding to the closest of the sampled standard deviations w.r.t. noise_std
    """
    values = np.abs(np.array([df_row["noise_1"], df_row["noise_2"], df_row["noise_3"]]) - noise_std)
    index = np.argmin(values)
    return df_row[2 * (index + 1) + 1]


def get_stoppage_coefficient(data: DataFrame, height: float, width: float, noise_std: float):
    """

    :param data: a DataFrame mapping each height, width and noise_std parameters to a specific linear coefficient
    :param height: the image height
    :param width: the image width
    :param noise_std: the estimated standard deviation of the image
    :return: the linear coefficient for the image of given height, width and noise standard deviation
    """
    filtered_h_w = data[(data["height"] <= height + 64) & (data["height"] >= height - 64)
                        & (data["width"] <= width + 64) & (data["width"] >= width - 64)]
    if filtered_h_w.empty:
        raise ValueError("The proposed height or weight values are too different to the DataFrame values.")
    closest_point = filtered_h_w.loc[filtered_h_w.apply(distance, axis=1, args=(height, width)).idxmin()]
    return closest_std_coefficient(closest_point, noise_std)
