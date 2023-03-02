# This module encapsulates functionality relating to reading coefficients for early stoppage
import math

import pandas as pd
from numpy import ndarray
from pandas import DataFrame
import numpy as np


# We load data whose headers are:
# | width | height | noise_1 | noise_1_coef | noise_2 | noise_2_coef | noise_3 | noise_3_coef |

def load_data(path) -> DataFrame:
    return pd.read_csv(path, header=0)


def distance(df: DataFrame, height: float, width: float):
    """

    :param df: a DataFrame
    :param width: of the image
    :param height: of the image
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


def get_surface_coefficients(db_index: int, noise_std: float, p_index: int | float, case='') -> list[float]:
    """

    :param db_index: synthetic image index.
    :param noise_std: standard deviation in Gaussian additive noise model.
    :param p_index: the value of p in our model.
    :param case: case for which to search coefficients (fidelity mask, diffusion mask or without mask)
    :return: a list containing the surface coefficients for the bicubic surface for the given standard deviation,
    value of p and synthetic image index.
    """
    def get_matrix_row(x: float, y: float) -> list[float]:
        return [1, x, y, x * y, x ** 2, y ** 2, x ** 2 * y, x * y ** 2, x ** 2 * y ** 2, x ** 3, x ** 3 * y,
                x ** 3 * y ** 2, x ** 3 * y ** 3, y ** 3, x * y ** 3, x ** 2 * y ** 3]

    def get_surface_matrix(A: ndarray, b: ndarray):
        return np.dot(np.linalg.pinv(A), b)

    dims = 64 + np.arange(10) * 64  # assume we have this configuration

    df = load_data(
        path=f'synth_images_testing/synth_img_{db_index}/results_log/coefficientsP{p_index}{case}.csv')

    res = []
    b = []
    max_dims = np.max(dims)
    for xx in dims:
        for yy in dims:
            res.append(get_matrix_row(xx / max_dims, yy / max_dims))
            b.append(get_stoppage_coefficient(df, xx, yy, noise_std))
    res = np.vstack(res)
    c = np.array(b).T

    return get_surface_matrix(res, c)


def get_surface_function(db_index: int, noise_std: float, p_index: int | float, case=''):
    """

    :param db_index: synthetic image index.
    :param noise_std: standard deviation in Gaussian additive noise model.
    :param p_index: the value of p in our model.
    :param case: case for which to search coefficients (fidelity mask, diffusion mask or without mask)
    :return: a function returning the surface associated to the parameters.
    """
    coefs = get_surface_coefficients(db_index, noise_std, p_index, case)
    from sympy.abc import x, y
    from sympy import lambdify
    base = [1, x, y, x * y, x ** 2, y ** 2, x ** 2 * y, x * y ** 2, x ** 2 * y ** 2, x ** 3, x ** 3 * y,
            x ** 3 * y ** 2, x ** 3 * y ** 3, y ** 3, x * y ** 3, x ** 2 * y ** 3]
    return lambdify([x, y], (coefs * base).sum())
