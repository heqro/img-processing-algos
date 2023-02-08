import cv2
import numpy as np


def load_normalized_image(path: str):
    """

    :param path: The path to the image we want to load.
    :return: Image normalized to [0,1].
    """
    image = cv2.imread(path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return cv2.normalize(image.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX)


def add_gaussian_noise(img, avg: float, std: float):
    """

    :param img: The image to contaminate.
    :param avg: Average for the gaussian noise.
    :param std: Standard deviation for the gaussian noise.
    :return:
    """
    return img + np.random.normal(avg, std, img.shape)
