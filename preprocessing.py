import cv2
import numpy as np


def load_normalized_image(path: str):
    image = cv2.imread(path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return cv2.normalize(image.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX)


def add_gaussian_noise(img, avg: float, std: float):
    return img + np.random.normal(avg, std, img.shape)
