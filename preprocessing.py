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


def tf_load_normalized_image(path: str):
    import tensorflow as tf
    img_original = load_normalized_image(path)
    H, W, C = img_original.shape
    img_original = np.reshape(img_original, newshape=[1, H, W, C])
    tf_img = tf.Variable(tf.zeros(shape=(1, H, W, C)), name="tf_img", trainable=False)
    tf_img.assign(img_original)
    return tf_img


def tf_add_gaussian_noise(img, avg: float, std: float):
    return img + 1 * np.random.normal(avg, std, img.shape)
