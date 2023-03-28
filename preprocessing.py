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
    import tensorflow as tf
    img = img + 1 * np.random.normal(avg, std, img.shape)
    tf_threshold = tf.where(img < 0, 0, img)
    tf_threshold = tf.where(tf_threshold > 1, 1, tf_threshold)
    img = tf_threshold
    return img


def tf_add_mask(tf_img, initial_value: float):
    import tensorflow as tf
    return tf.Variable(initial_value * tf.ones(shape=tf_img.shape), name="lambda_x", trainable=True)


def resize(img, width: int, height: int):
    return cv2.resize(img, (width, height))


def rgb_to_YCbCr(img):
    R, G, B = img[:, :, 0], img[:, :, 1], img[:, :, 2]
    new_img = np.zeros(shape=img.shape)
    new_img[:, :, 0] = 0.0 + .299 * R + .587 * G + .114 * B
    new_img[:, :, 1] = .5 + -.1687 * R - .3313 * G + .5 * B
    new_img[:, :, 2] = .5 + .5 * R - .4187 * G - .4392 * B
    return new_img


def load_gray_image(path: str):
    """

    :param path: The path to the image we want to load.
    :return: Image normalized to [0,1].
    """
    image = cv2.imread(path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.normalize(image.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX)
