import numpy as np
import scipy as sp

from enum import Enum

from numpy import ndarray


class GradientType(Enum):
    FORWARD = 0,
    CENTERED = 1,
    BACKWARD = 2


FWRD_X_GRADIENT = [[0.0, 0.0, 0.0], [0.0, -1.0, 1.0], [0.0, 0.0, 0.0]]
CENTER_X_GRADIENT = [[0.0, 0.0, 0.0], [0.5, 0.0, 0.5], [0.0, 0.0, 0.0]]
BWRD_X_GRADIENT = [[0.0, 0.0, 0.0], [-1.0, 1.0, 0.0], [0.0, 0.0, 0.0]]

FWRD_Y_GRADIENT = [[0.0, 1.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 0.0]]
CENTER_Y_GRADIENT = [[0.0, 0.5, 0.0], [0.0, 0.0, 0.0], [0.0, -0.5, 0.0]]
BWRD_Y_GRADIENT = [[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, -1.0, 0.0]]


def gradx(img, gradient_type: GradientType, use_convolution=True):
    """
    Calculates the gradient of an image along the x-axis.
    :param img: The image for which we calculate the gradient.
    :param gradient_type: The finite difference approximation formula
    we want to use.
    :param use_convolution: Whether to use a convolution matrix or to use
    explicit, pixel by pixel calculation.
    :return: The mapping of the image gradient along the x-axis
    using the finite approximation formula of type gradient_type.
    """
    img_x = np.zeros(img.shape)
    if not use_convolution:
        if gradient_type.value == GradientType.FORWARD.value:
            if len(img.shape) == 3:
                img_x[:, :-1, :] = img[:, 1:, :] - img[:, :-1, :]
            if len(img.shape) == 2:
                img_x[:, :-1] = img[:, 1:] - img[:, :-1]
        if gradient_type.value == GradientType.CENTERED.value:
            if len(img.shape) == 3:
                img_x[:, 1:-1, :] = (img[:, 2:, :] - img[:, :-2, :]) / 2
            if len(img.shape) == 2:
                img_x[:, 1:-1] = (img[:, 2:] - img[:, :-2]) / 2
        if gradient_type.value == GradientType.BACKWARD.value:
            if len(img.shape) == 3:
                img_x[:, 1:, :] = img[:, 1:, :] - img[:, :-1, :]
            if len(img.shape) == 2:
                img_x[:, 1:] = img[:, 1:] - img[:, :-1]
    else:
        if gradient_type.value == GradientType.FORWARD.value:
            img_x = convolve_image(img, np.array(FWRD_X_GRADIENT))
        if gradient_type.value == GradientType.CENTERED.value:
            img_x = convolve_image(img, np.array(CENTER_X_GRADIENT))
        if gradient_type.value == GradientType.BACKWARD.value:
            img_x = convolve_image(img, np.array(BWRD_X_GRADIENT))
    return img_x


def grady(img, gradient_type: GradientType, use_convolution=True):
    """
    Calculates the gradient of an image along the y-axis.
    :param img: The image for which we calculate the gradient.
    :param gradient_type: The finite difference approximation formula
    we want to use.
    :param use_convolution: Whether to use a convolution matrix or to use
    explicit, pixel by pixel calculation.
    :return: The mapping of the image gradient along the y-axis
    using the finite approximation formula of type gradient_type.
    """
    img_y = np.zeros(img.shape)
    if not use_convolution:
        if gradient_type.value == GradientType.FORWARD.value:
            if len(img.shape) == 3:
                img_y[:-1, :, :] = img[1:, :, :] - img[:-1, :, :]
            if len(img.shape) == 2:
                img_y[:-1, :] = img[1:, :] - img[:-1, :]
        if gradient_type.value == GradientType.CENTERED.value:
            if len(img.shape) == 3:
                img_y[1:-1, :, :] = (img[2:, :, :] - img[:-2, :, :]) / 2
            if len(img.shape) == 2:
                img_y[1:-1, :] = (img[2:, :] - img[:-2, :]) / 2
        if gradient_type.value == GradientType.BACKWARD.value:
            if len(img.shape) == 3:
                img_y[1:, :, :] = img[1:, :, :] - img[-1:, :, :]
            if len(img.shape) == 2:
                img_y[1:, :] = img[1:, :] - img[-1:, :]
    else:
        if gradient_type.value == GradientType.FORWARD.value:
            img_y = convolve_image(img, np.array(FWRD_Y_GRADIENT))
        if gradient_type.value == GradientType.CENTERED.value:
            img_y = convolve_image(img, np.array(CENTER_Y_GRADIENT))
        if gradient_type.value == GradientType.BACKWARD.value:
            img_y = convolve_image(img, np.array(BWRD_Y_GRADIENT))
    return img_y


def div(img_x, img_y, gradient_type=GradientType.BACKWARD, p=2, epsilon=0, use_convolution=False):
    """
    Calculates the divergence of an image.
    :param img_x: image gradient alongside the x-axis.
    :param img_y: image gradient alongside the y-axis.
    :param gradient_type: The type of gradient formula to apply. Defaults to CENTERED.
    :param p: p norm to consider. Defaults to 2 (linear case).
    :param epsilon: Optional parameter. Defaults to 0.
    :return: The mapping of the image p-Laplacian along the y-axis.
    """
    mod_p = np.sqrt(img_x ** 2 + img_y ** 2 + epsilon) ** (p - 2)
    return gradx(np.multiply(img_x, mod_p), gradient_type, use_convolution=use_convolution) + grady(np.multiply(img_y, mod_p), gradient_type, use_convolution=use_convolution)


def psnr(img_original, img_noise) -> float:
    """

    :param img_original: Noise-free image
    :param img_noise: Noisy image
    :return: Peak signal to noise ratio
    """
    mse = (1 / np.size(img_original)) * np.sum((img_original - img_noise) ** 2)
    # max_intensity = np.max(img_original)
    max_intensity = 1.0
    return 20 * np.log10(max_intensity / np.sqrt(mse))


def convolve_image(img, kernel):
    """

    :param img: RGB/grayscale image to which we apply convolution.
    :param kernel: The matrix to use for the convolution operation.
    :return: RGB/grayscale image with the convolution operator applied.
    """
    img_conv = np.zeros(img.shape)
    if len(img.shape) == 3: # we are assuming this is RGB case
        img_conv[:, :, 0] = sp.signal.convolve2d(img[:, :, 0], kernel, mode="same")
        img_conv[:, :, 1] = sp.signal.convolve2d(img[:, :, 1], kernel, mode="same")
        img_conv[:, :, 2] = sp.signal.convolve2d(img[:, :, 2], kernel, mode="same")
    elif len(img.shape) == 2:
        img_conv = sp.signal.convolve2d(img[:, :], kernel, mode="same")
    else:
        raise ValueError(f"Unexpected image shape. Expected len(image.shape) equal to 2 or 3 but received {len(img.shape)}")
    return img_conv


def fast_noise_std_estimation(img):
    """
    :param img: An RGB image
    :return: An estimation for the standard deviation (std) of the gaussian noise applied.
    This estimation is achieved by calculating the average of the std for the three channels.
    Reference for the std approximation for a grayscale image:
        Reference: J. Immerkaer, “Fast Noise Variance Estimation”,
        Computer Vision and Image Understanding,
        Vol. 64, No. 2, pp. 300-302, Sep. 1996 [PDF]
    """
    kernel = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]])
    convolution_result = np.sum(np.abs(convolve_image(img, kernel)))
    factor = np.sqrt(np.pi / 2) / (3 * 6 * (img.shape[0] - 2) * (img.shape[1] - 2)) * convolution_result
    return factor


def tf_grad_x(image, gradient_type: GradientType):
    """

    :param image: tensor image of shape (1,H,W,C)
    :param gradient_type: the kernel associated to the finite difference formula
    we want to use.
    :return: the mapping of the image gradient along the x-axis
    using the finite approximation formula of type gradient_type.
    """
    # calculate gradient of a
    import tensorflow as tf
    tf_x_filter = tf.Variable(tf.zeros(shape=(3, 3)), trainable=False)

    if gradient_type.value == GradientType.FORWARD.value:
        tf_x_filter.assign(FWRD_X_GRADIENT)

    if gradient_type.value == GradientType.CENTERED.value:
        tf_x_filter.assign(CENTER_X_GRADIENT)

    if gradient_type.value == GradientType.BACKWARD.value:
        tf_x_filter.assign(BWRD_X_GRADIENT)

    tf_x_filter = tf.reshape(tf_x_filter, [3, 3, 1, 1])
    tf_img_red_x = tf.nn.conv2d(tf.expand_dims(image[:, :, :, 0], -1), tf_x_filter,
                                strides=[1, 1, 1, 1], padding="SAME")
    tf_img_green_x = tf.nn.conv2d(tf.expand_dims(image[:, :, :, 1], -1), tf_x_filter,
                                  strides=[1, 1, 1, 1], padding="SAME")
    tf_img_blue_x = tf.nn.conv2d(tf.expand_dims(image[:, :, :, 2], -1), tf_x_filter,
                                 strides=[1, 1, 1, 1], padding="SAME")
    return tf.concat([tf_img_red_x, tf_img_green_x, tf_img_blue_x], -1)


def tf_grad_y(image, gradient_type: GradientType):
    """

    :param image: tensor image of shape (1,H,W,C)
    :param gradient_type: the kernel associated to the finite difference formula
    we want to use.
    :return: the mapping of the image gradient along the y-axis
    using the finite approximation formula of type gradient_type.
    """
    import tensorflow as tf
    tf_x_filter = tf.Variable(tf.zeros(shape=(3, 3)), trainable=False)

    if gradient_type.value == GradientType.FORWARD.value:
        tf_x_filter.assign(FWRD_Y_GRADIENT)

    if gradient_type.value == GradientType.CENTERED.value:
        tf_x_filter.assign(CENTER_Y_GRADIENT)

    if gradient_type.value == GradientType.BACKWARD.value:
        tf_x_filter.assign(BWRD_Y_GRADIENT)

    tf_x_filter = tf.reshape(tf_x_filter, [3, 3, 1, 1])
    tf_img_red_x = tf.nn.conv2d(tf.expand_dims(image[:, :, :, 0], -1), tf_x_filter,
                                strides=[1, 1, 1, 1], padding="SAME")
    tf_img_green_x = tf.nn.conv2d(tf.expand_dims(image[:, :, :, 1], -1), tf_x_filter,
                                  strides=[1, 1, 1, 1], padding="SAME")
    tf_img_blue_x = tf.nn.conv2d(tf.expand_dims(image[:, :, :, 2], -1), tf_x_filter,
                                 strides=[1, 1, 1, 1], padding="SAME")
    return tf.concat([tf_img_red_x, tf_img_green_x, tf_img_blue_x], -1)


def relationship_coefficient(img1, img2):
    return np.sum(img1 * img2)
