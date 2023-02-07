import numpy as np
import scipy as sp

from enum import Enum


class GradientType(Enum):
    FORWARD = 0,
    CENTERED = 1,
    BACKWARD = 2


def gradx(img, gradient_type: GradientType, use_convolution=False):
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
            img_x[:, :-1, :] = img[:, 1:, :] - img[:, :-1, :]
        if gradient_type.value == GradientType.CENTERED.value:
            img_x[:, 1:-1, :] = (img[:, 2:, :] - img[:, :-2, :]) / 2
        if gradient_type.value == GradientType.BACKWARD.value:
            img_x[:, 1:, :] = img[:, 1:, :] - img[:, :-1, :]
    else:
        if gradient_type.value == GradientType.FORWARD.value:
            img_x = convolve_image(img, np.array([[0, 0, 0],
                                                  [0, -1, 1],
                                                  [0, 0, 0]]))
        if gradient_type.value == GradientType.CENTERED.value:
            img_x = convolve_image(img, np.array([[0, 0, 0],
                                                  [0.5, 0, 0.5],
                                                  [0, 0, 0]]))
        if gradient_type.value == GradientType.BACKWARD.value:
            img_x = convolve_image(img, np.array([[0, 0, 0],
                                                  [-1, 1, 0],
                                                  [0, 0, 0]]))
    return img_x


def grady(img, gradient_type: GradientType, use_convolution=False):
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
            img_y[:-1, :, :] = img[1:, :, :] - img[:-1, :, :]
        if gradient_type.value == GradientType.CENTERED.value:
            img_y[1:-1, :, :] = (img[2:, :, :] - img[:-2, :, :]) / 2
        if gradient_type.value == GradientType.BACKWARD.value:
            img_y[1:, :, :] = img[1:, :, :] - img[-1:, :, :]
    else:
        if gradient_type.value == GradientType.FORWARD.value:
            img_y = convolve_image(img, np.array([[0, 1, 0],
                                                  [0, -1, 0],
                                                  [0, 0, 0]]))
        if gradient_type.value == GradientType.CENTERED.value:
            img_y = convolve_image(img, np.array([[0, 0.5, 0],
                                                  [0, 0, 0],
                                                  [0, -0.5, 0]]))
        if gradient_type.value == GradientType.BACKWARD.value:
            img_y = convolve_image(img, np.array([[0, 0, 0],
                                                  [0, 1, 0],
                                                  [0, -1, 0]]))
    return img_y


def div(img_x, img_y, gradient_type=GradientType.BACKWARD, p=2, epsilon=0):
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
    return gradx(np.multiply(img_x, mod_p), gradient_type) + grady(np.multiply(img_y, mod_p), gradient_type)


def psnr(img1, img2):
    mse = (1 / np.size(img1)) * np.sum((img1 - img2) ** 2)
    max_intensity = np.max(img1)
    return 20 * np.log10(max_intensity / np.sqrt(mse))


def convolve_image(img, matrix):
    img_x = np.zeros(img.shape)
    img_x[:, :, 0] = sp.signal.convolve2d(img[:, :, 0], matrix, mode="same")
    img_x[:, :, 1] = sp.signal.convolve2d(img[:, :, 1], matrix, mode="same")
    img_x[:, :, 2] = sp.signal.convolve2d(img[:, :, 2], matrix, mode="same")
    return img_x
