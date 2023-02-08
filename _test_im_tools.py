import unittest
import numpy as np

import im_tools
import preprocessing
from im_tools import GradientType


class MyTestCase(unittest.TestCase):
    def test_img_x_gradient(self):
        img = preprocessing.load_normalized_image('test_images/dali.jpg')
        im_x_no_conv = im_tools.gradx(img, gradient_type=GradientType.FORWARD, use_convolution=False)
        im_x_conv = im_tools.gradx(img, gradient_type=GradientType.FORWARD, use_convolution=True)
        assert np.equal(im_x_conv.all(), im_x_no_conv.all())
        im_x_no_conv = im_tools.gradx(img, gradient_type=GradientType.BACKWARD, use_convolution=False)
        im_x_conv = im_tools.gradx(img, gradient_type=GradientType.BACKWARD, use_convolution=True)
        assert np.equal(im_x_conv.all(), im_x_no_conv.all())
        im_x_no_conv = im_tools.gradx(img, gradient_type=GradientType.CENTERED, use_convolution=False)
        im_x_conv = im_tools.gradx(img, gradient_type=GradientType.CENTERED, use_convolution=True)
        assert np.equal(im_x_conv.all(), im_x_no_conv.all())

    def test_img_y_gradient(self):
        img = preprocessing.load_normalized_image('test_images/dali.jpg')
        im_y_no_conv = im_tools.grady(img, gradient_type=GradientType.FORWARD, use_convolution=False)
        im_y_conv = im_tools.grady(img, gradient_type=GradientType.FORWARD, use_convolution=True)
        assert np.equal(im_y_conv.all(), im_y_no_conv.all())
        im_y_no_conv = im_tools.grady(img, gradient_type=GradientType.BACKWARD, use_convolution=False)
        im_y_conv = im_tools.grady(img, gradient_type=GradientType.BACKWARD, use_convolution=True)
        assert np.equal(im_y_conv.all(), im_y_no_conv.all())
        im_y_no_conv = im_tools.grady(img, gradient_type=GradientType.CENTERED, use_convolution=False)
        im_y_conv = im_tools.grady(img, gradient_type=GradientType.CENTERED, use_convolution=True)
        assert np.equal(im_y_conv.all(), im_y_no_conv.all())

    def test_fast_kernel_noise_variation(self):
        img = preprocessing.load_normalized_image('test_images/dali.jpg')
        for i in range(5, 1000, 5):
            img_noise = preprocessing.add_gaussian_noise(img, 0, i / 100)
            noise_estimation = im_tools.fast_noise_std_estimation(img_noise)
            np.testing.assert_approx_equal(noise_estimation, i / 100, significant=1)

    def test_tensorflow_grad_x(self):
        img = preprocessing.load_normalized_image('test_images/dali.jpg')
        tf_img = preprocessing.tf_load_normalized_image('test_images/dali.jpg')

        im_x_conv = im_tools.gradx(img, gradient_type=GradientType.FORWARD)
        im_x_conv = np.reshape(im_x_conv, newshape=tf_img.shape)
        assert np.equal(im_x_conv.all(), im_tools.tf_grad_x(tf_img, gradient_type=GradientType.FORWARD).numpy().all())

        im_x_conv = im_tools.gradx(img, gradient_type=GradientType.CENTERED)
        im_x_conv = np.reshape(im_x_conv, newshape=tf_img.shape)
        assert np.equal(im_x_conv.all(), im_tools.tf_grad_x(tf_img, gradient_type=GradientType.CENTERED).numpy().all())

        im_x_conv = im_tools.gradx(img, gradient_type=GradientType.FORWARD)
        im_x_conv = np.reshape(im_x_conv, newshape=tf_img.shape)
        assert np.equal(im_x_conv.all(), im_tools.tf_grad_x(tf_img, gradient_type=GradientType.BACKWARD).numpy().all())

    def test_tensorflow_grad_y(self):
        img = preprocessing.load_normalized_image('test_images/dali.jpg')
        tf_img = preprocessing.tf_load_normalized_image('test_images/dali.jpg')

        im_y_conv = im_tools.grady(img, gradient_type=GradientType.FORWARD)
        im_y_conv = np.reshape(im_y_conv, newshape=tf_img.shape)
        assert np.equal(im_y_conv.all(), im_tools.tf_grad_y(tf_img, gradient_type=GradientType.FORWARD).numpy().all())

        im_y_conv = im_tools.grady(img, gradient_type=GradientType.CENTERED)
        im_y_conv = np.reshape(im_y_conv, newshape=tf_img.shape)
        assert np.equal(im_y_conv.all(), im_tools.tf_grad_y(tf_img, gradient_type=GradientType.CENTERED).numpy().all())

        im_y_conv = im_tools.grady(img, gradient_type=GradientType.FORWARD)
        im_y_conv = np.reshape(im_y_conv, newshape=tf_img.shape)
        assert np.equal(im_y_conv.all(), im_tools.tf_grad_y(tf_img, gradient_type=GradientType.BACKWARD).numpy().all())


if __name__ == '__main__':
    unittest.main()
