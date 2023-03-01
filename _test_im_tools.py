import math
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

    def test_tensorflow_gradients(self):
        from tensorflow import image, equal
        tf_img = preprocessing.tf_load_normalized_image('test_images/dali.jpg')
        grad_x_convolution = im_tools.tf_grad_x(tf_img, gradient_type=GradientType.FORWARD)
        grad_y_convolution = im_tools.tf_grad_y(tf_img, gradient_type=GradientType.FORWARD)
        grad_x, grad_y = image.image_gradients(tf_img)
        assert equal(grad_x.numpy().all(), grad_x_convolution.numpy().all())
        assert equal(grad_y.numpy().all(), grad_y_convolution.numpy().all())
    def test_fast_kernel_noise_variation(self):
        import time

        img_pool = 100
        std_values = np.arange(0.01, .51, 0.01)
        tests_amount = img_pool * len(std_values)
        successes = 0

        img_fail_index = []
        img_fail_value = []
        img_fail_estimation = []

        start = time.time()
        print("Timing started")
        for j in range(img_pool):
            img = preprocessing.load_normalized_image("noise_test_images/img_" + str(j))
            for i in np.arange(0.01, .51, 0.01):
                img_noise = preprocessing.add_gaussian_noise(img, 0, i)
                estimation = im_tools.fast_noise_std_estimation(img_noise)
                if math.isclose(estimation, i, abs_tol=0.01):
                    successes += 1
                else:
                    img_fail_index += [j]
                    img_fail_value += [i]
                    img_fail_estimation += [estimation]

        end = time.time()
        print("Time taken:", end - start)
        print("We have correctly estimated", successes, "out of", tests_amount, "noisy images, (",
              100 * successes / tests_amount, "%)")
        print("Failure results:")
        for index in range(len(img_fail_index)):
            print(img_fail_index[index], img_fail_value[index], img_fail_estimation[index])


if __name__ == '__main__':
    unittest.main()
