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


if __name__ == '__main__':
    unittest.main()
