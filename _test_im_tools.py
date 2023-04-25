import math
import unittest
import numpy as np

import im_tools
import preprocessing
from im_tools import GradientType


class MyTestCase(unittest.TestCase):
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
