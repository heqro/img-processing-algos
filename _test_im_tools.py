#########################################################################################
#   Copyright (C) 2023 by Hector Iglesias <hr.iglesias.2018@alumnos.urjc.es>            #
#                                                                                       #
#   This program is free software; you can redistribute it and/or modify                #
#   it under the terms of the GNU General Public License as published by                #
#   the Free Software Foundation; either version 3 of the License, or                   #
#   (at your option) any later version.                                                 #
#                                                                                       #
#   This program is distributed in the hope that it will be useful,                     #
#   but WITHOUT ANY WARRANTY; without even the implied warranty of                      #
#   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the                       #
#   GNU General Public License for more details.                                        #
#                                                                                       #
#   You should have received a copy of the GNU General Public License                   #
#   along with this program; if not, write to the                                       #
#   Free Software Foundation, Inc.,                                                     #
#   51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA .                      #
#########################################################################################

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
