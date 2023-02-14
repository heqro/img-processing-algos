import time

import numpy as np

import im_tools
import preprocessing
import tf_denoising_algorithms
from tf_denoising_algorithms import CostFunctionType

dimensions = 64 + np.arange(10) * 64
p = 1
epsilon = 1e-6
fidelity_coefficient = 0.5
time_step = 1e-3
iterations = 1000

for x_limit in [512, 576, 640]:  # ideally you want "dimensions" in here
    print("Processing images of width:", x_limit)
    for noise in [0.05, 0.1, 0.15]:
        print("Applying noise of value:", noise)
        for y_limit in dimensions:
            path = 'synth_images_testing/synth_img_' + str(x_limit) + "_" + str(y_limit) + ".png"
            start = time.time()
            print("Analyzing image", path)
            im = preprocessing.tf_load_normalized_image(path)
            im_noise = preprocessing.tf_add_gaussian_noise(im, 0, noise)
            estimated_variance = im_tools.fast_noise_std_estimation(im_noise[0]) ** 2
            u, energy, prior, fidelity, mass, psnr, stop, _, _ = tf_denoising_algorithms.tf_apply_denoising(
                tf_im_noise=im_noise, p=p,
                tf_lambda=fidelity_coefficient,
                epsilon=epsilon, dt=time_step,
                n_it=iterations,
                tf_im_orig=im, cost_function_type=CostFunctionType.NO_MASK)
            end = time.time()
            print(end - start, estimated_variance / fidelity[stop])
