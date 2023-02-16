import sys
import time

import numpy as np

import im_tools
import preprocessing
# import tf_denoising_algorithms
# from tf_denoising_algorithms import CostFunctionType
import p_laplacian_denoising_algorithms

dimensions = 64 + np.arange(10) * 64
p = 2
epsilon = 0
fidelity_coefficient = 0.5
time_step = 1e-2
iterations = 5000
noises = [0.05, 0.1, 0.15]

import pandas as pd

cols = {'width': [], 'height': [], 'noise_1': [], 'noise_1_coef': [], 'noise_2': [], 'noise_2_coef': [], 'noise_3': [],
        'noise_3_coef': []}
df = pd.DataFrame(cols)
######## TENSORFLOW VERSION
# for x_limit in dimensions:  # ideally you want "dimensions" in here
#     print("Processing images of width:", x_limit)
#     for y_limit in dimensions:
#         print("Height:", y_limit)
#         noise_coefs = [0] * len(noises)
#         for i in range(len(noises)):
#             noise = noises[i]
#             print("Applying noise of value:", noise)
#             path = 'synth_images_testing/synth_img_4/synth_img_' + str(x_limit) + "_" + str(y_limit) + ".png"
#             start = time.time()
#             im = preprocessing.tf_load_normalized_image(path)
#             im_noise = preprocessing.tf_add_gaussian_noise(im, 0, noise)
#             estimated_variance = im_tools.fast_noise_std_estimation(im_noise[0]) ** 2
#             u, energy, prior, fidelity, mass, psnr, stop, _, _ = tf_denoising_algorithms.tf_apply_denoising(
#                 tf_im_noise=im_noise, p=p,
#                 tf_lambda=fidelity_coefficient,
#                 epsilon=epsilon, dt=time_step,
#                 n_it=iterations,
#                 tf_im_orig=im, cost_function_type=CostFunctionType.NO_MASK)
#             end = time.time()
#             print(end - start, estimated_variance / fidelity[stop])
#             noise_coefs[i] = estimated_variance / fidelity[stop]
#         new_row = pd.Series({'width': x_limit, 'height': y_limit, 'noise_1': noises[0],
#                              'noise_1_coef': noise_coefs[0], 'noise_2': noises[1],
#                              'noise_2_coef': noise_coefs[1], 'noise_3': noises[2],
#                              'noise_3_coef': noise_coefs[2]})
#         temp_df = pd.concat([df, new_row.to_frame().T], ignore_index=True)
#         temp_df.to_csv('synth_images_testing/synth_img_4/results_log/coefficientsP2.csv', mode='a', index=False, header=False)
# df.to_csv('synth_images_testing/synth_img_3/results_log/coefficients.csv', index=False)  # if running for the first time
# df.to_csv('synth_images_testing/synth_img_1/results_log/coefficientsP2.csv', mode='a', index=False, header=False)  # else
#
#
stdout = sys.stdout
import os

# Get the current working directory
cwd = os.getcwd()

# Print the current working directory
print("Saving file to:", cwd)


# for index in [2, 3, 4]:
#     with open('output' + str(index) + '.txt') as f:
#         sys.stdout = f
#         for x_limit in dimensions:  # ideally you want "dimensions" in here
#             print("Processing images of width:", x_limit)
#             for y_limit in dimensions:
#                 print("height:", y_limit)
#                 noise_coefs = [0] * len(noises)
#                 omega_size = x_limit * y_limit * 3
#                 for i in range(len(noises)):
#                     noise = noises[i]
#                     print("Applying noise of value:", noise)
#                     path = 'synth_images_testing/synth_img_' + str(index) + '/synth_img_' + str(x_limit) + "_" + str(
#                         y_limit) + ".png"
#                     start = time.time()
#                     im = preprocessing.load_normalized_image(path)
#                     im_noise = preprocessing.add_gaussian_noise(im, 0, noise)
#                     # estimated_variance = im_tools.fast_noise_std_estimation(im_noise) ** 2
#                     u, energy, prior, fidelity, mass, psnr, stop, _ = p_laplacian_denoising_algorithms.p_laplacian_denoising(
#                         im_noise=im_noise, p=p,
#                         fidelity_coef=fidelity_coefficient,
#                         epsilon=epsilon, dt=time_step,
#                         n_it=iterations,
#                         im_orig=im)
#                     end = time.time()
#                     noise_coefs[i] = fidelity[stop] / (noise ** 2 * omega_size * fidelity_coefficient)
#                     print('T', end - start, 'Coef', noise_coefs[i], 'PSNR_end', psnr[-1], 'PSNR_start',
#                           psnr[0], 'it', len(fidelity))
#                 new_row = pd.Series({'width': x_limit, 'height': y_limit, 'noise_1': noises[0],
#                                      'noise_1_coef': noise_coefs[0], 'noise_2': noises[1],
#                                      'noise_2_coef': noise_coefs[1], 'noise_3': noises[2],
#                                      'noise_3_coef': noise_coefs[2]})
#                 temp_df = pd.concat([df, new_row.to_frame().T], ignore_index=True)
#                 temp_df.to_csv('synth_images_testing/synth_img_' + str(index) + '/results_log/coefficientsP2.csv',
#                                mode='a', index=False,
#                                header=False)
#         sys.stdout = stdout

def process_synth_img(index):
    # with open('output' + str(index) + '.txt') as f:
    #     sys.stdout = f
    for x_limit in dimensions:  # ideally you want "dimensions" in here
        # print("Processing images of width:", x_limit)
        for y_limit in dimensions:
            # print("height:", y_limit)
            noise_coefs = [0] * len(noises)
            omega_size = x_limit * y_limit * 3
            for i in range(len(noises)):
                noise = noises[i]
                # print("Applying noise of value:", noise)
                path = 'synth_images_testing/synth_img_' + str(index) + '/synth_img_' + str(x_limit) + "_" + str(
                    y_limit) + ".png"
                start = time.time()
                im = preprocessing.load_normalized_image(path)
                im_noise = preprocessing.add_gaussian_noise(im, 0, noise)
                # estimated_variance = im_tools.fast_noise_std_estimation(im_noise) ** 2
                u, energy, prior, fidelity, mass, psnr, stop, _ = p_laplacian_denoising_algorithms.p_laplacian_denoising(
                    im_noise=im_noise, p=p,
                    fidelity_coef=fidelity_coefficient,
                    epsilon=epsilon, dt=time_step,
                    n_it=iterations,
                    im_orig=im)
                end = time.time()
                noise_coefs[i] = fidelity[stop] / (noise ** 2 * omega_size * fidelity_coefficient)
                print('W_H_ID_NOISE',x_limit,y_limit,index,noise,'T', end - start, 'Coef', noise_coefs[i], 'PSNR_end', psnr[-1], 'PSNR_start',
                      psnr[0], 'it', len(fidelity))
            new_row = pd.Series({'width': x_limit, 'height': y_limit, 'noise_1': noises[0],
                                 'noise_1_coef': noise_coefs[0], 'noise_2': noises[1],
                                 'noise_2_coef': noise_coefs[1], 'noise_3': noises[2],
                                 'noise_3_coef': noise_coefs[2]})
            temp_df = pd.concat([df, new_row.to_frame().T], ignore_index=True)
            temp_df.to_csv('synth_images_testing/synth_img_' + str(index) + '/results_log/coefficientsP2.csv',
                           mode='a', index=False,
                           header=False)
        # sys.stdout = stdout


import concurrent.futures

indexes = [1, 8, 9, 10]
num_threads = 4
with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
    futures = []
    for i in range(num_threads):
        future = executor.submit(process_synth_img, indexes[i])
        futures.append(future)
    concurrent.futures.wait(futures)
