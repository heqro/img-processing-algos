import preprocessing
import time
import tf_denoising_algorithms
from tf_denoising_algorithms import CostFunctionType
import results_tools
import coefficients_data_handler

############################
#   Algorithm parameters   #
############################
p = 1
epsilon = 1e-6
time_step = 1e-3
iterations = 700
img_original = preprocessing.tf_load_normalized_image(path="noise_test_images/img_79")
img_noise = preprocessing.tf_add_gaussian_noise(img_original, avg=0, std=0.15)

#####################
#   Load database   #
#####################
df = coefficients_data_handler.load_data(path="synth_images_testing/results_log/coefficients.csv")
_, H, W, C = img_noise.shape
proposed_coefficient = coefficients_data_handler.get_stoppage_coefficient(df, height=H, width=W, noise_std=0.1)

img_approx, energy, prior, fidelity, mass, psnr, stop, psnr_image, _ = tf_denoising_algorithms.tf_apply_denoising(tf_im_noise=img_noise,
                                                                                                tf_lambda=0.5,
                                                                                                epsilon=epsilon, p=1,
                                                                                                n_it=iterations,
                                                                                                tf_im_orig=img_original,
                                                                                                dt=time_step,
                                                                                                proposed_coefficient=proposed_coefficient,
                                                                                                cost_function_type=CostFunctionType.NO_MASK)

results_tools.plot_denoising_results(img_orig=img_original[0], img_psnr=psnr_image[0], img_noise=img_noise[0],
                                     img_denoised=img_approx[0], energy=energy, prior=prior, fidelity=fidelity,
                                     mass=mass, psnr=psnr, stop=stop, time_step=time_step)
