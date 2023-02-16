import p_laplacian_denoising_algorithms
import preprocessing
import time
# import tf_denoising_algorithms
# from tf_denoising_algorithms import CostFunctionType
import results_tools
import coefficients_data_handler

############################
#   Algorithm parameters   #
############################
p = 2
epsilon = 0
time_step = 1e-2
iterations = 300
# img_original = preprocessing.tf_load_normalized_image(path="noise_test_images/img_79")
# img_noise = preprocessing.tf_add_gaussian_noise(img_original, avg=0, std=0.15)

#####################
#   Load database   #
#####################
my_spline = coefficients_data_handler.get_spline(1, 0.15)

# df = coefficients_data_handler.load_data(path="synth_images_testing/synth_img_1/results_log/coefficients.csv")
# _, H, W, C = img_noise.shape
# proposed_coefficient = coefficients_data_handler.get_stoppage_coefficient(df, height=H, width=W, noise_std=0.1)

# img_approx, energy, prior, fidelity, mass, psnr, stop, psnr_image, _ = tf_denoising_algorithms.tf_apply_denoising(tf_im_noise=img_noise,
#                                                                                                 tf_lambda=0.5,
#                                                                                                 epsilon=epsilon, p=p,
#                                                                                                 n_it=iterations,
#                                                                                                 tf_im_orig=img_original,
#                                                                                                 dt=time_step,
#                                                                                                 proposed_coefficient=-1,
#                                                                                                 cost_function_type=CostFunctionType.NO_MASK)
#
# results_tools.plot_denoising_results(img_orig=img_original[0], img_psnr=psnr_image[0], img_noise=img_noise[0],
#                                      img_denoised=img_approx[0], energy=energy, prior=prior, fidelity=fidelity,
#                                      mass=mass, psnr=psnr, stop=stop, time_step=time_step)

img_original = preprocessing.load_normalized_image(path="noise_test_images/img_78")
H, W, C = img_original.shape
coefficient = my_spline(W, H)[0, 0]
img_noise = preprocessing.add_gaussian_noise(img_original, avg=0, std=0.15)
img_approx, energy, prior, fidelity, mass, psnr, stop, psnr_image = p_laplacian_denoising_algorithms.p_laplacian_denoising(
    im_noise=img_noise,
    fidelity_coef=0.5,
    epsilon=epsilon, p=p,
    n_it=iterations,
    im_orig=img_original,
    dt=time_step,
    mu=coefficient)

results_tools.plot_denoising_results(img_orig=img_original, img_psnr=psnr_image, img_noise=img_noise,
                                     img_denoised=img_approx, energy=energy, prior=prior, fidelity=fidelity,
                                     mass=mass, psnr=psnr, stop=stop, time_step=time_step, save_pdf=True,
                                     pdf_name="SomeResult.pdf")
