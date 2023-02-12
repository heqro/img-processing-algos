import numpy as np
import preprocessing
import p_laplacian_denoising_algorithms
import results_tools

dimensions = 64 + np.arange(20) * 32
p = 1
epsilon = 1e-6
fidelity_coefficient = 0.5
time_step = 1e-3
iterations = 130

# for x_limit in dimensions:
#     for y_limit in dimensions:


####### "HOMEMADE" VERSION #######
# Load image
im = preprocessing.load_normalized_image('synth_images_testing/synth_img_256_256.png')
# Apply gaussian noise
im_noise = preprocessing.add_gaussian_noise(im, 0, 0.05)
# Apply p-laplacian denoising algorithm
u, energy, prior, fidelity, mass, psnr, stop, psnr_image = p_laplacian_denoising_algorithms.p_laplacian_denoising(
    im_noise=im_noise, p=p,
    fidelity_coef=fidelity_coefficient,
    epsilon=epsilon, dt=time_step,
    n_it=iterations,
    im_orig=im)
# [Show image], save values
# results_tools.plot_denoising_images([im, im_noise, u])
results_tools.plot_denoising_results(img_orig=im, img_noise=im_noise, img_denoised=u,
                                     energy=energy, prior=prior, fidelity=fidelity, mass=mass,
                                     time_step=time_step, psnr=psnr, img_psnr=psnr_image, stop=stop)

####### TENSORFLOW VERSION #######
# Load image
im = preprocessing.tf_load_normalized_image('synth_images_testing/synth_img_256_256.png')
# Apply gaussian noise
im_noise = preprocessing.tf_add_gaussian_noise(im, 0, 0.05)
# Apply p-laplacian denoising algorithm
u, energy, prior, fidelity, mass, psnr, stop, psnr_image = p_laplacian_denoising_algorithms.tf_p_laplacian_denoising(
    tf_im_noise=im_noise, p=p,
    fidelity_coef=fidelity_coefficient,
    epsilon=epsilon, dt=time_step,
    n_it=iterations,
    tf_im_orig=im)
# [Show image], save values
# results_tools.plot_denoising_images([im, im_noise, u])
results_tools.plot_denoising_results(img_orig=im[0], img_noise=im_noise[0], img_denoised=u[0],
                                     energy=energy, prior=prior, fidelity=fidelity, mass=mass,
                                     time_step=time_step, psnr=psnr, img_psnr=psnr_image[0], stop=stop)
