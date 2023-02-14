import preprocessing
import time
import tf_denoising_algorithms
from tf_denoising_algorithms import CostFunctionType
import results_tools

############################
#   Algorithm parameters   #
############################
p = 1
epsilon = 1e-6
time_step = 1e-3
iterations = 1000
img_original = preprocessing.tf_load_normalized_image(path="synth_images_testing/synth_img_512_352.png")
img_noise = preprocessing.tf_add_gaussian_noise(img_original, avg=0, std=0.1)

start = time.time()
print("Timing started")

tf_im_approx, energy, prior, fidelity, mass, psnr, stop, psnr_image, mask = tf_denoising_algorithms.tf_apply_denoising(
    tf_im_noise=img_noise, tf_lambda=preprocessing.tf_add_mask(img_original, .5),
    epsilon=epsilon,
    p=p, dt=time_step, n_it=iterations, tf_im_orig=img_original, cost_function_type=CostFunctionType.DIFFUSION_MASK)

end = time.time()
print("Time is", end - start)

mask = mask.numpy()[0]

results_tools.plot_denoising_results(img_orig=img_original[0], img_psnr=psnr_image[0], img_noise=img_noise[0],
                                     img_denoised=tf_im_approx[0], energy=energy, prior=prior, fidelity=fidelity,
                                     mass=mass, psnr=psnr, stop=stop, time_step=time_step)

results_tools.plot_simple_image(mask)
