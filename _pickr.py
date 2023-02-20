# This file is only necessary because the multiprocessing module from
# _test_no_mask_denoising.py will fail otherwise
import im_tools
import p_laplacian_denoising_algorithms
import preprocessing
import results_tools
import coefficients_data_handler
import numpy as np

############################
#   Algorithm parameters   #
############################
p = 2
epsilon = 0
time_step = 1e-2
iterations = 5000
fidelity_coef = 0.5


def process_img(img_index: int):
    img_original = preprocessing.load_normalized_image(path=f'noise_test_images/img_{img_index}')
    H, W, C = img_original.shape

    header = 'img_index,width,height,noise_std,noise_std_est,avg_mass_loss,t_diff,max_rel_gain'
    for index in range(1, 11):
        header += f',rel_gain_synth_{index},rel_error_synth_{index}'
    with open(f'natural_images_analysis/img_{img_index}/analysis.csv', mode='w') as file:
        file.writelines(f'{header}\n')

    for noise in [0.05, 0.1, 0.15]:
        img_noise = preprocessing.add_gaussian_noise(img_original, avg=0, std=noise)
        coefficients = np.zeros(10)
        noise_std_estimation = im_tools.fast_noise_std_estimation(img_noise)
        for index in range(1, 11):
            my_spline = coefficients_data_handler.get_spline(index, noise_std_estimation)
            coefficients[index - 1] = my_spline(W, H)[0, 0]
        print("COEFFICIENTS FOR NOISE", coefficients, noise)
        img_approx, energy, prior, fidelity, mass, psnr, stop, psnr_image = p_laplacian_denoising_algorithms.p_laplacian_denoising(
            im_noise=img_noise,
            fidelity_coef=fidelity_coef,
            epsilon=epsilon, p=p,
            n_it=iterations,
            im_orig=img_original,
            dt=time_step,
            mu=coefficients)
        stop_dict = {}
        for coef in range(len(stop)):
            stop_dict[f'synth_img_{coef + 1}'] = int(stop[coef])

        results_tools.plot_model_curves(energy=energy, prior=prior, fidelity=fidelity, time_step=time_step,
                                        stop_dict=stop_dict, psnr_values=psnr, mass=mass,
                                        title=f'Model curves for img {img_index}.\n Noise std={noise}',
                                        save_pdf=True, show_plot=False,
                                        pdf_name=f'natural_images_analysis/img_{img_index}/analysis_{img_index}_{noise}')
        with open(f'natural_images_analysis/img_{img_index}/analysis.csv', mode='a') as file:
            line = f'{img_index},{W},{H},{noise},{noise_std_estimation},{np.max(np.abs(mass)) / H * W * C},{time_step * len(mass)},'
            line += results_tools.print_psnr_data(psnr, stop_dict)
            file.writelines(line)
