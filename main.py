import preprocessing
import p_laplacian_denoising_algorithms
import results_tools

if __name__ == '__main__':
    ############################
    #   Algorithm parameters   #
    ############################
    p = 1
    epsilon = 10 ** -6
    fidelity_coefficient = 0.5
    time_step = 10 ** -3
    iterations = 130
    img_original = preprocessing.load_normalized_image(path="test_images/dali.jpg")
    img_noise = preprocessing.add_gaussian_noise(img_original, avg=0, std=0.1)

    u, energy, prior, fidelity, mass, psnr, stop, psnr_image = p_laplacian_denoising_algorithms.p_laplacian_denoising(
        im_noise=img_noise, p=p,
        fidelity_coef=fidelity_coefficient,
        epsilon=epsilon, dt=time_step,
        n_it=iterations,
        im_orig=img_original)

    ####################
    #   Show results   #
    ####################
    results_tools.print_denoising_images([img_original, img_noise, u], save_pdf=True,
                                         pdf_name="hacking_around_results/Image at the end")
    results_tools.print_denoising_images([img_original, img_noise, psnr_image], save_pdf=True,
                                         pdf_name="hacking_around_results/Image (premature PSNR stoppage)")
    results_tools.print_model_parameters(u, energy, prior, fidelity, mass, time_step,
                                         psnr, stop=stop, image_psnr=psnr_image, save_pdf=True,
                                         pdf_name="hacking_around_results/Result")
