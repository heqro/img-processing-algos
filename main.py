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
    img_original = preprocessing.load_normalized_image(path="test_images/person.jpg")
    img_noise = preprocessing.add_gaussian_noise(img_original, avg=0, std=0.1)

    import time

    start = time.time()
    print("Timing started")

    u, energy, prior, fidelity, mass, psnr, stop, psnr_image = p_laplacian_denoising_algorithms.p_laplacian_denoising(
        im_noise=img_noise, p=p,
        fidelity_coef=fidelity_coefficient,
        epsilon=epsilon, dt=time_step,
        n_it=iterations,
        im_orig=img_original)

    end = time.time()
    print("Time is", end - start)

    ####################
    #   Show results   #
    ####################
    results_tools.print_psnr_data(psnr, stop)
    results_tools.plot_image_subtraction(img_noise, u, "Removed noise (end of algorithm)")
    results_tools.plot_image_subtraction(img_noise, psnr_image, "Removed noise (early stoppage PSNR image)")
    results_tools.plot_image_subtraction(img_original, u, "Residual noise (end of algorithm)")
    results_tools.plot_image_subtraction(img_original, psnr_image, "Residual noise (early stoppage PSNR image)")
    # results_tools.plot_denoising_images([img_original, img_noise, u], save_pdf=True,
    #                                     pdf_name="hacking_around_results/Image at the end")
    # results_tools.plot_denoising_images([img_original, img_noise, psnr_image], save_pdf=True,
    #                                     pdf_name="hacking_around_results/Image (premature PSNR stoppage)")
    # results_tools.plot_model_parameters(u, energy, prior, fidelity, mass, time_step,
    #                                     psnr, stop=stop, image_psnr=psnr_image, save_pdf=True,
    #                                     pdf_name="hacking_around_results/Result")

    results_tools.plot_denoising_results(img_orig=img_original, img_psnr=psnr_image, img_noise=img_noise,
                                         img_denoised=u, energy=energy, prior=prior, fidelity=fidelity,
                                         mass=mass, psnr=psnr, stop=stop, time_step=time_step)
