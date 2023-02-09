import preprocessing
import p_laplacian_denoising_algorithms
import results_tools

if __name__ == '__main__':
    ############################
    #   Algorithm parameters   #
    ############################
    p = 1
    epsilon = 1e-6
    fidelity_coefficient = 0.5
    time_step = 1e-2
    iterations = 40
    img_original = preprocessing.tf_load_normalized_image(path="test_images/dali.jpg")
    img_noise = preprocessing.tf_add_gaussian_noise(img_original, avg=0, std=0.1)

    import time

    start = time.time()
    print("Timing started")

    tf_im_approx, energy, prior, fidelity, mass, psnr, stop, psnr_image = p_laplacian_denoising_algorithms.tf_p_laplacian_denoising(
        tf_im_noise=img_noise, fidelity_coef=fidelity_coefficient,
        epsilon=epsilon,
        p=p, dt=time_step, n_it=iterations, tf_im_orig=img_original)

    results_tools.print_psnr_data(psnr, stop)
    # results_tools.plot_image_subtraction(img_noise[0], tf_im_approx[0], "TF - Removed noise (end of algorithm)")
    results_tools.plot_image_subtraction(img_noise[0], psnr_image[0], "TF - Removed noise (proposed stoppage)")
    # results_tools.plot_image_subtraction(img_original[0], tf_im_approx[0], "TF - Residual noise (end of algorithm)")
    results_tools.plot_image_subtraction(img_original[0], psnr_image[0], "TF - Residual noise (proposed stoppage)")
    results_tools.plot_denoising_results(img_orig=img_original[0], img_psnr=psnr_image[0], img_noise=img_noise[0],
                                         img_denoised=tf_im_approx[0], energy=energy, prior=prior, fidelity=fidelity,
                                         mass=mass, psnr=psnr, stop=stop, time_step=time_step)

    u, energy, prior, fidelity, mass, psnr, stop, psnr_image = p_laplacian_denoising_algorithms.p_laplacian_denoising(
        im_noise=img_noise[0], p=p,
        fidelity_coef=fidelity_coefficient,
        epsilon=epsilon, dt=time_step,
        n_it=iterations,
        im_orig=img_original[0])

    end = time.time()
    print("Time is", end - start)

    # results_tools.plot_denoising_images([img_original[0], im_noise_cast, im_approx_cast])

    # tf_im_approx_2, tf_im_noise_2 = p_laplacian_denoising_algorithms.tf_p_laplacian_denoising(tf_im_noise=img_noise,
    #                                                                          fidelity_coef=fidelity_coefficient,
    #                                                                          epsilon=epsilon,
    #                                                                          p=p, dt=time_step, n_it=iterations)
    # import numpy as np
    # print( np.not_equal(tf_im_approx_2.numpy().all(), tf_im_approx.numpy().all()))
    ####################
    #   Show results   #
    ####################
    results_tools.print_psnr_data(psnr, stop)
    results_tools.plot_image_subtraction(img_noise[0], u, "Removed noise (end of algorithm)")
    # results_tools.plot_image_subtraction(img_noise[0], psnr_image, "Removed noise (proposed stoppage)")
    results_tools.plot_image_subtraction(img_original[0], u, "Residual noise (end of algorithm)")
    # results_tools.plot_image_subtraction(img_original[0], psnr_image, "Residual noise (proposed stoppage)")
    # results_tools.plot_denoising_images([img_original, img_noise, u], save_pdf=True,
    #                                     pdf_name="hacking_around_results/Image at the end")
    # results_tools.plot_denoising_images([img_original, img_noise, psnr_image], save_pdf=True,
    #                                     pdf_name="hacking_around_results/Image (premature PSNR stoppage)")
    # results_tools.plot_model_parameters(u, energy, prior, fidelity, mass, time_step,
    #                                     psnr, stop=stop, image_psnr=psnr_image, save_pdf=True,
    #                                     pdf_name="hacking_around_results/Result")

    results_tools.plot_denoising_results(img_orig=img_original[0], img_psnr=psnr_image, img_noise=img_noise[0],
                                         img_denoised=u, energy=energy, prior=prior, fidelity=fidelity,
                                         mass=mass, psnr=psnr, stop=stop, time_step=time_step)
