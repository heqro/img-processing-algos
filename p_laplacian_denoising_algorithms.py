from numpy import ndarray

import im_tools
from im_tools import GradientType
import numpy as np


def p_laplacian_denoising(im_noise, fidelity_coef: float, epsilon: float, p: float, dt: float, n_it: int, mu: ndarray,
                          im_orig=None):
    def p_energy() -> tuple[float, float, float]:
        """

        :return: The values for energy, prior and fidelity of the problem at the current iteration.
        """

        prior_value = (1 / p) * np.sum(np.sqrt(im_x ** 2 + im_y ** 2) ** p)
        fidelity_value = fidelity_coef * np.sum((im_approx - im_noise) ** 2)
        energy_value = prior_value + fidelity_value
        return energy_value, prior_value, fidelity_value

    def verify_mass_conservation():
        """

        :return: The difference in mass between the current image and the initial image.
        It serves as another check for the correctness of the algorithm.
        """
        return np.sum(im_approx) - np.sum(im_noise)

    # Initialization
    omega_size = im_noise.shape[0] * im_noise.shape[1] * im_noise.shape[2]
    energy_values = []
    prior_values = []
    fidelity_values = []
    mass_loss_values = []
    psnr_values = []
    im_approx = im_noise

    estimated_variance = im_tools.fast_noise_std_estimation(img=im_noise) ** 2
    proposed_coefficients = np.zeros(len(mu))
    psnr_images = [None] * len(mu)

    for i in range(n_it):
        im_x = im_tools.gradx(im_approx, GradientType.FORWARD)
        im_y = im_tools.grady(im_approx, GradientType.FORWARD)

        # Save values for current iteration
        energy, prior, fidelity = p_energy()
        energy_values += [energy]
        prior_values += [prior]
        fidelity_values += [fidelity]
        mass_loss_values += [verify_mass_conservation()]

        if im_orig is not None:
            psnr_values += [im_tools.psnr(im_orig, im_approx)]

        #### Early stoppage (stop whenever psnr starts declining)
        # if i > 2 and psnr_values[-1] < psnr_values[-2]:
        #     proposed_coefficients = [i - 1]
        #     psnr_images = None
        #     break

        threshold = fidelity_values[-1] / (fidelity_coef * estimated_variance * omega_size)
        indices = np.where(threshold < mu)[0]
        if len(indices) > 0:
            for index in indices:
                proposed_coefficients[index] = i
                psnr_images[index] = im_approx
        else:
            break

        # Calculate next iteration
        lap = im_tools.div(img_x=im_x, img_y=im_y, p=p, epsilon=epsilon)
        # PDE calculation
        pde_value = lap - fidelity_coef * (im_approx - im_noise)
        # Gradient descent iteration
        im_approx = im_approx + dt * pde_value
    return im_approx, np.array(energy_values), np.array(prior_values), np.array(fidelity_values), \
        np.array(mass_loss_values), np.array(psnr_values), proposed_coefficients, psnr_images

