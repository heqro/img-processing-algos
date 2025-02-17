#########################################################################################
#   Copyright (C) 2023 by Hector Iglesias <hr.iglesias.2018@alumnos.urjc.es>            #
#                                                                                       #
#   This program is free software; you can redistribute it and/or modify                #
#   it under the terms of the GNU General Public License as published by                #
#   the Free Software Foundation; either version 3 of the License, or                   #
#   (at your option) any later version.                                                 #
#                                                                                       #
#   This program is distributed in the hope that it will be useful,                     #
#   but WITHOUT ANY WARRANTY; without even the implied warranty of                      #
#   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the                       #
#   GNU General Public License for more details.                                        #
#                                                                                       #
#   You should have received a copy of the GNU General Public License                   #
#   along with this program; if not, write to the                                       #
#   Free Software Foundation, Inc.,                                                     #
#   51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA .                      #
#########################################################################################

from numpy import ndarray

import im_tools
from im_tools import GradientType
import numpy as np

from results_tools import plot_simple_image
from skimage.metrics import structural_similarity as ssim


def p_laplacian_denoising(im_noise, fidelity_coef: float, epsilon: float, p: float, dt: float, n_it: int,
                          mu: ndarray | None = None, im_orig=None, interactive=False, compute_ssim=False) -> dict:
    """
    Applies ``n_it`` iterations of the gradient method to the generalized Tikhonov regularization model.
    :param interactive: whether to stop after a few iterations.
    :param im_noise: an image with Gaussian noise n ~ N(0, sigma)
    :param fidelity_coef: lambda fidelity coefficient that multiplies the fidelity term.
    :param epsilon: tolerance to add to the gradient; useful if `p` = 1.
    :param p: the `p` parameter in the Tikhonov regularization model.
    :param dt: discretization step in time.
    :param n_it: number of iterations to apply of the gradient
    :param im_orig: the original image.
    :param mu: if ``im_orig is not None``, a list of proposed stoppage coefficients to achieve the maximum for the PSNR value.
    :return:
    """

    def p_energy() -> tuple[float, float, float]:
        """

        :return: The values for energy, prior and fidelity of the problem at the current iteration.
        """

        prior_value = (1 / p) * np.sum(np.sqrt(im_x ** 2 + im_y ** 2) ** p)
        fidelity_value = fidelity_coef * np.sum((im_approx - im_noise) ** 2) / 2
        energy_value = prior_value + fidelity_value
        return energy_value, prior_value, fidelity_value

    def verify_mass_conservation():
        """

        :return: The difference in mass between the current image and the initial image.
        It serves as another check for the correctness of the algorithm.
        """
        return np.sum(im_approx) - np.sum(im_noise)

    def get_results_dict() -> dict:
        """

        :return: A dictionary containing the results for the algorithm.
        """
        return {'energy': np.array(energy_values), 'prior': np.array(prior_values),
                'fidelity': np.array(fidelity_values), 'mass': np.array(mass_loss_values),
                'psnr': np.array(psnr_values), 'img_denoised': im_approx,
                'coefficients': proposed_coefficients, 'psnr_images': psnr_images,
                'ssim_indices': ssim_indices}

    # Initialization
    omega_size = im_noise.shape[0] * im_noise.shape[1] * im_noise.shape[2]
    energy_values = []
    prior_values = []
    fidelity_values = []
    mass_loss_values = []
    psnr_values = []
    im_approx = im_noise

    estimated_variance = im_tools.fast_noise_std_estimation(img=im_noise) ** 2 if mu is not None else -1
    proposed_coefficients = -np.ones(len(mu)) if mu is not None else []
    psnr_images = [None] * len(mu) if mu is not None else []
    max_psnr = False
    ssim_indices = []

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
            if compute_ssim:
                # clip numbers above 1 and below 0
                ssim_indices += [
                    ssim(np.clip(a=im_approx, a_min=0.0, a_max=1.0), im_orig, channel_axis=2, data_range=1.0)]

        #### Early stoppage (stop whenever psnr starts declining)
        if i > 2 and psnr_values[-1] < psnr_values[-2]:
            # proposed_coefficients = [i - 1]
            # psnr_images = None
            # break
            max_psnr = True
            if mu is None:
                break

        if mu is not None:
            threshold = 2 * fidelity_values[-1] / (fidelity_coef * estimated_variance * omega_size)
            indices = np.where(threshold < mu)[0]
            if len(indices) > 0:
                for index in indices:
                    proposed_coefficients[index] = i
                    psnr_images[index] = im_approx
            else:
                if max_psnr:
                    break

        if interactive and i % 10 == 0:
            plot_simple_image(im_approx)
            if input(f'It {i} - Press 0 to stop') == '0':
                break

        # Calculate next iteration
        lap = im_tools.div(img_x=im_x, img_y=im_y, p=p, epsilon=epsilon)
        # PDE calculation
        pde_value = lap - fidelity_coef * (im_approx - im_noise)
        # Gradient descent iteration
        im_approx = im_approx + dt * pde_value

    # Format return values
    return get_results_dict()
