import im_tools
from im_tools import GradientType
import numpy as np


def p_laplacian_denoising(im_noise, fidelity_coef: float, epsilon: float, p: float, dt: float, n_it: int, im_orig= None):
    def p_energy() -> tuple[float, float, float]:
        """

        :return: The values for energy, prior and fidelity.
        """
        omega_size = im_approx.shape[0] * im_approx.shape[1]
        im_x = im_tools.gradx(im_approx, gradient_type=GradientType.FORWARD)
        im_y = im_tools.grady(im_approx, gradient_type=GradientType.FORWARD)

        prior = (1 / p) * np.sum(np.sqrt(im_x ** 2 + im_y ** 2) ** p) / omega_size
        fidelity = fidelity_coef * np.sum((im_approx - im_noise) ** 2) / omega_size
        energy = prior + fidelity
        return energy, prior, fidelity

    # Initialization
    energy_values = np.zeros(n_it)
    prior_values = np.zeros(n_it)
    fidelity_values = np.zeros(n_it)
    im_approx = im_noise

    for i in range(n_it):
        print('Iteración',i)
        im_x = im_tools.gradx(im_approx, GradientType.FORWARD)
        im_y = im_tools.grady(im_approx, GradientType.FORWARD)
        lap = im_tools.div(im_x, im_y, GradientType.CENTERED, p, epsilon)
        # PDE calculation
        pde_value = lap - fidelity_coef * (im_approx - im_noise)
        # Gradient descent iteration
        im_approx = im_approx + dt * pde_value
        # Save values
        energy_values[i], prior_values[i], fidelity_values[i] = p_energy()
        print('En:', energy_values[i], 'Pr:', prior_values[i], 'Fi:', fidelity_values[i], end='')
        if im_orig is not None:
            print(" PSNR", im_tools.psnr(im_approx, im_orig), end='')
        print('')
    return im_approx
