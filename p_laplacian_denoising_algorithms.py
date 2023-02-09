import im_tools
from im_tools import GradientType
import numpy as np


def p_laplacian_denoising(im_noise, fidelity_coef: float, epsilon: float, p: float, dt: float, n_it: int, im_orig=None):
    def p_energy() -> tuple[float, float, float]:
        """

        :return: The values for energy, prior and fidelity of the problem at the current iteration.
        """

        prior_value = (1 / p) * np.sum(np.sqrt(im_x ** 2 + im_y ** 2) ** p) / omega_size
        fidelity_value = fidelity_coef * np.sum((im_approx - im_noise) ** 2) / omega_size
        energy_value = prior_value + fidelity_value
        return energy_value, prior_value, fidelity_value

    def verify_mass_conservation():
        """

        :return: The difference in mass between the current image and the initial image.
        It serves as another check for the correctness of the algorithm.
        """
        return (np.sum(im_approx) - np.sum(im_noise)) / omega_size

    # Initialization
    omega_size = im_noise.shape[0] * im_noise.shape[1]
    energy_values = []
    prior_values = []
    fidelity_values = []
    mass_loss_values = []
    psnr_values = []
    im_approx = im_noise

    estimated_variance = im_tools.fast_noise_std_estimation(img=im_noise) ** 2
    proposed_stop = -1
    psnr_image = None

    # fidelity = 0
    # while fidelity < estimated_variance: # early stoppage
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
        if fidelity < estimated_variance:
            proposed_stop = i
            psnr_image = im_approx

        # Calculate next iteration
        lap = im_tools.div(img_x=im_x, img_y=im_y, p=p, epsilon=epsilon)
        # PDE calculation
        pde_value = lap - fidelity_coef * (im_approx - im_noise)
        # Gradient descent iteration
        im_approx = im_approx + dt * pde_value
    return im_approx, np.array(energy_values), np.array(prior_values), np.array(fidelity_values), \
        np.array(mass_loss_values), np.array(psnr_values), proposed_stop, psnr_image


def tf_p_laplacian_denoising(tf_im_noise, fidelity_coef: float, epsilon: float, p: float, dt: float, n_it: int,
                             tf_im_orig=None):
    def tf_cost():
        tf_im_x = im_tools.tf_grad_x(tf_im_approx, GradientType.FORWARD)
        tf_im_y = im_tools.tf_grad_y(tf_im_approx, GradientType.FORWARD)

        mod = tf.sqrt(tf.square(tf_im_x) + tf.square(tf_im_y) + epsilon)

        prior_value = tf.reduce_sum(fidelity_coef * tf.pow(mod, p))
        fidelity_value = tf.reduce_sum(tf.square(tf_im_noise - tf_im_approx))
        energy_value = prior_value + fidelity_value

        return energy_value, fidelity_value, prior_value

    def tf_mass_conservation():
        return tf.reduce_sum(tf_im_noise) - tf.reduce_sum(tf_im_approx)

    import tensorflow as tf

    # Initialization
    omega_size = tf_im_noise.shape[1] * tf_im_noise.shape[2]
    energy_values = []
    prior_values = []
    fidelity_values = []
    mass_loss_values = []
    psnr_values = []
    opt = tf.keras.optimizers.SGD(learning_rate=dt)

    tf_im_approx = tf.Variable(tf.zeros(shape=tf_im_noise.shape),
                               name="tf_im_approx", trainable=True)
    tf_im_approx.assign(tf_im_noise)

    estimated_variance = im_tools.fast_noise_std_estimation(img=tf_im_approx[0]) ** 2
    proposed_stop = -1
    psnr_image = None

    for i in range(n_it):
        with tf.GradientTape() as g:
            energy, fidelity, prior = tf_cost()
            dCost_dIm_approx = g.gradient(energy, tf_im_approx)

        # Save values for plotting (if needed)
        energy_values += [energy.numpy() / omega_size]
        prior_values += [prior.numpy() / omega_size]
        fidelity_values += [fidelity.numpy() / omega_size]
        mass_loss_values += [tf_mass_conservation() / omega_size]

        opt.apply_gradients([(dCost_dIm_approx, tf_im_approx)])

        if tf_im_orig is not None:
            # psnr_values += [im_tools.psnr(tf_im_orig[0], tf_im_approx[0])]
            psnr_values += [tf.image.psnr(tf_im_orig, tf_im_approx, max_val=1.0)]

        if fidelity.numpy() < estimated_variance * omega_size:
            proposed_stop = i
            psnr_image = tf.constant(tf_im_approx)

    return tf_im_approx, np.array(energy_values), np.array(prior_values), np.array(fidelity_values), \
        np.array(mass_loss_values), np.array(psnr_values), proposed_stop, psnr_image
