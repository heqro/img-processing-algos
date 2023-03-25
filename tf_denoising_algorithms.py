from numpy import ndarray

import im_tools
import results_tools
from im_tools import GradientType
import numpy as np
import tensorflow as tf
from enum import Enum


class CostFunctionType(Enum):
    NO_MASK = 0,
    FIDELITY_MASK = 1,
    DIFFUSION_MASK = 2,
    DIFFUSION_SQUARE_MASK = 3


def tf_calculate_grad_mod_p(tf_im_approx: tf.Variable, p=2.0, epsilon=0.0):
    tf_im_x, tf_im_y = tf.image.image_gradients(tf_im_approx)
    mod = tf.sqrt(tf.square(tf_im_x) + tf.square(tf_im_y) + epsilon)
    return tf.pow(mod, p)


def tf_constant_lambda_cost(tf_im_approx: tf.Variable, tf_im_noise: tf.constant, tf_lambda: float,
                            p=2.0, epsilon=0.0):
    grad_mod_p = tf_calculate_grad_mod_p(tf_im_approx, p, epsilon)

    prior_value = tf.reduce_sum(tf_lambda * grad_mod_p)
    fidelity_value = tf.reduce_sum(tf.square(tf_im_noise - tf_im_approx)) / 2
    energy_value = prior_value + fidelity_value

    return energy_value, fidelity_value, prior_value


def tf_fidelity_mask_cost(tf_im_approx: tf.Variable, tf_lambda: tf.Variable, tf_im_noise: tf.constant,
                          p=2.0, epsilon=0.0):
    grad_mod_p = tf_calculate_grad_mod_p(tf_im_approx, p, epsilon)

    prior_value = (1 / p) * tf.reduce_sum(grad_mod_p)
    fidelity_value = tf.reduce_sum(tf_lambda * tf.square(tf_im_noise - tf_im_approx)) / 2
    energy_value = prior_value + fidelity_value

    return energy_value, fidelity_value, prior_value


def tf_diffusion_mask_cost(tf_im_approx: tf.Variable, tf_lambda: tf.Variable, tf_im_noise: tf.constant,
                           p=2.0, epsilon=0.0):
    grad_mod_p = tf_calculate_grad_mod_p(tf_im_approx, p, epsilon)

    prior_value = (1 / p) * tf.reduce_sum(tf_lambda * grad_mod_p)
    fidelity_value = tf.reduce_sum(tf.square(tf_im_noise - tf_im_approx))
    energy_value = prior_value + fidelity_value

    return energy_value, fidelity_value, prior_value


def tf_diffusion_square_mask_cost(tf_im_approx: tf.Variable, tf_lambda: tf.Variable, tf_im_noise: tf.constant,
                                  p=2.0, epsilon=0.0):
    grad_mod_p = tf_calculate_grad_mod_p(tf_im_approx, p, epsilon)
    prior_value = (1 / p) * tf.reduce_sum(tf.square(tf_lambda) * grad_mod_p)
    fidelity_value = tf.reduce_sum(tf.square(tf_im_noise - tf_im_approx))
    energy_value = prior_value + fidelity_value

    return energy_value, fidelity_value, prior_value


def tf_mass_conservation(tf_im_approx: tf.Variable, tf_im_noise: tf.constant):
    return tf.reduce_sum(tf_im_noise) - tf.reduce_sum(tf_im_approx)


def tf_apply_denoising(tf_im_noise: tf.constant, tf_lambda: tf.Variable | float, dt: float, n_it: int,
                       p=2.0, epsilon=0.0, cost_function_type: CostFunctionType = CostFunctionType.NO_MASK,
                       tf_im_orig=None, interactive=True, mu: ndarray | None = None):
    def get_cost_function():
        if cost_function_type.value == CostFunctionType.NO_MASK.value:
            return tf_constant_lambda_cost
        if cost_function_type.value == CostFunctionType.FIDELITY_MASK.value:
            return tf_fidelity_mask_cost
        if cost_function_type.value == CostFunctionType.DIFFUSION_MASK.value:
            return tf_diffusion_mask_cost
        if cost_function_type.value == CostFunctionType.DIFFUSION_SQUARE_MASK.value:
            return tf_diffusion_square_mask_cost

    def get_tape_args():
        if cost_function_type.value != CostFunctionType.NO_MASK.value:
            return [tf_im_approx, tf_lambda]
        return [tf_im_approx]

    def get_pairs(derivatives):
        if cost_function_type.value != CostFunctionType.NO_MASK.value:
            return [(derivatives[0], tf_im_approx), (derivatives[1], tf_lambda)]
        return [(derivatives[0], tf_im_approx)]

    def get_results_dict() -> dict:
        return {'energy': np.array(energy_values), 'prior': np.array(prior_values),
                'fidelity': np.array(fidelity_values), 'mass': np.array(mass_loss_values),
                'psnr': np.array(psnr_values), 'img_denoised': tf_im_approx[0], 'mask': masks,
                'coefficients': proposed_coefficients, 'psnr_images': psnr_images,
                'udt': u_dt, 'resto': resto}
    
    def print_u_dt():
        u_dt.append((tf.reduce_sum(tf_im_approx**2) - tf.reduce_sum(im_prev**2)) / (2 * dt))
    def print_resto():
        sigma = im_tools.fast_noise_std_estimation(tf_im_approx[0])
        resto.append(sigma * np.sum(tf_lambda * tf_im_approx))

    # Initialization
    omega_size = tf_im_noise.shape[1] * tf_im_noise.shape[2] * tf_im_noise.shape[3]
    # Return values
    energy_values = []
    prior_values = []
    fidelity_values = []
    mass_loss_values = []
    psnr_values = []
    # Restriction
    u_dt, resto = [], []
    # Tflow initialization
    opt = tf.keras.optimizers.SGD(learning_rate=dt)
    tf_cost = get_cost_function()

    tf_im_approx = tf.Variable(tf.zeros(shape=tf_im_noise.shape),
                               name="tf_im_approx", trainable=True)
    tf_im_approx.assign(tf_im_noise)
    im_prev = tf.constant(tf_im_approx)
    tape_args = get_tape_args()
    opt.build(tape_args)

    estimated_variance = im_tools.fast_noise_std_estimation(tf_im_noise[0]) ** 2 if mu is not None else -1
    proposed_coefficients = -np.ones(len(mu)) if mu is not None else []
    psnr_images = [None] * len(mu) if mu is not None else []
    max_psnr = False
    masks = [None] * len(mu) if mu is not None else [tf_lambda]

    for i in tf.range(n_it):
        with tf.GradientTape() as tape:
            energy, fidelity, prior = tf_cost(tf_im_approx=tf_im_approx, tf_im_noise=tf_im_noise,
                                              tf_lambda=tf_lambda, p=p, epsilon=epsilon)
            energy_values += [energy.numpy()]
            prior_values += [prior.numpy()]
            fidelity_values += [fidelity.numpy()]
            mass_loss_values += [tf_mass_conservation(tf_im_noise=tf_im_noise, tf_im_approx=tf_im_approx)]

            derivatives_cost = tape.gradient(energy, tape_args)

        opt.apply_gradients(get_pairs(derivatives_cost))

        if tf_im_orig is not None:
            psnr_values += [tf.image.psnr(tf_im_orig, tf_im_approx, max_val=1.0)]

        if i > 2 and psnr_values[-1] < psnr_values[-2]:
            max_psnr = True
            if mu is None:
                break

        if interactive and i % 10 == 0:
            results_tools.plot_simple_image(tf_im_approx[0])
            if input(f'It {i} - Press 0 to stop') == '0':
                break

        if mu is not None:
            threshold = 2 * fidelity_values[-1] / (tf_lambda * estimated_variance * omega_size) # assume tf_lambda starts equal to 1
            indices = np.where(threshold < mu)[0]
            if len(indices) > 0:
                for index in indices:
                    proposed_coefficients[index] = i
                    psnr_images[index] = tf.constant(tf_im_approx)
                    masks[index] = tf.constant(tf_lambda)
            else:
                if max_psnr:
                    break
        print_u_dt()
        print_resto()
        im_prev = tf.constant(tf_im_approx)

    return get_results_dict()
