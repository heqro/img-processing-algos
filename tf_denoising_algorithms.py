import im_tools
from im_tools import GradientType
import numpy as np
import tensorflow as tf
from enum import Enum


class CostFunctionType(Enum):
    NO_MASK = 0,
    FIDELITY_MASK = 1,
    DIFFUSION_MASK = 2


def tf_calculate_grad_mod_p(tf_im_approx: tf.Variable, p=2.0, epsilon=0.0):
    tf_im_x = im_tools.tf_grad_x(tf_im_approx, GradientType.FORWARD)
    tf_im_y = im_tools.tf_grad_y(tf_im_approx, GradientType.FORWARD)

    mod = tf.sqrt(tf.square(tf_im_x) + tf.square(tf_im_y) + epsilon)
    return tf.pow(mod, p)


def tf_constant_lambda_cost(tf_im_approx: tf.Variable, tf_im_noise: tf.constant, tf_lambda: float,
                            p=2.0, epsilon=0.0):
    grad_mod_p = tf_calculate_grad_mod_p(tf_im_approx, p, epsilon)

    prior_value = tf.reduce_sum(tf_lambda * grad_mod_p)
    fidelity_value = tf.reduce_sum(tf.square(tf_im_noise - tf_im_approx))
    energy_value = prior_value + fidelity_value

    return energy_value, fidelity_value, prior_value


def tf_fidelity_mask_cost(tf_im_approx: tf.Variable, tf_lambda: tf.Variable, tf_im_noise: tf.constant,
                          p=2.0, epsilon=0.0):
    grad_mod_p = tf_calculate_grad_mod_p(tf_im_approx, p, epsilon)

    prior_value = (1 / p) * tf.reduce_sum(grad_mod_p)
    fidelity_value = tf.reduce_sum(tf_lambda * tf.square(tf_im_noise - tf_im_approx))
    energy_value = prior_value + fidelity_value

    return energy_value, fidelity_value, prior_value


def tf_diffusion_mask_cost(tf_im_approx: tf.Variable, tf_lambda: tf.Variable, tf_im_noise: tf.constant,
                           p=2.0, epsilon=0.0):
    grad_mod_p = tf_calculate_grad_mod_p(tf_im_approx, p, epsilon)

    prior_value = (1 / p) * tf.reduce_sum(tf_lambda * grad_mod_p)
    fidelity_value = tf.reduce_sum(tf.square(tf_im_noise - tf_im_approx))
    energy_value = prior_value + fidelity_value

    return energy_value, fidelity_value, prior_value


def tf_mass_conservation(tf_im_approx: tf.Variable, tf_im_noise: tf.constant):
    return tf.reduce_sum(tf_im_noise) - tf.reduce_sum(tf_im_approx)


def tf_apply_denoising(tf_im_noise: tf.constant, tf_lambda: tf.Variable | float, dt: float, n_it: int,
                       p=2.0, epsilon=0.0, cost_function_type: CostFunctionType = CostFunctionType.NO_MASK,
                       tf_im_orig=None, proposed_coefficient=-1.0):
    def get_cost_function():
        if cost_function_type.value == CostFunctionType.NO_MASK.value:
            return tf_constant_lambda_cost
        if cost_function_type.value == CostFunctionType.FIDELITY_MASK.value:
            return tf_fidelity_mask_cost
        if cost_function_type.value == CostFunctionType.DIFFUSION_MASK.value:
            return tf_diffusion_mask_cost

    def get_tape_args():
        if cost_function_type.value != CostFunctionType.NO_MASK.value:
            return [tf_im_approx, tf_lambda]
        return [tf_im_approx]

    def get_pairs(derivatives):
        if cost_function_type.value != CostFunctionType.NO_MASK.value:
            return [(derivatives[0], tf_im_approx), (derivatives[1], tf_lambda)]
        return [(derivatives[0], tf_im_approx)]

    # Initialization
    omega_size = tf_im_noise.shape[1] * tf_im_noise.shape[2]
    # Return values
    energy_values = []
    prior_values = []
    fidelity_values = []
    mass_loss_values = []
    psnr_values = []
    # Tflow initialization
    opt = tf.keras.optimizers.SGD(learning_rate=dt)
    tf_cost = get_cost_function()

    tf_im_approx = tf.Variable(tf.zeros(shape=tf_im_noise.shape),
                               name="tf_im_approx", trainable=True)
    tf_im_approx.assign(tf_im_noise)

    tape_args = get_tape_args()
    opt.build(tape_args)

    estimated_variance = im_tools.fast_noise_std_estimation(img=tf_im_approx[0]) ** 2
    proposed_stop = -1
    psnr_image = None

    for i in tf.range(n_it):
        print("Iteration", i.numpy())
        with tf.GradientTape() as tape:
            energy, fidelity, prior = tf_cost(tf_im_approx=tf_im_approx, tf_im_noise=tf_im_noise,
                                              tf_lambda=tf_lambda, p=p, epsilon=epsilon)
            energy_values += [energy.numpy() / omega_size]
            prior_values += [prior.numpy() / omega_size]
            fidelity_values += [fidelity.numpy() / omega_size]
            mass_loss_values += [tf_mass_conservation(tf_im_noise=tf_im_noise, tf_im_approx=tf_im_approx) / omega_size]

            derivatives_cost = tape.gradient(energy, tape_args)

        opt.apply_gradients(get_pairs(derivatives_cost))

        if tf_im_orig is not None:
            psnr_values += [tf.image.psnr(tf_im_orig, tf_im_approx, max_val=1.0)]

        #### Early stoppage (stop whenever psnr starts declining)
        # if i.numpy() > 2 and psnr_values[-1] < psnr_values[-2]:
        #     proposed_stop = i.numpy() - 1
        #     psnr_image = None
        #     return tf_im_approx, np.array(energy_values), np.array(prior_values), np.array(fidelity_values), \
        #         np.array(mass_loss_values), np.array(psnr_values), proposed_stop, psnr_image, tf_lambda

        if proposed_coefficient * fidelity.numpy() < estimated_variance * omega_size: # if proposed coefficient == -1 then we run to the end
            proposed_stop = i.numpy()
            psnr_image = tf.constant(tf_im_approx)
    return tf_im_approx, np.array(energy_values), np.array(prior_values), np.array(fidelity_values), \
        np.array(mass_loss_values), np.array(psnr_values), proposed_stop, psnr_image, tf_lambda
