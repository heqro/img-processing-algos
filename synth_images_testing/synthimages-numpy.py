import numpy as np
import matplotlib.pyplot as plt

import sys

import p_laplacian_denoising_algorithms

sys.path.append('..')

# Now you can import modules from the parent directory
import preprocessing

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
#
x = np.linspace(0, 1, 450)
y = np.linspace(0, 1, 450)
X, Y = np.meshgrid(x, y)
Z = np.exp(-(X - .5) ** 2 - (Y - .5) ** 2)

def z_to_img(Z):
    channel = Z / 3
    return np.concatenate((channel, channel, channel), axis=0).reshape((450, 450,3))
    # return a

im_orig = z_to_img(Z)

# im_dirty = preprocessing.add_gaussian_noise(Z, 0, 0.15)
# ax.plot_surface(X, Y, Z, cmap=plt.cm.YlGnBu_r)
#
#
# plt.show()



# dirty_conversed = z_to_img(im_dirty)
# print(dirty_conversed.shape)
fig = plt.figure()
ax = fig.add_subplot()
plt.imshow(im_orig)
# # plt.contour(X, Y, Z, levels=120)
plt.show()
# orig_conversed = z_to_img(dirty_conversed)
#
# im_approx, energy, prior, fidelity, mass, psnr, stop, psnr_image = p_laplacian_denoising_algorithms.p_laplacian_denoising(
#     dirty_conversed, 0.5, 0, 2, 1e-2, 40, orig_conversed
# )
#
# #
# fig = plt.figure()
# ax = fig.add_subplot()
# plt.imshow
# # plt.contour(X, Y, Z, levels=120)
# plt.show()
# plt.contourf(X, Y, im_approx, levels=120)
#
# plt.show()
