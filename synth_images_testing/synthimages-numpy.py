import numpy as np
import matplotlib.pyplot as plt

import sys

import p_laplacian_denoising_algorithms

sys.path.append('..')

# Now you can import modules from the parent directory

dimensions = 64 + np.arange(20)*32

for x_limit in dimensions:
    for y_limit in dimensions:
        x = np.linspace(0, 1, x_limit)
        y = np.linspace(0, 1, y_limit)
        X, Y = np.meshgrid(x, y)
        Z = np.exp(-(X - .5) ** 2 - (Y - .5) ** 2)
        plt.contourf(X, Y, Z, levels=240)
        plt.axis('off')
        plt.axis('scaled')
        plt.imsave("synth_img_"+str(x_limit)+"_"+str(y_limit)+".png", arr=Z, cmap=plt.cm.YlGnBu_r, format="png")
