import numpy as np
import matplotlib.pyplot as plt


dimensions = 64 + np.arange(10) * 64
index = 10

for x_limit in dimensions:
    for y_limit in dimensions:
        x = np.linspace(0, 1, x_limit)
        y = np.linspace(0, 1, y_limit)
        X, Y = np.meshgrid(x, y)
        Z = 2*X**2 + 1/np.exp(Y**10)
        plt.contourf(X, Y, Z, levels=240)
        plt.axis('off')
        plt.axis('scaled')
        plt.imsave(f'synth_img_{index}/synth_img_{x_limit}_{y_limit}.png', arr=Z, cmap=plt.cm.YlGnBu_r, format="png")
