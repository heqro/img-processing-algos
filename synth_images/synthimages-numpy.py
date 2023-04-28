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

import numpy as np
import matplotlib.pyplot as plt


dimensions = 64 + np.arange(10) * 64
index = 15

for x_limit in dimensions:
    for y_limit in dimensions:
        x = np.linspace(0, 1, x_limit)
        y = np.linspace(0, 1, y_limit)
        X, Y = np.meshgrid(x, y)
        Z = (1/(np.abs(X-.5)+1) + np.sqrt(Y/9) > .95) * np.tan(Y) * np.cos(X*Y) + np.tan(X) * ((X-.25)**2 + (Y-.75)**2 < .05)
        plt.contourf(X, Y, Z, levels=240)
        plt.axis('off')
        plt.axis('scaled')
        plt.imsave(f'synth_img_{index}/synth_img_{x_limit}_{y_limit}.png', arr=Z, cmap=plt.cm.YlGnBu_r, format="png")
        plt.show()
