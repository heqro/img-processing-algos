import coefficients_data_handler
import numpy as np
import matplotlib.pyplot as plt

import preprocessing

#####################
#   Load database   #
#####################
db_index = 1 # 2 3 4 8
p = 1
case='-fid'
df = coefficients_data_handler.load_data(
    path=f'synth_images_testing/synth_img_{db_index}/results_log/coefficientsP{p}{case}.csv')

dims = 64 + np.arange(10) * 64
x = dims
y = dims

X, Y = np.meshgrid(x, y)
noise_levels = [0.05, 0.10, 0.15]
Z = []
for noise in noise_levels:
    Z.append(np.zeros(X.shape))

for xx in range(len(x)):
    for yy in range(len(y)):
        for index in range(len(noise_levels)):  # load points for each noise level
            Z[index][xx, yy] = coefficients_data_handler.get_stoppage_coefficient(
                df, X[xx, yy], Y[xx, yy], noise_levels[index])

################################
# Plot points + bicubic spline #
################################
spline_funs = []
x_points = np.arange(64, 640, 0.5)
y_points = np.arange(64, 640, 0.5)
grid_X, grid_Y = np.meshgrid(x_points, y_points)
for index in range(len(noise_levels)): # load bicubic splines
    spline_fun = coefficients_data_handler.get_surface_function(db_index, noise_levels[index], p, case)
    spline_funs.append(spline_fun)

fig = plt.figure(figsize=(30, 7))
colors = ['green', 'goldenrod', 'red']

ax = fig.add_subplot(1, 1 + len(noise_levels), 1)
img = preprocessing.load_normalized_image(f'synth_images_testing/synth_img_{db_index}/synth_img_256_256.png')
ax.imshow(img)
ax.axis('off')
ax.axis('scaled')
plt.title(f'Synthetic image {db_index}')


for index in range(len(noise_levels)):
    ax = fig.add_subplot(1, 1 + len(noise_levels), index + 2, projection='3d')
    ax.scatter(X, Y, Z[index], c=colors[index], marker='o', label=f'std={noise_levels[index]}')
    ax.plot_surface(grid_X, grid_Y, spline_funs[index](grid_X / 640, grid_Y / 640), cmap='plasma', alpha=.5)
    ax.set_xticks(dims)
    ax.set_xlabel('width')
    ax.set_yticks(dims)
    ax.set_ylabel('height')
    ax.set_zlabel('coefficients')
    ax.invert_xaxis()
    ax.legend(loc="upper left")

fig.suptitle(f'PSNR coefficients surface\n Synthetic image {db_index}, p={p}. Fidelity mask.')

# Save to pdf
from matplotlib.backends.backend_pdf import PdfPages
plt.tight_layout()
pp = PdfPages(f'database_plots/synthetic_database_{db_index}_{p}_{case[1:]}.pdf')
pp.savefig(fig)
pp.close()

# Show plot

# plt.show()
