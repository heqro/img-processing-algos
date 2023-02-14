import coefficients_data_handler
import numpy as np
import matplotlib.pyplot as plt

#####################
#   Load database   #
#####################
df = coefficients_data_handler.load_data(path="synth_images_testing/results_log/coefficients.csv")

dims = 64 + np.arange(10) * 64
x = dims
y = dims

X, Y = np.meshgrid(x, y)
Z_1 = np.zeros(X.shape)
Z_2 = np.zeros(X.shape)
Z_3 = np.zeros(X.shape)

for xx in range(len(dims)):
    for yy in range(len(dims)):
        Z_1[xx, yy] = coefficients_data_handler.get_stoppage_coefficient(df, X[xx, yy], Y[xx, yy], 0.05)
        Z_2[xx, yy] = coefficients_data_handler.get_stoppage_coefficient(df, X[xx, yy], Y[xx, yy], 0.1)
        Z_3[xx, yy] = coefficients_data_handler.get_stoppage_coefficient(df, X[xx, yy], Y[xx, yy], 0.15)

fig = plt.figure(figsize=(7, 7))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X, Y, Z_3, c='red', marker='o', label="std=0.15")
ax.scatter(X, Y, Z_1, c='green', marker='o', label="std=0.1")
ax.scatter(X, Y, Z_2, c='orange', marker='o', label="std=0.05")
ax.set_xticks(dims)
ax.set_yticks(dims)
ax.legend(loc="upper left")
plt.title("Proposed PSNR stoppage linear coefficients")

from matplotlib.backends.backend_pdf import PdfPages

pp = PdfPages('database_plots/AllPointsTogether.pdf')
pp.savefig(fig)
pp.close()

plt.show()
