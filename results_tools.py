import matplotlib.pyplot as plt
import numpy as np


def print_denoising_images(images, show_plot=True, save_pdf=False, pdf_name=""):
    fig = plt.figure(figsize=(30, 10))

    rows = 1
    columns = 3

    titles = ["Original", "Original + noise", "Result"]

    for i in range(columns):
        fig.add_subplot(rows, columns, 1 + i)
        plt.imshow(images[i])
        plt.title(titles[i])
        plt.axis('off')
        plt.axis('scaled')

    if show_plot:
        plt.show()

    if save_pdf:
        from matplotlib.backends.backend_pdf import PdfPages
        pp = PdfPages(pdf_name + '.pdf')
        pp.savefig(fig)
        pp.close()


def print_model_parameters(image, energy, prior, fidelity, mass, iterations, time_step,
                           psnr=None, show_plot=True, save_pdf=False, pdf_name=""):
    fig = plt.figure(figsize=(30, 10))

    rows = 1
    columns = 2 + (psnr is not None)

    fig.add_subplot(rows, columns, 1)
    plt.imshow(image)
    plt.axis('off')
    plt.title("Result")
    plt.axis('scaled')

    x_axis = np.arange(iterations) * time_step

    fig.add_subplot(rows, columns, 2)
    plt.plot(x_axis, energy, label="Energy")
    plt.plot(x_axis, prior, label="Prior")
    plt.plot(x_axis, fidelity, label="Fidelity")
    plt.plot(x_axis, mass, label="Mass")
    plt.legend(loc="upper right")
    plt.xlabel('time')

    if psnr is not None:
        fig.add_subplot(rows, columns, 3)
        plt.plot(x_axis, psnr)
        plt.title("PSNR (dB)")
        plt.xlabel('time')

    if show_plot:
        plt.show()

    if save_pdf:
        from matplotlib.backends.backend_pdf import PdfPages
        pp = PdfPages(pdf_name + '.pdf')
        pp.savefig(fig)
        pp.close()
