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


def print_model_parameters(image, energy, prior, fidelity, mass, time_step,
                           psnr=None, image_psnr=None, show_plot=True, save_pdf=False, pdf_name="", stop=-1):
    def step2it(step):
        return step / time_step

    def it2step(it):
        return it * time_step

    fig = plt.figure(figsize=(30, 10))

    rows = 1
    columns = 2 + (psnr is not None) + (image_psnr is not None)

    # Print image when the algorithm ends
    fig.add_subplot(rows, columns, 1)
    plt.imshow(image)
    plt.axis('off')
    plt.title("Result at end")
    plt.axis('scaled')

    # Print energy, prior, fidelity, mass values
    x_axis = np.arange(len(energy)) * time_step

    ax = fig.add_subplot(rows, columns, 2)
    plt.plot(x_axis, energy, label="Energy")
    plt.plot(x_axis, prior, label="Prior")
    plt.plot(x_axis, fidelity, label="Fidelity")
    plt.plot(x_axis, mass, label="Mass")
    plt.legend(loc="upper right")
    plt.xlabel('time')

    sec_x = ax.secondary_xaxis('top', functions=(step2it, it2step))
    sec_x.set_xlabel('iterations')

    # Print psnr data

    if psnr is not None:
        ax = fig.add_subplot(rows, columns, 3)
        plt.plot(x_axis, psnr)
        if stop != -1:
            plt.plot(x_axis[stop], psnr[stop], "s")
        plt.title("PSNR (dB)")
        plt.xlabel('time')
        sec_x = ax.secondary_xaxis('top', functions=(step2it, it2step))
        sec_x.set_xlabel('iterations')

    if image_psnr is not None:
        fig.add_subplot(rows, columns, 4)
        plt.imshow(image_psnr)
        plt.axis('off')
        plt.title("Result at proposed point")
        plt.axis('scaled')

    if show_plot:
        plt.show()

    if save_pdf:
        from matplotlib.backends.backend_pdf import PdfPages
        pp = PdfPages(pdf_name + '.pdf')
        pp.savefig(fig)
        pp.close()
