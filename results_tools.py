import matplotlib.pyplot as plt
import numpy as np


def plot_denoising_images(images, show_plot=True, save_pdf=False, pdf_name=""):
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


def plot_denoising_results(img_orig, img_noise, img_denoised, energy, prior, fidelity, mass, time_step,
                           psnr, img_psnr, show_plot=True, save_pdf=False, pdf_name="", stop=-1):
    from matplotlib.gridspec import GridSpec
    def step2it(step):
        return step / time_step

    def it2step(it):
        return it * time_step

    fig = plt.figure(figsize=(30, 10))
    gs = GridSpec(nrows=2, ncols=4)

    x_axis = np.arange(len(energy)) * time_step

    ax0 = fig.add_subplot(gs[0, 0])
    ax0.imshow(img_orig)
    plt.axis('off')
    plt.title("Original image")
    plt.axis('scaled')

    ax1 = fig.add_subplot(gs[1, 0])
    ax1.imshow(img_noise)
    plt.axis('off')
    plt.title("Noisy image")
    plt.axis('scaled')

    ax2 = fig.add_subplot(gs[0, 1])
    ax2.imshow(img_psnr)
    plt.axis('off')
    plt.title("Denoised image (proposed stoppage)")
    plt.axis('scaled')

    ax2 = fig.add_subplot(gs[1, 1])
    ax2.imshow(img_denoised)
    plt.axis('off')
    plt.title("Denoised image (algorithm end)")
    plt.axis('scaled')

    ax3 = fig.add_subplot(gs[:, 3])
    plt.plot(x_axis, psnr)
    if stop != -1:
        plt.plot(x_axis[stop], psnr[stop], "s", label="proposed stoppage point")
    plt.plot(x_axis[np.argmax(psnr)], psnr[np.argmax(psnr)], "s", label="psnr maximum")
    plt.title("PSNR (dB)")
    plt.xlabel('time')
    sec_x = ax3.secondary_xaxis('top', functions=(step2it, it2step))
    sec_x.set_xlabel('iterations')
    plt.legend(loc="lower right")

    ax4 = fig.add_subplot(gs[:, 2])
    plt.plot(x_axis, energy, label="Energy")
    plt.plot(x_axis, prior, label="Prior")
    plt.plot(x_axis, fidelity, label="Fidelity")
    plt.plot(x_axis, mass, label="Mass")
    plt.legend(loc="upper right")
    plt.xlabel('time')

    sec_x = ax4.secondary_xaxis('top', functions=(step2it, it2step))
    sec_x.set_xlabel('iterations')

    if show_plot:
        plt.show()

    if save_pdf:
        from matplotlib.backends.backend_pdf import PdfPages
        pp = PdfPages(pdf_name + '.pdf')
        pp.savefig(fig)
        pp.close()


def plot_model_parameters(image, energy, prior, fidelity, mass, time_step,
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
            plt.plot(x_axis[stop], psnr[stop], "s", label="proposed stoppage point")
        plt.title("PSNR (dB)")
        plt.xlabel('time')
        sec_x = ax.secondary_xaxis('top', functions=(step2it, it2step))
        sec_x.set_xlabel('iterations')
        plt.legend(loc="lower right")

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


def print_psnr_data(psnr_values, proposed_stoppage_dict: dict):
    max_psnr = np.max(psnr_values)
    start_psnr = psnr_values[0]
    msg = f'{np.abs(max_psnr - start_psnr) / np.abs(start_psnr)}'  # max relative gain
    msg += f',{start_psnr}'  # starting psnr
    for entry in proposed_stoppage_dict.keys():
        value = psnr_values[proposed_stoppage_dict.get(entry)]
        msg += f',{np.abs(value - start_psnr) / np.abs(start_psnr)},'  # relative gain (coefficient)
        msg += f'{np.abs(max_psnr - value) / np.abs(max_psnr)}'  # relative loss (coefficient)
    msg += "\n"
    return msg


def plot_image_subtraction(img1, img2, title="Image subtraction results", show_plot=True, save_pdf=False, pdf_name=""):
    subtraction = 1 - np.abs(img1 - img2)
    fig = plt.figure()
    plt.imshow(subtraction)
    plt.axis('off')
    plt.title(title)
    plt.axis('scaled')

    if show_plot:
        plt.show()

    if save_pdf:
        from matplotlib.backends.backend_pdf import PdfPages
        pp = PdfPages(pdf_name + '.pdf')
        pp.savefig(fig)
        pp.close()


def plot_simple_image(img, show_plot=True, save_pdf=False, pdf_name="", cmap=None):
    fig = plt.figure()
    plt.imshow(img, cmap=cmap)
    plt.axis('off')
    plt.axis('scaled')
    plt.tight_layout()

    if show_plot:
        plt.show()

    if save_pdf:
        from matplotlib.backends.backend_pdf import PdfPages
        pp = PdfPages(pdf_name + '.pdf')
        pp.savefig(fig)
        pp.close()
    plt.close()


def plot_model_curves(energy, prior, fidelity, mass, time_step,
                      psnr_values, stop_dict: dict, title, show_plot=True, save_pdf=False, pdf_name=""):
    def step2it(step):
        return step / time_step

    def it2step(it):
        return it * time_step

    x_axis = np.arange(len(energy)) * time_step

    fig = plt.figure(figsize=(10, 10))

    ax = fig.add_subplot(1, 2, 1)

    plt.plot(x_axis, energy, label="Energy")
    plt.plot(x_axis, prior, label="Prior")
    plt.plot(x_axis, fidelity, label="Fidelity")
    plt.plot(x_axis, mass, label="Mass")
    plt.legend(loc="upper right")
    plt.xlabel('time')

    sec_x = ax.secondary_xaxis('top', functions=(step2it, it2step))
    sec_x.set_xlabel('iterations')

    ax = fig.add_subplot(1, 2, 2)
    plt.plot(x_axis, psnr_values)

    i = 1
    for key in stop_dict.keys():
        plt.plot(x_axis[stop_dict.get(key)], psnr_values[stop_dict.get(key)], marker=6 if i <= 10 else 7, label=key)
        i += 1
    plt.plot(x_axis[np.argmax(psnr_values)], psnr_values[np.argmax(psnr_values)], marker="o", color="black",
             label="Max")
    plt.title("PSNR (dB)")
    plt.xlabel('time')
    sec_x = ax.secondary_xaxis('top', functions=(step2it, it2step))
    sec_x.set_xlabel('iterations')
    plt.legend(loc="lower right")

    plt.suptitle(title)

    if show_plot:
        plt.show()

    if save_pdf:
        from matplotlib.backends.backend_pdf import PdfPages
        pp = PdfPages(f'{pdf_name}.pdf')
        pp.savefig(fig)
        pp.close()
    plt.close()


def plot_restriction(results: dict, dt: float, img_index: int, std: float, p: int, show_17: bool, show_plot=True,
                     save_pdf=False, pdf_name=""):
    import numpy as np

    # Plot zoomed in to verify restriction
    starting_index = int(np.min([np.argmax(results['psnr'])] + results['coefficients'].tolist()))

    fig = plt.figure(figsize=(15, 10))
    x_axis = np.arange(starting_index, len(results['udt']), 1)

    # Plot u_dt, prior, rest and sum curves
    sum_curve = results['udt'] + results['prior'][1:] - results['resto']

    plt.plot(x_axis, results['udt'][starting_index:], label=r"$\frac{1}{2} \frac{d \| u_k \|^2_2 (t)}{d t}$")
    plt.plot(x_axis, results['prior'][(starting_index + 1):], label=r"$\int_{\Omega} | \nabla u_k|^p$")
    plt.plot(x_axis, [-x for x in results['resto'][starting_index:]], label=r"$-\sigma_{\varepsilon} \lambda \int_{\Omega} u_k$")
    plt.plot(x_axis, sum_curve[starting_index:], label="sum")

    # Plot x-axis
    plt.axhline(y=0, color="black", lw=.35)

    for index in range(len(results['coefficients']) - 1 + show_17):
        x_index = int(results['coefficients'][index])
        plt.scatter([x_index] * 4, [results['udt'][x_index], results['prior'][x_index + 1], 
                                    sum_curve[x_index], -results['resto'][x_index],], marker=6 if index <= 10 else 7, label=f'synth_img_{index + 1}')
    psnr_index = int(np.argmax(results['psnr']))
    plt.scatter([psnr_index] * 4, [results['udt'][psnr_index], results['prior'][psnr_index + 1], -results['resto'][psnr_index],
                                sum_curve[psnr_index]], marker="o", color="black", label="Max")
    
    # Add labels
    plt.xlabel(r'iterations with $\Delta t=$' + f'{dt}')
    eqn = "\\frac{1}{2} \\frac{d \| u_k \|^2_2 (t)}{d t} + \int_{\Omega} | \\nabla u_k|^p - \sigma_{\\varepsilon} \lambda \int_{\Omega} u_k = 0"
    plt.title(r"Equation $" + eqn + "$" + f'. Case p={p}.\nAnalysis for img {img_index}. $\sigma={std}$.')
    plt.legend(loc=(1.04, 0))
    plt.tight_layout()

    if show_plot:
        plt.show()

    if save_pdf:
        from matplotlib.backends.backend_pdf import PdfPages
        pp = PdfPages(f'{pdf_name}.pdf')
        pp.savefig(fig)
        pp.close()
    plt.close()


def tf_plot_curves(results: dict, time_step: float, title: str, show_plot=True, save_pdf=False, pdf_name=""):
    def step2it(step):
        return step / time_step

    def it2step(it):
        return it * time_step

    x_axis = np.arange(len(results['energy'])) * time_step

    fig = plt.figure(figsize=(10, 10))

    ax = fig.add_subplot(1, 1, 1)

    plt.plot(x_axis, results['energy'], label="Energy")
    plt.plot(x_axis, results['prior'], label="Prior")
    plt.plot(x_axis, results['fidelity'], label="Fidelity")
    plt.plot(x_axis, results['mass'], label="Mass")
    plt.legend(loc="upper right")
    plt.xlabel('time')

    sec_x = ax.secondary_xaxis('top', functions=(step2it, it2step))
    sec_x.set_xlabel('iterations')

    if show_plot:
        plt.show()
