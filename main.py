import matplotlib.pyplot as plt
import preprocessing
import processing_algorithms

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    img_original = preprocessing.load_normalized_image(path="test_images/dali.jpg")
    img_noise = preprocessing.add_gaussian_noise(img_original, avg=0, std=0.1)

    # u = processing_algorithms.p_laplacian_denoising(im_noise=img_noise,p=2,fidelity_coef=0.5,
    #                                                epsilon=0,dt=10**-2, n_it=800,
    #                                                 im_orig=img_original)
    u = processing_algorithms.p_laplacian_denoising(im_noise=img_noise, p=1, fidelity_coef=0.5,
                                                    epsilon=10 ** -6, dt=10 ** -3,
                                                    n_it=200, im_orig=img_original)

    fig = plt.figure(figsize=(30, 10))

    # setting values to rows and column variables
    rows = 1
    columns = 3

    fig.add_subplot(rows, columns, 1)
    plt.imshow(img_original)
    plt.axis('off')
    plt.title("Original")
    plt.axis('scaled')
    # plt.savefig('orign')

    fig.add_subplot(rows, columns, 2)
    plt.imshow(img_noise)
    plt.axis('off')
    plt.title("Noise")
    plt.axis('scaled')

    # plt.savefig('noise')

    fig.add_subplot(rows, columns, 3)
    plt.imshow(u)
    plt.axis('off')
    plt.title("Sol")
    plt.axis('scaled')

    # plt.savefig('sol')
    plt.show()
