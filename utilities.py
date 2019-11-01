import matplotlib.pyplot as plt
import numpy as np


def show_image(img, title="", cmap=""):
    plt.title(title)

    if cmap.lstrip().rstrip() == "":
        plt.imshow(img)
    else:
        plt.imshow(img, cmap=cmap)

    plt.show()


def norm(a, p):
    if type(a) == np.ndarray:
        return np.power(np.power(np.abs(a), p).sum(-1), 1/p)
    else:
        return np.power(np.power(np.abs(a), p).sum(-1)[0], 1/p)


def addRandNoise(image, magnitude):
    """
    Adds uniformly distributed random noise to the image.
    The magnitude would be the standard deviation.
    :param image: the image to add noise to
    :param magnitude: the interval of the normal distribution
    :return:
    """
    img_m = np.array(image)
    # We generate a matrix (of same shape as image) that contains
    # uniformly distributed values and add them to the image
    noisy_img = img_m + np.random.normal(magnitude * -1, magnitude, img_m.shape)
    return noisy_img
