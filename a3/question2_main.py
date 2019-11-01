import numpy as np
from shapely.geometry.point import Point
from skimage.draw import circle_perimeter_aa
import matplotlib.pyplot as plt
from os import path
MAIN_DIR_PATH = "drive/My Drive/Colab Notebooks/assignment3"

import a3.circlenet as cn


def draw_circle(img, row, col, rad):
    rr, cc, val = circle_perimeter_aa(row, col, rad)
    valid = (
        (rr >= 0) &
        (rr < img.shape[0]) &
        (cc >= 0) &
        (cc < img.shape[1])
    )
    img[rr[valid], cc[valid]] = val[valid]


def noisy_circle(size, radius, noise):
    img = np.zeros((size, size), dtype=np.float)

    # Circle
    row = np.random.randint(size)
    col = np.random.randint(size)
    rad = np.random.randint(10, max(10, radius))
    draw_circle(img, row, col, rad)

    # Noise
    img += noise * np.random.rand(*img.shape)
    return (row, col, rad), img


def find_circle(model, img):
    # Fill in this function
    y_pred = model.predict(img.reshape(img.shape[0], img.shape[1], 1))
    plt.imshow(y_pred)
    plt.show()
    return 100, 100, 30


def iou(params0, params1):
    row0, col0, rad0 = params0
    row1, col1, rad1 = params1

    shape0 = Point(row0, col0).buffer(rad0)
    shape1 = Point(row1, col1).buffer(rad1)

    return (
        shape0.intersection(shape1).area /
        shape0.union(shape1).area
    )


def binary_circle_mask(size, params):
    """
    Generates a binary mask for the given circle.

    :param size: the size of the mask
    :param params: the parameters of the circle (row, col, rad)
    """
    img = np.zeros((size, size), dtype=np.float)
    row, col, rad = params
    draw_circle(img, row, col, rad)
    return img


def generate_predictor():
    if path.exists(MAIN_DIR_PATH + "/c_model.hdf5"):
        c_model = cn.circle_model((200, 200, 1))
        c_model.load_weights(MAIN_DIR_PATH + "/c_model.hdf5")
        return c_model

    train_x = []
    train_y = []
    # Start be getting some training data
    for i in range(1000):
        params, img = noisy_circle(200, 50, 2)
        mask = binary_circle_mask(200, params)

        train_x.append(img.reshape((img.shape[0], img.shape[1], 1)))
        train_y.append(mask.reshape((mask.shape[0], mask.shape[1], 1)))

    train_x = np.array(train_x)
    train_y = np.array(train_y)
    return cn.train_model(train_x=train_x, train_y=train_y, model_name="circle_model", augment=False, lr=1e-4, epochs=100,
                       valid_split=0.1, early_stop=True)[0]


def main(model=None):
    results = []
    if model is None:
      model = generate_predictor()
    for _ in range(1):
        params, img = noisy_circle(200, 50, 2)
        detected = find_circle(model, img)
        results.append(iou(params, detected))
    results = np.array(results)
    print((results > 0.7).mean())
