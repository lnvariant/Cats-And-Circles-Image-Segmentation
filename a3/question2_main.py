import numpy as np
import cv2
from shapely.geometry.point import Point
from skimage.draw import circle_perimeter_aa
import matplotlib.pyplot as plt
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
    """
    Finds the circle in the given img using the model's prediction.

    Returns the (rows, cols, radius) of the circle, image with circle drawn, predicted mask mask by model

    :param model: the model to predict the circle
    :param img: the image on which to find the circle
    """
    row, col, rad = 0, 0, 0

    # Get prediction
    y_pred = model.predict(np.expand_dims(img.reshape(img.shape[0], img.shape[1], 1), axis=0))

    p_mask = y_pred.squeeze()
    p_mask[p_mask >= 0.5] = 255
    p_mask[p_mask < 0.5] = 0
    p_mask = p_mask.astype(np.uint8)

    # Apply cv2.threshold() to get a binary image
    _, thresh = cv2.threshold(p_mask, 50, 255, cv2.THRESH_BINARY)

    # Detect circles in the image
    circles = cv2.HoughCircles(thresh, method=cv2.HOUGH_GRADIENT, dp=1, minDist=thresh.shape[0] / 2, param1=10,
                               param2=8)

    n_img = img.copy()
    # Ensure at least some circles were found
    if circles is not None:
        # Convert the (x, y) coordinates and radius of the circles to integers
        circles = np.round(circles[0, :]).astype("int")
        # Loop over the (x, y) coordinates and radius of the circles
        for (x, y, r) in circles:
            row, col, rad = y, x, r
            cv2.circle(n_img, (x, y), r, (0, 255, 0), 2)

    return (row, col, rad), n_img, thresh


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
    return cn.train_model(train_x=train_x, train_y=train_y, model_name="circle_model", augment=False, lr=1e-4,
                          epochs=100,
                          valid_split=0.1, early_stop=True)


def main(model=None):
    results = []
    if model is None:
        model = generate_predictor()
    for i in range(10):
        params, img = noisy_circle(200, 50, 2)
        detected_params, new_img, mask = find_circle(model, img)

        int_over_uni = iou(params, detected_params)
        results.append(int_over_uni)

        figure = plt.figure(figsize=(10, 10))
        plt.subplot(231), plt.imshow(img)
        plt.subplot(232), plt.imshow(new_img)
        plt.subplot(233), plt.imshow(mask)
        plt.title("IOU: " + str(int_over_uni))
        plt.show()
        figure.savefig(cn.MAIN_DIR_PATH + "/result_" + str(i) + ".png", dpi=100, bbox_inches='tight')

    results = np.array(results)
    print("Mean IOU: " + str((results > 0.7).mean()))
