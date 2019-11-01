import numpy as np
import cv2
import glob
import random
import datetime

from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.engine.saving import load_model
from keras.models import Model
from keras.layers import *
from keras.optimizers import Adam
import matplotlib.pyplot as plt
import keras.backend as kb
from livelossplot.keras import PlotLossesCallback

MAIN_DIR_PATH = "drive/My Drive/Colab Notebooks/assignment3"
IMG_WIDTH = 200
IMG_HEIGHT = 200


def plot_model_history(model_results):
    """
    Plots Loss and Accuracy graphs for the given model results.

    :param model_results: the model results from which to retrieve graph data
    """
    plt.figure(figsize=(8, 9))

    plt.subplot(2, 1, 1)
    if "loss" in model_results.history.keys():
        plt.plot(model_results.history["loss"], label="Training Loss")
    if "val_loss" in model_results.history.keys():
        plt.plot(model_results.history["val_loss"], label="Validation Loss")
    plt.ylabel("Loss")
    plt.xlabel("Epoch")
    plt.legend(loc="upper right")
    plt.grid(True)
    plt.title("Training and Validation Loss")

    plt.subplot(2, 1, 2)
    if "acc" in model_results.history.keys():
        plt.plot(model_results.history["acc"], label="Training Accuracy")
    if "val_acc" in model_results.history.keys():
        plt.plot(model_results.history["val_acc"], label="Validation Accuracy")
    if "dice_coef" in model_results.history.keys():
        plt.plot(model_results.history['dice_coef'], label="Dice Coefficient")
    plt.ylabel("Accuracy")
    plt.xlabel("Epoch")
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.title("Training and Validation Accuracy")

    plt.show()


def visualize_segmentations_with_contours(images, ground_masks, pred_masks, save_path=""):
    """
    Visualizes the predicted masks by drawing them as contours on their corresponding images.

    :param images: the list of images
    :param ground_masks: the list of ground truth masks for the image
    :param pred_masks: the list of predicted masks for the image
    :param save_path: path to save all the images (if empty, then images won't be saved)
    """
    for i in range(len(pred_masks)):
        p_mask = pred_masks[i].squeeze()
        p_mask[p_mask >= 0.5] = 255
        p_mask[p_mask < 0.5] = 0
        p_mask = p_mask.astype(np.uint8)

        # Apply cv2.threshold() to get a binary image
        _, thresh = cv2.threshold(p_mask, 50, 255, cv2.THRESH_BINARY)
        _, contours, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

        img = images[i].copy()
        for contour in contours:
            cv2.drawContours(img, contour, -1, (0, 255, 0), 1)

        figure = plt.figure(figsize=(10, 10))
        plt.subplot(231), plt.imshow(p_mask)
        plt.subplot(222), plt.imshow(img)

        #dice_index = dice_coef_np(ground_masks[i].squeeze(), p_mask)
        #plt.title("Dice Index: " + str(dice_index))

        plt.show()
        if save_path is not None or save_path != "":
            figure.savefig(save_path + "/result_" + str(i) + ".png", dpi=100, bbox_inches='tight')


def visualize_segmentations(images, ground_masks, pred_masks, save_path=""):
    """
    Visualizes the predicted masks by drawing the images, ground truth masks, and predicted masks separately in
    a single figure.

    :param images: the list of images
    :param ground_masks: the list of ground truth masks for the image
    :param pred_masks: the list of predicted masks for the image
    :param save_path: path to save all the images (if empty, then images won't be saved)
    """
    for i in range(len(pred_masks)):
        p_mask = pred_masks[i].squeeze()
        p_mask[p_mask >= 0.5] = 1
        p_mask[p_mask < 0.5] = 0

        figure = plt.figure(figsize=(10, 10))
        plt.subplot(231), plt.imshow(images[i].squeeze())
        plt.subplot(232), plt.imshow(ground_masks[i].squeeze())
        plt.subplot(233), plt.imshow(p_mask)

        #dice_index = dice_coef_np(ground_masks[i].squeeze(), p_mask)
        #plt.title("Dice Index: " + str(dice_index))

        plt.show()
        if save_path is not None or save_path != "":
            figure.savefig(save_path + "/result_" + str(i) + ".png", dpi=100, bbox_inches='tight')


def conv2d_block(name, x, filters, kernel_size, padding, activation):
    conv = Conv2D(name=name, filters=filters, kernel_size=kernel_size, padding=padding)(x)
    conv = BatchNormalization()(conv)
    conv = Activation(activation=activation)(conv)
    return conv


def circle_model(input_size):
    inputs = Input(name="input1", shape=input_size)
    conv1 = conv2d_block(name="conv1", x=inputs, filters=64, kernel_size=3, padding="same", activation="relu")
    maxpool1 = MaxPool2D(name="maxpool1", pool_size=(2, 2), strides=2)(conv1)

    conv2 = conv2d_block(name="conv2", x=maxpool1, filters=128, kernel_size=3, padding="same", activation="relu")
    maxpool2 = MaxPool2D(name="maxpool2", pool_size=(2, 2), strides=2)(conv2)

    conv3 = conv2d_block(name="conv3", x=maxpool2, filters=256, kernel_size=3, padding="same", activation="relu")
    upconv1 = UpSampling2D(name="upconv1", size=(2, 2))(conv3)

    concat1 = concatenate(name="concat1", inputs=[conv2, upconv1], axis=3)
    conv4 = conv2d_block(name="conv4", x=concat1, filters=128, kernel_size=3, padding="same", activation="relu")
    upconv2 = UpSampling2D(name="upconv2", size=(2, 2))(conv4)

    concat2 = concatenate(name="concat2", inputs=[conv1, upconv2], axis=3)
    conv5 = conv2d_block(name="conv5", x=concat2, filters=64, kernel_size=3, padding="same", activation="relu")

    conv6 = Conv2D(name="conv6", filters=1, kernel_size=1, activation="sigmoid")(conv5)
    u_model = Model(input=inputs, output=conv6)

    u_model.summary()

    return u_model


def test_model(model, test_x, test_y, visualize_func=None):
    """
    Tests the model on the given data and optionally visualizes it.

    :param model: the model to test with
    :param test_x: input data (ie. test_x)
    :param test_y: output data (ie. test_y)
    :param visualize_func: a function that takes (images, ground_masks, pred_masks, save_path) and visualizes them
    """
    pred_y = model.predict(test_x)

    if visualize_func is not None:
        visualize_func(test_x, test_y, pred_y, MAIN_DIR_PATH + "/results")


def train_model(train_x, train_y, model_name="circle_model", augment=False, lr=1e-4, epochs=10, valid_split=0,
                early_stop=False, model=None):
    """
    Returns a trained model and the training history.

    :param input_path: path to all the input data (ie. train_X)
    :param output_path: path to all the output data (ie. train_y)
    :param augment: whether or not the augment the data
    :param model_name: name of the new model (weights will be saved under this name)
    :param lr: learning rate for model
    :param epochs: number of epochs to run for
    :param valid_split: percentage of data that should be used for validation
    :param early_stop: whether or not to stop early if the validation and training curves diverge too much
    :param model: an existing model to train
    """
    if model is None:
        model = circle_model((IMG_WIDTH, IMG_HEIGHT, 1))
        model.compile(optimizer=Adam(lr=lr), loss="mean_squared_error", metrics=["accuracy"])

    # Setup training callbacks
    callbacks = []
    if early_stop:
        callbacks.append(EarlyStopping(monitor='val_loss', verbose=1, patience=50))
    callbacks.append(ModelCheckpoint(MAIN_DIR_PATH + "/" + model_name + ".hdf5", save_weights_only=True))
    callbacks.append(PlotLossesCallback())

    history = model.fit(train_x, train_y, epochs=epochs, validation_split=valid_split, callbacks=callbacks)

    plot_model_history(history)

    return model, history

