import tensorflow as tf
import numpy as np
from random import *
import random
import matplotlib.pyplot as plt


# Resizes all number images from 28x28 to 14x14
# Rounds each pixel above 128 to 2, and below 128 to -2
# Resizes all number images from 28x28 to 14x14
# Rounds each pixel above 128 to 1.0 and below to 0.0
def resize_number_image(number):
    new_number = np.zeros((14, 14))
    for i in range(14):
        for j in range(14):
            avg_val = (int(number[i*2, j*2]) + int(number[i*2 + 1, j*2]) + int(number[i*2, j*2 + 1]) + int(number[i*2 + 1, j*2 + 1])) / 4
            new_number[i, j] = 2 if avg_val > 60 else -2  # avg_val / 255

    return new_number


# Retrieves the Mnist dataset, rescales it and saves it as .npy file
def create_mnist_array(filename):
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    numbers_array = []
    for num in range(0, 10):
        print(num)
        num_arr = []
        for number, label in zip(x_train, y_train):
            if label == num:
                num_arr.append(np.asarray(resize_number_image(number).flatten()))
        numbers_array.append(np.asarray(num_arr))

    num_arr = np.asarray(numbers_array)

    np.save(filename, num_arr, allow_pickle=True)


# retrieves the array with the given name
def load_mnist_data(filename):
    return np.load(filename, allow_pickle=True)


# Hides part of a number image (given percentage)
def hide_img_partial(number_img, portion, img_shape=(14, 14)):
    min_row = int((1.0 - portion) * img_shape[0])
    for i in range(min_row, img_shape[0]):
        for j in range(img_shape[1]):
            number_img[i * img_shape[0] + j] = 2


# Flips random bits of the image with a given percentage (gaussian)
def add_img_noise(number_img, perc):
    for idx in range(len(number_img)):
        r_num = random.uniform(0, 1)
        number_img[idx] = 0 - number_img[idx] if r_num < perc else number_img[idx]


# returns a given amount of images per given label
def get_random_images(amnt, num_arr, labels=None):
    if labels is None:
        labels = list(range(10))

    images_arr = []
    labels_arr = []
    for label in labels:
        for _ in range(amnt):
            rand_idx = randint(0, len(num_arr[label])-1)
            images_arr.append(num_arr[label][rand_idx])
            labels_arr.append(label)

    return np.asarray(images_arr)


# Replaces 2 with 1.0 and -2 with 0.0 (useful for training a classifier)
def image_to_floats(number_img):
    return float(number_img > 0)


# converges to an image iteratively
def converge(pattern, weights, max_it=50):
    pattern = (np.sign(pattern) + 1) / 2
    dim_size = len(pattern)
    max_val = np.max(weights)
    weights -= max_val/2

    for _ in range(max_it):
        old_pat = pattern.copy()
        bits = list(range(dim_size))
        shuffle(bits)

        for b in bits:
            act = np.sum(weights[b] * pattern)
            if act > 0:
                pattern[b] = 1.0
            else:
                pattern[b] = 0.0

        if np.all(pattern == old_pat):
            return pattern


def convert_mOja_img(pattern):
    img = np.zeros((14, 14))
    for i in range(0, 14):
        for j in range(0, 14):
            if pattern[i * 14 + j] > 0:
                img[i, j] = 1.0
            else:
                img[i, j] = 0.0

    return img


# Plot multiple digit images
def show_digit_images(patterns, title, annotations=""):
    # Determine amount of rows and columns (row amnt. = col amnt.)
    pat_amnt = len(patterns)
    fig = plt.figure(figsize=(14, 14))
    fig.suptitle(title, fontsize=30)
    columns = int(np.sqrt(pat_amnt) + 0.5)
    rows = columns

    # Add every pattern as an image to the figure
    for i in range(1, pat_amnt+1):
        img = convert_mOja_img(patterns[i-1])
        fig.add_subplot(rows, columns, i)
        # Add the FFN prediction of the image as a green digit in the top corner
        if annotations != "":
            plt.text(0, 3, str(annotations[i-1]), color="lime", fontsize=30, fontstyle="oblique")
        plt.imshow(img, cmap=plt.get_cmap("gray"))
        #fig.set_size_inches(2, 2)
        #plt.savefig('digits1.png', dpi=1200)
        plt.axis('off')



