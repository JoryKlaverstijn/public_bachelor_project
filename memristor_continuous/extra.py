import numpy as np
from matplotlib import image
from matplotlib import pyplot
from memristor_continuous.formulas import *


# Load in images
def load_image_jpg(path, color=False):
    img_mario_tmp = np.asarray(image.imread(path))
    img_mario = img_mario_tmp.reshape(-1, img_mario_tmp.shape[-1])
    size = len(img_mario)
    img_mario_array = []
    for rgb in img_mario:
        if not color:
            avg = float(int(rgb[0]) + int(rgb[1]) + int(rgb[2]))/3
        if color:
            img_mario_array.append(float(rgb[0]))
            img_mario_array.append(float(rgb[1]))
            img_mario_array.append(float(rgb[2]))
        else:
            img_mario_array.append(avg)

    return np.asarray(img_mario_array)

# Convert gray values to an image of rgb values
def array_to_image(gray, width, height, color=False, norm=False):
    image = np.zeros(shape=(height, width, 3))
    ind = 0
    row = 0
    for row in range(0, width):
        for col in range(0, height):
            if not color:
                image[row, col, 0] = float(gray[col+row*width]) / (1 + ((1-norm) * 255))
                image[row, col, 1] = float(gray[col+row*width]) / (1 + ((1-norm) * 255))
                image[row, col, 2] = float(gray[col+row*width]) / (1 + ((1-norm) * 255))
            if color:
                image[row, col, 0] = float(gray[(col + row * width) * 3 + 0]) / (1 + ((1-norm) * 255))
                image[row, col, 1] = float(gray[(col + row * width) * 3 + 1]) / (1 + ((1-norm) * 255))
                image[row, col, 2] = float(gray[(col + row * width) * 3 + 2]) / (1 + ((1-norm) * 255))

    return image


# Add noise to gray values and clamp between 0 and 255
def add_noise(gray, noise):
    new_gray = gray.copy()
    for i in range(len(new_gray)):
        new_val = new_gray[i] + np.random.normal(0, noise, 1) * new_gray[i]
        new_val = min(max(new_val, 0), 255)
        new_gray[i] = float(new_val)

    return new_gray


# Hide part of the image
def hide_image(gray, perc, value=0):
    new_gray = gray.copy()
    for i in range(len(new_gray)):
        if i > len(new_gray) * (1-perc):
            new_gray[i] = float(value)

    return new_gray


# Print the energy of the network with current state
def print_energy(beta, attractors, pattern, it):
    print(f"Iteration {it + 0},", "Energy is: ", calc_energy(beta, attractors, pattern))

