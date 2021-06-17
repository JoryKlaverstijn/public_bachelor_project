import numpy as np
from memristor_continuous.formulas import *
from memristor_continuous.extra import *
from memristor_continuous.figures.alphanum5x5 import *
from memristor_continuous.blur import *
import matplotlib.pyplot as plt

# Image parameters
width = 5
height = 5

img = digits[1]

new_img = blur_image(img, blur3x3[2])
print(new_img)

f1, ax1 = plt.subplots(1, 2)
ax1[0].imshow(array_to_image(img.flatten(), height, width, False, True))
ax1[1].imshow(array_to_image(new_img.flatten(), height, width, False, True))
pyplot.show()

