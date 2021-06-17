import numpy as np
from memristor_continuous.formulas import *
from memristor_continuous.extra import *
import matplotlib.pyplot as plt



# Image parameters
width = 30
height = 30
color = True

# Hyper parameters
beta = 2e-3
iterations = 1

# Load in images as array of brightness values (0-255)
bunny = load_image_jpg('figures/bunny.jpg', color)
cat = load_image_jpg('figures/cat.jpg', color)
dog = load_image_jpg('figures/dog.jpg', color)
lizard = load_image_jpg('figures/lizard.jpg', color)
mouse = load_image_jpg('figures/mouse.jpg', color)
rabbit = load_image_jpg('figures/rabbit.jpg', color)
snake = load_image_jpg('figures/snake.jpg', color)
tiger = load_image_jpg('figures/tiger.jpg', color)

# Stored patterns
attractors = [bunny, cat, lizard, rabbit, snake, mouse, tiger, dog]
attractors = np.asarray(attractors)

# Input pattern
pattern = np.asarray(hide_image(add_noise(tiger, 0.25), 0.0, 0.0))
original_pattern = pattern.copy()

# Print general information
print("Amount of attractors:\t", attractors.shape[0])
max_dist, max_vec = max_distance(attractors.T)
print("Max vector distance: Pattern", max_vec, "with distance", max_dist)

# Print how separated each pattern is from the rest
print()
for i in range(attractors.shape[0]):
    min_sep, attrac = min_separation(attractors.T, i)
    print("Min separation pattern", i, ":", (i, attrac), "Seperation:", np.round(min_sep, 2))

# Print energy of each pattern
print()
for i in range(attractors.shape[0]):
    energy = calc_energy(beta, attractors.T, attractors[i].T)
    print("Energy of pattern", i, ":", energy)

# Apply update rule multiple times
print()
print_energy(beta, attractors.T, pattern.T, 0)
for i in range(iterations):
    pattern = update_pattern(pattern.T, attractors.T, beta)
    print_energy(beta, attractors.T, pattern.T, i+1)

# Plot before and after updates (and all stored pictures)
f1, ax1 = plt.subplots(1, 2)
ax1[0].imshow(array_to_image(original_pattern, height, width, color))
ax1[0].set_title('Query pattern')
ax1[1].imshow(array_to_image(pattern.T, height, width, color))
ax1[1].set_title('Retrieved pattern')

f2, (ax2, ax3) = plt.subplots(2, 4)
ax2[0].set_title("stored patterns")
ax2[0].imshow(array_to_image(bunny.T, height, width, color))
ax2[0].tick_params(axis = "x", which = "both", bottom = False, top = False)
ax2[1].imshow(array_to_image(cat.T, height, width, color))
ax2[2].imshow(array_to_image(lizard.T, height, width, color))
ax2[3].imshow(array_to_image(rabbit.T, height, width, color))
ax3[0].imshow(array_to_image(snake.T, height, width, color))
ax3[1].imshow(array_to_image(mouse.T, height, width, color))
ax3[2].imshow(array_to_image(tiger.T, height, width, color))
ax3[3].imshow(array_to_image(dog.T, height, width, color))
for ax in ax1:
    ax.set_xticks([])
    ax.set_yticks([])

for ax in ax2:
    ax.set_xticks([])
    ax.set_yticks([])

for ax in ax3:
    ax.set_xticks([])
    ax.set_yticks([])

pyplot.show()

# W_K = Memristors(width*height, width*height, 1000, [0.01, 0.01, 0.01, 0.01])
# print(W_K.get_weights())


