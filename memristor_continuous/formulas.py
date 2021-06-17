import numpy as np
from scipy.special import softmax
from scipy.special import logsumexp
from scipy import signal
import random

# Update rule
def update_pattern(pattern, attractors, beta):
    dot_prod = beta * np.dot(attractors.T, pattern)
    for i in range(len(dot_prod)):
        dot_prod[i] /= np.linalg.norm(attractors.T[i])
    p = softmax(dot_prod, axis=0)
    new_pat = np.dot(attractors, p)

    return new_pat

# Gets highest distance column vector from matrix
def max_distance(vectors):
    max_dist = 0
    max_vec = -1
    vec_ind = -1
    for vec in np.transpose(vectors):
        vec_ind += 1
        dist = np.linalg.norm(vec)
        if dist > max_dist:
            max_dist = dist
            max_vec = vec_ind

    return max_dist, max_vec


# Calculates current energy of a hopfield network
def calc_energy(beta, attractors, pattern):
    pattern_amnt = np.size(attractors, 1)
    max_dist, _ = max_distance(attractors)
    E_comp1 = -logsumexp(beta * np.dot(np.transpose(attractors), pattern)) / beta
    E_comp2 = 0.5 * np.dot(np.transpose(pattern), pattern)
    E_comp3 = 1/beta * np.log(pattern_amnt)
    E_comp4 = 0.5 * max_dist * max_dist
    Energy = E_comp1 + E_comp2 + E_comp3 + E_comp4

    return Energy


def min_separation(vectors, i):
    max_dot2 = -np.Inf
    vector_amnt = np.size(vectors, 1)
    vectors_t = np.transpose(vectors)
    attractor2 = 0

    for j in range(0, vector_amnt):
        if j != i:
            dot2 = np.dot(vectors_t[i], np.transpose(vectors_t[j]))
            if dot2 > max_dot2:
                attractor2 = j
                max_dot2 = dot2

    dot1 = np.dot(vectors_t[i], np.transpose(vectors_t[i]))
    min_sep = np.abs(dot1 - dot2)

    return min_sep, attractor2


def blur_image(img, H, fill_val=1.0):
    return signal.convolve2d(img, H, boundary='fill', fillvalue=fill_val, mode='same')
