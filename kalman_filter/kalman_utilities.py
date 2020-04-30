import numpy as np

import math

global delta_t
global G
sigma_a = 0.1


def init_global_variables(frames_per_seconds):
    global delta_t, G
    delta_t = 1 / frames_per_seconds
    G = np.array([0.5 * delta_t ** 2, 0.5 * delta_t ** 2, delta_t, delta_t]).T


def init_state_transition(sigma_a):
    ak = np.random.normal(0, sigma_a)
    return np.array([[1, 0, delta_t, 0], [0, 1, 0, delta_t], [0, 0, 1, 0], [0, 0, 0, 1]])


def init_state_vector(initial_location):
    return np.array([[initial_location[0]], [initial_location[1]], [0], [0]], dtype=float)


def init_measuring_matrix():
    return np.array([[1, 0, 0, 0], [0, 1, 0, 0]])


def init_covariance_matrix():
    return np.array([[100000, 0, 0, 0], [0, 10000, 0, 0], [0, 0, 100000, 0], [0, 0, 0, 10000]])


def init_process_noise_covariance():
    return np.array([[1, 0], [0, 1]])


def init_measurement_noise():
    return G * G.T * sigma_a


def init_R_maximum():
    return np.array([[1000000, 0], [0, 1000000]])
