import cv2
import numpy as np
from scipy.special import logsumexp
import math
from processing_tracking.perform_tracking_utilities import *

window_w = 60
window_h = 60
target_w = 5
target_h = 5


def normalizeArray(window):
    """
    Normalize the given window array to values between 0 and 1.
    Return a numpy array of floats (of the same shape as given)
    """
    width, height = window.shape
    min_val = window.min()
    if min_val < 0:  # shift the whole matrix to being positive values
        window = window + abs(min_val)
    max_val = window.max()
    norm_window = np.zeros(window.shape, 'd')
    for x in range(0, width):
        for y in range(0, height):
            if window[x, y] == 0:
                norm_window[x, y] = 0
            norm_window[x, y] = round(window[x, y] / max_val,2)
    return norm_window


def correlation(window, target):
    """
    Calculate the correlation coefficients between the given pixel arrays.
    window - a matrix representing the current search window
    target - the matrix representing the image of the target to be tracked
    """
    coef_mat = np.zeros(window.shape)  # a matrix to store the coefficients
    targ_mean = target.mean()
    window_w, window_h = window.shape  # get window width and height
    targ_w, targ_h = target.shape  # get target width and height

    # start searching for the target in the search window via correlation
    for i in range(0, window_w):
        for j in range(0, window_h):

            # find the left, right, top and bottom of the sub-image
            if i - targ_w / 2 <= 0:
                left = 0
            elif window_w - i < targ_w:
                left = window_w - targ_w
            else:
                left = i

            right = left + targ_w

            if j - targ_h / 2 <= 0:
                top = 0
            elif window_h - j < targ_h:
                top = window_h - targ_h
            else:
                top = j

            bottom = top + targ_h

            # match part of the search window to compare to the target matrix
            sub = window[left:right, top:bottom]
            # assert sub.shape == target.shape, "SubImages must be same size"
            local_mean = sub.mean()
            temp = (sub - local_mean) * (target - targ_mean)
            s1 = temp.sum()
            temp = (sub - local_mean) * (sub - local_mean)
            s2 = temp.sum()
            temp = (target - targ_mean) * (target - targ_mean)
            s3 = temp.sum()
            denom = s2 * s3
            if denom == 0:
                temp = 0
            else:
                temp = s1 / math.sqrt(denom)

            coef_mat[i, j] = float("{:.2f}".format(temp))
    return coef_mat


def get_correlation_prediction(x, y, search_window, target, top_left_corner_x, top_left_corner_y):
    corr = correlation(search_window, target)
    #corr = normalizeArray(corr)
    x_max, y_max = np.unravel_index(np.argmax(corr), corr.shape)  # find the relative coordinates of highest correlation
    if x < window_w:
        x = x_max
    else:
        x = top_left_corner_x + x_max
    if y < window_h:
        y = y_max
    else:
        y = top_left_corner_y + y_max
    return x, y
