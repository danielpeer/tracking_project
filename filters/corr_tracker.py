
from processing_tracking.perform_tracking_utilities import *


def correlation(window, target):
    """
       Calculate the correlation coefficients between the given pixel arrays.
       window - a matrix representing the current search window
       target - the matrix representing the image of the target to be tracked
       """
    coef_mat = np.zeros(window.shape)  # a matrix to store the coefficients
    targ_mean = target.mean()
    window_h, window_w = window.shape  # get frame width and height
    targ_h, targ_w = target.shape  #

    # start searching for the target in the search window via correlation
    for i in range(0, window_h):
        for j in range(0, window_w):

            # find the left, right, top and bottom of the sub-image

            # the sub_image not include the whole object
            if i + targ_h >= window_h or j + targ_w >= window_w:
                coef_mat[i, j] = 0
                continue

            # the sub_image includes the whole object
            left = i
            right = left + targ_h

            top = j
            bottom = top + targ_w

            # match part of the search window to compare to the target matrix
            sub = window[left:right, top:bottom]
            # assert sub.shape == target.shape, "SubImages must be same size"
            local_mean = sub.mean()
            temp = np.multiply(np.subtract(sub, local_mean), np.subtract(target, targ_mean))
            s1 = temp.sum()
            temp = np.multiply(np.subtract(sub, local_mean), np.subtract(sub, local_mean))
            s2 = temp.sum()
            temp = np.multiply(np.subtract(target, targ_mean), np.subtract(target, targ_mean))
            s3 = temp.sum()
            denom = np.multiply(s2,  s3)
            if denom == 0:
                temp = 0
            else:
                temp = np.divide(s1, math.sqrt(denom))

            coef_mat[i, j] = temp
    return coef_mat


def get_correlation_prediction(target_info, search_window_info):
    search_window = search_window_info.search_window
    target = target_info.target
    corr = correlation(search_window, target)
    y_max, x_max = np.unravel_index(np.argmax(corr), corr.shape)   # find the relative coordinates of highest correlation
    y_max += math.floor(target.shape[0] / 2)
    x_max += math.floor(target.shape[1] / 2)

    return (int(search_window_info.top_left_corner_x + x_max), int(search_window_info.top_left_corner_y +
                                                                               y_max))



