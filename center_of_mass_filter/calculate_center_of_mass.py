import cv2
import numpy as np

window_w = 60
window_h = 60


def get_center_of_mass_prediction(search_window_info):
    search_window = search_window_info.search_window
    (x_width, y_width) = search_window.shape
    m = np.zeros((x_width, y_width))
    for x in range(x_width):
        for y in range(y_width):
            m[x, y] = not (np.array_equal(search_window[(x, y)], 0))
    m = m / np.sum(m)
    # marginal distributions
    dx = np.sum(m, 1)
    dy = np.sum(m, 0)

    # expected values
    x_max = int(np.sum(dx * np.arange(x_width)))
    y_max = int(np.sum(dy * np.arange(y_width)))
    return (int(search_window_info.top_left_corner_x + x_max), int(search_window_info.top_left_corner_y +
                                                                               y_max))
