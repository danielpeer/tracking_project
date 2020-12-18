import numpy as np


def get_center_of_mass_prediction(search_window_info):
    search_window = search_window_info.search_window
    (y_width, x_width) = search_window.shape
    m = np.zeros((y_width, x_width))
    for y in range(y_width):
        for x in range(x_width):
            m[y, x] = not (np.array_equal(search_window[(y, x)], 0))
    m = m / np.sum(m)
    # marginal distributions
    dx = np.sum(m, 0)
    dy = np.sum(m, 1)

    # expected values
    y_max = int(np.sum(dy * np.arange(y_width)))
    x_max = int(np.sum(dx * np.arange(x_width)))
    return (int(search_window_info.top_left_corner_x + x_max), int(search_window_info.top_left_corner_y +
                                                                   y_max))
