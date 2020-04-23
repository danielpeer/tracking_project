import cv2
import numpy as np


def get_center_of_mass_prediction(x, y, frame, top_left_corner_x, top_left_corner_y, x_width, y_width):
    m = np.zeros((1200, 1200))
    for x in range(1200):
        for y in range(1200):
            m[x, y] = not (np.array_equal(frame[(x, y)], (0, 0, 0)))
    print(m.sum())
    m = m / np.sum(m)
    # marginal distributions
    dx = np.sum(m, 1)
    dy = np.sum(m, 0)

    # expected values
    cx = int(np.sum(dx * np.arange(1200)))
    cy = int(np.sum(dy * np.arange(1200)))
    return cx,cy


def get_center_of_mass_prediction2(x, y, search_window, top_left_corner_x, top_left_corner_y, x_width, y_width):
            m = np.zeros((x_width, y_width))
            for x in range(x_width):
                for y in range(y_width):
                    m[x, y] = not(np.array_equal(search_window[(x, y)], (0, 0, 0)))
            print(m.sum())
            m = m / np.sum(m)
            # marginal distributions
            dx = np.sum(m, 1)
            dy = np.sum(m, 0)

            # expected values
            cx = int(np.sum(dx * np.arange(x_width)))
            cy = int(np.sum(dy * np.arange(y_width)))
            if x < window_w:
                x = cx
            else:
                x = top_left_corner_x + cx
            if y < window_h:
                y = cy
            else:
                y = top_left_corner_y + cy
            return x, y
