import cv2
import numpy as np

window_w = 60
window_h = 60


def get_center_of_mass_prediction(x, y, search_window, top_left_corner_x, top_left_corner_y):
            (x_width, y_width) = search_window.shape
            m = np.zeros((x_width, y_width))
            for x in range(x_width):
                for y in range(y_width):
                    m[x, y] = not(np.array_equal(search_window[(x, y)], 0))
            m = m / np.sum(m)
            # marginal distributions
            dx = np.sum(m, 1)
            dy = np.sum(m, 0)

            # expected values
            cx = int(np.sum(dx * np.arange(x_width)))
            cy = int(np.sum(dy * np.arange(y_width)))
            if x < window_h/2:
              x = cx
            elif x + window_h/2>1200:
                x = 1200- window_h/2 + cx
            else:
                x = top_left_corner_x + cx
            if y < window_w / 2:
                y = cy
            elif y + window_w / 2 > 1200:
                y = 1200 - window_h / 2 + cy
            else:
                y = top_left_corner_y + cy
            return np.array([[int(x)], [int(y)]])
