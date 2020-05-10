import cv2
import math
import numpy as np

window_w = 60
window_h = 60
target_w = 30
target_h = 30


############################################################### Search Window #####################################################

def create_window(x, y, window_w, window_h, gray):
    """
    creating the search window for the correlation algorithm
    x,y - the coordinates of the target which was chosen by the user
    window_w, window_h - width and height of the search window. default numbers
    gray - the frame in grayscale
    returns - the top left corner coordinates of the search window and the search windows itself
    """
    x_window = x
    y_window = y
    gray_width, gray_height = gray.shape
    top_left_corner_x = x - (window_w / 2)
    top_left_corner_y = y - (window_h / 2)
    if x - (window_w / 2) < 0:
        x_window = window_w / 2
        top_left_corner_x = 0
    if y - (window_h / 2) < 0:
        y_window = window_h / 2
        top_left_corner_y = 0
    if x + (window_w / 2) > gray_width:
        x_window = gray_width - (window_w / 2)
        top_left_corner_x = gray_width - window_w
    if y + (window_h / 2) > gray_height:
        y_window = gray_height - (window_h / 2)
        top_left_corner_y = gray_height - window_h
    search_window = gray[int(x_window - math.floor(window_w / 2)): int(x_window + math.floor(window_w / 2)),
                    int(y_window - math.floor(window_h / 2)): int(y_window + math.floor(window_h / 2))]
    return top_left_corner_x, top_left_corner_y, search_window


######################################### detect mouse clicks #####################################################################
refPt = (0, 0)
pressed = False


def click(event, x, y, flags, param):
    global refPt, pressed
    if event == cv2.EVENT_LBUTTONDOWN:
        refPt = (y, x)
        pressed = True



############################################################################################################

def add_gaussian_noise(search_window_info):
    image = search_window_info.search_window
    mean = 0.0  # some constant
    std = 1.0  # some constant (standard deviation)
    noisy_img = image + np.random.normal(mean, std, image.shape)
    noisy_img_clipped = np.clip(noisy_img, 0, 255)
    search_window_info.search_window = noisy_img_clipped

########################################################################################################

