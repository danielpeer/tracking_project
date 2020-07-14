import math
import cv2
import numpy as np


class SearchWindow:
    def __init__(self, target):
        self.search_window = None
        self.top_left_corner_x = None
        self.top_left_corner_y = None
        self.window_w = target.target_w * 2.5
        self.window_h = target.target_h * 2.5


    def update_search_window(self, target_info, mask):
        """
            creating the search window for the correlation algorithm
            x,y - the coordinates of the target which was chosen by the user
            window_w, window_h - width and height of the search window. default numbers
            gray - the frame in grayscale
            returns - the top left corner coordinates of the search window and the search windows itself
         """
        y, x = target_info.current_pos
        x_window = x
        y_window = y
        window_w = target_info.target_w
        window_h = target_info.target_h
        # window_w = self.window_w
        # window_h = self.window_h
        gray_width, gray_height = mask.shape
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
        search_window = mask[int(x_window - math.floor(window_w / 2)): int(x_window + math.floor(window_w / 2)),
                        int(y_window - math.floor(window_h / 2)): int(y_window + math.floor(window_h / 2))]
        self.top_left_corner_x = top_left_corner_x
        self.top_left_corner_y = top_left_corner_y
        self.window_h = window_h
        self.window_w = window_w
        self.search_window = search_window

