from queue import Queue

import cv2
import math
import numpy as np
from random import randint


# from processing_tracking.perform_tracking_utilities import get_square_center, click
from colorthief import ColorThief

from filters.corr_tracker import get_correlation_prediction
from filters.kalman_filter import KalmanFilter
from processing_tracking_objects.search_window import SearchWindow
from processing_tracking_objects.state_machine import StateMachine, get_center_of_mass_prediction
from processing_tracking_objects.targetinfo import TargetInfo


class Target:
    def __init__(self, frame, mask, fps, points, incoming=False):
        self.target_info = TargetInfo(frame, mask, points)
        self.search_window = SearchWindow(self.target_info)
        self.kalman_filter = KalmanFilter(self.target_info, fps)
        self.state_holder = StateMachine(self.target_info)
        self.previous_histogram = Queue()
        self.best_histogram = Queue()
        self.calc_x_pos = None
        self.calc_y_pos = None
        self.target_image = None
        self.detection = None
        self.outgoing = False
        self.incoming = incoming

    def update_search_window(self, mask):
        self.search_window.update_search_window(self.target_info, mask)

    def get_correlation_prediction(self, results):
        results[0] = get_correlation_prediction(self.target_info, self.search_window)

    def get_center_of_mass_prediction(self, results):
        results[1] = get_center_of_mass_prediction(self.search_window)

    def update_target_image(self, target_mask, color_image):
        self.target_image = np.bitwise_and(color_image,target_mask)
