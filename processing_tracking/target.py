import cv2
import math
import numpy as np
from processing_tracking.perform_tracking_utilities import get_square_center, click


class Target:
    def __init__(self, gray):
        center_x, center_y = get_square_center(gray)
        object_dims = get_object_dimensions(center_x, center_y, gray)
        x, y, w, h = object_dims[0]
        target_area = object_dims[1]
        target = gray[y:y + h, x:x + w]
        self.target = gray[y:y + h, x:x + w]
        self.current_pos = (center_x, center_y)
        self.target_area = target_area
        self.target_w = int(target.shape[0])
        self.target_h = int(target.shape[1])

    def update_position(self, x, y):
        self.current_pos = (x, y)


def get_object_dimensions(x, y, gray):
    object_contour = None
    ret, thresh = cv2.threshold(gray, 127, 255, 0)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        if cv2.pointPolygonTest(contour, (y, x), False) >= 0:
            object_contour = contour
            break
    contour_dim = cv2.boundingRect(object_contour)
    contour_area = cv2.contourArea(object_contour)
    return contour_dim, contour_area



