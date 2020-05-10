import cv2
import math
import numpy as np


# from processing_tracking.perform_tracking_utilities import get_square_center, click

class Target:
    def __init__(self, frame, mask):
        x, y, w, h = get_target(frame)
        self.target = mask[y:y + h, x:x + w]
        self.current_pos = (x + int(w / 2), y + int(h / 2))
        self.target_area = get_object_dimensions(self.current_pos[0], self.current_pos[1], mask)
        self.target_w = int(self.target.shape[0])
        self.target_h = int(self.target.shape[1])

    def update_position(self, x, y):
        self.current_pos = (x, y)


def get_object_dimensions(x, y, mask):
    red = (0,0,255)
    object_contour = None
    ret, thresh = cv2.threshold(mask, 70, 255, 0)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        if cv2.pointPolygonTest(contour, (x, y), False) >= 0:
            object_contour = contour
            break
    contour_area = cv2.contourArea(object_contour)
    return contour_area


def get_target(gray):
    from_center = False
    (x, y, w, h) = cv2.selectROI(
        "Drag the rect from the top left to the bottom right corner of the forground object,"
        " then press ENTER.",
        gray, from_center)
    cv2.destroyAllWindows()
    return x, y, w, h
