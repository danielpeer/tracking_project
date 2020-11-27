import cv2
import math
import numpy as np
from random import randint


# from processing_tracking.perform_tracking_utilities import get_square_center, click

class Target_Info:
    def __init__(self, frame, mask, points):
        if not points:
            x, y, w, h = get_target(frame)
        else:
            x, y, w, h = points[0], points[1], points[2] - points[0], points[3] - points[1]

        self.target = mask[y:y + h, x:x + w]
        self.current_pos = (x + int(w / 2), y + int(h / 2))
        if (x != 0) and (y != 0):
            self.target_area = get_object_dimensions(self.current_pos[0], self.current_pos[1], self.target)
        else:
            self.target_area = 0
        self.target_w = int(self.target.shape[0])
        self.target_h = int(self.target.shape[1])
        self.x = x
        self.y = y
        self.w = w
        self.h = h

    def update_position(self, x, y):
        self.current_pos = (x, y)
        self.x = x - int(self.target_w / 2)
        self.y = y - int(self.target_h / 2)



def get_object_dimensions(x, y, mask):
    red = (0, 0, 255)
    object_contour = None
    ret, thresh = cv2.threshold(mask, 70, 255, 0)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = merge_contours(contours, mask)
    object_contour = contours[0]
    contour_area = cv2.contourArea(object_contour)
    return contour_area


def get_target(gray):
    from_center = False
    target = cv2.selectROI(
        "Drag the rect from the top left to the bottom right corner of the forground object,"
        " then press ENTER.",
        gray, from_center)
    cv2.destroyAllWindows()
    return target


def find_if_close(cnt1, cnt2):
    row1, row2 = cnt1.shape[0], cnt2.shape[0]
    for i in range(row1):
        for j in range(row2):
            dist = np.linalg.norm(cnt1[i] - cnt2[j])
            if abs(dist) < 50:
                return True
            elif i == row1 - 1 and j == row2 - 1:
                return False


def merge_contours(contours, mask):
    LENGTH = len(contours)
    status = np.zeros((LENGTH, 1))

    for i, cnt1 in enumerate(contours):
        x = i
        if i != LENGTH - 1:
            for j, cnt2 in enumerate(contours[i + 1:]):
                x = x + 1
                dist = find_if_close(cnt1, cnt2)
                if dist == True:
                    val = min(status[i], status[x])
                    status[x] = status[i] = val
                else:
                    if status[x] == status[i]:
                        status[x] = i + 1

    unified = []
    maximum = int(status.max()) + 1
    for i in range(maximum):
        pos = np.where(status == i)[0]
        if pos.size != 0:
            cont = np.vstack([contours[i] for i in pos])
            hull = cv2.convexHull(cont)
            unified.append(hull)
    return unified
