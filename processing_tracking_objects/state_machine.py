from filters.calculate_center_of_mass import *
import math
from scipy.spatial.distance import euclidean

VISIBLE_OBJECT = 0
OVERLAP = 1
CONCEALMENT = 2
EXIT_OVERLAP = 3
MAXIMUM_DISTANCE = 20


class StateMachine:
    def __init__(self, target):
        self.previous_areas = [target.target_area]
        self.previous_area = target.target_area
        self.previous_pos = target.current_pos
        self.previous_state = VISIBLE_OBJECT
        self.corr_ratio = 0
        self.center_of_mass_ratio = 0

    def get_current_state(self, search_window_info, center_of_mass, correlation_prediction):
        center_of_mass_window = (center_of_mass[1] - search_window_info.top_left_corner_y,
                                 center_of_mass[0] - search_window_info.top_left_corner_x)

        correlation_prediction_window = (correlation_prediction[1] - search_window_info.top_left_corner_y,
                                         correlation_prediction[0] - search_window_info.top_left_corner_x)
        object_contour = None
        search_window = search_window_info.search_window
        contours, hierarchy = cv2.findContours(search_window, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours = merge_contours(contours)
        for contour in contours:
            corr_dist = cv2.pointPolygonTest(contour, correlation_prediction_window, True)
            if corr_dist >= 0:
                object_contour = contour
                object_current_pos = (correlation_prediction[1], correlation_prediction[0])
                break
        closest_contour = None
        closest_contour_distance = math.inf
        if object_contour is None:
            for contour in contours:
                M = cv2.moments(contour)
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                if euclidean((cX, cY), correlation_prediction_window) < closest_contour_distance:
                    closest_contour = contour
                    closest_contour_distance = euclidean((cX, cY), correlation_prediction_window)
            object_contour = closest_contour
        M = cv2.moments(object_contour)
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        corr_dist = 2 ** euclidean((cX, cY), correlation_prediction_window)
        center_of_mass_dist = 2 ** euclidean((cX, cY), center_of_mass_window)
        current_object_area = cv2.contourArea(object_contour)
        self.previous_area = max(sum(self.previous_areas) / len(self.previous_areas), 1)
        if cv2.pointPolygonTest(contour, center_of_mass_window, True) > 0:
            self.corr_ratio = center_of_mass_dist /(corr_dist + center_of_mass_dist)
            self.center_of_mass_ratio = corr_dist /(corr_dist + center_of_mass_dist)
        else:
            self.corr_ratio = 1
            self.center_of_mass_ratio = 0
        print(self.corr_ratio, self.center_of_mass_ratio)
        pervious_before = False
        if self.previous_state == VISIBLE_OBJECT:
            self._get_current_state_from_visible_object(current_object_area, object_current_pos)
            pervious_before = True

        elif self.previous_state == OVERLAP:
            self._get_current_state_from_overlap(current_object_area)

        elif self.previous_state == CONCEALMENT:
            self._get_current_state_from_concealment(current_object_area)

        if self.previous_state == VISIBLE_OBJECT and pervious_before:
            if len(self.previous_areas) == 5:
                self.previous_areas.pop(0)
            self.previous_areas.append(current_object_area)
        return self.previous_state

    def _get_current_state_from_visible_object(self, current_object_area, object_current_pos):
       # print("object is visible")
        if current_object_area / self.previous_area < 0.6:
            self.previous_state = CONCEALMENT
        elif current_object_area / self.previous_area > 1.1:
            self.previous_state = OVERLAP
        else:
            self.previous_state = VISIBLE_OBJECT

    def _get_current_state_from_overlap(self, current_object_area):
        print("overlap")
        if current_object_area < self.previous_area:
            self.previous_state = VISIBLE_OBJECT
        else:
            self.previous_state = OVERLAP

    def _get_current_state_from_concealment(self, current_object_area):
       # print("concealment")
        if current_object_area < self.object_area * 0.7:
            self.previous_state = CONCEALMENT
        else:
            self.previous_state = VISIBLE_OBJECT

    def update_previous_pos(self, pos):
        self.previous_pos = pos


def find_if_close(cnt1, cnt2):
    row1, row2 = cnt1.shape[0], cnt2.shape[0]
    for i in range(row1):
        for j in range(row2):
            dist = np.linalg.norm(cnt1[i] - cnt2[j])
            if abs(dist) < 15:
                return True
            elif i == row1 - 1 and j == row2 - 1:
                return False


def merge_contours(contours):
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
