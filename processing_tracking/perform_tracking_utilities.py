import cv2
import math
import numpy as np

window_w = 60
window_h = 60
target_w = 30
target_h = 30
scale_percent = 50
black = [0, 0, 0]


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

def frame_scaling(frame):
    width = int(frame.shape[1] * scale_percent / 100)
    height = int(frame.shape[0] * scale_percent / 100)
    dim = (width, height)
    resize_frame = cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)
    return resize_frame


def get_frame_resize_dim(frame_shape):
    width = int(frame_shape[1] * scale_percent / 100)
    height = int(frame_shape[0] * scale_percent / 100)
    return width, height


def get_target_from_mask(image, mask):
    result = cv2.bitwise_and(image, mask)
    cv2.imshow("image", result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def detect_new_targets(image, object_detections, suspected_targets, targets_lst):
    targets = []
    image_mask = np.zeros(image.shape)
    image_mask.fill(255)  # or img[:] = 255
    cv2.rectangle(image_mask, (70, 0), (image.shape[1] - 70, image.shape[0] - 120), black, -1)
    mask = np.bitwise_and(image_mask.astype(int), image)
    targets_dims = object_detections.get_targets_on_the_sides(mask)

    for target_dim in targets_dims:
        target_left = (target_dim[0], target_dim[1])
        target_right = (target_dim[2], target_dim[3])
        if target_dim[0] < 20 or target_dim[2] > 1260 or target_dim[1] < 20 or target_dim[3] > 700:
            if target_dim[3] < 680 or target_dim[3] - target_dim[1] > 100:
                overlap = False
                should_check_second_loop = True
                for target in suspected_targets:
                    target_left_before = (target[0], target[1])
                    target_right_before = (target[2], target[3])
                    if do_overlap(target_left, target_right, target_left_before, target_right_before, True):
                        if detect_incoming_targets(target, target_dim):
                            do_overlap(target_left, target_right, target_left_before, target_right_before, True)
                            targets.append(target_dim)
                            suspected_targets.remove(target)
                        else:
                            suspected_targets.remove(target)
                        should_check_second_loop = False
                        break
                if should_check_second_loop:
                    overlap = False
                    for target in targets_lst:
                        current_target_l = (target.target_info.current_pos[0], target.target_info.current_pos[1])
                        current_target_r = (
                            target.target_info.current_pos[0] + target.target_info.w, target.target_info.current_pos[1] + target.target_info.target_h)
                        if do_overlap(target_left, target_right, current_target_l, current_target_r, target.incoming):
                            overlap = True
                    if not overlap:
                        suspected_targets.append(target_dim)
    return targets


def detect_incoming_targets(frame_before_target_dim, frame_after_target_dim):
    if frame_before_target_dim[3] > 700:
        if frame_after_target_dim[1] >= frame_before_target_dim[1]-3 or frame_before_target_dim[2] > 1160:
            return False

    elif frame_before_target_dim[2] > 1260:
        if frame_after_target_dim[0] >= frame_before_target_dim[0]-2 or frame_after_target_dim[3] < 400:
            return False

    elif frame_before_target_dim[0] < 20:
        if frame_after_target_dim[2] <= frame_before_target_dim[2] or frame_before_target_dim[1] < 100:
            return False
    return True


def detect_outgoing_targets(target):
    if not target.incoming and (target.calc_y_pos + target.target_info.target_h/2 >= 1280 or target.calc_y_pos -
                                target.target_info.target_h/2 <= 0 or target.calc_x_pos +
                                target.target_info.target_w/2 >= 720 or target.calc_x_pos -
                                target.target_info.target_w/2 <= 0) and not target.target_info.start_on_side:
        target.outgoing = True


def get_target(targets):
    not_outgoing_targets = []
    for target in targets:
        if not target.outgoing:
            not_outgoing_targets.append(target)
    return not_outgoing_targets


def do_overlap(l1, r1, l2, r2, same_range):
    # If one rectangle is on left side of other
    if same_range:
        x1 = range(l1[0], r1[0])
        x2 = range(l2[0], r2[0])
        xs = set(x1)
        z1 = xs.intersection(x2)

        y1 = range(l1[1], r1[1])
        y2 = range(l2[1], r2[1])
        ys = set(y1)
        z2 = ys.intersection(y2)

        if len(z1) > 0 and len(z2):
            return True
    else:
        x1 = range(l1[0] + 20, r1[0] - 20)
        x2 = range(l2[0] + 20, r2[0] - 20)
        xs = set(x1)
        z1 = xs.intersection(x2)

        y1 = range(l1[1] + 20, r1[1] - 20)
        y2 = range(l2[1] + 20, r2[1] - 20)
        ys = set(y1)
        z2 = ys.intersection(y2)

        if len(z1) > 0 and len(z2) > 0:
            return True

    return False
