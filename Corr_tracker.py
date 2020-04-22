from PIL import Image
import cv2
import numpy as np
from scipy import signal
import math
import sys
from detect_mouse_clicks import *
from create_target_window import *



def normalizeArray(window):
    """
    Normalize the given window array to values between 0 and 1.
    Return a numpy array of floats (of the same shape as given)
    """
    width, height = window.shape
    min_val = window.min()
    if min_val < 0:  # shift the whole matrix to being positive values
        window = window + abs(min_val)
    max_val = window.max()
    norm_window = np.zeros(window.shape, 'd')
    for x in range(0, width):
        for y in range(0, height):
            norm_window[x, y] = float("{:.2f}".format(window[x, y])) / max_val
    return norm_window


def correlation(window, target):
    """
    Calculate the correlation coefficients between the given pixel arrays.
    window - a matrix representing the current search window
    target - the matrix representing the image of the target to be tracked
    """
    coef_mat = np.zeros(window.shape)  # a matrix to store the coefficients
    targ_mean = target.mean()
    window_w, window_h = window.shape  # get frame width and height
    targ_w, targ_h = target.shape  # get target width and height

    # start searching for the target in the search window via correlation
    for i in range(0, window_w):
        for j in range(0, window_h):

            # find the left, right, top and bottom of the sub-image
            if i - targ_w / 2 <= 0:
                left = 0
            elif window_w - i < targ_w:
                left = window_w - targ_w
            else:
                left = i

            right = left + targ_w

            if j - targ_h / 2 <= 0:
                top = 0
            elif window_h - j < targ_h:
                top = window_h - targ_h
            else:
                top = j

            bottom = top + targ_h

            # match part of the search window to compare to the target matrix
            sub = window[left:right, top:bottom]
            # assert sub.shape == target.shape, "SubImages must be same size"
            local_mean = sub.mean()
            temp = (sub - local_mean) * (target - targ_mean)
            s1 = temp.sum()
            temp = (sub - local_mean) * (sub - local_mean)
            s2 = temp.sum()
            temp = (target - targ_mean) * (target - targ_mean)
            s3 = temp.sum()
            denom = s2 * s3
            if denom == 0:
                temp = 0
            else:
                temp = s1 / math.sqrt(denom)

            coef_mat[i, j] = temp
    return coef_mat

def main():
    input_video = "output.avi"
    cap = cv2.VideoCapture(input_video)
    select_target_flag = False
    first_flag = False

    # Check if opened successfully
    if not cap.isOpened():
        print("Error opening video stream or file")
        return

    # retrieving Resolution
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Retrieving fps
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # Defining the codec and creating VideoWriter object. The output is stored in 'Vid1_Binary.avi' file.
    out1 = cv2.VideoWriter('Corr_Tracker.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), fps,
                           (frame_width, frame_height))
    red = [0, 0, 255]
    # Read until video is completed
    while cap.isOpened():
        # Capture frame-by-frame
        ret, frame = cap.read()

        if ret:
            # converting to grayscale in order to calculate correlation
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            if not select_target_flag:  # creating the target only once
                x, y, target = create_target(gray)
                select_target_flag = True

            top_left_corner_x, top_left_corner_y, search_window = create_window(x, y, window_w, window_h, gray)  # creating the search window for the current frame
            corr = correlation(search_window, target)
            corr = normalizeArray(corr)
            x_max, y_max = np.unravel_index(np.argmax(corr),
                                            corr.shape)  # find the relative coordinates of highest correlation
            if x < window_w:
                x = x_max
            else:
                x = top_left_corner_x + x_max
            if y < window_h:
                y = y_max
            else:
                y = top_left_corner_y + y_max

            print(x, y, x_max, y_max)
            cv2.circle(frame, (y, x), 3, red, -1)

            # Write the frame into the file 'output.avi'
            out1.write(frame)

            # Display the resulting frame
            cv2.imshow('Frame', frame)
            # Press Q on keyboard to  exit
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
        else:
            break
    # When everything done, release the video capture object
    cap.release()
    out1.release()

    # Closes all the frames
    cv2.destroyAllWindows()


main()
