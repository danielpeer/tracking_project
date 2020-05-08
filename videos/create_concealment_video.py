import math

import cv2
import numpy as np
import random
from cv2 import VideoWriter, VideoWriter_fourcc
import operator

width = 1200
height = 1200
FPS = 10
seconds = 30
color = (255, 255, 255)
step_size = 15
paint_h = int((height / 2) - 15)
square_width = 30
fourcc = VideoWriter_fourcc(*'MP42')
'''
# video #1 - one concealment
video = VideoWriter('conceal1.avi', fourcc, float(FPS), (width, height))
for paint_x in range(0, width - square_width, 6):
    frame = np.zeros((height, width, 3), np.uint8)
    concealment = cv2.rectangle(frame, (150, 0), (350, 1200), color, -1)
    frame = frame + concealment
    cv2.rectangle(frame, (paint_x, paint_h), (paint_x + square_width, square_width + paint_h), color, -1)
    video.write(frame)
video.release()

# video #2 - two concealments
video = VideoWriter('conceal2.avi', fourcc, float(FPS), (width, height))
for paint_x in range(0, width - square_width, 6):
    frame = np.zeros((height, width, 3), np.uint8)
    concealment1 = cv2.rectangle(frame, (400, 0), (450, 1200), color, -1)
    concealment2 = cv2.rectangle(frame, (800, 0), (850, 1200), color, -1)
    frame = frame + concealment1 + concealment2
    cv2.rectangle(frame, (paint_x, paint_h), (paint_x + square_width, square_width + paint_h), color, -1)
    video.write(frame)
video.release()

# video #3 - two concealments + diagonal move
video = VideoWriter('conceal3.avi', fourcc, float(FPS), (width, height))
diag_h = paint_h
for paint_x in range(0, width - square_width, 6):
    frame = np.zeros((height, width, 3), np.uint8)
    concealment1 = cv2.rectangle(frame, (400, 0), (450, 1200), color, -1)
    concealment2 = cv2.rectangle(frame, (800, 0), (850, 1200), color, -1)
    frame = frame + concealment1 + concealment2
    if paint_x < 500:
        cv2.rectangle(frame, (paint_x, paint_h), (paint_x + square_width, square_width + paint_h), color, -1)
    else:
        diag_h = diag_h - 3
        cv2.rectangle(frame, (paint_x, diag_h), (paint_x + square_width, square_width + diag_h), color, -1)
    video.write(frame)
video.release()

# video #4 - diagonal run
video = VideoWriter('conceal4.avi', fourcc, float(FPS), (width, height))
diag_h = paint_h
for paint_x in range(0, width - square_width, 6):
    frame = np.zeros((height, width, 3), np.uint8)
    cv2.rectangle(frame, (paint_x, diag_h), (paint_x + square_width, square_width + diag_h), color, -1)
    diag_h = diag_h - 3
    video.write(frame)
video.release()
'''
# video #5 - sine wave run
video = VideoWriter('conceal5.avi', fourcc, float(FPS), (width, height))
i = 0
for paint_x in range(0, width - square_width, 6):
    frame = np.zeros((height, width, 3), np.uint8)
    # Compute the x and y coordinates for points on a sine curve
    x = np.arange(0, 3 * np.pi, 0.2)
    y = np.sin(x)
    cv2.rectangle(frame, (paint_x, paint_h + math.floor(y[i] * 90)),
                  (paint_x + square_width, paint_h + math.floor(y[i] * 90) - square_width), color,
                  -1)
    if x[i] == 9.4:
        i = 0
    else:
        i += 1
    video.write(frame)
video.release()


