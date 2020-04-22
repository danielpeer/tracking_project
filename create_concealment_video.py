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
video = VideoWriter('conceal.avi', fourcc, float(FPS), (width, height))
for paint_x in range(0, width - square_width, 6):
    frame = np.zeros((height, width, 3), np.uint8)
    concealment = cv2.rectangle(frame, (500, 0), (700, 1200), color, -1)
    frame = frame + concealment
    cv2.rectangle(frame, (paint_x, paint_h), (paint_x + square_width, square_width + paint_h), color, -1)
    video.write(frame)
video.release()
