import cv2
import numpy as np
from videos import *

cap = cv2.VideoCapture(".\\..\\videos\\walking_alone.mp4")
kernel = np.ones((5, 5), np.uint8)
subtractor = cv2.createBackgroundSubtractorMOG2(history=0, varThreshold=50, detectShadows=True)

while True:
    scale_percent = 50
    _, frame = cap.read()
    width = int(frame.shape[1] * scale_percent / 100)
    height = int(frame.shape[0] * scale_percent / 100)
    dim = (width, height)
    resized_frame = cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2GRAY)
    mask = subtractor.apply(gray)
    cv2.imshow("mask", mask)
    #mask = cv2.morphologyEx(mask, cv2.MORPH_GRADIENT, kernel)
    mask = cv2.GaussianBlur(mask, (3, 3), 0)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    #mask = cv2.erode(mask, kernel, iterations=1)


    #mask = cv2.erode(mask, kernel, iterations=1)
    #mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)


    cv2.imshow("Frame", resized_frame)
    cv2.imshow("mask", mask)

    key = cv2.waitKey(5)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()
