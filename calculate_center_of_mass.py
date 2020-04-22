import cv2
import numpy as np
from detect_mouse_clicks import *


def example(input_video):
    # opening video
    cap = cv2.VideoCapture(input_video)

    # Check if opened successfully
    if not cap.isOpened():
        print("Error opening video stream or file")

    # retrieving Resolution
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Retrieving fps
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # Saving new video, defining the codec and creating VideoWriter object. The output is stored in 'Vid1_Binary.avi' file.
    out = cv2.VideoWriter('Vid1_Binary.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), fps,
                      (frame_width, frame_height))

    i = 0
    # Read until video is completed
    while cap.isOpened():
        # Capture frame-by-frame
        ret, frame = cap.read()

        if ret :
            m = np.zeros((frame_width, frame_height))
            for x in range(frame_width):
                for y in range(frame_height):
                    m[x, y] = not(np.array_equal(frame[(x, y)], (0, 0, 0)))
            m = m / np.sum(m)
            # marginal distributions
            dx = np.sum(m, 1)
            dy = np.sum(m, 0)

            # expected values
            cx = np.sum(dx * np.arange(frame_width))
            cy = np.sum(dy * np.arange(frame_height))
            frame[(int(cx), int(cy))] = (255, 0, 0)

            out.write(frame)

            # Display the resulting frame
            cv2.imshow('Frame', frame)

            # Press Q on keyboard to  exit
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
        else:
            break
    # When everything done, release the video capture object
    cap.release()
    out.release()

    # Closes all the frames
    cv2.destroyAllWindows()
    return

example("C:\\Users\\danielpeer\\Downloads\\Output.avi")