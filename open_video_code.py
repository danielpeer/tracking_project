import cv2
import numpy as np


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

# Read until video is completed
while cap.isOpened():
    # Capture frame-by-frame
    ret, frame = cap.read()

    if ret:
        # Here you can do stuff on each frame

        # Write the frame into the file 'output.avi'
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