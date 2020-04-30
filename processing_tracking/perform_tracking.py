import cv2
from processing_tracking.perform_tracking_utilities import *
from correlation_filter.corr_tracker import *
from center_of_mass_filter.calculate_center_of_mass import *
from kalman_filter.kalman_filter import *

system_mode = "debug "


def perform_tracking():
    if system_mode != "debug ":
        input_video = input("Please enter a video path:\n")
    else:
        input_video = "C:\\Users\\Z41\\PycharmProjects\\tracking_project\\videos\\conceal3.avi"
    try:
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
        out1 = cv2.VideoWriter('Corr_Tracker_conceal3.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), fps,
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
                    x, y, target = create_object_target(gray)
                    # x, y, target = create_target(gray)
                    select_target_flag = True
                    kalman = kalman_filter((x, y), fps)
                    window_w = target.shape[0] * 6
                    window_h = target.shape[1] * 6
                # creating the search window for the current frame
                top_left_corner_x, top_left_corner_y, search_window = create_window(x, y, window_w, window_h, gray)
                measurment_x, measurment_y = get_correlation_prediction(x, y, search_window, target, top_left_corner_x,
                                                                        top_left_corner_y)
                measurment = kalman.get_prediction(np.array([[measurment_x], [measurment_y]]))
                x, y = int(measurment[0]), int(measurment[1])
                cv2.circle(frame, (y, x), 3, red, -1)
                # Display the resulting frame
                cv2.imshow('Frame', frame)

                # Write the frame into the file
                out1.write(frame)

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
    except IOError:
        print(IOError)
        print("File not accessible")


perform_tracking()
