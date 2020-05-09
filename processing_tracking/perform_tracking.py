import cv2
from processing_tracking.perform_tracking_utilities import *
from correlation_filter.corr_tracker import *
from center_of_mass_filter.calculate_center_of_mass import *
from kalman_filter.kalman_filter import *
from processing_tracking.state_machine import *
from processing_tracking.target import *
from processing_tracking.SearchWindow import *
from videos import *


system_mode = "debug "
should_add_gaussian_noise = False


def perform_tracking():
    if system_mode != "debug ":
        input_video = input("Please enter a video path:\n")
    else:
        input_video = ".\\..\\videos\\conceal1.avi"
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
        out1 = cv2.VideoWriter('Corr_Tracker_atrium.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), fps,
                               (frame_width, frame_height))
        red = [0, 0, 255]
        # Read until video is completed
        retries = 0
        while cap.isOpened():
            # Capture frame-by-frame
            ret, frame = cap.read()

            if ret:
                # converting to grayscale in order to calculate correlation
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                if not select_target_flag:  # creating the target only once
                    target_info = Target(gray)
                    search_window_info = SearchWindow(target_info)
                    kalman = KalmanFilter(target_info, fps)
                    state_holder = StateMachine(target_info)
                    select_target_flag = True
                # creating the search window for the current frame
                search_window_info.update_search_window(target_info, gray)

                if should_add_gaussian_noise:
                    add_gaussian_noise(search_window_info)

                correlation_prediction = get_correlation_prediction(target_info, search_window_info)
                center_of_mass_prediction = get_center_of_mass_prediction(search_window_info)
                current_state = state_holder.get_current_state(search_window_info, center_of_mass_prediction,
                                                               correlation_prediction)
                prediction = get_integrated_prediction(correlation_prediction, center_of_mass_prediction, state_holder)

                if current_state == OVERLAP or current_state == CONCEALMENT:
                    kalman.base_kalman_prior_prediction()
                else:
                    kalman.base_measurement()


                final_prediction = kalman.get_prediction(prediction)
                x, y = final_prediction[0][0], final_prediction[1][0]
                cv2.rectangle(frame, (y - target_info.target_h, x - target_info.target_w),
                              (y + target_info.target_h, x + target_info.target_w), red, 1)

                target_info.update_position(x, y)
                state_holder.update_previous_pos((x, y))
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


def get_integrated_prediction(corr_prediction, center_of_mass_prediction, state_machine):
    if state_machine.use_center_of_mass_prediction and state_machine.use_correlation_prediction:
        center_of_mass_prediction = 0.5 * np.array([[center_of_mass_prediction[0]], [center_of_mass_prediction[1]]])
        corr_prediction = 0.5 * np.array([[corr_prediction[0]], [corr_prediction[1]]])
        prediction = center_of_mass_prediction + corr_prediction
    elif state_machine.use_correlation_prediction:
        prediction = np.array([[corr_prediction[0]], [corr_prediction[1]]])
    else:
        prediction = np.array([[center_of_mass_prediction[0]], [center_of_mass_prediction[1]]])
    return prediction

perform_tracking()
