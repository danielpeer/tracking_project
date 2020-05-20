
from correlation_filter.corr_tracker import *
from center_of_mass_filter.calculate_center_of_mass import *
from kalman_filter.kalman_filter import *
from processing_tracking.state_machine import *
from processing_tracking.target import *
from processing_tracking.SearchWindow import *
from processing_tracking.GUI import *
from processing_tracking.perform_tracking_utilities import *
from processing_tracking.stabilize import *
import time


system_mode = "debug "
should_add_gaussian_noise = False


def perform_tracking():
    start_time_prog = time.time()
    if system_mode != "debug ":
        input_video = input("Please enter a video path:\n")
    else:
        input_video = ".\\..\\videos\\walking.mp4"
    try:
        cap = cv2.VideoCapture(input_video)
        select_target_flag = False
        first_flag = False

        # Check if opened successfully
        if not cap.isOpened():
            print("Error opening video stream or file")
            return

        #stabilize video
        '''
        video_stabilization(cap)
        cap.release()
        input_video = ".\\..\\process_tracking\\stabilized.avi"
        cap = cv2.VideoCapture(input_video)
        if not cap.isOpened():
            print("Error opening video stream or file")
            return
        '''
        # background subtraction
        # Randomly select 25 frames to create background for background subtraction
        frameIds = cap.get(cv2.CAP_PROP_FRAME_COUNT) * np.random.uniform(size=25)

        # Store selected frames in an array
        frames = []
        for fid in frameIds:
            cap.set(cv2.CAP_PROP_POS_FRAMES, fid)
            ret, frame = cap.read()
            resized_frame = frame_scaling(frame)
            frames.append(resized_frame)

        # Calculate the median along the time axis
        background = np.median(frames, axis=0).astype(dtype=np.uint8)
        gray_background = cv2.cvtColor(background, cv2.COLOR_BGR2GRAY)
        cap.release()

        cap = cv2.VideoCapture(input_video)
        # retrieving Resolution
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Retrieving fps
        fps = int(cap.get(cv2.CAP_PROP_FPS))

        # Defining the codec and creating VideoWriter object. The output is stored in 'Vid1_Binary.avi' file.
        out1 = cv2.VideoWriter('berlin_walk.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), fps,
                               (frame_width, frame_height))
        red = [0, 0, 255]

        # Read until video is completed
        retries = 0
        while cap.isOpened():
            # Capture frame-by-frame
            ret, frame = cap.read()

            if ret:
                # adjusting frame size to fit screen properly
                resized_frame = frame_scaling(frame)

                # converting to grayscale in order to calculate correlation and applying background subtraction mask
                gray = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2GRAY)

                mask = cv2.absdiff(gray_background, gray)
                _, mask = cv2.threshold(mask, 25, 255, cv2.THRESH_BINARY)

                if not select_target_flag:  # creating the target only once
                    target_info = Target(resized_frame, mask)
                    search_window_info = SearchWindow(target_info)
                    kalman = KalmanFilter(target_info, fps)
                    state_holder = StateMachine(target_info)
                    select_target_flag = True
                # creating the search window for the current frame
                search_window_info.update_search_window(target_info, mask)

                if should_add_gaussian_noise:
                    add_gaussian_noise(search_window_info)
                start_time_corr = time.time()
                correlation_prediction = get_correlation_prediction(target_info, search_window_info)
                start_time_cmass = time.time()
                center_of_mass_prediction = get_center_of_mass_prediction(search_window_info)
                start_time_state = time.time()
                current_state = state_holder.get_current_state(search_window_info, center_of_mass_prediction,
                                                               correlation_prediction)
                prediction = get_integrated_prediction(correlation_prediction, center_of_mass_prediction, state_holder)

                if current_state == OVERLAP or current_state == CONCEALMENT:
                    kalman.base_kalman_prior_prediction()
                else:
                    kalman.base_measurement()

                final_prediction = kalman.get_prediction(prediction)
                x, y = final_prediction[0][0], final_prediction[1][0]
                cv2.rectangle(resized_frame, (y - int(target_info.target_h / 2), x - int(target_info.target_w / 2)),
                              (y + int(target_info.target_h / 2), x + int(target_info.target_w / 3)), red, 1)

                target_info.update_position(y, x)
                state_holder.update_previous_pos((x, y))
                # Display the resulting frame
                cv2.imshow('Frame', resized_frame)

                # Write the frame into the file
                out1.write(resized_frame)

                # Press Q on keyboard to  exit
                if cv2.waitKey(25) & 0xFF == ord('q'):
                    break
                print("correlation took", str(time.time() - start_time_corr), "sec to run")
                print("center of mass took", str(time.time() - start_time_cmass), "sec to run")
                print("state machine", str(time.time() - start_time_state), "sec to run")
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
    print("The program took", str(time.time() - start_time_prog), "sec to run")

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
