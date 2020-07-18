import threading
from queue import Queue

from filters.corr_tracker import *
from filters.kalman_filter import *
from processing_tracking_objects.target import Target
from processing_tracking_objects.target_info import *
from processing_tracking_objects.SearchWindow import *
from processing_tracking_objects.state_machine import *
from processing_tracking.perform_tracking_utilities import *
from image_processing.stabilize import *
import time

system_mode = "debug "
should_add_gaussian_noise = False

def get_prediction(target, color_image):
    results = [None, None]
    thread_correlation = threading.Thread(target=target.get_correlation_prediction, args=(results,))
    thread_correlation.start()
    thread_center_of_mass = threading.Thread(target=target.get_center_of_mass_prediction, args=(results,))
    thread_center_of_mass.start()
    thread_correlation.join()
    thread_center_of_mass.join()
    center_of_mass_prediction = results[1]
    correlation_prediction = results[0]
    current_state = target.state_holder.get_current_state(target.search_window,
                                                          center_of_mass_prediction, correlation_prediction)
    prediction = get_integrated_prediction(correlation_prediction, center_of_mass_prediction, target.state_holder)
    if current_state == OVERLAP or current_state == CONCEALMENT:
        target.kalman_filter.base_kalman_prior_prediction()
    else:
        target.kalman_filter.base_measurement()
    final_prediction = target.kalman_filter.get_prediction(prediction)
    x, y = final_prediction[0][0], final_prediction[1][0]
    target.calc_x_pos = x
    target.calc_y_pos = y
    target.target_info.update_position(y, x)
    target.state_holder.update_previous_pos((x, y))

def perform_tracking():
    j = 0
    start_time_prog = time.time()
    if system_mode != "debug ":
        input_video = input("Please enter a video path:\n")
    else:
        input_video = "C:\\Users\\danielpeer\\Downloads\\p.mp4"
    try:
        cap = cv2.VideoCapture(input_video)
        select_target_flag = False
        first_flag = False

        # Check if opened successfully
        if not cap.isOpened():
            print("Error opening video stream or file")
            return

        # stabilize video

        video_stabilization(cap)
        cap.release()
        input_video = 'C:\\Users\\danielpeer\\Downloads\\stabilized.avi'
        cap = cv2.VideoCapture(input_video)
        if not cap.isOpened():
            print("Error opening video stream or file")
            return

        # background subtraction
        # Randomly select 25 frames to create background for background subtraction
        frameIds = cap.get(cv2.CAP_PROP_FRAME_COUNT) * np.random.uniform(size=25)

        # Store selected frames in an array
        frames = []
        for fid in frameIds:
            cap.set(cv2.CAP_PROP_POS_FRAMES, fid)
            ret, frame = cap.read()
            # resized_frame = frame_scaling(frame)
            resized_frame = frame
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

        resized_frame_dim = get_frame_resize_dim(frame.shape)
        # Defining the codec and creating VideoWriter object. The output is stored in 'Vid1_Binary.avi' file.
        out1 = cv2.VideoWriter('berlin_walk.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), fps,
                               (resized_frame_dim[0], resized_frame_dim[1]))
        red = [0, 0, 255]

        # Read until video is completed
        retries = 0

        while cap.isOpened():
            j += 1
            if (j%7!=0):
                continue

            # Capture frame-by-frame
            ret, frame = cap.read()

            if ret:
                # adjusting frame size to fit screen properly
               # resized_frame = frame_scaling(frame)
                resized_frame = frame
                # converting to grayscale in order to calculate correlation and applying background subtraction mask
                gray = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2GRAY)
                mask = cv2.absdiff(gray_background, gray)
                _, mask = cv2.threshold(mask, 50, 255, cv2.THRESH_BINARY)
                cv2.imshow('mask', mask)
                if not select_target_flag:  # creating the target only once
                    targets_lst = []
                    threads_lst = []
                    i = 0
                    while True:

                        targets_lst.append(Target(resized_frame, mask, fps))
                        if targets_lst[i].target_info.target_w == 0 & targets_lst[i].target_info.target_h == 0:
                            targets_lst.pop(i)  # once c pressed, another null object is added to the list of
                            # targets, therefore remove
                            break
                        i += 1
                    select_target_flag = True

                # creating the search windows for the current frame
                for current_target in targets_lst:
                    current_target.update_search_window(mask)
                  #  if should_add_gaussian_noise:
                   #    add_gaussian_noise(search_window_lst[i])
                start_processing = time.time()
                for target in targets_lst:
                    thread = threading.Thread(target=get_prediction, args=(target,resized_frame))
                    threads_lst.append(thread)
                    thread.start()
                for thread in threads_lst:
                    thread.join()
                end_processing = time.time()
            #    print("total time", str(start_processing - end_processing), "sec to run")
                # calculating predictions for each target
                for target in targets_lst:
                    cv2.rectangle(resized_frame, (
                        target.calc_y_pos - int(target.target_info.target_h / 2),
                        target.calc_x_pos - int(target.target_info.target_w / 2)),
                                  (target.calc_y_pos + int(target.target_info.target_h / 2),
                                   target.calc_x_pos + int(target.target_info.target_w / 3)), red, 1)

                # Display the resulting frame
                cv2.imshow('Frame', resized_frame)
                # Write the frame into the file
                out1.write(resized_frame)
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
