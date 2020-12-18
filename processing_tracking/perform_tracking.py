import threading
from queue import Queue

from filters.corr_tracker import *
from filters.kalman_filter import *
from processing_tracking_objects.target import Target
from processing_tracking_objects.targetinfo import *
from processing_tracking_objects.SearchWindow import *
from processing_tracking_objects.state_machine import *
from perform_tracking_utilities import *
from image_processing.stabilize import *
from ObjetRecognition.ObjectDetection import *
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
    current_state = target.state_holder.get_current_state(target, target.search_window,
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

    detect_outgoing_targets(target)
    target.target_info.update_position(x, y)
    target.state_holder.update_previous_pos((x, y))

def perform_tracking():
    start_time_prog = time.time()
    if system_mode != "debug ":
        input_video = input("Please enter a video path:\n")
    else:
        input_video = "C:\\Users\\97252\\Downloads\\pp.mp4"
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
        input_video = 'C:\\Users\\97252\\Downloads\\stabilized.avi'
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
                               (frame.shape[1], frame.shape[0]))
        red = [0, 0, 255]

        white = [255, 255, 255]

        # Read until video is completed
        retries = 0

        while cap.isOpened():

            # Capture frame-by-frame
            ret, frame = cap.read()

            if ret:
                # adjusting frame size to fit screen properly
               # resized_frame = frame_scaling(frame)
                resized_frame = frame
                # converting to grayscale in order to calculate correlation and applying background subtraction mask
                gray = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2GRAY)
                mask = cv2.absdiff(gray_background, gray)
                _, mask = cv2.threshold(mask, 60, 255, cv2.THRESH_BINARY)
                cv2.imshow('mask', mask)
                if not select_target_flag:  # creating the target only once
                    targets_lst = []
                    threads_lst = []
                    suspected_targets = []
                    i = 0
                    while True:
                        targets_lst.append(Target(resized_frame, mask, fps, []))
                        if targets_lst[i].target_info.target_w == 0 & targets_lst[i].target_info.target_h == 0:
                            targets_lst.pop(i)  # once c pressed, another null object is added to the list of
                            # targets, therefore remove
                            break
                        i += 1
                    #object_detector = ObjectDetector()
                    select_target_flag = True

                else:
                   #incoming_targets = detect_new_targets(resized_frame, object_detector, suspected_targets,targets_lst)
                   incoming_targets =[]
                   for target in incoming_targets:
                        target = Target(resized_frame, mask, fps, target, True)
                        if target.target_info.target_area > 0:
                            targets_lst.append(target)
                # creating the search windows for the current fra
                current_frame_target = get_target(targets_lst)
                for current_target in current_frame_target:
                    current_target.update_search_window(mask)
                for target in current_frame_target:
                    if not target.outgoing:
                        thread = threading.Thread(target=get_prediction, args=(target,resized_frame))
                        threads_lst.append(thread)
                        thread.start()
                for thread in threads_lst:
                    thread.join()
                # calculating predictions for each target
                for target in current_frame_target:
                    target_mask = np.zeros(frame.shape)
                    cv2.rectangle(target_mask, (
                         max(target.calc_y_pos - int(target.target_info.target_h / 2) - 20, 0),
                         max(target.calc_x_pos - int(target.target_info.target_w / 2) - 20, 0)),
                         (min(target.calc_y_pos + int(target.target_info.target_h / 2 + 20), 1280),
                          min(target.calc_x_pos + int(target.target_info.target_w / 3) + 20, 720)), white, -1)
                    target.update_target_image(target_mask.astype(int), frame)
                    if target.detection is None:
                       #target.detection = object_detector.get_target_detect(target.target_image)
                       a=5

                for target in current_frame_target:
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    cv2.putText(resized_frame, target.detection, (target.calc_y_pos - int(target.target_info.target_h / 2),target.calc_x_pos - int(target.target_info.target_w / 2)), font, 1, red, 2, cv2.LINE_AA)
                    cv2.rectangle(resized_frame, (
                        target.calc_x_pos - int(target.target_info.target_w / 2),
                        target.calc_y_pos - int(target.target_info.target_h / 2)),
                        (target.calc_x_pos + int(target.target_info.target_w / 2),
                         target.calc_y_pos + int(target.target_info.target_h / 2)), red, 1)

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

    except IOError as e:
        print(e)
        print("File not accessible")
    print("The program took", str(time.time() - start_time_prog), "sec to run")


def get_integrated_prediction(corr_prediction, center_of_mass_prediction, state_machine):
    center_of_mass_prediction = state_machine.center_of_mass_ratio * np.array(
        [[center_of_mass_prediction[0]], [center_of_mass_prediction[1]]])
    corr_prediction = state_machine.corr_ratio * np.array([[corr_prediction[0]], [corr_prediction[1]]])
    return center_of_mass_prediction + corr_prediction


perform_tracking()
