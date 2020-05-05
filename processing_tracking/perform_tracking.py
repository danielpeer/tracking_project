import cv2
from processing_tracking.perform_tracking_utilities import *
from correlation_filter.corr_tracker import *
from center_of_mass_filter.calculate_center_of_mass import *
from kalman_filter.kalman_filter import *
from videos import *

system_mode = "debug "


def perform_tracking():
    if system_mode != "debug ":
        input_video = input("Please enter a video path:\n")
    else:
        input_video = ".\\..\\videos\\conceal4.avi"
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
        out1 = cv2.VideoWriter('Corr_Tracker_conceal5.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), fps,
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
                    x, y, target = create_object_target(gray)
                    # x, y, target = create_target(gray)
                    select_target_flag = True
                    kalman = kalman_filter((x, y), fps)
                    window_w = target.shape[0] * 6
                    window_h = target.shape[1] * 6
                # creating the search window for the current frame
                top_left_corner_x, top_left_corner_y, search_window = create_window(x, y, window_w, window_h, gray)
                prior_prediction = kalman.get_prior_estimate()
                search_window = add_gaussian_noise(search_window)
                correlation_predictions = get_correlation_prediction(x, y, search_window, target, top_left_corner_x,
                                                                     top_left_corner_y)

                if not(np.array_equal(correlation_predictions, np.array([[-1], [-1]]))):
                    #object is not hidden
                    kalman.update_process_noise_covariance(0)
                    center_of_mass_predictions = get_center_of_mass_prediction(x, y, search_window, top_left_corner_x,
                                                                               top_left_corner_y)
                    filter_predictions = get_integrated_prediction(center_of_mass_predictions, correlation_predictions,
                                                                   prior_prediction)
                    posterior_prediction = kalman.get_prediction(filter_predictions)

                else:
                    #object is hidden
                    kalman.update_process_noise_covariance(1)
                    posterior_prediction = kalman.get_prediction(prior_prediction)

                x, y = int(posterior_prediction[0]), int(posterior_prediction[1])
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
