import cv2

############################################################### Search Window #####################################################


window_w = 60
window_h = 60
target_w = 5
target_h = 5

def create_window(x, y, window_w, window_h, gray):
    """
    creating the search window for the correlation algorithm
    x,y - the coordinates of the target which was chosen by the user
    window_w, window_h - width and height of the search window. default numbers
    gray - the frame in grayscale
    returns - the top left corner coordinates of the search window and the search windows itself
    """
    x_window = x
    y_window = y
    gray_width, gray_height = gray.shape
    top_left_corner_x = x - window_w
    top_left_corner_y = y - window_h
    if x - window_w < 0:
        x_window = window_w
        top_left_corner_x = 0
    if y - window_h < 0:
        y_window = window_h
        top_left_corner_y = 0
    if x + window_w > gray_width:
        x_window = gray_width
    if y + window_h > gray_height:
        y_window = gray_height
    search_window = gray[x_window - window_w: x_window + window_w, y_window - window_h: y_window + window_h]
    return top_left_corner_x, top_left_corner_y, search_window,search_window.shape[0], search_window.shape[1]

######################################### detect mouse clicks #####################################################################
refPt =(0,0)
pressed = False


def click(event, x, y, flags, param):
    global refPt, pressed
    if event == cv2.EVENT_LBUTTONDOWN:
        refPt = (y, x)
        pressed = True


def get_square_center(first_frame):
    global refPt, pressed
    cv2.namedWindow("first frame")
    cv2.startWindowThread()
    cv2.setMouseCallback("first frame", click)
    while True:
        # display the image and wait for a keypress
        cv2.imshow("first frame", first_frame)
        key = cv2.waitKey(1) & 0xFF
        if pressed:
            break
        # if the 'c' key is pressed, break from the loop
        if key == ord("c"):
            break
    print(refPt)
    return refPt




################################################ target position ##################################################################

def create_target(gray):
    """
    creating the target matrix for the correlation algorithm
    gray - the frame in grayscale
    returns -  x,y - the coordinates of the target which was chosen by the user and the target matrix itself
    """
    x, y = get_square_center(gray)
    gray_width, gray_height = gray.shape
    x_targ = x
    y_targ = y
    if x_targ - target_w < 0:
        x_targ = target_w
    if y_targ - target_h < 0:
        y_targ = target_w
    if x_targ + 15 > gray_width:
        x_targ = gray_width
    if y_targ + 15 > gray_height:
        y_targ = gray_height
    target = gray[x_targ - target_w: x_targ + target_w, y_targ - target_h: y_targ + target_h]
    return x, y, target

##########################################################################################################

