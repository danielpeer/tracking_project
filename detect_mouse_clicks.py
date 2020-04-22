import cv2

refPt = (0, 0)
pressed = False


def click(event, x, y, flags, param):
    global refPt, pressed
    if event == cv2.EVENT_LBUTTONDOWN:
        refPt = (y, x)
        pressed = True


def get_square_center(image_path):
    global refPt, pressed
    #image = cv2.imread(image_path)
    clone = image_path.copy()
    cv2.namedWindow("image")
    cv2.startWindowThread()
    cv2.setMouseCallback("image", click)
    while True:
        # display the image and wait for a keypress
        cv2.imshow("image", image_path)
        key = cv2.waitKey(1) & 0xFF
        if pressed:
            break
        # if the 'c' key is pressed, break from the loop
        if key == ord("c"):
            break
    print(refPt)
    return refPt
