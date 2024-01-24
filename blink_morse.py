from imutils.video import VideoStream
import cv2
import time
from face_detector import eye_blink_detector, convert_rectangles2array, get_areas, bounding_box
from morse_to_text import morse_to_text
import imutils
import numpy as np

# Instantiate detector
detector = eye_blink_detector()

# Initialize variables for blink detector
counter = 0
morse = ''
text = ''
clear_text = False

# Initialize video stream
vs = VideoStream(src=0).start()

# Button coordinates
button_x1, button_y1 = 10, 370
button_x2, button_y2 = 90, 340

def click_event(event, x, y, flags, param):
    global clear_text
    if event == cv2.EVENT_LBUTTONDOWN and button_x1 <= x <= button_x2 and button_y2 <= y <= button_y1:
        clear_text = True

cv2.namedWindow('blink-morse')
cv2.setMouseCallback('blink-morse', click_event)

while True:
    start_time = time.time()

    # Read frame from video stream
    frame = vs.read()
    frame = cv2.flip(frame, 1)
    frame = imutils.resize(frame, width=720)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    rectangles = detector.detector_faces(gray, 0)
    boxes_face = convert_rectangles2array(rectangles, frame)

    if len(boxes_face) != 0:
        # Select the face with the largest area
        index = np.argmax(get_areas(boxes_face))
        rectangles = rectangles[index]
        boxes_face = np.expand_dims(boxes_face[index], axis=0)

        # Blink detector
        counter, morse = detector.eye_blink(gray, rectangles, counter, morse)
        if morse.endswith('/'):
            print(morse)
            text += morse_to_text(morse)
            morse = ''

        # Add bounding box
        frame = bounding_box(frame, boxes_face)
    else:
        img_post = frame

    # Display FPS
    end_time = time.time() - start_time
    fps = 1 / end_time
    cv2.putText(frame, f"FPS: {round(fps, 3)}", (5, 25), cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 0, 255), 2)

    # Display text
    text_position = (10, frame.shape[0] - 10)
    cv2.putText(frame, f"Text: {text}", text_position, cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)

    # Draw clear button
    cv2.rectangle(frame, (button_x1, button_y1), (button_x2, button_y2), (2, 2, 134), -1)
    cv2.putText(frame, "Clear", (button_x1, button_y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, (222, 222, 222), 2)
    cv2.imshow('blink-morse', frame)

    if clear_text:
        morse = ''
        text = ''
        clear_text = False

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
vs.stop()
