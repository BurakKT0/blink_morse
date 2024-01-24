import config as cfg
import dlib
import cv2
import time
import numpy as np
from imutils import face_utils
from scipy.spatial import distance as dist


class eye_blink_detector():
    def __init__(self):
        self.start_time = None
        self.detector_faces = dlib.get_frontal_face_detector()
        self.predictor_eyes = dlib.shape_predictor(cfg.eye_landmarks)

    def eye_blink(self, gray, rect, counter, morse):
        (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
        (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

        shape = self.predictor_eyes(gray, rect)
        shape = face_utils.shape_to_np(shape)

        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]
        leftEAR = self.eye_aspect_ratio(leftEye)
        rightEAR = self.eye_aspect_ratio(rightEye)
        ear = (leftEAR + rightEAR) / 2.0

        # Check for a blink
        if ear < cfg.EYE_AR_THRESH:
            counter += 1
            if self.start_time is None:
                self.start_time = time.time()
        else:
            # Check if the blink duration meets the criteria
            if counter >= cfg.EYE_AR_CONSEC_FRAMES:
                if self.start_time is not None:
                    end_time = time.time()
                    blink_duration = end_time - self.start_time
                    print(blink_duration)
                    if blink_duration <= 0.5:
                        morse += '.'
                    elif blink_duration < 1 and blink_duration > 0.5:
                        morse += '-'
                    else:
                        morse += '/'
                    self.start_time = None
            counter = 0
        return counter, morse

    # Calculate the eye aspect ratio
    def eye_aspect_ratio(self, eye):
        A = dist.euclidean(eye[1], eye[5])
        B = dist.euclidean(eye[2], eye[4])
        C = dist.euclidean(eye[0], eye[3])
        ear = (A + B) / (2.0 * C)
        return ear

# Convert rectangles to a numpy array
def convert_rectangles2array(rectangles,image):
    res = np.array([])
    for box in rectangles:
        [x0,y0,x1,y1] = max(0, box.left()), max(0, box.top()), min(box.right(), image.shape[1]), min(box.bottom(), image.shape[0])
        new_box = np.array([x0,y0,x1,y1])
        if res.size == 0:
            res = np.expand_dims(new_box,axis=0)
        else:
            res = np.vstack((res,new_box))
    return res

# Calculate areas of the given boxes
def get_areas(boxes):
    areas = []
    for box in boxes:
        x0,y0,x1,y1 = box
        area = (y1-y0)*(x1-x0)
        areas.append(area)
    return areas

# Draw bounding boxes on the image
def bounding_box(img,box,match_name=[]):
    for i in np.arange(len(box)):
        x0,y0,x1,y1 = box[i]
        img = cv2.rectangle(img,
                    (x0,y0),
                    (x1,y1),
                    (0,255,0),3);
        if not match_name:
            continue
        else:
            cv2.putText(img, match_name[i], (x0, y0-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)
    return img