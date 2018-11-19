# import the necessary packages

from __future__ import division
from scipy.spatial import distance as dist
import numpy as np
import time
import dlib
import cv2
from collections import OrderedDict


ii=0
num_of_sleep=0

# define two constants, one for the eye aspect ratio to indicate
# blink and then a second constant for the number of consecutive
# frames the eye must be below the threshold
EYE_AR_THRESH = 0.23
EYE_AR_CONSEC_FRAMES = 35

# initialize the frame counters and the total number of blinks
COUNTER = 0

# define a dictionary that maps the indexes of the facial
# landmarks to specific face regions
FACIAL_LANDMARKS_IDXS = OrderedDict([
    ("mouth", (48, 68)),
    ("right_eyebrow", (17, 22)),
    ("left_eyebrow", (22, 27)),
    ("right_eye", (36, 42)),
    ("left_eye", (42, 48)),
    ("nose", (27, 35)),
    ("jaw", (0, 17))
])

def Eye_AR(eye):
    # compute the euclidean distances between the two sets of
    # vertical eye landmarks (x, y)-coordinates
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
 
    # compute the euclidean distance between the horizontal
    # eye landmark (x, y)-coordinates
    C = dist.euclidean(eye[0], eye[3])
 
    # compute the eye aspect ratio
    EAR = (A + B) / (2.0 * C)

## The Eye Aspect Ratio is a constant value when the eye is open,
## but rapidly falls to 0 when the eye is closed.
## A program can determine if a person’s eyes are closed
## if the Eye Aspect Ratio falls below a certain threshold.
    
 
    # return the eye aspect ratio
    return EAR

def shapping(shape, dtype="int"):
    # initialize the list of (x, y)-coordinates
    coords = np.zeros((68, 2), dtype=dtype)

    # loop over the 68 facial landmarks and convert them
    # to a 2-tuple of (x, y)-coordinates
    for i in range(0, 68):
        coords[i] = (shape.part(i).x, shape.part(i).y)

    # return the list of (x, y)-coordinates
    return coords

camera = cv2.VideoCapture(0)

predictor_path = 'C:\Users\Eslam\Downloads\\shape_predictor_68_face_landmarks.dat'

print('[INFO] Downloading face detector and facial landmarks predictor...')
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)

# grab the indexes of the facial landmarks for the left and
# right eye, respectively
(lStart, lEnd) = FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = FACIAL_LANDMARKS_IDXS["right_eye"]

vs = cv2.VideoCapture(0)
fileStream = True
time.sleep(1.0)

# loop over frames from the video stream
while True:
    # grab the frame from the threaded video file stream, resize
    # it, and convert it to grayscale
    # channels)
    ret, frame = camera.read()
    if ret == False:
        print('Failed to capture frame from camera. Check camera index in cv2.VideoCapture(0) \n')
        break

    global ratio
    w, h, _ = frame.shape
    width=240
    ratio = width / w
    height = int(h * ratio)
    frame_resized = cv2.resize(frame, (height, width))


    ##frame_resized = resize(frame, width=240)
    frame_gray = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2GRAY)

    # detect faces in the grayscale frame
    rects = detector(frame_gray, 0)
    # loop over the face detections
    for rect in rects:
        # determine the facial landmarks for the face region, then
        # convert the facial landmark (x, y)-coordinates to a NumPy
        # array
        shape = predictor(frame_gray, rect)
        shape = shapping(shape)
 
        # extract the left and right eye coordinates, then use the
        # coordinates to compute the eye aspect ratio for both eyes
        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]
        leftEAR = Eye_AR(leftEye)
        rightEAR = Eye_AR(rightEye)
 
        # take the minimum eye aspect ratio
        EAR = min([leftEAR,rightEAR])
        # check to see if the eye aspect ratio is below the blink
        # threshold, and if so, increment the blink frame counter
        if EAR < EYE_AR_THRESH:
            COUNTER += 1

            if 70 >= COUNTER >= EYE_AR_CONSEC_FRAMES:
                cv2.putText(frame, "Warning! Seems he is trying to sleep", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                print "Warning! Seems he is trying to sleep"

            if COUNTER > 70 :
                cv2.putText(frame, "ALARM!!! ALARM!!! HE\'S SLEEPING", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                print "HE\'S SLEEPING"
                if ii is 0:
                    num_of_sleep=num_of_sleep+1
                    ii=1
                print num_of_sleep," is the number of sleeping times"

        else:
           # reset the eye frame counter
            COUNTER = 0
            ii=0


        for (x, y) in np.concatenate((leftEye, rightEye), axis=0):
            cv2.circle(frame, (int(x / ratio), int(y / ratio)), 2, (255, 255, 255), -1)

    # show the frame
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF
 
    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break

 
# close all
cv2.destroyAllWindows()
vs.release()
