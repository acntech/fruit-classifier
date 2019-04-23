"""
This file contains the inference algorithm
"""

from keras.preprocessing.image import img_to_array
from keras.models import load_model
from threading import Thread
import subprocess
import numpy as np
import time
import cv2


MODEL_PATH = 'model.h5'
# initialize the total number of frames that *consecutively* contain
# santa along with threshold required to trigger the santa alarm
TOTAL_CONSEC = 0
TOTAL_THRESH = 20


def send_signal():
    print('This is a signal!!!')


# initialize if the alarm has been triggered
ALARM = False
ALARM_ON = 'Apple'
CUR_LABEL = None
PREV_LABEL = None

# load the model
print("[INFO] loading model...")
model = load_model(MODEL_PATH)


# initialize the video stream and allow the camera sensor to warm up
print("[INFO] starting video stream...")
# NOTE: sudo modprobe bcm2835-v4l2 must be called prior to opencv to work
subprocess.run('sudo modprobe bcm2835-v4l2'.split(' '))

# The argument is referring to the camera number
cap = cv2.VideoCapture(0)
# NOTE: This is not full resolution
# https://picamera.readthedocs.io/en/release-1.12/fov.html
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

time.sleep(2.0)

# loop over the frames from the video stream
while True:
    # grab the frame from the threaded video stream and resize it
    # to have a maximum width of 400 pixels
    _, frame = cap.read()

    # prepare the image to be classified by our deep learning network
    image = cv2.resize(frame, (28, 28))
    image = image.astype("float") / 255.0
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)

    # classify the input image and initialize the label and
    # probability of the prediction
    probabilities = model.predict(image)[0]
    label = np.argmax(probabilities)
    probability = probabilities[label]

    # build the label
    if label == 0:
        label = 'Apple'
    elif label == 1:
        label = 'Banana'
    elif label == 2:
        label = 'Orange'

    CUR_LABEL = label

    if PREV_LABEL == label:
        # increment the total number of consecutive frames that
        # contain santa
        TOTAL_CONSEC += 1

        if not ALARM and TOTAL_CONSEC >= TOTAL_THRESH:
            ALARM = True

            thread = Thread(target=send_signal, args=())
            thread.daemon = True
            thread.start()

    else:
        TOTAL_CONSEC = 0
        ALARM = False

    PREV_LABEL = CUR_LABEL

    # build the label and draw it on the frame
    label = "{}: {:.2f}%".format(label, probability * 100)
    frame = cv2.putText(frame,
                        label,
                        (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 0),
                        2)

    # show the output frame
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
            break

# do a bit of cleanup
print("[INFO] cleaning up...")
cap.release()
cv2.destroyAllWindows()
