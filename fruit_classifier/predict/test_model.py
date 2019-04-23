# import the necessary packages
from keras.preprocessing.image import img_to_array
from keras.models import load_model
from skimage.transform import resize
import numpy as np
import argparse
import cv2

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()

ap.add_argument('-i', '--image', required=True,
                help='path to input image')
args = vars(ap.parse_args())

# load the image
image = cv2.imread(args['image'])
orig = image.copy()

# pre-process the image for classification
image = cv2.resize(image, (28, 28))
image = image.astype('float') / 255.0
image = img_to_array(image)
image = np.expand_dims(image, axis=0)

# load the trained convolutional neural network
print('[INFO] loading network...')
model = load_model('model.h5')

# classify the input image
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

probability_text = '{}: {:.2f}%'.format(label, probability * 100)

# draw the label on the image
orig_shape = np.array(orig.shape)
width = 400
height = (orig_shape[1]*(width/orig_shape[0])).astype(int)
output = resize(orig,
                output_shape=(width, height),
                mode='reflect',
                anti_aliasing=True)
cv2.putText(output,
            probability_text,
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2)

# show the output image
cv2.imshow('Output', output)
cv2.waitKey(0)
