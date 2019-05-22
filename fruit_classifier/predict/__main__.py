import argparse
import cv2
import numpy as np
from pathlib import Path
from fruit_classifier.predict.predict_utils import draw_class_on_image
from fruit_classifier.predict.predict_utils import pick_random_image
from fruit_classifier.predict.predict_utils import classify
from fruit_classifier.predict.predict_utils import load_classifier
from fruit_classifier.utils.image_utils import open_image
from fruit_classifier.preprocessing.preprocessing_utils import \
    preprocess_image


def main(image_path='', show_image=False):
    """
    Predict the class of an image

    Parameters
    ----------
    image_path : str
        The image path as a string
    show_image : bool
        Whether or not to use cv2.imshow to display the image

    Returns
    -------
    output : np.array, shape (height, width, channels)
        The original image annotated with the class label and confidence
    """

    if image_path == '':
        image_path = pick_random_image(from_directory='cleaned_data')

    # Load the image
    image = open_image(Path(image_path))
    orig = image.copy()

    # Pre-process the image for classification
    image = preprocess_image(image)
    # Expand the dimension (i.e. make the batch size = 1)
    image = np.expand_dims(image, axis=0)

    # Load the trained convolutional neural network
    model = load_classifier()

    # Classify the input image
    labels, probabilities = classify(model, image)
    label = labels[0]
    probability = np.max(probabilities[0])

    probability_text = '{}: {:.2f}%'.format(label, probability * 100)

    # Draw the label on the image
    output = draw_class_on_image(orig, probability_text)

    if show_image:
        # Show the output image
        cv2.imshow('Output', output)
        cv2.waitKey(0)

    return output


if __name__ == '__main__':
    # Construct the argument parse and parse the arguments
    parser = argparse.ArgumentParser(description='Predict the class '
                                                 'of an image')
    parser.add_argument('-i',
                        '--image',
                        required=False,
                        help='Path to input image',
                        default='')
    args = parser.parse_args()

    main(args.image, show_image=True)

