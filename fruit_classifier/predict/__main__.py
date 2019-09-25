import argparse
import cv2
import numpy as np
from pathlib import Path
from fruit_classifier.predict.predict_utils import classify_many
from fruit_classifier.predict.predict_utils import draw_class_on_image
from fruit_classifier.predict.predict_utils import inverse_encode
from fruit_classifier.predict.predict_utils import load_classifier
from fruit_classifier.utils.image_utils import open_image
from fruit_classifier.preprocessing.preprocessing_utils import \
    resize_image


def main(image_path, model_files_dir, model_name, show_image=False):
    """
    Predict the class of an image

    Parameters
    ----------
    image_path : str
        The image path as a string
    model_files_dir : Path
        Directory to the model files
    model_name : str
        Name of model
        Will be loaded from
        model_files_dir/models/model_name/model.h5
    show_image : bool
        Whether or not to use cv2.imshow to display the image

    Returns
    -------
    output : np.array, shape (height, width, channels)
        The original image annotated with the class label and confidence
    """

    # Load the image
    image = open_image(Path(image_path))
    orig = image.copy()

    # Pre-process the image for classification
    image = resize_image(image)
    # Expand the dimension (i.e. make the batch size = 1)
    image = np.expand_dims(image, axis=0)

    model = load_classifier(model_files_dir, model_name)
    labels, probabilities = classify_many(model, image)
    labels = inverse_encode(labels, model_files_dir, model_name)

    label = labels[0]
    probability = np.max(probabilities[0])

    probability_text = '{}: {:.2f}%'.format(label, probability * 100)

    # Draw the label on the image
    output = draw_class_on_image(orig, probability_text)

    if show_image:
        # Show the output image (convert from RGB to GBR)
        cv2.imshow('Output', output[..., ::-1])
        cv2.waitKey(0)

    return output


if __name__ == '__main__':
    # Construct the argument parse and parse the arguments
    parser = argparse.ArgumentParser(description='Predict the class '
                                                 'of an image')
    parser.add_argument('-i',
                        '--image',
                        required=True,
                        help='Path to input image')
    parser.add_argument('-m',
                        '--model_files_dir',
                        required=False,
                        default=None,
                        help='Path to the model files directory')
    parser.add_argument('-n',
                        '--model_name',
                        required=False,
                        help='Name of the resulting model')
    args = parser.parse_args()

    if args.model_files_dir is None:
        model_files_dir_ = \
            Path(__file__).absolute().parents[2].\
            joinpath('model_files')
    else:
        model_files_dir_ = args.model_files_dir

    if args.model_name is None:
        model_name_ = 'basic'
    else:
        model_name_ = args.model_name

    main(args.image, model_files_dir_, model_name_, show_image=True)
