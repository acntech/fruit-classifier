import pickle
import random
import cv2
import numpy as np
from pathlib import Path
from keras.engine.saving import load_model
from skimage.transform import resize


def draw_class_on_image(image, probability_text):
    """
    Draws the class and confidence on the image

    Parameters
    ----------
    image : np.array, shape (height, width, channels)
        The image to draw on
    probability_text : str
        The text to print

    Returns
    -------
    output_image : np.array, shape (height, width, channels)
        The image with text
    """
    orig_shape = np.array(image.shape)
    width = 400
    height = (orig_shape[1] * (width / orig_shape[0])).astype(int)

    output_image = resize(image,
                          output_shape=(width, height),
                          mode='reflect',
                          anti_aliasing=True).astype(np.uint8)

    cv2.putText(output_image,
                probability_text,
                (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2)

    return output_image


def classify(model, images):
    """
    Classifies a single image and returns the label and probability

    Parameters
    ----------
    model : Sequential
        The model to predict from
    images : np.array (examples,  height, width, channels)
        The images to predict

    Returns
    -------
    labels : np.array, shape (examples, )
        The label with the highest probability
    probabilities : np.array, shape (examples, n_classes)
        The probabilities of all the classes according to the label
        encoder
    """
    probabilities = model.predict(images)
    labels = np.argmax(probabilities, axis=1)

    # Load the label encoder
    encoder_dir = \
        Path(__file__).absolute().parents[2].joinpath('generated_data',
                                                      'encoders')
    encoder_path = encoder_dir.joinpath('encoder.pkl')

    with encoder_path.open('rb') as f:
        label_encoder = pickle.load(f)

    labels = label_encoder.inverse_transform(labels)

    return labels, probabilities


def load_classifier():
    """
    Loads the classifier

    Returns
    -------
    model : Sequential
        The model to classify from
    """
    print('[INFO] loading network...')
    model_dir = \
        Path(__file__).absolute().parents[2].joinpath('generated_data',
                                                      'models')
    model_path = model_dir.joinpath('model.h5')
    model = load_model(str(model_path))

    return model


def pick_random_image(from_directory):
    """
    Picks a random image from a random directory in "from_directory"

    Note, this includes images from both training set and test set.

    Parameters
    ----------
    from_directory : str
        The directory where an image is found

    Returns
    -------
    image_path : str
        the path to the image

    """
    generated_data_dir = \
        Path(__file__).absolute().parents[2].joinpath(
            'generated_data')
    image_dir = generated_data_dir.joinpath(from_directory)
    p = Path(image_dir)
    random_directory = random.choice(
        [x for x in p.iterdir() if x.is_dir()])
    sub_dir = image_dir.joinpath(random_directory)
    f = Path(sub_dir)
    image_path = random.choice(
        [x for x in f.iterdir() if not x.is_dir()])
    return image_path
