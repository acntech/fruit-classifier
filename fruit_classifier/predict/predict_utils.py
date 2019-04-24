import pickle
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
                          anti_aliasing=True)

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
    labels
    probabilities
    """
    probabilities = model.predict(images)
    labels = np.argmax(probabilities)

    # Load the label encoder
    encoder_dir = \
        Path(__file__).absolute().parents[1].joinpath('generated_data',
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
        Path(__file__).absolute().parents[1].joinpath('generated_data',
                                                      'models')
    model_path = model_dir.joinpath('model.h5')
    model = load_model(str(model_path))

    return model
