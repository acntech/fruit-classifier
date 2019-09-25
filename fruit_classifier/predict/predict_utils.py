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
                          anti_aliasing=True).astype(np.uint8)

    cv2.putText(output_image,
                probability_text,
                (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2)

    return output_image


def classify_many(model, images):
    """
    Classifies a images and returns the labels and probabilities

    Parameters
    ----------
    model : Sequential
        The model to predict from
    images : np.array (examples,  height, width, channels)
        The images to predict

    Returns
    -------
    labels : np.array, shape (examples, n_classes)
        The labels with the highest probability
    probabilities : np.array, shape (examples, n_classes)
        The probabilities of all the classes according to the label
        encoder
    """

    probabilities = model.predict(images)
    labels = np.argmax(probabilities, axis=1)

    return labels, probabilities


def inverse_encode(labels, model_files_dir, model_name):
    """
    Inverse encodes the labels

    Parameters
    ----------
    labels
    model_files_dir
    model_name

    Returns
    -------

    """
    # Load the label encoder
    encoder_path = model_files_dir.joinpath('encoders',
                                           model_name,
                                           'encoder.pkl')
    with encoder_path.open('rb') as f:
        label_encoder = pickle.load(f)
    labels = label_encoder.inverse_transform(labels)

    return labels


def load_classifier(model_files_dir, model_name='basic'):
    """
    Loads the classifier

    Parameters
    ----------
    model_files_dir : Path
        Path to the model files directory
    model_name : str
        Name of the model

    Returns
    -------
    model : Sequential
        The model to classify_many from
    """

    print('[INFO] loading network...')

    model_path = model_files_dir.joinpath('models',
                                         model_name,
                                         'model.h5')
    model = load_model(str(model_path))

    return model
