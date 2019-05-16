import pickle
import cv2
import numpy as np
from pathlib import Path
from keras.engine.saving import load_model
from skimage.transform import resize


def draw_text_with_background(output_image,
                              text="text",
                              text_offset_x=10,
                              text_offset_y=25,
                              font_scale=.7,
                              font_thickness=2,
                              font_color=(0, 0, 0),
                              font_bg_color=(255, 255, 255),
                              font=cv2.FONT_HERSHEY_SIMPLEX,
                              padding=5):
    """
    Draws text inside a colored box ontop of an image

    Parameters
    ----------
    output_image    : cv2 image -  The image
    text            : str - The text
    text_offset_x   : integer - text start position x axis
    text_offset_y   : integer - text start position y axis
    font_scale      : integer - font scale used
    font_thickness  : integer - font thickness used
    font_color      : integer array - text color
    font_bg_color   : integer array - background box color
    font            : cv2 font - font of text
    padding         : integer - margin of the box around text
    """

    # get the width and height of the text box from the text
    (text_width, text_height) = \
        cv2.getTextSize(text, font, fontScale=font_scale,
                        thickness=font_thickness)[0]

    # make the coords of the box with a small padding
    box_coords = ((text_offset_x - padding,
                   text_offset_y + padding),
                  (text_offset_x + (text_width + padding),
                   text_offset_y - (text_height + padding)))

    # Draw the background box onto the image variable
    cv2.rectangle(output_image, box_coords[0], box_coords[1],
                  font_bg_color, cv2.FILLED)

    # Draw the text onto the image variable and display it
    cv2.putText(output_image, text, (text_offset_x, text_offset_y),
                font, fontScale=font_scale, color=font_color,
                thickness=font_thickness)


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

    draw_text_with_background(output_image, probability_text)

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
