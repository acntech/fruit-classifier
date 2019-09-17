import random

import cv2
from keras.preprocessing.image import img_to_array


def open_image(image_path):
    """
    Opens the image at the given path

    Parameters
    ----------
    image_path : Path
        The path to the image

    Returns
    -------
    image_array : np.array, shape (height, width, channels)
        The image as a numpy array
    """

    image = cv2.imread(str(image_path))
    image_array = img_to_array(image)

    return image_array


def get_image_paths(path):
    """
    # FIXME: Move shuffle to train_utils
    Returns a list of random shuffled image paths

    Parameters
    ----------
    path : Path
        Path to the training images

    Returns
    -------
    image_paths : list
        Random shuffled image paths
    """

    # FIXME: Move this print statement
    print('[INFO] Loading images...')

    image_paths = sorted(path.glob('**/*'))
    image_paths = [p for p in image_paths if p.is_file()]
    random.seed(42)
    random.shuffle(image_paths)

    return image_paths
