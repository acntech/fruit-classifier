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
        The image as a numpy array (RGB order)
    """

    image = cv2.imread(str(image_path))
    # Cast to array and convert from BGR (OpenCV standard) to RGB
    image_array = img_to_array(image)[..., ::-1]

    return image_array


def get_image_paths(path):
    """
    Returns a list of image paths

    Parameters
    ----------
    path : Path
        Path to the training images

    Returns
    -------
    image_paths : list
        A list of image paths
    """

    image_paths = sorted(path.glob('**/*'))
    image_paths = [p for p in image_paths if p.is_file()]

    return image_paths
