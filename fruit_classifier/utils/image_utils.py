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
