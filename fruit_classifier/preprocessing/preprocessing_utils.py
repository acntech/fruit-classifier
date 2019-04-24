from skimage.transform import resize
from pathlib import Path
import shutil
import cv2


def remove_non_images(raw_dir, clean_dir):
    """
    Removes images which are not readable

    Parameters
    ----------
    raw_dir : Path
        Path to the raw dataset
    clean_dir : Path
        Path for the cleaned dataset
    """

    shutil.copytree(str(raw_dir), str(clean_dir))

    # Find all image_paths
    image_paths = sorted(clean_dir.glob('**/*'))
    image_paths = [image_path for image_path in image_paths if
                   image_path.is_file()]
    for image_path in image_paths:
        image = cv2.imread(str(image_path))

        if image is None:
            print('Un-linking {}'.format(image_path))
            image_path.unlink()
    dirs = sorted(raw_dir.glob('*'))
    dirs = [d for d in dirs if d.is_dir()]
    for d in dirs:
        all_files = list(d.glob('*'))
        print('{} files found in {}'.format(d, len(all_files)))


def preprocess_image(image):
    """
    Pre-processes a single image

    Parameters
    ----------
    image : np.array, shape (height, width, channels)
        The image to resize

    Returns
    -------
    preprocessed_image : np.array, shape (new_h, new_w, new_c)
        The preprocessed image
    """

    preprocessed_image = resize(image,
                                output_shape=(28, 28),
                                mode='reflect',
                                anti_aliasing=True)
    preprocessed_image = preprocessed_image / 255.0

    return preprocessed_image
