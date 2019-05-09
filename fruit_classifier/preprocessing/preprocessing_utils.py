import cv2
from keras.preprocessing.image import ImageDataGenerator
from tqdm import tqdm
from skimage.transform import resize
from pathlib import Path
from fruit_classifier.utils.file_utils import copytree
import os


def truncate_filenames(raw_dir):
    folder_list = os.listdir(raw_dir)
    print("Reducing length of filenames so that combined path to a "
          "file is maximum 255 characters long")
    max_windows_path_length = 255

    for folder in folder_list:
        raw_sub_dir = raw_dir.joinpath(folder)
        # length of folders path + "/"
        sub_dir_len = len(str(raw_sub_dir)) + 1
        # length available for image file name including type
        available_max_len = max_windows_path_length - sub_dir_len
        # Get all files in folder
        file_list = os.listdir(raw_sub_dir)
        num_renamed = 0
        for filename in file_list:
            name_len = len(filename)
            if name_len <= available_max_len:
                continue

            possible_types = filename.split('.')
            file_type = possible_types[-1]
            cut_position = available_max_len - len(file_type) - 1
            new_name = filename[0:cut_position] + '.' + file_type

            old_path = raw_sub_dir.joinpath(filename)
            new_path = raw_sub_dir.joinpath(new_name)

            os.rename(old_path, new_path)
            num_renamed = num_renamed + 1
        print('Truncated ' + str(num_renamed) + ' filenames in folder: '
              + folder)


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

    copytree(raw_dir, clean_dir)

    # Find all image_paths
    image_paths = sorted(clean_dir.glob('**/*'))
    image_paths = [image_path for image_path in image_paths if
                   image_path.is_file()]

    for image_path in tqdm(image_paths, desc='Checking images'):
        image = cv2.imread(str(image_path))

        if image is None:
            print('Un-linking {}'.format(image_path))
            image_path.unlink()

    raw_dirs = sorted(raw_dir.glob('*'))
    raw_dirs = [d for d in raw_dirs if d.is_dir()]

    clean_dirs = sorted(clean_dir.glob('*'))
    clean_dirs = [d for d in clean_dirs if d.is_dir()]

    print('\nResult of cleaning:')
    for r, c in zip(raw_dirs, clean_dirs):
        n_raw = len(list(r.glob('*')))
        n_clean = len(list(c.glob('*')))
        print('    {}/{} remaining in {}'.format(n_clean, n_raw, c))


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


def get_image_generator(rotation_range=30,
                        width_shift_range=0.1,
                        height_shift_range=0.1,
                        shear_range=0.2,
                        zoom_range=0.2,
                        horizontal_flip=True,
                        fill_mode='nearest'):
    """
    Returns the image generator

    Parameters
    ----------
    rotation_range : int
        Degree range for random rotations
    width_shift_range : float
        Fraction of total width, if < 1, or pixels if >= 1
    height_shift_range : float
        Fraction of total height, if < 1, or pixels if >= 1
    shear_range : float
        Shear intensity
    zoom_range : float
        Range for random zoom
    horizontal_flip : bool
        Randomly flip inputs horizontally
    fill_mode : ["constant"|"nearest"|"reflect"|"wrap"]
        How points outside the boundaries of the input should be filled

    Returns
    -------
    image_generator : ImageDataGenerator
        Generator used for batches
    """

    image_generator =\
        ImageDataGenerator(rotation_range=rotation_range,
                           width_shift_range=width_shift_range,
                           height_shift_range=height_shift_range,
                           shear_range=shear_range,
                           zoom_range=zoom_range,
                           horizontal_flip=horizontal_flip,
                           fill_mode=fill_mode)

    return image_generator
