import cv2
import imghdr
from keras.preprocessing.image import ImageDataGenerator
from tqdm import tqdm
from skimage.transform import resize
from skimage.io import imsave
from pathlib import Path
from fruit_classifier.utils.file_utils import copytree
from fruit_classifier.utils.image_utils import get_image_paths, \
    open_image


def copy_valid_images(raw_dir, interim_dir):
    """
    Removes images which are not readable

    Parameters
    ----------
    raw_dir : Path
        Path to the raw dataset
    interim_dir : Path
        Path for the cleaned dataset
    """

    copytree(raw_dir, interim_dir)

    # Find all image_paths
    image_paths = sorted(interim_dir.glob('**/*'))
    image_paths = [image_path for image_path in image_paths if
                   image_path.is_file()]

    for image_path in tqdm(image_paths, desc='Checking images'):
        image = cv2.imread(str(image_path))

        if image is None:
            print('Un-linking {}'.format(image_path))
            image_path.unlink()

    raw_dirs = sorted(raw_dir.glob('*'))
    raw_dirs = [d for d in raw_dirs if d.is_dir()]

    interim_dirs = sorted(interim_dir.glob('*'))
    interim_dirs = [d for d in interim_dirs if d.is_dir()]

    print('\nResult of cleaning:')
    for r, c in zip(raw_dirs, interim_dirs):
        n_raw = len(list(r.glob('*')))
        n_clean = len(list(c.glob('*')))
        print('    {}/{} remaining in {}'.format(n_clean, n_raw, c))


def resize_image(image, height=28, width=28):
    """
    Resize a single image

    Parameters
    ----------
    image : np.array, shape (height, width, channels)
        The image to resize
    height : int
        Height of the resized image
    width : int
        Width of the resized image

    Returns
    -------
    resized_image : np.array, shape (new_h, new_w, new_c)
        The resized image
    """

    resized_image = resize(image,
                           output_shape=(height, width),
                           mode='reflect',
                           anti_aliasing=True)

    return resized_image.astype('uint8')


def resize_images(path, height=28, width=28):
    """
    Overwrites the images in `path` with the resized version

    Parameters
    ----------
    path : Path
        The path to the image files
    height : int
        Height of the resized images
    width : int
        Width of the resized images
    """
    image_paths = get_image_paths(path)

    for image_path in tqdm(image_paths, desc='Resizing images'):
        tqdm.write(str(image_path))
        image_array = open_image(image_path)
        resized_array = resize_image(image_array, height, width)
        # Determine image type as imsave is sensitive to the file
        # extension
        extension = imghdr.what(image_path)
        extension = f'.{extension}' if extension is not None else '.png'
        if extension == image_path.suffix:
            save_path = image_path
        else:
            save_path = \
                image_path.parent.joinpath(image_path.stem + extension)
            image_path.unlink()

        imsave(save_path, resized_array)


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
