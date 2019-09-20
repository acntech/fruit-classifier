import argparse
from pathlib import Path
from fruit_classifier.preprocessing.preprocessing_utils import \
    copy_valid_images
from fruit_classifier.preprocessing.preprocessing_utils import \
    resize_images
from fruit_classifier.utils.file_utils import truncate_filenames


def main(dataset_name='basic', height=28, width=28):
    """"
    Pre-processes the images in `data/raw`

    The resulting images are stored in `data/interim/dataset_name`

    Parameters
    ----------
    dataset_name : str
        Name of the dataset
    height : int
        Height of the resized image
    width : int
        Width of the resized image
    """

    data_dir = \
        Path(__file__).absolute().parents[2].joinpath('data')
    raw_dir = data_dir.joinpath('raw')
    interim_dir = data_dir.joinpath('interim', dataset_name)

    # Shorten filenames if they are so long that Windows protests
    truncate_filenames(raw_dir)

    if not interim_dir.is_dir():
        interim_dir.mkdir(parents=True, exist_ok=True)

    copy_valid_images(raw_dir, interim_dir)

    # FIXME: Resize based on input
    # FIXME: Shape must reflect shape in model
    resize_images(interim_dir, height, width)


if __name__ == '__main__':
    # Construct the argument parse and parse the arguments
    parser = argparse.ArgumentParser(description='Preprocess a dataset')
    parser.add_argument('-d',
                        '--dataset_name',
                        required=False,
                        help='Name of the resulting dataset')
    parser.add_argument('-h',
                        '--height',
                        default=28,
                        help='Height of the resized images')
    parser.add_argument('-w',
                        '--width',
                        default=28,
                        help='Width of the resized images')
    args = parser.parse_args()

    if args.dataset_name is None:
        dataset_name_ = 'basic'
    else:
        dataset_name_ = args.dataset_name

    main(dataset_name_, args.width, args.heigth)
