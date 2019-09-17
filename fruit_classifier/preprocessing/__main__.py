from pathlib import Path
from fruit_classifier.preprocessing.preprocessing_utils import \
    remove_non_images
from fruit_classifier.preprocessing.preprocessing_utils import \
    truncate_filenames


def main():
    """"
    Pre-processes the images in raw_data

    The resulting images are stored in cleaned_data
    """

    data_dir = \
        Path(__file__).absolute().parents[2].joinpath('data')
    raw_dir = data_dir.joinpath('raw')
    interim_dir = data_dir.joinpath('interim')

    # Shorten filenames if they are so long that Windows protests
    truncate_filenames(raw_dir)

    if not interim_dir.is_dir():
        interim_dir.mkdir(parents=True, exist_ok=True)

    remove_non_images(raw_dir, interim_dir)


if __name__ == '__main__':
    main()
