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

    generated_data_dir = \
        Path(__file__).absolute().parents[2].joinpath('generated_data')
    raw_dir = generated_data_dir.joinpath('raw_data')
    cleaned_dir = generated_data_dir.joinpath('cleaned_data')

    # Shorten filenames if they are so long that Windows protests
    truncate_filenames(raw_dir)

    if not cleaned_dir.is_dir():
        cleaned_dir.mkdir(parents=True, exist_ok=True)

    remove_non_images(raw_dir, cleaned_dir)


if __name__ == '__main__':
    main()
