from pathlib import Path
from fruit_classifier.preprocessing.image_cleaner import \
    remove_non_images


def main():
    """"
    Preprocesses the images in raw_data

    The resulting images are stored in cleaned_data
    """

    generated_data_dir = \
        Path(__file__).absolute().parents[1].joinpath('generated_data')
    raw_dir = generated_data_dir.joinpath('raw_data')
    cleaned_dir = generated_data_dir.joinpath('cleaned_data')

    if not cleaned_dir.is_dir():
        cleaned_dir.mkdir(parents=True, exist_ok=True)

    remove_non_images(raw_dir, cleaned_dir)


if __name__ == '__main__':
    main()
