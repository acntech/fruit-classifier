import argparse
import shutil
from pathlib import Path
from google_images_download import google_images_download


def main(categories=('bananas', 'apples', 'oranges'),
         limit=700):
    """
    Scrapes google for the images given in keywords

    Parameters
    ----------
    categories : array-like
        The categories to scrape
    limit : int
        The maximum amount of images to scrape for each category
    """

    root_dir = Path(__file__).absolute().parents[2]
    response = google_images_download.googleimagesdownload()

    keywords = ','.join(categories)

    arguments = \
        {'keywords': keywords,
         'limit': limit,
         'print_urls': True,
         'chromedriver': root_dir.joinpath('chromedriver')}

    _ = response.download(arguments)

    destination_dir = root_dir.joinpath('raw_data')

    if not destination_dir.is_dir():
        destination_dir.mkdir(parents=True, exist_ok=True)

    shutil.move(str(root_dir.joinpath('downloads')),
                str(destination_dir))


if __name__ == '__main__':
    # Construct the argument parse and parse the arguments
    parser = argparse.ArgumentParser(description='Scrape images')
    parser.add_argument('-c',
                        '--categories',
                        nargs='+',
                        required=False,
                        help='Categories to scrape. Example: '
                             'data_scraping -c bananas apples oranges')
    parser.add_argument('-l',
                        '--limit',
                        type=int,
                        required=False,
                        help='The maximum amount of images to scrape '
                             'for each category')
    args = parser.parse_args()

    if args.categories is None:
        categories = ('bananas', 'apples', 'oranges')
    else:
        categories = args.categories

    if args.limit is None:
        limit = 700
    else:
        limit = args.limit

    main(categories, limit)
