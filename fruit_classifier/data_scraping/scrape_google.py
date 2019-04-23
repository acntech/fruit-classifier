import shutil
from pathlib import Path
from google_images_download import google_images_download

this_dir = Path(__file__).absolute().parent

response = google_images_download.googleimagesdownload()

arguments = \
    {'keywords': 'bananas,apples,oranges',
     'limit': 700,
     'print_urls': True,
     'chromedriver': this_dir.joinpath('chromedriver')}

_ = response.download(arguments)

destination_dir = this_dir.parent.joinpath('raw_data')

if not destination_dir.is_dir():
    destination_dir.mkdir(parents=True, exist_ok=True)

shutil.move(this_dir, destination_dir)
