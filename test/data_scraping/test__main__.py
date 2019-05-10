import unittest
from unittest.mock import patch
from fruit_classifier.data_scraping.__main__ import main
from pathlib import Path


def mocked_googleimagesdownload():

    class MockGoogleimagesdownload:
        def __init__(self):
            pass

        def download(self, arguments):
            return 'nothing'

    return MockGoogleimagesdownload()


class TestDataScraping(unittest.TestCase):

    def setUp(self):
        self.root_dir = Path(__file__).absolute().parents[2]
        self.destination_dir = self.root_dir.joinpath('generated_data',
                                                      'raw_data')

    @patch("google_images_download.google_images_download.googleimagesdownload",
           side_effect=mocked_googleimagesdownload)
    def test_main(self, mock_download):
        main()

        mock_download.assert_called_once()
        self.assertTrue(self.destination_dir.is_dir())


if __name__ == '__main__':
    unittest.main()
