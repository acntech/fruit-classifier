import unittest
from unittest.mock import patch, Mock
from pathlib import Path


class TestDataScraping(unittest.TestCase):

    def setUp(self):
        self.root_dir = Path(__file__).absolute().parents[2]
        self.limit = 3
        self.categories = ('bananas', 'apples')

    @patch('fruit_classifier.data_scraping.__main__.main',
           return_value=Path(__file__).absolute().parents[2].joinpath(
                                                           'test',
                                                           'test_data',
                                                           'raw_data'))
    def test_main(self, mocked_main):
        self.data_dir = mocked_main(categories=self.categories,
                                    limit=self.limit)

        # Check that the function was called once with the correct
        # parameters
        mocked_main.assert_called_once_with(
                                    categories=('bananas', 'apples'),
                                    limit=3)

        # Check that all folders and files exist and the amount
        # is correct
        self.assertIsNotNone(True, msg='Image folder is empty')
        self.assertTrue(self.data_dir.joinpath('bananas').is_dir())
        self.assertTrue(self.data_dir.joinpath('apples').is_dir())

        for cat in self.categories:
            count = 0

            for i, file in enumerate(self.data_dir.joinpath(cat).glob('*')):
                self.assertTrue(file.is_file())
                count = i+1
            self.assertEqual(count, self.limit)
