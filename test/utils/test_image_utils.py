import unittest
from pathlib import Path
import numpy as np

from fruit_classifier.utils.image_utils import open_image


class TestImageUtils(unittest.TestCase):

    def setUp(self):

        test_dir = Path(__file__).absolute().parents[1]

        self.jpg_image_file_name = \
            test_dir.joinpath("test_data",
                              "original_test_image.jpg")
        self.png_image_file_name = \
            test_dir.joinpath("test_data",
                              "original_test_image.png")

        self.test_orig_shape = [115, 73, 3]
        self.test_orig_max = 255
        self.test_orig_min = 0

    def test_open_image(self):
        self.open_image_function(self.jpg_image_file_name)
        self.open_image_function(self.png_image_file_name)

    def open_image_function(self, file):
        image = open_image(file)

        self.assertEqual(tuple(self.test_orig_shape), image.shape)

        self.assertEqual(self.test_orig_max, np.amax(image))
        self.assertEqual(self.test_orig_min, np.amin(image))


if __name__ == '__main__':
    unittest.main()
