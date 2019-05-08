import unittest
import cv2
from pathlib import Path
import numpy as np
import shutil
from fruit_classifier.preprocessing.preprocessing_utils \
    import truncate_filenames
from fruit_classifier.preprocessing.preprocessing_utils \
    import preprocess_image


class TestPreprocessingUtils(unittest.TestCase):

    def setUp(self):

        self.test_dir = Path(__file__).absolute().parents[1]
        self.jpg_image_file_name = self.test_dir.joinpath(
            "test_data", "original_test_image.jpg")

        # Correct dimensions
        self.test_raw_shape = [115, 73, 3]
        self.test_comp_shape = [28, 28, 3]

        # Read test image from file
        self.raw = cv2.imread(str(self.jpg_image_file_name))

        # Create a compressed version
        self.comp = preprocess_image(self.raw)

    def test_assert_dimensions(self):
        # Assert that dimensions on original image are correct
        self.assertEqual(self.raw.shape, tuple(self.test_raw_shape))

        # Assert that dimensions on original image are correct
        self.assertEqual(self.comp.shape, tuple(self.test_comp_shape))

    def test_verify_legal_max_min(self):
        self.assertLessEqual(np.amax(self.raw), 255)
        self.assertGreaterEqual(np.amin(self.raw), 0)

        self.assertLessEqual(np.amax(self.comp), 1.)
        self.assertGreaterEqual(np.amin(self.comp), 0.)

    def test_verify_pixel_variation(self):
        self.assertGreater(np.amax(self.raw), np.amin(self.raw))
        self.assertGreater(np.amax(self.comp), np.amin(self.comp))

    def test_truncate_filenames(self):
        # Select a unique folder name for a new folder
        folder_name = 'temp_folder'
        self.folder_name = self.test_dir.joinpath(folder_name)
        while self.folder_name.is_dir():
            folder_name = folder_name + '_'
            self.folder_name = self.test_dir.joinpath(folder_name)
        self.folder_name.mkdir(parents=True, exist_ok=True)
        short_image_dest_filename = self.folder_name.\
            joinpath("short_image_name.jpg")
        shutil.copy(self.jpg_image_file_name, short_image_dest_filename)
        long_image_dest_filename = self.folder_name.\
            joinpath("lllllllllllllllllllllllllllllllllllllllllllllllll"
                     "ooooooooooooooooooooooooooooooooooooooooooooooooo"
                     "nnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnn"
                     "ggggggggggggggggggggggggggggggggggggggggggggggggg"
                     "_image_name.jpg")
        shutil.copy(self.jpg_image_file_name, long_image_dest_filename)
        truncate_filenames(self.folder_name)
        

if __name__ == '__main__':
    unittest.main()


