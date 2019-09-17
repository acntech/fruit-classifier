import unittest
import cv2
from pathlib import Path
import numpy as np
import shutil
from fruit_classifier.preprocessing.preprocessing_utils \
    import truncate_filenames
from fruit_classifier.preprocessing.preprocessing_utils \
    import resize_image


class TestPreprocessingUtils(unittest.TestCase):

    def setUp(self):

        self.test_dir = Path(__file__).absolute().parents[1]
        self.jpg_image_file_name = self.test_dir.joinpath(
            "test_data", "original_test_image.jpg")
        self.tmp_dir_path = self.test_dir.joinpath('tmp_dir')

        # Correct dimensions
        self.test_raw_shape = [115, 73, 3]
        self.test_comp_shape = [28, 28, 3]

        # Read test image from file
        self.raw = cv2.imread(str(self.jpg_image_file_name))

        # Create a compressed version
        self.comp = resize_image(self.raw)

    def tearDown(self):
        # Tear down the tmp_dir if it has been created
        if self.tmp_dir_path.is_dir():
            # Delete the temporary directory and its contents
            shutil.rmtree(self.tmp_dir_path)

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
        """
        Test that truncate_filenames does not crash

        This test creates a directory structure with an image in it,
        truncates filenames inside it, and basically checks that it
        does not crash.
        """

        # Make a dir of this name
        self.tmp_dir_path.mkdir(parents=True, exist_ok=True)

        # Create an inner directory, to  mimic the directory structure
        # that truncate_filenames() requires.
        inner_dir_path = self.tmp_dir_path.joinpath('inner_dir')
        inner_dir_path.mkdir(parents=True, exist_ok=True)

        # Copy an image into this directory
        short_image_dest_filename = inner_dir_path.\
            joinpath("short_image_name.jpg")
        shutil.copy(self.jpg_image_file_name, short_image_dest_filename)

        # Run the function to be tested
        # No files should be truncated, unless this directory's path is
        # long in terms of characters (windows API allows max 260)
        truncate_filenames(self.tmp_dir_path)

        # Extract list of files present in directory
        file_list = list(Path(inner_dir_path).glob('*'))

        # Make sure the right number of files exist in that directory
        self.assertEqual(len(file_list), 1)

        # Make sure no path name is longer than 255 characters
        for filepath in file_list:
            self.assertLessEqual(len(str(filepath)), 255)


if __name__ == '__main__':
    unittest.main()


