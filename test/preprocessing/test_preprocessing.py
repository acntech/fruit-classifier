import unittest
import cv2
from pathlib import Path
import numpy as np
import shutil
import os
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
        '''
        This test creates a folder structure with an image in it,
        truncates filenames inside it, and basically checks that it
        does not crash.

        Ideally, this test would create a file with too long filename,
        run truncate_filenames, and check that the filename now is
        sufficiently short. However, it was hard to create a file wih
        long filename, so I did not spend more time on this.
        '''

        # Select a unique folder name for a new folder
        outer_folder_name = 'temp_folder'
        outer_folder_path = self.test_dir.joinpath(outer_folder_name)
        while outer_folder_path.is_dir():
            outer_folder_name = outer_folder_name + '_'
            outer_folder_path = self.test_dir.joinpath(outer_folder_name)
        outer_folder_path.mkdir(parents=True, exist_ok=True)
        inner_folder_name = outer_folder_path.joinpath('inner_folder')
        inner_folder_name.mkdir(parents=True, exist_ok=True)

        # Copy an image into this folder
        short_image_dest_filename = inner_folder_name.\
            joinpath("short_image_name.jpg")
        shutil.copy(self.jpg_image_file_name, short_image_dest_filename)

        # Run the function to be tested
        truncate_filenames(outer_folder_path)

        # Extract list of files present in folder
        file_list = os.listdir(inner_folder_name)

        # Make sure the right number of files exist in that folder
        self.assertEqual(len(file_list), 1)

        # Make sure no path name is longer than 255 characters
        for filename in file_list:
            filepath = inner_folder_name.joinpath(filename)
            self.assertLessEqual(len(str(filepath)), 255)

        # Delete the temporary directory and its contents
        shutil.rmtree(outer_folder_path)


if __name__ == '__main__':
    unittest.main()


