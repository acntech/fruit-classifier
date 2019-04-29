import unittest
import cv2
from pathlib import Path
import numpy as np
from fruit_classifier.preprocessing.preprocessing_utils import preprocess_image


class TestPreprocessingUtils(unittest.TestCase):

    def setUp(self):

        test_dir = Path(__file__).absolute().parents[1]
        self.orig_image_file_name = test_dir.joinpath("test_data",
                                                      "original_test_image.jpg")
        self.test_orig_shape = [115, 73, 3]
        self.test_orig_max = 255
        self.test_orig_min = 0

        self.test_comp_shape = [28, 28, 3]
        self.test_comp_max = 0.00392156862745098
        self.test_comp_min = 0.00019912376715391965

    def test_preprocess_image(self):

        # Read test image from file
        raw = cv2.imread(str(self.orig_image_file_name))

        # Create a compressed version
        comp = preprocess_image(raw)

        # Assert that dimensions on original image are correct
        self.assertEqual(np.size(raw, 0), self.test_orig_shape[0])
        self.assertEqual(raw.shape[1], self.test_orig_shape[1])
        self.assertEqual(raw.shape[2], self.test_orig_shape[2])

        self.assertAlmostEqual(np.amax(raw), self.test_orig_max, 7)
        self.assertAlmostEqual(np.amin(raw), self.test_orig_min, 7)

        # Assert that dimensions on original image are correct

        self.assertEqual(comp.shape[0], self.test_comp_shape[0])
        self.assertEqual(comp.shape[1], self.test_comp_shape[1])
        self.assertEqual(comp.shape[2], self.test_comp_shape[2])

        self.assertAlmostEqual(np.amax(comp), self.test_comp_max, 7)
        self.assertAlmostEqual(np.amin(comp), self.test_comp_min, 7)


if __name__ == '__main__':
    unittest.main()


