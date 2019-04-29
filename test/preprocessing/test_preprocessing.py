import unittest
import cv2
from pathlib import Path
import numpy as np
from fruit_classifier.preprocessing.preprocessing_utils import preprocess_image


class TestPreprocessingUtils(unittest.TestCase):

    def setUp(self):

        test_dir = Path(__file__).absolute().parents[1]
        self.jpg_image_file_name = test_dir.joinpath("test_data",
                                                     "original_test_image.jpg")
        self.png_image_file_name = test_dir.joinpath("test_data",
                                                     "original_test_image.png")

        self.test_orig_shape = [115, 73, 3]
        self.test_orig_max = 255
        self.test_orig_min = 0

        self.test_comp_shape = [28, 28, 3]
        self.test_comp_max = 0.00392156862745098
        self.test_comp_min = 0.00019912376715391965
        self.test_comp_vals = [0.00325285, 0.00281178, 0.00246107]

    def test_preprocess_image(self):
        self.preprocess(self.jpg_image_file_name)
        self.preprocess(self.png_image_file_name)

    def preprocess(self, file_name):
        # Read test image from file
        raw = cv2.imread(str(file_name))

        # Create a compressed version
        comp = preprocess_image(raw)

        # Assert that dimensions on original image are correct
        for i in range(3):
            self.assertEqual(np.size(raw, i), self.test_orig_shape[i])

        # Compare maximum and minimum values in original image
        self.assertAlmostEqual(np.amax(raw), self.test_orig_max, 7)
        self.assertAlmostEqual(np.amin(raw), self.test_orig_min, 7)

        # Assert that dimensions on original image are correct
        for i in range(3):
            self.assertEqual(comp.shape[i], self.test_comp_shape[i])

        # Compare maximum and minimum values in compressed image
        self.assertAlmostEqual(np.amax(comp), self.test_comp_max, 7)
        self.assertAlmostEqual(np.amin(comp), self.test_comp_min, 7)

        # Compare exact values in compressed image
        for i in range(3):
            self.assertAlmostEqual(comp[14, 4, i], self.test_comp_vals[i], 7)


if __name__ == '__main__':
    unittest.main()


