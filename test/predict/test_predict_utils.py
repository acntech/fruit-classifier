import unittest
import cv2
from pathlib import Path
from fruit_classifier.predict.predict_utils import pick_random_image


class TestPredictUtils(unittest.TestCase):
    def setUp(self):
        self.testDir = Path(__file__).absolute().parents[1].\
            joinpath('test_data')

    def test_pick_random_image(self):
        image_path = pick_random_image(self.testDir)
        image = cv2.imread(str(image_path))
        self.assertGreater(image.shape[0], 0)
        self.assertGreater(image.shape[1], 0)


if __name__ == '__main__':
    unittest.main()
