import unittest
from fruit_classifier.train.train_utils import get_image_paths
from pathlib import Path


class TestTrainUtils(unittest.TestCase):
    def setUp(self):
        test_dir = Path(__file__).absolute().parents[1]
        self.folderName = test_dir.joinpath('temp_folder')
        while self.folderName.is_dir():
            self.folderName = self.folderName + "_"

        self.folderName.mkdir(parents=True, exist_ok=True)

    def test_not_crashing(self):
        get_image_paths(self.folderName)
        pass

    def tearDown(self):
        folder_to_remove = self.folderName
        folder_to_remove.rmdir()


if __name__ == "__main__":
    unittest.main()
