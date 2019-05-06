import unittest
from fruit_classifier.train.train_utils import get_image_paths
from fruit_classifier.train.train_utils import get_data_and_labels
from pathlib import Path
import shutil


class TestTrainUtils(unittest.TestCase):
    def setUp(self):
        test_dir = Path(__file__).absolute().parents[1]
        folder_name = 'temp_folder'
        self.folderName = test_dir.joinpath(folder_name)
        while self.folderName.is_dir():
            folder_name = folder_name + '_'
            self.folderName = test_dir.joinpath(folder_name)

        self.folderName.mkdir(parents=True, exist_ok=True)
        self.folderName.joinpath('A').mkdir(parents=True, exist_ok=True)
        self.folderName.joinpath('B').mkdir(parents=True, exist_ok=True)

        orig_file_path = test_dir.joinpath("test_data").\
            joinpath("original_test_image.jpg")
        shutil.copy(str(orig_file_path),
                    str(self.folderName.joinpath('A')))
        shutil.copy(str(orig_file_path),
                    str(self.folderName.joinpath('B')))

    def test_not_crashing(self):
        image_paths = get_image_paths(self.folderName)
        self.assertGreater(len(image_paths), 0)
        data, labels = get_data_and_labels(image_paths)

    def tearDown(self):
        folder_to_remove = self.folderName
        shutil.rmtree(folder_to_remove)


if __name__ == "__main__":
    unittest.main()
