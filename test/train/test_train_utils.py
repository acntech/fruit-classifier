import unittest
from fruit_classifier.train.train_utils import get_image_paths
from fruit_classifier.train.train_utils import get_data_and_labels
from fruit_classifier.train.train_utils import get_model_input
from fruit_classifier.train.train_utils import get_model
from fruit_classifier.train.train_utils import train_model
from fruit_classifier.preprocessing.preprocessing_utils import \
    get_image_generator
from pathlib import Path
import shutil


class TestTrainUtils(unittest.TestCase):
    def setUp(self):
        test_dir = Path(__file__).absolute().parents[1]

        # Select a unique folder name for a new folder
        folder_name = 'temp_folder'
        self.folderName = test_dir.joinpath(folder_name)
        while self.folderName.is_dir():
            folder_name = folder_name + '_'
            self.folderName = test_dir.joinpath(folder_name)

        # Make directory and two subdirectories
        self.folderName.mkdir(parents=True, exist_ok=True)
        self.folderName.joinpath('A').mkdir(parents=True, exist_ok=True)
        self.folderName.joinpath('B').mkdir(parents=True, exist_ok=True)

        # Copy an image into each of those subdirectories
        orig_file_path = test_dir.joinpath("test_data").\
            joinpath("original_test_image.jpg")
        shutil.copy(str(orig_file_path),
                    str(self.folderName.joinpath('A')))
        shutil.copy(str(orig_file_path),
                    str(self.folderName.joinpath('B')))

    def test_does_not_crash(self):
        # Test get_image_paths
        image_paths = get_image_paths(self.folderName)
        self.assertGreater(len(image_paths), 0)

        # Test get_data_and_labels
        data, labels = get_data_and_labels(image_paths)
        self.assertGreater(len(data), 0)
        self.assertGreater(len(labels), 0)

        # Test get_model_input
        x_train, x_val, y_train, y_val = get_model_input(data, labels)
        self.assertEqual(1, len(x_train))
        self.assertEqual(1, len(x_val))
        self.assertEqual(1, len(y_train))
        self.assertEqual(1, len(y_val))

        # Test get_image_generator
        image_generator = get_image_generator()
        self.assertEqual(image_generator.rotation_range, 30)
        self.assertEqual(image_generator.width_shift_range, .1)
        self.assertEqual(image_generator.height_shift_range, .1)
        self.assertEqual(image_generator.shear_range, .2)
        self.assertEqual(image_generator.horizontal_flip, True)
        self.assertEqual(image_generator.fill_mode, 'nearest')

        # Test get_model
        model = get_model(len(set(labels)), epochs = 1)
        self.assertTrue(model._built)

        history = train_model(model,
                              image_generator,
                              x_train,
                              y_train,
                              x_val,
                              y_val)
        self.assertEqual(history.params['epochs'], 25)

    def tearDown(self):
        folder_to_remove = self.folderName
        shutil.rmtree(folder_to_remove)


if __name__ == "__main__":
    unittest.main()
