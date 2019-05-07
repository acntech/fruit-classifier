import unittest
from fruit_classifier.train.train_utils import get_image_paths
from fruit_classifier.train.train_utils import get_data_and_labels
from fruit_classifier.train.train_utils import get_model_input
from fruit_classifier.train.train_utils import get_model
from fruit_classifier.train.train_utils import train_model
from fruit_classifier.train.train_utils import plot_training
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

        # Run get_image_paths()
        image_paths = get_image_paths(self.folderName)
        self.numImagePaths = len(image_paths)

        # Run get_data_and_labels
        data, labels = get_data_and_labels(image_paths)
        self.numData = len(data)
        self.numLabels = len(labels)

        # Run get_model_input
        x_train, x_val, y_train, y_val = get_model_input(data, labels)
        self.num_x_train = len(x_train)
        self.num_x_val = len(x_val)
        self.num_y_train = len(y_train)
        self.num_y_val = len(y_val)

        # Run get_image_generator
        self.image_generator = get_image_generator()

        # Run get_model
        model = get_model(len(set(labels)), epochs=1)
        self.model_was_built = model._built

        # Run train_model
        history = train_model(model,
                              self.image_generator,
                              x_train,
                              y_train,
                              x_val,
                              y_val)
        self.num_epochs = history.params['epochs']

        # Run plot_training
        plot_training(history)

    def test_get_image_paths(self):
        # Test get_image_paths output
        self.assertGreater(self.numImagePaths, 0)

    def test_get_data_and_labels(self):
        # Test get_data_and_labels output
        self.assertGreater(self.numData, 0)
        self.assertGreater(self.numLabels, 0)

    def test_get_model_input(self):
        # Test get_model_input output
        self.assertEqual(1, self.num_x_train)
        self.assertEqual(1, self.num_x_val)
        self.assertEqual(1, self.num_y_train)
        self.assertEqual(1, self.num_y_val)

    def test_image_generator(self):
        # Test get_image_generator output
        self.assertEqual(self.image_generator.rotation_range, 30)
        self.assertEqual(self.image_generator.width_shift_range, .1)
        self.assertEqual(self.image_generator.height_shift_range, .1)
        self.assertEqual(self.image_generator.shear_range, .2)
        self.assertEqual(self.image_generator.horizontal_flip, True)
        self.assertEqual(self.image_generator.fill_mode, 'nearest')

    def test_model(self):
        # Test get_model output
        self.assertTrue(self.model_was_built)

    def test_train_model(self):
        # Test train_model output
        self.assertEqual(self.num_epochs, 25)

    def tearDown(self):
        folder_to_remove = self.folderName
        shutil.rmtree(folder_to_remove)


if __name__ == "__main__":
    unittest.main()
