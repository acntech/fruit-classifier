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
        self.folder_name = test_dir.joinpath(folder_name)
        while self.folder_name.is_dir():
            folder_name = folder_name + '_'
            self.folder_name = test_dir.joinpath(folder_name)

        # Make directory and two subdirectories
        self.folder_name.mkdir(parents=True, exist_ok=True)
        self.folder_name.joinpath('A').mkdir(parents=True, exist_ok=True)
        self.folder_name.joinpath('B').mkdir(parents=True, exist_ok=True)

        # Copy an image into each of those subdirectories
        orig_file_path = test_dir.joinpath("test_data").\
            joinpath("original_test_image.jpg")
        shutil.copy(str(orig_file_path),
                    str(self.folder_name.joinpath('A')))
        shutil.copy(str(orig_file_path),
                    str(self.folder_name.joinpath('B')))

    def test_train_model(self):
        # Run get_image_paths() and verify outputs
        image_paths = get_image_paths(self.folder_name)
        num_image_paths = len(image_paths)
        self.assertGreater(num_image_paths, 0)

        # Run get_data_and_labels and verify outputs
        data, labels = get_data_and_labels(image_paths)
        self.assertGreater(len(data), 0)
        self.assertGreater(len(labels), 0)

        # Run get_model_input and verify outputs
        x_train, x_val, y_train, y_val = get_model_input(data, labels)
        self.assertEqual(1, len(x_train))
        self.assertEqual(1, len(x_val))
        self.assertEqual(1, len(y_train))
        self.assertEqual(1, len(y_val))

        # Run get_image_generator and verify outputs
        image_generator = get_image_generator()
        self.assertEqual(image_generator.rotation_range, 30)
        self.assertEqual(image_generator.width_shift_range, .1)
        self.assertEqual(image_generator.height_shift_range, .1)
        self.assertEqual(image_generator.shear_range, .2)
        self.assertEqual(image_generator.horizontal_flip, True)
        self.assertEqual(image_generator.fill_mode, 'nearest')

        # Run get_model and verify outputs
        num_intended_epochs = 2
        model = get_model(len(set(labels)), epochs=num_intended_epochs)
        self.assertTrue(model._built)

        # Run train_model and verify outputs
        history = train_model(model,
                              image_generator,
                              x_train,
                              y_train,
                              x_val,
                              y_val,
                              epochs=num_intended_epochs)
        num_epochs = history.params['epochs']
        self.assertEqual(num_epochs, num_intended_epochs)

        # Run plot_training and verify it does not crash
        plot_training(history)

    def tearDown(self):
        # Delete the temporary folder its contents
        shutil.rmtree(self.folder_name)


if __name__ == "__main__":
    unittest.main()
