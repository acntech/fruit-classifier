import unittest
from fruit_classifier.utils.image_utils import get_image_paths
from fruit_classifier.train.train_utils import get_data_and_labels
from fruit_classifier.train.train_utils import get_model_input
from fruit_classifier.train.train_utils import get_model
from fruit_classifier.train.train_utils import train_model
from fruit_classifier.train.train_utils import plot_training
from fruit_classifier.preprocessing.preprocessing_utils import \
    get_image_generator
from fruit_classifier.preprocessing.preprocessing_utils import \
    resize_images
from pathlib import Path
import shutil


class TestTrainUtils(unittest.TestCase):
    def setUp(self):
        test_dir = Path(__file__).absolute().parents[2]

        # Select a unique directory name for a new directory
        self.tmp_dir = test_dir.joinpath('data', 'tmp_dir')

        # Make directory and two subdirectories
        classes = ('class_a', 'class_b')
        self.n_classes = len(classes)

        for c in classes:
            self.tmp_dir.joinpath(c).mkdir(parents=True,
                                           exist_ok=True)

        # Copy an image into each of those subdirectories
        orig_file_path = test_dir.joinpath("data").\
            joinpath("original_test_image.jpg")

        for c in classes:
            shutil.copy(str(orig_file_path),
                        str(self.tmp_dir.joinpath(c)))

        # Get the image_paths
        self.image_paths = get_image_paths(self.tmp_dir)

        # Set number of epochs globally
        self.num_intended_epochs = 2

    def test_get_image_paths(self):
        # Run get_image_paths() and verify outputs
        num_image_paths = len(self.image_paths)
        self.assertGreater(num_image_paths, 0)

    def test_get_data_and_labels(self):
        # Run get_data_and_labels and verify outputs
        data, labels = get_data_and_labels(self.image_paths)
        self.assertGreater(len(data), 0)
        self.assertGreater(len(labels), 0)

    def test_get_model_input(self):
        # Run get_model_input and verify outputs
        data, labels = get_data_and_labels(self.image_paths)
        x_train, x_val, y_train, y_val = get_model_input(data, labels)
        self.assertEqual(1, len(x_train))
        self.assertEqual(1, len(x_val))
        self.assertEqual(1, len(y_train))
        self.assertEqual(1, len(y_val))

    def test_get_image_generator(self):
        # Run get_image_generator and verify outputs
        rotation_range = 30
        width_shift_range = 0.1
        height_shift_range = 0.1
        shear_range = 0.2
        zoom_range = 0.2
        horizontal_flip = True
        fill_mode = 'nearest'

        image_generator = \
            get_image_generator(rotation_range=rotation_range,
                                width_shift_range=width_shift_range,
                                height_shift_range=height_shift_range,
                                shear_range=shear_range,
                                zoom_range=zoom_range)

        self.assertEqual(image_generator.rotation_range, rotation_range)
        self.assertEqual(image_generator.width_shift_range,
                         width_shift_range)
        self.assertEqual(image_generator.height_shift_range,
                         height_shift_range)
        self.assertEqual(image_generator.shear_range,
                         shear_range)
        self.assertEqual(image_generator.horizontal_flip,
                         horizontal_flip)
        self.assertEqual(image_generator.fill_mode,
                         fill_mode)

    def test_get_model(self):
        # Run get_model and verify outputs
        model = get_model(self.n_classes,
                          epochs=self.num_intended_epochs)
        self.assertTrue(model._built)

    def test_train_model(self):
        # Resize the images
        resize_images(self.tmp_dir)
        # Re-run image paths after resizing (extension may be altered)
        image_paths = get_image_paths(self.tmp_dir)
        # Run train_model and verify outputs
        data, labels = get_data_and_labels(image_paths)

        x_train, x_val, y_train, y_val = get_model_input(data, labels)
        image_generator = get_image_generator()

        model = get_model(self.n_classes,
                          epochs=self.num_intended_epochs)

        history = train_model(model,
                              image_generator,
                              x_train,
                              y_train,
                              x_val,
                              y_val,
                              epochs=self.num_intended_epochs)
        num_epochs = history.params['epochs']
        self.assertEqual(num_epochs, self.num_intended_epochs)

    def test_plot_training(self):
        # Run plot_training and verify it does not crash
        class History(object):
            pass

        history = History
        history.history = dict(loss=(1, 2),
                               val_loss=(1, 2),
                               acc=(1, 2),
                               val_acc=(1, 2),)
        plot_training(history)

    def tearDown(self):
        # Delete the temporary directory its contents
        shutil.rmtree(self.tmp_dir)


if __name__ == "__main__":
    unittest.main()
