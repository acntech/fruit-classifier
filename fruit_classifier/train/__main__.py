import pickle
from pathlib import Path
from fruit_classifier.utils.image_utils import get_image_paths
from fruit_classifier.train.train_utils import get_data_and_labels
from fruit_classifier.train.train_utils import get_model_input
from fruit_classifier.train.train_utils import get_model
from fruit_classifier.train.train_utils import train_model
from fruit_classifier.train.train_utils import plot_training
from fruit_classifier.preprocessing.preprocessing_utils import \
    get_image_generator


def main():
    """
    This is the main module for training the fruit-classifier

    This method will
    1. Load all the images from the 'raw_dir' directory
        - NOTE: The images must be sorted in directories according to
          class in 'raw_dir'
    2. Split the data in train and validate
    3. Initialize a model
    4. Train the model
    5. Plot the training
    """

    data_dir = \
        Path(__file__).absolute().parents[2].joinpath('data')
    interim_dir = data_dir.joinpath('interim')
    processed_dir = data_dir.joinpath('processed')

    # Grab the image paths and randomly shuffle them
    image_paths = get_image_paths(interim_dir)

    # Load the data and and label and split to train and validation
    data_path = processed_dir.joinpath('data.pkl')
    labels_path = processed_dir.joinpath('labels.pkl')
    if data_path.is_file() and labels_path.is_file():
        with data_path.open('rb') as f:
            data = pickle.load(f)
        with labels_path.open('rb') as f:
            labels = pickle.load(f)
    else:
        data, labels = get_data_and_labels(image_paths)

    x_train, x_val, y_train, y_val = \
        get_model_input(data, labels)

    # Construct the image generator for data augmentation
    image_generator = get_image_generator()

    # Initialize the model
    model = get_model(len(set(labels)))

    # Train the network
    history = train_model(model,
                          image_generator,
                          x_train,
                          y_train,
                          x_val,
                          y_val)

    # Plot the training loss and accuracy
    plot_training(history)


if __name__ == '__main__':
    main()
