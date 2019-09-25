import argparse
import random
import json
from pathlib import Path
from fruit_classifier.utils.image_utils import get_image_paths
from fruit_classifier.train.train_utils import get_processed_data
from fruit_classifier.train.train_utils import get_model
from fruit_classifier.train.train_utils import train_model
from fruit_classifier.train.train_utils import plot_training
from fruit_classifier.preprocessing.preprocessing_utils import \
    get_image_generator


# FIXME: Add issues to github

def main(dataset_name='basic',
         model_name='basic',
         model_setup=None,
         optimizer_setup=None):
    """
    This is the main module for training the fruit-classifier

    This method will
    1. Load all the images from the 'interim' directory
    2. Split the data in train and validate
    3. Initialize a model
    4. Train the model
    5. Plot the training

    Parameters
    ----------
    dataset_name : str
        Dataset to train from
    model_name : str
        Name of model
        The model will be stored in
        model_files/models/model_name/model.h5
    model_setup : None or dict
        Dictionary of model specific setup.
        Must contain the key `model_type` with a string value
        corresponding to one of the implemented models found in
        fruit_classifier.models.factory.
        The rest of the keys corresponds to model hyperparameters.
        Valid parameters can be found in
        fruit_classifier.models.models for a given `model_type`.
        If set to None, defaults will be used
    optimizer_setup : None or dict
        Dictionary for optimizer setup.
        See input parameters of
        fruit_classifier.train.train_utils.get_model for details

    Returns
    -------
    history : History
        The training History object containing
        - loss
        - val_loss
        - acc
        - val_acc
    x_test : np.array, shape (n_test, height, width, channels)
        The data of the test set
    y_test : np.array, shape (n_test,)
        The labels of the test set
    """

    root_dir = Path(__file__).absolute().parents[2]
    data_dir = root_dir.joinpath('data')
    interim_dir = data_dir.joinpath('interim', dataset_name)
    processed_dir = data_dir.joinpath('processed', dataset_name)
    model_files_dir = root_dir.joinpath('model_files')
    plot_dir = root_dir.joinpath('reports', 'figures', model_name)

    # Grab the image paths and randomly shuffle them
    image_paths = get_image_paths(interim_dir)
    random.seed(42)
    random.shuffle(image_paths)

    # Get the data
    x_train,\
        x_val,\
        x_test,\
        y_train,\
        y_val,\
        y_test = \
        get_processed_data(image_paths,
                           model_files_dir,
                           model_name,
                           processed_dir)

    # Construct the image generator for data augmentation
    image_generator = get_image_generator()

    # Initialize the model
    if model_setup is None:
        model_setup = dict()

    # Infer some of the setup to use in the models
    model_setup['height'],\
        model_setup['width'],\
        model_setup['channels'] = x_train.shape[1:]

    # FIXME: This is also done in get_model :/
    if optimizer_setup is None:
        optimizer_setup = dict(initial_learning_rate=1e-3,
                               epochs=2,
                               batch_size=32)

    model = get_model(len(set(y_train)), model_setup, optimizer_setup)

    # Train the network
    history = train_model(model,
                          image_generator,
                          model_files_dir,
                          x_train,
                          y_train,
                          x_val,
                          y_val,
                          model_name,
                          batch_size=optimizer_setup['batch_size'],
                          epochs=optimizer_setup['epochs'])

    # Plot the training loss and accuracy
    plot_training(history,
                  plot_dir,
                  f'{dataset_name}_{model_name}')

    return history, x_test, y_test


if __name__ == '__main__':
    # Construct the argument parse and parse the arguments
    parser = argparse.ArgumentParser(description='Preprocess a dataset')
    parser.add_argument('-d',
                        '--dataset_name',
                        required=False,
                        default=None,
                        help='Name of the resulting dataset')
    parser.add_argument('-m',
                        '--model_name',
                        required=False,
                        default=None,
                        help='Name of the resulting model')
    parser.add_argument('-s',
                        '--model_setup',
                        required=False,
                        default=None,
                        type=json.loads,
                        help='Model setup as a json string. I.e. in '
                             'the form {"key1": val1, "key2": val2}')
    parser.add_argument('-o',
                        '--optimizer_setup',
                        required=False,
                        default=None,
                        type=json.loads,
                        help='Optimizer setup as a json string. I.e. '
                             'in the form {"key1": val1, "key2": val2}')

    args = parser.parse_args()

    if args.dataset_name is None:
        dataset_name_ = 'basic'
    else:
        dataset_name_ = args.dataset_name

    if args.model_name is None:
        model_name_ = 'basic'
    else:
        model_name_ = args.model_name

    main(dataset_name_,
         model_name_,
         args.model_setup,
         args.optimizer_setup)
