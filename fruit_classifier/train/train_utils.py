import pickle
import numpy as np
from pathlib import Path
from tqdm import tqdm
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from fruit_classifier.models.factory import ModelFactory
from fruit_classifier.utils.image_utils import open_image


def get_data_and_labels(image_paths):
    """
    Returns the data and the labels from the input paths

    Parameters
    ----------
    image_paths : list
        List of Paths of the image paths

    Returns
    -------
    data : np.array, shape (len(image_paths), height, width, channels)
        The images as numpy array
    labels : np.array, shape (len(image_paths,))
        The corresponding labels
    """

    data = list()
    labels = list()
    # Loop over the input images
    for image_path in tqdm(image_paths, desc='Loading the images'):
        tqdm.write(str(image_path))
        # Load the image, pre-process it, and store it in the data list
        image_array = open_image(image_path)
        data.append(image_array)

        # Extract the class label from the image path and update the
        # labels list
        label = image_path.parts[-2]
        labels.append(label)
    # Scale the raw pixel intensities to the range [0, 1]
    data = np.array(data, dtype='float')/255
    labels = np.array(labels)

    return data, labels


def store_data_and_labels(data, labels, processed_dir):
    """
    Store the dataset and label

    Parameters
    ----------
    processed_dir : Path
        Path to the directory where the final datasets are stored
    data : np.array, shape (len(image_paths), height, width, channels)
        The images as numpy array
    labels : np.array, shape (len(image_paths,))
        The corresponding labels
    """
    # Pickle the data and labels
    if not processed_dir.is_dir():
        processed_dir.mkdir(parents=True, exist_ok=True)

    data_path = processed_dir.joinpath('data.pkl')
    with data_path.open('wb') as f:
        pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
        print('[INFO] Saved to {}'.format(data_path))

    labels_path = processed_dir.joinpath('labels.pkl')
    with labels_path.open('wb') as f:
        pickle.dump(labels, f, pickle.HIGHEST_PROTOCOL)
        print('[INFO] Saved to {}'.format(labels_path))


def get_data_split(data, labels):
    """
    Splits the data in train and test (or validation)

    Parameters
    ----------
    data : np.array, shape (n_images, height, width, channels)
        The images as a numpy array
    labels : np.array, shape (n_images,)
        The corresponding labels

    Returns
    -------
    x_train : np.array, shape (n_train, height, width, channels)
        The training data
    x_test : np.array, shape (n_test, height, width, channels)
        The test or validation data
    y_train : np.array, shape (n_train, n_classes)
        The training labels
    y_test : np.array, shape (n_test, n_classes)
        The test or validation labels
    """

    # Partition the data into training and testing splits using 75% of
    # the data for training and the remaining 25% for validation
    (x_train, x_test, y_train, y_test) = \
        train_test_split(data, labels, test_size=0.25, random_state=42)

    return x_train, x_test, y_train, y_test


def get_processed_data(image_paths,
                       model_files_dir,
                       model_name,
                       processed_dir):
    """
    Returns the processed data

    Parameters
    ----------
    image_paths : list
        List of Paths of the image paths
    model_files_dir : Path
        Path to the model_files
    model_name : str
        Name of the model
    processed_dir : Path
        Path to the processed directory

    Returns
    -------
    x_train : np.array, shape (n_train, height, width, channels)
        The training data
    x_val : np.array, shape (n_val, height, width, channels)
        The validation data
    test_data: np.array, shape (n_test, height, width, channels)
        The test data
    y_train : np.array, shape (n_train, n_classes)
        The training labels
    y_val : np.array, shape (n_val, n_classes)
        The validation labels
    test_labels : np.array, shape (n_test, n_classes)
        The test labels
    """

    if processed_dir.joinpath('y_val.pkl').is_file():
        x_train, x_val, x_test, y_train, y_val, y_test =\
            load_data_sets(processed_dir)
    else:
        data, labels = get_data_and_labels(image_paths)
        store_data_and_labels(data, labels, processed_dir)

        encoded_labels = encode_labels(labels,
                                       model_files_dir,
                                       model_name)
        train_data, x_test, train_labels, y_test = \
            get_data_split(data, encoded_labels)

        x_train, x_val, y_train, y_val = \
            get_data_split(train_data, train_labels)

        store_data_sets(x_train,
                        x_val,
                        x_test,
                        y_train,
                        y_val,
                        y_test,
                        processed_dir)

    return x_train, x_val, x_test, y_train, y_val, y_test


def store_data_sets(x_train,
                    x_val,
                    x_test,
                    y_train,
                    y_val,
                    y_test,
                    processed_dir):
    """
    Saves and returns the train and the test

    Parameters
    ----------
    x_train : np.array, shape (n_train, height, width, channels)
        The training data
    x_val : np.array, shape (n_val, height, width, channels)
        The validation data
    x_test: np.array, shape (n_test, height, width, channels)
        The test data
    y_train : np.array, shape (n_train, n_classes)
        The training labels
    y_val : np.array, shape (n_val, n_classes)
        The validation labels
    y_test : np.array, shape (n_test, n_classes)
        The test labels
    processed_dir : Path
        Path to the processed directory
    """

    data_sets = (x_train, x_val, x_test, y_train, y_val, y_test)
    names = ('x_train', 'x_val', 'x_test', 'y_train', 'y_val', 'y_test')

    for data_set, name in zip(data_sets, names):
        path = processed_dir.joinpath(f'{name}.pkl')
        with path.open('wb') as f:
            pickle.dump(data_set, f, pickle.HIGHEST_PROTOCOL)
            print('[INFO] Saved to {}'.format(path))


def load_data_sets(processed_dir):
    """
    Saves and returns the train and the test

    Parameters
    ----------
    processed_dir : Path
        Path to the processed directory

    Returns
    -------
    x_train : np.array, shape (n_train, height, width, channels)
        The training data
    x_val : np.array, shape (n_val, height, width, channels)
        The validation data
    x_test: np.array, shape (n_test, height, width, channels)
        The test data
    y_train : np.array, shape (n_train, n_classes)
        The training labels
    y_val : np.array, shape (n_val, n_classes)
        The validation labels
    y_test : np.array, shape (n_val, n_classes)
        The test labels
    """

    names = ('x_train', 'x_val', 'x_test', 'y_train', 'y_val', 'y_test')
    data_sets = list()

    for name in names:
        path = processed_dir.joinpath(f'{name}.pkl')
        with path.open('rb') as f:
            data_sets.append(pickle.load(f))

    x_train, x_val, x_test, y_train, y_val, y_test = data_sets

    return x_train, x_val, x_test, y_train, y_val, y_test


def encode_labels(labels, model_files_dir, model_name):
    """
    Encode the labels

    Parameters
    ----------
    labels : np.array, shape (n_labels,)
        The labels to encode
    model_files_dir : Path
        Path to the model_files
    model_name : str
        Name of the model

    Returns
    -------
    encoded_labels : np.array, shape (n_labels, n_classes)
        The encoded labels
    """

    label_encoder = OneHotEncoder()
    labels = labels.reshape(len(labels), 1)
    label_encoder.fit(labels)
    encoded_labels = label_encoder.transform(labels).toarray()
    encoder_dir = model_files_dir.joinpath('encoders', model_name)

    if not encoder_dir.is_dir():
        encoder_dir.mkdir(parents=True, exist_ok=True)
    encoder_path = encoder_dir.joinpath('encoder.pkl')

    with encoder_path.open('wb') as f:
        pickle.dump(label_encoder, f, pickle.HIGHEST_PROTOCOL)
        print('[INFO] Saved to {}'.format(encoder_path))

    return encoded_labels


def get_model(n_classes, model_setup=None, optimizer_setup=None):
    """
    Returns a compiled model

    Parameters
    ----------
    n_classes : int
        Number of classes to use in the model
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
        Currently only `initial_learning_rate` and `epochs` can be set

    Returns
    -------
    model : Sequential
        The compiled model
    """

    print('[INFO] compiling model...')

    if model_setup is None:
        model_setup = dict(model_type='leNet',
                           width=28,
                           height=28,
                           channels=3)
    elif 'model_type' not in model_setup.keys():
        # If only width, height, channels is given
        model_setup['model_type'] = 'leNet'

    model_type = model_setup.pop('model_type')

    model_setup['classes'] = n_classes
    model = ModelFactory.create_model(model_type, model_setup)

    if optimizer_setup is None:
        optimizer_setup = dict(initial_learning_rate=1e-3,
                               epochs=25)

    initial_learning_rate = optimizer_setup['initial_learning_rate']
    epochs = optimizer_setup['epochs']

    opt = Adam(lr=initial_learning_rate,
               decay=initial_learning_rate / epochs)

    model.compile(loss='categorical_crossentropy',
                  optimizer=opt,
                  metrics=['accuracy'])

    return model


def train_model(model,
                image_generator,
                model_files_dir,
                x_train,
                y_train,
                x_val,
                y_val,
                model_name='basic',
                batch_size=32,
                epochs=25):
    """
    Trains and saves the model

    Parameters
    ----------
    model : Sequential
        The model to train
    image_generator : ImageDataGenerator
        The image data generator to use
    model_files_dir : Path
        Directory to store the model
    x_train : np.array, shape (n_train, height, width, channels)
        The training data
    x_val : np.array, shape (n_val, height, width, channels)
        The validation data
    y_train : np.array, shape (n_train,)
        The training labels
    y_val : np.array, shape (n_val,)
        The validation labels
    model_name : str
        Name of the model
    batch_size : int
        The batch size
    epochs : int
        The number of epochs

    Returns
    -------
    history : History
        History object containing
        - loss
        - val_loss
        - accuracy
        - val_accuracy
    """

    print('[INFO] Training network...')

    train_batch_ratio = len(x_train) // batch_size
    steps_per_epoch = train_batch_ratio if train_batch_ratio > 0 else 1

    history = \
        model.fit_generator(image_generator.flow(x_train,
                                                 y_train,
                                                 batch_size=batch_size),
                            validation_data=(x_val, y_val),
                            steps_per_epoch=steps_per_epoch,
                            epochs=epochs,
                            verbose=1)

    # Save the model to disk
    print('[INFO] Serializing network...')

    model_dir = model_files_dir.joinpath('models', model_name)
    model_path = model_dir.joinpath('model.h5')

    if not model_dir.is_dir():
        model_dir.mkdir(parents=True, exist_ok=True)

    model.save(str(model_path))
    print('[INFO] Saved to {}'.format(model_path))

    return history


def plot_training(history, plot_dir, plot_name='last'):
    """
    Plots the training loss and accuracy

    The plot is saved in the 'reports/figures' directory

    Parameters
    ----------
    history : History
        History object containing
        - loss
        - val_loss
        - accuracy
        - val_accuracy
    plot_dir : Path
        Directory where to store the plot
    plot_name : str
        Name of plot
    """

    plt.style.use('ggplot')
    plt.figure()
    n_epochs = np.arange(0, len(history.history['loss']))
    plt.plot(n_epochs, history.history['loss'], label='Training '
                                                      'loss')
    plt.plot(n_epochs, history.history['val_loss'], label='Validation '
                                                          'loss')
    plt.plot(n_epochs, history.history['accuracy'],
             label='Training accuracy')
    plt.plot(n_epochs, history.history['val_accuracy'],
             label='Validation accuracy')
    plt.title('Training Loss and Accuracy for fruit classifier')
    plt.xlabel('Epoch #')
    plt.ylabel('Loss/Accuracy')
    plt.legend(loc='lower left')

    if not plot_dir.is_dir():
        plot_dir.mkdir(parents=True, exist_ok=True)

    plot_path = plot_dir.joinpath(f'{plot_name}_training_history.png')

    plt.savefig(str(plot_path))
    print('[INFO] Saved to {}'.format(plot_path))
