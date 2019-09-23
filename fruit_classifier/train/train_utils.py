import pickle
import numpy as np
from pathlib import Path
from tqdm import tqdm
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import to_categorical
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from fruit_classifier.models.factory import ModelFactory
from fruit_classifier.utils.image_utils import open_image


def get_data_and_labels(image_paths, processed_dir):
    """
    Returns the data and the labels from the input paths

    Parameters
    ----------
    image_paths : list
        List of Paths of the image paths
    processed_dir : Path
        Path to the directory where the final datasets are stored

    Returns
    -------
    data : np.array, shape (len(image_paths), height, width, channels)
        The images as numpy array
    labels : np.array, shape (len(image_paths,)
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

    return data, labels


def get_model_input(data, labels, model_files_dir, model_name):
    """
    Returns the input to the model

    Parameters
    ----------
    data : np.array, shape (n_images, height, width, channels)
        The images as a numpy array
    labels : np.array, shape (n_images,)
        The corresponding labels
    model_files_dir : Path
        Path to the model files
    model_name : str
        Name of model

    Returns
    -------
    x_train : np.array, shape (n_train, height, width, channels)
        The training data
    x_val : np.array, shape (n_val, height, width, channels)
        The validation data
    y_train : np.array, shape (n_train,)
        The training labels
    y_val : np.array, shape (n_val,)
        The validation labels
    """

    label_encoder = LabelEncoder()
    label_encoder.fit(labels)
    encoded_labels = label_encoder.transform(labels)

    encoder_dir = model_files_dir.joinpath('encoders', model_name)
    if not encoder_dir.is_dir():
        encoder_dir.mkdir(parents=True, exist_ok=True)

    encoder_path = encoder_dir.joinpath('encoder.pkl')

    with encoder_path.open('wb') as f:
        pickle.dump(label_encoder, f, pickle.HIGHEST_PROTOCOL)
        print('[INFO] Saved to {}'.format(encoder_path))

    num_classes = len(set(labels))

    # Partition the data into training and testing splits using 75% of
    # the data for training and the remaining 25% for testing
    (x_train, x_val, y_train, y_val) = train_test_split(data,
                                                        encoded_labels,
                                                        test_size=0.25,
                                                        random_state=42)

    # Convert the labels from integers to vectors
    y_train = to_categorical(y_train, num_classes=num_classes)
    y_val = to_categorical(y_val, num_classes=num_classes)

    return x_train, x_val, y_train, y_val


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

    # FIXME: Enable setup knobs outside of just initial_learning rate
    #        and epochs
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
        - acc
        - val_acc
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


def plot_training(history, plot_dir):
    """
    Plots the training loss and accuracy

    The plot is saved in the 'reports/figures' directory

    Parameters
    ----------
    history : History
        History object containing
        - loss
        - val_loss
        - acc
        - val_acc
    plot_dir : Path
        Directory where to store the plot
    """

    plt.style.use('ggplot')
    plt.figure()
    n_epochs = np.arange(0, len(history.history['loss']))
    plt.plot(n_epochs, history.history['loss'], label='Training '
                                                      'loss')
    plt.plot(n_epochs, history.history['val_loss'], label='Validation '
                                                          'loss')
    plt.plot(n_epochs, history.history['acc'],
             label='Training accuracy')
    plt.plot(n_epochs, history.history['val_acc'], label='Validation '
                                                         'accuracy')
    plt.title('Training Loss and Accuracy for fruit classifier')
    plt.xlabel('Epoch #')
    plt.ylabel('Loss/Accuracy')
    plt.legend(loc='lower left')

    if not plot_dir.is_dir():
        plot_dir.mkdir(parents=True, exist_ok=True)

    plot_path = plot_dir.joinpath('training_history.png')

    plt.savefig(str(plot_path))
    print('[INFO] Saved to {}'.format(plot_path))
