from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense


def get_lenet(height, width, channels, classes):
    """
    Implementation of a LeNet like architecture with input (h, w, c)

    Parameters
    ----------
    height : int
        Pixel height of the image
    width : int
        Pixel width of the image
    channels : int
        Number of channels
    classes : int
        Number of prediction classes

    Returns
    -------
    model : Sequential
        The network architecture

    References
    ----------
    LeCun et al. - Gradient-Based Learning Applied to Document
    Recognition
    http://yann.lecun.com/exdb/publis/pdf/lecun-01a.pdf
    """
    model = Sequential()
    input_shape = (height, width, channels)

    model.add(Conv2D(20,
                     (5, 5),
                     input_shape=input_shape,
                     padding='same',
                     activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2),
                           strides=(2, 2)))

    model.add(Conv2D(50,
                     (5, 5),
                     padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2), 
                           strides=(2, 2)))

    model.add(Flatten())
    model.add(Dense(500, activation='relu'))
    model.add(Dense(classes, activation='softmax'))

    return model
