import itertools
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
from fruit_classifier.predict.predict_utils import inverse_encode


def plot_confusion_matrix(y_true_sorted,
                          y_pred_sorted,
                          plot_dir,
                          plot_name='last',
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues,
                          figsize=None):
    """
    Plots the confusion matrix

    Parameters
    ----------
    y_true_sorted : array-like
        The ground truth sorted alphabetically
    y_pred_sorted : array-like
        The predicted values sorted alphabetically
    normalize : bool
        Whether to have absolute count or percentage
    title : str
        The title of the plot
    cmap : Colormap
        The colormap to use
    figsize : None or tuple
        Width and height of the figure

    Returns
    -------
    plot_path : Path
        Path to the stored figure
    fig : Figure
        The figure
    ax : Axes
        The axes

    References
    ----------
    http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html#sphx-glr-auto-examples-model-selection-plot-confusion-matrix-py
    https://matplotlib.org/gallery/images_contours_and_fields/image_annotated_heatmap.html#sphx-glr-gallery-images-contours-and-fields-image-annotated-heatmap-py
    """
    sns.set(style='darkgrid')

    conf_mat = confusion_matrix(y_pred_sorted, y_true_sorted)

    class_names = set(y_pred_sorted)
    class_names.update(y_true_sorted)
    class_names = sorted(class_names)

    if normalize:
        conf_mat = conf_mat.astype('float') / \
                   conf_mat.sum(axis=1)[:, np.newaxis]

    fig, ax = plt.subplots(figsize=figsize)

    im_ax = ax.imshow(conf_mat,
                      interpolation='nearest',
                      cmap=cmap)

    fig.suptitle(title)
    _ = fig.colorbar(im_ax)

    # Make proper ticks
    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))
    ax.set_xticklabels(class_names, rotation=90)
    ax.set_yticklabels(class_names)

    # Add text
    fmt = '.2f' if normalize else 'd'
    thresh = conf_mat.max() / 2.
    for i, j in itertools.product(range(conf_mat.shape[0]),
                                  range(conf_mat.shape[1])):
        ax.text(j, i, format(conf_mat[i, j], fmt),
                horizontalalignment="center",
                color="white" if conf_mat[i, j] > thresh else "black")

    ax.set_ylabel('Predicted label')
    ax.set_xlabel('True label')
    ax.grid(False)
    fig.tight_layout()

    plot_path = plot_dir.joinpath(f'{plot_name}_confusion.png')

    fig.savefig(str(plot_path))
    print('[INFO] Saved to {}'.format(plot_path))

    return plot_path, fig, ax


def get_y_true_and_y_pred_sorted(model_files_dir,
                                 model_name,
                                 y_true,
                                 y_pred):
    """
    Returns the sorted version of the ground truth and prediction

    Parameters
    ----------
     model_files_dir : Path
        Path to the model_files
    model_name : str
        Name of the model
    y_true : np.array, shape (n, n_classes)
        The ground truth labels
    y_pred : np.array, shape (n, n_classes)
        The predicted labels

    Returns
    -------
    y_true_sorted : np.array, shape (n,)
        The sorted ground truth labels
    y_pred_sorted : np.array, shape (n,)
        The predicted labels sorted after y_true_sorted
    """

    y_true = inverse_encode(y_true, model_files_dir, model_name)
    y_pred = inverse_encode(y_pred, model_files_dir, model_name)

    # Sort in alphabetical order
    y_true_sorted, y_pred_sorted =\
        zip(*sorted(zip(y_true, y_pred)))

    return tuple(y_true_sorted), tuple(y_pred_sorted)
