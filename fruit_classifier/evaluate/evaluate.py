import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
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
    plot_dir : Path
        Path to the plotting directory
    plot_name : str
        Name of the plot
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

    conf_mat = confusion_matrix(y_pred_sorted, y_true_sorted)

    class_names = set(y_pred_sorted)
    class_names.update(y_true_sorted)
    class_names = sorted(class_names)

    if normalize:
        conf_mat = conf_mat.astype('float') / \
                   conf_mat.sum(axis=1)[:, np.newaxis]

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(conf_mat, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)

    ax.set(xticks=np.arange(conf_mat.shape[1]),
           yticks=np.arange(conf_mat.shape[0]),
           xticklabels=class_names,
           yticklabels=class_names,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')
    ax.grid(False)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(),
             rotation=45,
             ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = conf_mat.max() / 2.
    for i in range(conf_mat.shape[0]):
        for j in range(conf_mat.shape[1]):
            ax.text(j, i, format(conf_mat[i, j], fmt),
                    ha="center",
                    va="center",
                    color="white" if conf_mat[i, j] > thresh
                    else "black")
    fig.tight_layout()

    plot_path = plot_dir.joinpath(f'{plot_name}_confusion.png')

    fig.savefig(str(plot_path))
    print('[INFO] Saved to {}'.format(plot_path))

    return plot_path, fig, ax


def get_y_true_and_y_pred_sorted(model_files_dir,
                                 model_name,
                                 y_true_inv,
                                 y_pred_inv):
    """
    Returns the sorted version of the ground truth and prediction

    Parameters
    ----------
     model_files_dir : Path
        Path to the model_files
    model_name : str
        Name of the model
    y_true_inv : np.array, shape (n, n_classes)
        The ground truth labels
    y_pred_inv : np.array, shape (n, n_classes)
        The predicted labels

    Returns
    -------
    y_true_sorted : np.array, shape (n,)
        The sorted ground truth labels
    y_pred_sorted : np.array, shape (n,)
        The predicted labels sorted after y_true_sorted
    """

    y_true_inv = inverse_encode(y_true_inv, model_files_dir, model_name)
    y_pred_inv = inverse_encode(y_pred_inv, model_files_dir, model_name)

    # Sort in alphabetical order
    y_true_sorted, y_pred_sorted =\
        zip(*sorted(zip(y_true_inv.ravel(), y_pred_inv.ravel())))

    return tuple(y_true_sorted), tuple(y_pred_sorted)
