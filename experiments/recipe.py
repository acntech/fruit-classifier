from pathlib import Path
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import accuracy_score
from sacred import Experiment
from sacred.observers import MongoObserver
from sacred.utils import apply_backspaces_and_linefeeds
from fruit_classifier.predict.predict_utils import load_classifier
from fruit_classifier.predict.predict_utils import classify_many
from fruit_classifier.evaluate.evaluate import \
    get_y_true_and_y_pred_sorted
from fruit_classifier.evaluate.evaluate import plot_confusion_matrix
from fruit_classifier.preprocessing.__main__ import main as pre_main
from fruit_classifier.train.__main__ import main as train_main
from experiments.utils import get_configuration
from experiments.utils import log_history


# Add experiments to mongodb
ex = Experiment()
ex.observers.append(
    MongoObserver.create(
        url=f'mongodb://sample:password@localhost:27017/?authMechanism=SCRAM-SHA-1',
        db_name='db'))

# Avoid linefeed capture
ex.captured_out_filter = apply_backspaces_and_linefeeds


@ex.main
def experiment_recipe(**_):
    """
    Runs a classifier experiment

    Notes
    -----
    The final **_ is used to capture parts of the configuration which is
    just added to the omniboard.
    See classifier_experiments_runner.add_ocr_options_to_omniboard for details

    Parameters
    ----------
    experiment_file : str
        The name of the experiment file which contains the parameters of
        the experiment.
        The file must be found in experiment_files/
    """

    # NOTE: *arg and **kwarg cannot be used to capture the
    #       configurations given in the caller routine
    #       As we prefer .ini files over @ex.config,
    #       the configuration is read anew (__main__.run_experiment
    #       just adds the configuration to sacred)
    config = get_configuration(ex.path)

    pre_main(config['preprocessing']['dataset_name'],
             config['preprocessing']['height'],
             config['preprocessing']['width'])

    model_setup = {'model_type': config['train']['model_type'],
                   **config['model_setup']}

    model_name = config['train']['model_name']

    history, test_data, test_labels = \
        train_main(config['preprocessing']['dataset_name'],
                   model_name,
                   model_setup,
                   config['optimizer_setup'])

    # Setup paths
    root_dir = Path(__file__).absolute().parents[1]
    model_files_dir = root_dir.joinpath('model_files')
    plot_dir = root_dir.joinpath('reports', 'figures', model_name)

    # Run prediction
    model = load_classifier(model_files_dir, model_name)
    y_pred, _ = classify_many(model, test_data)

    # Run evaluation
    y_true_sorted, y_pred_sorted =\
        get_y_true_and_y_pred_sorted(model_files_dir,
                                     model_name,
                                     test_labels,
                                     y_pred)

    cm_path = plot_confusion_matrix(y_true_sorted,
                                    y_pred_sorted,
                                    plot_dir,
                                    plot_name=model_name)

    # Add to sacred
    log_history(ex, history)
    ex.add_artifact(str(cm_path), name=cm_path.name)
    ex.log_scalar('cohens_kappa',
                  cohen_kappa_score(y_true_sorted, y_pred_sorted))

    # Will appear as "Result in omniboard"
    return accuracy_score(y_true_sorted, y_pred_sorted)

