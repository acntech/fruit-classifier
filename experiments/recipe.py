from sacred import Experiment
from sacred.observers import MongoObserver
from sacred.utils import apply_backspaces_and_linefeeds
from fruit_classifier.preprocessing.__main__ import main as pre_main
from fruit_classifier.train.__main__ import main as train_main
from experiments.utils import get_configuration


# Add experiments to mongodb
ex = Experiment()
#ex.observers.append(
#    MongoObserver.create(url='host.docker.internal:27017',
#                         db_name='fruit_classifier'))

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
    train_main(config['preprocessing']['dataset_name'],
               config['train']['model_name'],
               model_setup,
               config['optimizer_setup'])

    # cohens_kappa, confusion_path, confidence_path = \
    #     run_evaluate_with_config(data_src,
    #                              img_preprocessing_option,
    #                              ocr_type,
    #                              ocr_option,
    #                              txt_processing_option,
    #                              dataset_type,
    #                              dataset_option,
    #                              classifier_type,
    #                              classifier_option)
    #
    # # Add to sacred
    # ex.add_artifact(str(confusion_path), name=confusion_path.name)
    # ex.add_artifact(str(confidence_path), name=confidence_path.name)
    # ex.log_scalar('cohens_kappa', cohens_kappa)
