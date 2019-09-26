import ast
import configparser
from pathlib import Path


def reset_config(ex):
    """
    Resets the configuration of an experiment

    This needs to be done when experiments are being performed
    as part of a for loop in order to avoid residues

    Parameters
    ----------
    ex : Experiment
        The experiment object to reset

    Returns
    -------
    ex : Experiment
        The cleaned experiment object
    """
    for i in range(len(ex.configurations)):
        ex.configurations[i]._conf = {}
    return ex


def add_to_sacred(ex, options):
    """
    Adds the parameters of the input dictionary to sacred

    Notes
    -----
    The configuration added here is not used in ex.main, as ex.main will
    unwrap these configurations on demand

    Parameters
    ----------
    ex : Experiment
        The experiment to add the config to
    options : dict
        The classifier option to use
        The options are stored in a json file under
        /config/classifier/classifier_type/
    """

    for keyword, value in options.items():
        ex.add_config({keyword: value})


def get_configuration(experiment_file):
    """
    Returns the configuration

    Parameters
    ----------
    experiment_file : str
        The name of the experiment file which contains the parameters of
        the experiment.
        The file must be found in experiment_files/

    Returns
    -------
    config_dict : dict
        A dictionary based on the read ConfigParser object
    """

    # Setup the directories
    experiment_files = Path(__file__).parents[1].absolute(). \
        joinpath('experiment_files')
    experiment_file_path = experiment_files.joinpath(experiment_file)

    # Parse the configurations
    config = configparser.ConfigParser()
    config.read(experiment_file_path)

    # Guess the type
    config_dict = dict()
    for section in config.sections():
        config_dict[section] = dict()
        for keyword in config[section].keys():
            config_dict[section][keyword] = \
                ast.literal_eval(config[section][keyword])

    return config_dict


def log_history(ex, history):
    """
    Logs the history after the run

    For real time logging, see for example [1]_

    Parameters
    ----------
    ex : Experiment
        The experiment object to use
    history : History
        The training History object containing
        - loss
        - val_loss
        - accuracy
        - val_accuracy

    References
    ----------
    [1]
    https://www.hhllcks.de/blog/2018/5/4/version-your-machine-learning-models-with-sacred
    """

    h = history.history

    for loss, acc, val_loss, val_acc in zip(h['loss'],
                                            h['accuracy'],
                                            h['val_loss'],
                                            h['val_accuracy']):
        ex.log_scalar('loss', loss)
        ex.log_scalar('acc', acc)
        ex.log_scalar('val_loss', val_loss)
        ex.log_scalar('val_acc', val_acc)
