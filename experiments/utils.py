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
