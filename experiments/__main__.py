from experiments.recipe import ex as ex_
from experiments.utils import reset_config
from experiments.utils import get_configuration
from experiments.utils import add_to_sacred


def run_experiment(ex, experiment_file='first_experiment.ini'):
    """
    Sets up the experiment and run it them

    Parameters
    ----------
    ex : Experiment
        The experiment object to use
    experiment_file : str
        The name of the experiment file which contains the parameters of
        the experiment.
        The file must be found in experiment_files/
    """

    config = get_configuration(experiment_file)

    ex = reset_config(ex)
    ex.path = experiment_file

    # Add the information to sacred
    for section in config.keys():
        add_to_sacred(ex, config[section])

    ex.run()


if __name__ == '__main__':
    run_experiment(ex_)
