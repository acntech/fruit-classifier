import argparse
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
    # Construct the argument parse and parse the arguments
    parser = argparse.ArgumentParser(description='Run an experiment')
    parser.add_argument('-e',
                        '--experiment_file_name',
                        required=False,
                        default=None,
                        help='Name of experiment file (located in '
                             'experiment_files/)')

    args = parser.parse_args()

    if args.experiment_file_name is None:
        experiment_file_ = 'first_experiment.ini'
    else:
        experiment_file_ = args.experiment_file_name

    run_experiment(ex_, experiment_file_)
