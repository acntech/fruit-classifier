"""
Contains the Model Factory class
"""

from fruit_classifier.models.models import lenet


class ModelFactory(object):
    """
    Factory which creates the models
    """

    @staticmethod
    def create_model(name, setup=None):
        """
        Creates an model

        Parameters
        ----------
        name : str
            Name of the model to use
            Current available models:
            - leNet
        setup : dict or None
            Dict of the setup
            See the various implementations of the engines for details

        Returns
        -------
        model : model-like
            The model must have a fit and predict method
        """

        if setup is None:
            # Avoid mutable default input arguments
            setup = dict()

        implemented = ('leNet',)

        if name == 'leNet':
            model = lenet(**setup)
        else:
            msg = f'{name} is not a valid model, choose from ' \
                  f'{implemented}'
            raise NotImplementedError(msg)

        return model
