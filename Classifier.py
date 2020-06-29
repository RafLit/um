"""@package docstring
Documentation for this module.

More details.
"""
class Classifier:
    """Documentation for a class.

        More details.
    """
    def __init__(self):
        """Documentation for a method."""
        pass

    def predict(self, testSet):
        """Documentation for a methodMore details."""
        raise NotImplementedError

    ## Documentation for a method.
    # @param trainSet test
    def fit(self, trainSet, trainLab):
        raise NotImplementedError