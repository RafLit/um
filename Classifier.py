##
# @author Rafal Litka
# @file Classifier.py


## Klasa bazowa klasyfikatora
class Classifier:
    def __init__(self):
        pass

    ## testowanie klasyfikatora
    # @param testSet zbiór testowy
    def predict(self, testSet):
        raise NotImplementedError

    ## trenowanie klasyfikatora
    # @param trainSet zbiór treningowy
    # @param trainLab etykiety dla zbioru treningowego
    def fit(self, trainSet, trainLab):
        raise NotImplementedError