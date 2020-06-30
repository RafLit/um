##
# @author Rafal Litka
# @file NaiveBayes.py

from Classifier import Classifier
import pandas as pd

## Naiwny klasyfikator Bayesa
class NaiveBayes(Classifier):
    def __init__(self):
        super().__init__()
        # prawdopodobieństwo warunkowe P(X|Y)
        self.PXWhenClass = {}
        # prawdopodobieństwo P(Y)
        self.Pclass = pd.Series()

    def fit(self, trainSet, trainLab):
        ClassCounts = trainLab.value_counts()
        sumClassCounts = ClassCounts.sum()
        for feature in trainSet.columns:
            self.PXWhenClass[feature] = pd.DataFrame(columns=trainLab.unique())
            for cls in trainLab.unique():
                reducedCol = trainSet.loc[trainLab==cls, feature]
                col = trainSet.loc[:, feature]
                valueCounts = reducedCol.value_counts()
                valueCounts = {val:valueCounts[val] if val in reducedCol.unique() else 0 for val in col.unique()}
                for valName, valCount in valueCounts.items():
                    self.PXWhenClass[feature].loc[valName, cls] = valCount / ClassCounts[cls]
        self.Pclass = ClassCounts / sumClassCounts

    def predict(self, testSet):
        n = testSet.shape[0]
        classProbs = pd.concat([self.Pclass]*n, axis=1, ignore_index=True).T
        for i in range(n):
            sample = testSet.iloc[i]
            for feature in testSet.columns:
                # jezeli cechy nie bylo cechy w zbiorze treningowym, to nie jest uwzgledniana
                if feature not in self.PXWhenClass.keys():
                    pass
                elif sample[feature] not in self.PXWhenClass[feature].index:
                    classProbs.iloc[i] = 0
                    continue
                else:
                    # obliczenie P(Y|X) na podstawie wzoru Bayesa
                    classProbs.iloc[i] *= self.PXWhenClass[feature].loc[sample[feature], :]
                    classProbs.iloc[i] /= sum(classProbs.iloc[i])
        classRes = classProbs.idxmax(axis=1)
        classRes.name = 'class'
        classRes.index = testSet.index
        return classRes


