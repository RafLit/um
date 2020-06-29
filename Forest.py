from Classifier import Classifier
from DecisionTree import DecisionTree
from NaiveBayes import NaiveBayes
import pandas as pd
import numpy as np
import random

class Forest(Classifier):
    def __init__(self, ntrees, nbayes):
        super().__init__()
        self.classifiers = []
        for i in range(ntrees):
            self.classifiers.append(DecisionTree())
        for i in range(nbayes):
            self.classifiers.append(NaiveBayes())

    def fit(self, trainSet, trainLab):
        for i in range(len(self.classifiers)):
            tset, tlab = self.bagData(trainSet, trainLab)
            tset = self.chooseFeatures(tset)
            print(tset, tlab)
            self.classifiers[i].fit(tset, tlab)

    def predict(self, testSet):
        votes = pd.DataFrame(index=testSet.index)
        for i in range(len(self.classifiers)):
            res = self.classifiers[i].predict(testSet)
            print(res)
            res.name += str(i)
            votes = votes.join(res)
        with pd.option_context('display.max_rows', None, 'display.max_columns',
                               None):  # more options can be specified also
            print(votes)
        voteCount = votes.apply(pd.value_counts, axis=1)

        return voteCount.idxmax(axis=1)



    def bagData(self, trainSet, trainLab, part=1):
        n = trainSet.shape[0]
        randindex = np.random.randint(0,n,int(part*n))
        newTrainSet = trainSet.iloc[randindex]
        newTrainLab = trainLab.iloc[randindex]

        return newTrainSet, newTrainLab

    def chooseFeatures(self, trainSet):
        nfeatures = trainSet.shape[1]
        i = np.array(range(nfeatures))
        i = np.random.permutation(i)
        i = i[:nfeatures//2]
        newTrainSet = trainSet.iloc[:,i]
        return newTrainSet





