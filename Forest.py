from Classifier import Classifier
from DecisionTree import DecisionTree
from NaiveBayes import NaiveBayes
import pandas as pd
import numpy as np
import random

class Forest(Classifier):
    def __init__(self, ntrees, nbayes, pruneSplit=0.0):
        super().__init__()
        self.classifiers = []
        for i in range(ntrees):
            self.classifiers.append(DecisionTree(pruneSplit=pruneSplit))
        for i in range(nbayes):
            self.classifiers.append(NaiveBayes())

    def fit(self, trainSet, trainLab):
        for i in range(len(self.classifiers)):
            tset, tlab = self.bagData(trainSet, trainLab)
            tset = self.chooseFeatures(tset)
            self.classifiers[i].fit(tset, tlab)

    def predict(self, testSet):
        votes = pd.DataFrame(index=testSet.index)
        for i in range(len(self.classifiers)):
            res = self.classifiers[i].predict(testSet)
            res.name += str(i)
            votes = votes.join(res)
        voteCount = votes.apply(pd.value_counts, axis=1)

        return voteCount.idxmax(axis=1)
    def prune(self, pruneSet, pruneLab):
        for i in range(len(self.classifiers)):
            self.classifiers[i].prune(pruneSet, pruneLab)



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





