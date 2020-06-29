from Classifier import Classifier
import math
import numpy as np
import pandas as pd
class TreeElement(Classifier):

    def __init__(self):
        super().__init__()
        self.splitFeature = ''
        self.branches = {}

class TreeNode(TreeElement):
    def __init__(self):
        super().__init__()
        self.subElements = []


    def fit(self, trainSet, trainLab):
        entropies = {feature:0 for feature in trainSet.columns}
        for feature in trainSet.columns:
            column = trainSet[feature]
            for value in column.unique():
                reducedSet = trainSet.loc[column==value,:]
                reducedLab = trainLab[column==value]
                entropy = 0.
                for count in reducedLab.value_counts():

                    x = count/len(reducedLab)
                    if not (x == 0. or x == 1.):
                        entropy -= x*math.log2(x)
                entropies[feature] += len(reducedLab)/len(trainLab)*entropy
        self.splitFeature = min(entropies,key=entropies.get)
        column = trainSet[self.splitFeature]
        for value in trainSet[self.splitFeature].unique():
            reducedSet = trainSet.loc[column == value, :]
            reducedLab = trainLab[column == value]
            entropy = 0.
            for count in reducedLab.value_counts():
                x = count / len(reducedLab)
                if not (x == 0. or x == 1.):
                    entropy -= x * math.log2(x)
            if entropy == 0.0 or len(trainSet.columns) == 1:
                self.branches[value] = TreeLeaf()
                self.branches[value].fit(reducedSet, reducedLab)
            else:
                self.branches[value] = TreeNode()
                self.branches[value].fit(reducedSet.drop(labels=[self.splitFeature],axis='columns'), reducedLab)


    def predict(self, testSet):
        values = testSet.loc[:,self.splitFeature].unique()
        tLab = pd.Series(None,index=testSet.index, name='class')
        print(tLab)
        for value in [val for val in values if val in self.branches]:
            reducedSet = testSet.loc[testSet.loc[:,self.splitFeature] == value,:]
            tLab[testSet[self.splitFeature] == value] = self.branches[value].predict(reducedSet)
        return tLab


class TreeLeaf(TreeElement):

    def __init__(self):
        super().__init__()
        self.decision = None

    def fit(self, trainSet, trainLab):
        counts = trainLab.value_counts()
        self.decision = counts.index[counts.argmax()]
        print(self.decision)

    def predict(self, testSet):
        print(self.decision)
        return self.decision

class DecisionTree(Classifier):

    def __init__(self):
        super().__init__()
        self.rootNode = TreeNode()

    def fit(self, trainSet, trainLab):
        self.rootNode.fit(trainSet, trainLab)

    def predict(self, testSet):
        return self.rootNode.predict(testSet)
