
import pandas as pd
from NaiveBayes import NaiveBayes
from DecisionTree import DecisionTree
from GetData import getData
from Forest import Forest
import numpy as np
from getCreditData import getCreditData
from sklearn.metrics import confusion_matrix
# np.random.seed(1)
# tree = DecisionTree()
# trainSet, testSet, trainLab, testLab = getData()
#tree.fit(trainSet, trainLab)
#tlab = tree.predict(testSet)
#bayes = NaiveBayes()
#bayes.fit(trainSet, trainLab)
#tlab = bayes.predict(testSet)
#
# forest = Forest(6,6)
# forest.fit(trainSet, trainLab)
# tlab = forest.predict(testSet)
# print(sum(testLab != tlab))
trainSet, testSet, trainLab, testLab = getCreditData()
forest = DecisionTree()
forest.fit(trainSet, trainLab)
# tlab = forest.predict(testSet)
# tlab2 = forest.predict(trainSet)
# print(sum(trainLab != tlab2))
# print(sum(testLab != tlab))
# forest = NaiveBayes()
# forest.fit(trainSet, trainLab)
# tlab = forest.predict(testSet)
# tlab2 = forest.predict(trainSet)
# print(tlab.describe())
# print(tlab2.describe())
# forest = Forest(10,0)
# forest.fit(trainSet, trainLab)
#
tlab = forest.predict(testSet)
tlab2 = forest.predict(trainSet)
with pd.option_context('display.max_rows', None, 'display.max_columns',
                       None):  # more options can be specified also
    print(tlab2)
    print(tlab)
print(confusion_matrix(trainLab,tlab2))
print(confusion_matrix(testLab, tlab))






