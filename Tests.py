import pandas as pd
from NaiveBayes import NaiveBayes
from DecisionTree import DecisionTree
from GetData import getData
from Forest import Forest
import numpy as np
import random
from getCreditData import getCreditData
from sklearn.metrics import confusion_matrix
from sklearn import tree
from sklearn import preprocessing

def treeTest(trainSet, trainLab, testSet, testLab, pruneSplit = 0,  visualizeName = None):
    tree = DecisionTree(pruneSplit = pruneSplit)
    tree.fit(trainSet, trainLab)
    predLab = tree.predict(testSet)
    if visualizeName:
        tree.visualize(fileName=visualizeName)
    print("tree errors:")
    print(sum(testLab != predLab))
    print(1-(sum(testLab != predLab)/len(testLab)))
    print("tree confusion matrix:")
    print(confusion_matrix(testLab, predLab))

def nbcTest(trainSet, trainLab, testSet, testLab):
    nbc = DecisionTree()
    nbc.fit(trainSet, trainLab)
    predLab = nbc.predict(testSet)
    print("nbc errors:")
    print(sum(testLab != predLab))
    print(1-sum(testLab != predLab)/len(testLab))
    print("nbc confusion matrix:")
    print(confusion_matrix(testLab, predLab))

def forestTest(trainSet, trainLab, testSet, testLab, ntree, nbayes, pruneTreeSplit = 0.0):
    forest = Forest(ntree, nbayes, pruneSplit=pruneTreeSplit)
    forest.fit(trainSet, trainLab)
    predLab = forest.predict(testSet)
    print("forest (trees: {}, bayes:{}) errors:".format(ntree, nbayes))
    print(sum(testLab != predLab))
    print(1-sum(testLab != predLab)/len(testLab))
    print("forest (trees: {}, bayes:{}) confusion matrix:".format(ntree, nbayes))
    print(confusion_matrix(testLab, predLab))
