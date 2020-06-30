
from GetData import getData
import numpy as np
import random
from getCreditData import getCreditData
from Tests import treeTest, nbcTest, forestTest
if __name__ == '__main__':
    random.seed(1)
    np.random.seed(1)
    trainSet, testSet, trainLab, testLab = getData(split=0.2)
    treeTest(trainSet, trainLab, testSet, testLab, visualizeName="grzyby.pdf")
    nbcTest(trainSet, trainLab, testSet, testLab)

    trainSet, testSet, trainLab, testLab = getCreditData(split=0.25)
    treeTest(trainSet, trainLab, testSet, testLab, visualizeName="kredyt.pdf")
    treeTest(trainSet, trainLab, testSet, testLab, pruneSplit=0.3, visualizeName="kredytPruned.pdf")
    nbcTest(trainSet, trainLab, testSet, testLab)
    forestTest(trainSet,trainLab,testSet,testLab, 5, 0)
    forestTest(trainSet,trainLab,testSet,testLab, 5, 0, pruneTreeSplit=0.3)
    forestTest(trainSet,trainLab,testSet,testLab, 0, 5)
    forestTest(trainSet,trainLab,testSet,testLab, 5, 5)
    forestTest(trainSet,trainLab,testSet,testLab, 5, 5,pruneTreeSplit= 0.3)
    forestTest(trainSet,trainLab,testSet,testLab, 10, 10)
    forestTest(trainSet,trainLab,testSet,testLab, 10, 10,pruneTreeSplit= 0.3)
    forestTest(trainSet,trainLab,testSet,testLab, 20, 20)






