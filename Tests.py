##
# @author Rafal Litka
# @file Tests.py

from DecisionTree import DecisionTree
from Forest import Forest
from sklearn.metrics import confusion_matrix

## Test drzewa decyzyjnego na podanych danych
# @param pruneSplit czesc zbioru treningowego, ktora ma zostac uzyta do przyciecia drzewa
# @param visualizeName Nazwa pliku, w ktorym zostanie umieszczona wizualizacja drzewa
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
    print()

## Test Naiwnego klasyfikatora bayesowskiego na podanych danych
def nbcTest(trainSet, trainLab, testSet, testLab):
    nbc = DecisionTree()
    nbc.fit(trainSet, trainLab)
    predLab = nbc.predict(testSet)
    print("nbc errors:")
    print(sum(testLab != predLab))
    print(1-sum(testLab != predLab)/len(testLab))
    print("nbc confusion matrix:")
    print(confusion_matrix(testLab, predLab))
    print()

## Test Lasu losowego na podanych danych
# @param ntree ilosc drzew decyzyjnych w lesie
# @param nbayes ilosc klasyfikatorow NBC w lesie
# @param pruneTreeSplit czesc zbioru treningowego, ktora ma zostac uzyta do przyciecia drzewa
def forestTest(trainSet, trainLab, testSet, testLab, ntree, nbayes, pruneTreeSplit = 0.0):
    forest = Forest(ntree, nbayes, pruneSplit=pruneTreeSplit)
    forest.fit(trainSet, trainLab)
    predLab = forest.predict(testSet)
    print("forest (trees: {}, bayes:{}) errors:".format(ntree, nbayes))
    print(sum(testLab != predLab))
    print(1-sum(testLab != predLab)/len(testLab))
    print("forest (trees: {}, bayes:{}) confusion matrix:".format(ntree, nbayes))
    print(confusion_matrix(testLab, predLab))
    print()
